import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from tqdm import tqdm

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, self.embedding_dim)
        distances = (flat_z.pow(2).sum(1, keepdim=True) 
                     + self.embeddings.weight.pow(2).sum(1)
                     - 2 * torch.matmul(flat_z, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), z.permute(0, 3, 1, 2))
        q_latent_loss = F.mse_loss(quantized, z.permute(0, 3, 1, 2).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z.permute(0, 3, 1, 2) + (quantized - z.permute(0, 3, 1, 2)).detach()

        return quantized, loss

class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, embedding_dim, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, 256, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

class CLIPDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_dataset = ImageFolder(image_folder, transform=transform)
        self.text_data = [f"A photo of a {label}" for _, label in self.image_dataset.samples]
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        text = self.text_data[idx]
        return image, text

def train(model, dataloader, optimizer, clip_model, device):
    model.train()
    total_loss = 0.0
    
    for images, texts in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        
        # Forward pass through VQVAE
        output_data, vq_loss = model(images)
        
        # Calculate CLIP loss
        image_features = clip_model.encode_image(output_data)
        text_features = clip_model.encode_text(texts)
        clip_loss = F.cosine_similarity(image_features, text_features).mean()
        
        # Calculate total loss
        recon_loss = F.mse_loss(output_data, images)
        total_loss = recon_loss + vq_loss + clip_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_loss += total_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, clip_model, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, texts in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            
            # Forward pass through VQVAE
            output_data, vq_loss = model(images)
            
            # Calculate CLIP loss
            image_features = clip_model.encode_image(output_data)
            text_features = clip_model.encode_text(texts)
            clip_loss = F.cosine_similarity(image_features, text_features).mean()
            
            # Calculate total loss
            recon_loss = F.mse_loss(output_data, images)
            total_loss = recon_loss + vq_loss + clip_loss
            
            total_loss += total_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # Hyperparameters
    in_channels = 3
    embedding_dim = 512
    num_embeddings = 512
    commitment_cost = 0.25
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize CLIP model
    clip_model = SentenceTransformer('clip-ViT-B-32').to(device)
    
    # Initialize VQVAE model
    model = VQVAE(in_channels, embedding_dim, num_embeddings, commitment_cost).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load dataset
    transform = ToTensor()
    train_dataset = CLIPDataset("path/to/train/images", transform=transform)
    val_dataset = CLIPDataset("path/to/val/images", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, clip_model, device)
        val_loss = evaluate(model, val_dataloader, clip_model, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "vqvae_clip_model.pth")

if __name__ == "__main__":
    main()
