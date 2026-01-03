import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import numpy as np
import torchvision.models as models

# --- 1. DATASET WITH GAUSSIAN HEATMAPS ---
class HeatmapDataset(Dataset):
    def __init__(self, json_file, video_path, img_size=224, sigma=5):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.cap = cv2.VideoCapture(video_path)
        self.img_size = img_size
        self.sigma = sigma # Controls the 50x50 "spread"

    def create_heatmap(self, center_x, center_y, w, h):
        # Create a 2D Gaussian heatmap
        grid_y, grid_x = np.mgrid[0:self.img_size, 0:self.img_size]
        # Scale coordinates to img_size
        mu_x = center_x * (self.img_size / w)
        mu_y = center_y * (self.img_size / h)
        
        d2 = (grid_x - mu_x)**2 + (grid_y - mu_y)**2
        heatmap = np.exp(-d2 / (2 * self.sigma**2))
        return heatmap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame'])
        ret, frame = self.cap.read()
        
        h_orig, w_orig = frame.shape[:2]
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0

        # Create 7 heatmaps (one for each point)
        heatmaps = []
        for pt in item['points']:
            heatmaps.append(self.create_heatmap(pt[0], pt[1], w_orig, h_orig))
        
        return torch.FloatTensor(img), torch.FloatTensor(np.array(heatmaps))

# --- 2. THE MODEL (Fully Convolutional) ---
class KeypointHeatmapNet(nn.Module):
    def __init__(self):
        super(KeypointHeatmapNet, self).__init__()
        # Backbone: ResNet minus the global pooling and FC layers
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Decoder: Upsample features back to 224x224 heatmap
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14->28
            nn.ReLU(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 7, kernel_size=3, padding=1) # 7 Output channels
        )

    def forward(self, x):
        x = self.features(x)
        return self.decoder(x)

# --- 3. TRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = HeatmapDataset('training_data.json', 'demo_vids/demo3.webm')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = KeypointHeatmapNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss() # Compare predicted heatmap to Gaussian target

print(f"Training on {device}...")
for epoch in range(100):
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'heatmap_model.pth')
print("Model saved as heatmap_model.pth")