import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import numpy as np

# --- 1. DATASET SETUP ---
class KeypointDataset(Dataset):
    def __init__(self, json_file, video_path):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.cap = cv2.VideoCapture(video_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame'])
        ret, frame = self.cap.read()
        
        # Resize to 224x224 for the model
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (224, 224))
        img = frame.transpose(2, 0, 1) / 255.0 # CHW format
        
        # Normalize labels to [0, 1]
        pts = np.array(item['points'], dtype=np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        return torch.FloatTensor(img), torch.FloatTensor(pts.flatten())

# --- 2. THE MODEL (ResNet based) ---
class KeypointDetector(nn.Module):
    def __init__(self):
        super(KeypointDetector, self).__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 14) # 7 points * 2 (x,y)

    def forward(self, x):
        return self.backbone(x)

# --- 3. TRAINING LOOP ---
dataset = KeypointDataset('training_data.json', 'demo_vids/demo3.webm')
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = KeypointDetector()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("Training started...")
for epoch in range(50): # Adjust epochs as needed
    for imgs, labels in loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'keypoint_model.pth')
print("Model Saved!")