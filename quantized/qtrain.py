import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
import cv2
import os
from torchvision import models

# --- 1. THE VLA-STYLE MODEL ---
class VLAActionPredictor(nn.Module):
    def __init__(self):
        super(VLAActionPredictor, self).__init__()
        # Using ResNet18 as the Vision Encoder (Embeddings)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Remove classification layer
        
        # Freeze backbone to preserve pre-trained features
        for param in self.vision_backbone.parameters():
            param.requires_grad = False

        # Fusion & Action Head
        # 512 (Goal Features) + 512 (State Features) = 1024
        self.policy_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # Outputs 6 Joint Angles
        )

    def forward(self, goal_img, state_img):
        # Extract 512-dim embeddings for both images
        goal_emb = torch.flatten(self.vision_backbone(goal_img), 1)
        state_emb = torch.flatten(self.vision_backbone(state_img), 1)
        
        # Concatenate (The VLA 'Thought' process)
        combined = torch.cat((goal_emb, state_emb), dim=1)
        return self.policy_head(combined)

# --- 2. DATASET LOADER ---
class VLADataset(Dataset):
    def __init__(self, csv_file, goal_video, sim_video):
        self.data = pd.read_csv(csv_file)
        self.goal_video = goal_video
        self.sim_video = sim_video
        
        # Open videos to get total frame counts
        cap_g = cv2.VideoCapture(goal_video)
        cap_s = cv2.VideoCapture(sim_video)
        self.goal_len = int(cap_g.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sim_len = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_g.release()
        cap_s.release()

    def __len__(self):
        return len(self.data)

    def process_frame(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: return np.zeros((3, 224, 224))
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose((2, 0, 1)) / 255.0 # HWC to CHW
        return frame

    def __getitem__(self, idx):
        # 1. Map CSV index to video frames (Temporal Alignment)
        g_idx = int((idx / len(self)) * (self.goal_len - 1))
        s_idx = int((idx / len(self)) * (self.sim_len - 1))
        
        cap_g = cv2.VideoCapture(self.goal_video)
        cap_s = cv2.VideoCapture(self.sim_video)
        
        goal_frame = self.process_frame(cap_g, g_idx)
        state_frame = self.process_frame(cap_s, s_idx)
        
        cap_g.release()
        cap_s.release()

        # 2. Get target joint angles
        joints = np.array(ast.literal_eval(self.data.iloc[idx]['joint_angles']))
        
        return (torch.tensor(goal_frame, dtype=torch.float32), 
                torch.tensor(state_frame, dtype=torch.float32), 
                torch.tensor(joints, dtype=torch.float32))

# --- 3. TRAINING CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VLAActionPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Update these paths to your actual files
dataset = VLADataset(
    csv_file="joint_logs/joints_20260101_005629.csv", 
    goal_video="../demo_vids/demo4.webm", 
    sim_video="../sim_recs/sim_20260101_005629.mp4"
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# --- 4. TRAINING LOOP ---
print(f"Training on {device}...")
model.train()
for epoch in range(50):
    epoch_loss = 0
    for goals, states, targets in loader:
        goals, states, targets = goals.to(device), states.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(goals, states)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.6f}")

# --- 5. SAVE ---
torch.save(model.state_dict(), "vla_model.pth")
print("Model saved as vla_model.pth")