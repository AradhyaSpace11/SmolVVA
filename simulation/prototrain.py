import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
import numpy as np

# --- 1. DATA PREPROCESSING ---
print("Loading data from training_data.csv...")
df = pd.read_csv("training_data.csv")

def parse_list(string):
    # Safely convert strings like "[[x,y]...]" or "[j1, j2...]" to flat arrays
    data = ast.literal_eval(string)
    return np.array(data).flatten()

X = np.array([parse_list(row) for row in df['demo_points']])
y = np.array([parse_list(row) for row in df['joint_angles']])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# --- 2. MODEL DEFINITION ---
class RobotBrain(nn.Module):
    def __init__(self):
        super(RobotBrain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.net(x)

model = RobotBrain()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. TRAINING LOOP ---
epochs = 50000 # Increased for better accuracy
print(f"Training AI Model for {epochs} epochs...")

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}")

# --- 4. SAVE MODEL ---
torch.save(model.state_dict(), "robot_model.pth")
print("\nSuccess: Model saved as 'robot_model.pth'")