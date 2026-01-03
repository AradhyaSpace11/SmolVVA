import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import pybullet_data
from torchvision import models

# --- 1. ARCHITECTURE (Must match qtrain exactly) ---
class SmolVLA_Policy(nn.Module):
    def __init__(self, num_bins=256):
        super(SmolVLA_Policy, self).__init__()
        self.num_bins = num_bins
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.action_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6 * num_bins) 
        )

    def forward(self, goal_img, state_img):
        g_emb = torch.flatten(self.vision_backbone(goal_img), 1)
        s_emb = torch.flatten(self.vision_backbone(state_img), 1)
        combined = torch.cat((g_emb, s_emb), dim=1)
        logits = self.action_head(combined)
        return logits.view(-1, 6, self.num_bins)

# --- 2. INITIALIZATION ---
device = torch.device("cpu")
model = SmolVLA_Policy()
# Ensure the .pth filename matches what you saved in Colab
model.load_state_dict(torch.load("smol_vla_model.pth", map_location=device))
model.eval()

# Simulation Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.loadURDF("plane.urdf")
robot = p.loadURDF("../3D_models/gripper_arm.urdf", [0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", [0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 1, 1])

# Video & Logic Constants
cap_demo = cv2.VideoCapture("../demo_vids/demo4.webm")
fps = cap_demo.get(cv2.CAP_PROP_FPS) or 30
min_ang, max_ang = -1.7, 1.7 # Must match training exactly
smooth_angles = np.zeros(6)
alpha = 0.2  # Smoothing factor (0.1 = very smooth, 1.0 = raw output)

print("SmolVLA Inference Engine Running...")

# --- 3. INFERENCE LOOP ---
try:
    while True:
        start_t = time.time()
        
        ret, demo_frame = cap_demo.read()
        if not ret:
            cap_demo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Prepare Tensors
        demo_in = cv2.resize(demo_frame, (224, 224)).transpose((2, 0, 1)) / 255.0
        goal_t = torch.tensor(demo_in).float().unsqueeze(0)
        
        # Get Live Sim View
        view = p.computeViewMatrix([0.75, -0.75, 0.75], [0.15, 0.0, 0.15], [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(45, 1.0, 0.05, 5.0)
        _, _, rgb, _, _ = p.getCameraImage(224, 224, view, proj)
        sim_in = np.reshape(rgb, (224, 224, 4))[:, :, :3].transpose((2, 0, 1)) / 255.0
        sim_t = torch.tensor(sim_in).float().unsqueeze(0)

        # AI PREDICTION (Quantized)
        with torch.no_grad():
            logits = model(goal_t, sim_t) # [1, 6, 256]
            best_bins = torch.argmax(logits, dim=2).squeeze() # [6]
            
            # De-quantize: Bin -> Radians
            raw_angles = (best_bins.float() / 255.0) * (max_ang - min_ang) + min_ang
            raw_angles = raw_angles.numpy()

        # 4. FILTERING & CONTROL
        # Apply smoothing to prevent jitter between bins
        smooth_angles = (alpha * raw_angles) + ((1 - alpha) * smooth_angles)
        
        p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, smooth_angles)
        p.stepSimulation()

        # Visual Feedback
        demo_disp = cv2.resize(demo_frame, (224, 224))
        sim_disp = cv2.cvtColor(np.reshape(rgb, (224, 224, 4))[:, :, :3], cv2.COLOR_RGB2BGR)
        combined = np.hstack((demo_disp, sim_disp))
        cv2.putText(combined, "GOAL", (10, 20), 0, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "SMOL-VLA ACTION", (234, 20), 0, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("VLA Quantized Inference", combined)

        # Sync timing
        elapsed = time.time() - start_t
        if elapsed < (1.0 / fps):
            time.sleep((1.0 / fps) - elapsed)
            
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap_demo.release()
    cv2.destroyAllWindows()
    p.disconnect()