import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import pybullet_data
from torchvision import models

# --- 1. MODEL ARCHITECTURE (Must match qtrain.py exactly) ---
class VLAActionPredictor(nn.Module):
    def __init__(self):
        super(VLAActionPredictor, self).__init__()
        # 1. Vision Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Policy Head (Must match qtrain.py exactly)
        self.policy_head = nn.Sequential(
            nn.Linear(1024, 512),      # layer 0
            nn.ReLU(),                 # layer 1
            nn.Dropout(0.1),           # layer 2 (This was missing or in a different spot)
            nn.Linear(512, 128),       # layer 3
            nn.ReLU(),                 # layer 4
            nn.Linear(128, 6)          # layer 5
        )

    def forward(self, goal_img, state_img):
        goal_emb = torch.flatten(self.vision_backbone(goal_img), 1)
        state_emb = torch.flatten(self.vision_backbone(state_img), 1)
        combined = torch.cat((goal_emb, state_emb), dim=1)
        return self.policy_head(combined)

# --- 2. LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VLAActionPredictor().to(device)
model.load_state_dict(torch.load("vla_model.pth", map_location=device))
model.eval()
print(f"VLA Inference Engine Active on {device}")

# --- 3. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1, rgbaColor=[0.6, 0.6, 0.6, 1.0])
robot = p.loadURDF("../3D_models/gripper_arm.urdf", [0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", [0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 1, 1])

def get_sim_frame():
    # Identical camera settings used during data collection
    view = p.computeViewMatrix([0.75, -0.75, 0.75], [0.15, 0.0, 0.15], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(45, 1.0, 0.05, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(224, 224, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (224, 224, 4))[:, :, :3]
    # Normalize and convert to Tensor for AI
    tensor = torch.tensor(frame.transpose((2, 0, 1)) / 255.0, dtype=torch.float32).unsqueeze(0)
    return frame, tensor.to(device)

# --- 4. VIDEO SETUP ---
cap_demo = cv2.VideoCapture("../demo_vids/demo4.webm")
fps = cap_demo.get(cv2.CAP_PROP_FPS) or 30
target_dt = 1.0 / fps

# --- 5. INFERENCE LOOP ---
print("Starting AI Control...")
try:
    while True:
        start_time = time.time()
        
        ret, demo_frame = cap_demo.read()
        if not ret: 
            cap_demo.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue
        
        # Prepare Demo Frame Tensor
        demo_resized = cv2.resize(demo_frame, (224, 224))
        demo_tensor = torch.tensor(demo_resized.transpose((2, 0, 1)) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get Live Sim Frame and Tensor
        sim_raw, sim_tensor = get_sim_frame()
        
        # AI PREDICTION
        with torch.no_grad():
            # Model takes (Goal, Current_State)
            predicted_angles = model(demo_tensor, sim_tensor).squeeze().cpu().numpy()
        
        # APPLY TO ROBOT
        p.setJointMotorControlArray(
            robot, range(6), p.POSITION_CONTROL, 
            targetPositions=predicted_angles
        )
        p.stepSimulation()
        
        # VISUAL FEEDBACK
        sim_disp = cv2.cvtColor(sim_raw, cv2.COLOR_RGB2BGR)
        combined = np.hstack((cv2.resize(demo_frame, (224, 224)), sim_disp))
        cv2.putText(combined, "GOAL (Demo)", (10, 20), 0, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "AI ACTION (Sim)", (234, 20), 0, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("VLA Inference - Goal vs Action", combined)
        
        # Maintain original FPS
        elapsed = time.time() - start_time
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap_demo.release()
    cv2.destroyAllWindows()
    p.disconnect()