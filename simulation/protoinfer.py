import time
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import cv2
import numpy as np
import math

# --- 1. MODEL ARCHITECTURE ---
# Must match your prototrain.py architecture exactly
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

# --- 2. INITIALIZE AND LOAD MODEL ---
model = RobotBrain()
try:
    model.load_state_dict(torch.load("robot_model.pth"))
    model.eval()
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure your RobotBrain class matches your training script!")
    exit()

# --- 3. CONFIGURATION ---
VIDEO_PATH = "../demo_vids/demo4.webm"
RENDER_W, RENDER_H = 500, 500
KEYPOINT_NAMES = ["Marker", "Elbow", "Wrist", "Gripper", "Prong1", "Prong2", "Object"]
demo_points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(demo_points) < 7:
        demo_points.append([float(x), float(y)])
        print(f"Marked: {KEYPOINT_NAMES[len(demo_points)-1]}")

# --- 4. PYBULLET SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Cleaner view

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1, rgbaColor=[0.5, 0.5, 0.5, 1.0])
robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

# --- 5. VIDEO MARKING STAGE ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 30
target_frame_time = 1.0 / fps

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video.")
    exit()

first_frame = cv2.resize(first_frame, (RENDER_W, RENDER_H))
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("AI Vision - Label Demo")
cv2.setMouseCallback("AI Vision - Label Demo", on_mouse)

print("STEP 1: Mark the 7 keypoints on the video. Press ENTER to start AI Inference.")
while len(demo_points) < 7:
    img = first_frame.copy()
    for i, pt in enumerate(demo_points):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        cv2.putText(img, KEYPOINT_NAMES[i], (int(pt[0]), int(pt[1])-10), 0, 0.4, (0, 255, 0), 1)
    cv2.imshow("AI Vision - Label Demo", img)
    if cv2.waitKey(1) == ord('q'): exit()

cv2.waitKey(0) # Press ENTER to confirm labels
cv2.destroyWindow("AI Vision - Label Demo")

# --- 6. AI INFERENCE & CONTROL LOOP (SYNCED) ---
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
cv2.namedWindow("AI Running")

print(f"AI Running at {fps} FPS. Press 'q' to stop.")

while True:
    start_time = time.time() # Start timing this frame
    
    ret, frame = cap.read()
    if not ret: 
        print("End of video stream.")
        break
    
    frame = cv2.resize(frame, (RENDER_W, RENDER_H))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Computer Vision: Track the robot in the video
    p0 = np.array(demo_points, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    demo_points = p1.reshape(-1, 2).tolist()
    old_gray = frame_gray.copy()

    # 2. AI Processing: Pixels to Angles
    input_vector = np.array(demo_points).flatten()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        predicted_angles = model(input_tensor).squeeze().numpy()

    # 3. Physics Control: Execute moves in simulation
    p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=predicted_angles)
    p.stepSimulation()

    # UI Feedback
    for pt in demo_points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
    cv2.imshow("AI Running", frame)
    
    # --- TIME SYNC: Prevent "Sped Up" look ---
    elapsed = time.time() - start_time
    if elapsed < target_frame_time:
        time.sleep(target_frame_time - elapsed)

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
p.disconnect()