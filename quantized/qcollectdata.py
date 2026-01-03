import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
import serial
import csv
import os
from datetime import datetime

# --- 1. CONFIGURATION & FOLDERS ---
DEMO_PATH = "../demo_vids/demo4.webm"
SAVE_DIR = "sim_recs"
DATA_LOG_DIR = "joint_logs"
RENDER_W, RENDER_H = 224, 224 
SPEED_MULTIPLIER = 0.5  # Half speed for easier imitation

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_LOG_DIR, exist_ok=True)

# State storage for Potentiometer Arm
serial_state = {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "end": 0.0, "button": 0}

# --- 2. SERIAL LISTENER ---
def serial_listener(port="/dev/ttyUSB0", baud=115200):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=0.1)
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    try:
                        parts = line.split(",")
                        if len(parts) == 5:
                            serial_state.update({
                                "yaw": float(parts[0]), "shoulder": float(parts[1]), 
                                "elbow": float(parts[2]), "end": float(parts[3]), "button": int(parts[4])
                            })
                    except: pass
        except: time.sleep(1.0)

threading.Thread(target=serial_listener, daemon=True).start()

# --- 3. PYBULLET SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

# Grey floor, no grid
plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1, rgbaColor=[0.6, 0.6, 0.6, 1.0])

robot = p.loadURDF("../3D_models/gripper_arm.urdf", [0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", [0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 1, 1]) 

def get_sim_frame():
    # Camera view from the side
    view = p.computeViewMatrix([0.75, -0.75, 0.75], [0.15, 0.0, 0.15], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(45, 1.0, 0.05, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3], cv2.COLOR_RGB2BGR)

# --- 4. PRE-RECORDING PREP ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(SAVE_DIR, f"sim_{timestamp}.mp4")
csv_filename = os.path.join(DATA_LOG_DIR, f"joints_{timestamp}.csv")

cap_demo = cv2.VideoCapture(DEMO_PATH)
original_fps = cap_demo.get(cv2.CAP_PROP_FPS) or 30
target_dt = (1.0 / original_fps) / SPEED_MULTIPLIER 

# Simulation Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_sim = cv2.VideoWriter(video_filename, fourcc, original_fps, (RENDER_W, RENDER_H))

print(f"Ready. Saving to: {video_filename}")
print("Press ENTER to start recording...")
input()

# --- 5. COLLECTION LOOP ---
all_joint_data = []
start_time = time.time()
frame_idx = 0

while True:
    ret, demo_frame = cap_demo.read()
    if not ret: break
    
    # Update Robot with Pot-Arm
    pos = [math.radians(serial_state["yaw"]), math.radians(serial_state["shoulder"]), 
           -math.radians(serial_state["elbow"]), math.radians(serial_state["end"]),
           0.5 if serial_state["button"] == 1 else 0.0, -0.5 if serial_state["button"] == 1 else 0.0]
    
    p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=pos)
    p.stepSimulation()
    
    # Capture & Write Sim Video
    sim_frame = get_sim_frame()
    out_sim.write(sim_frame)
    
    # Store Joint Angles
    current_joints = [p.getJointState(robot, j)[0] for j in range(6)]
    all_joint_data.append(current_joints)
    
    # Display Side-by-Side Feedback
    demo_disp = cv2.resize(demo_frame, (RENDER_W, RENDER_H))
    combined = np.hstack((demo_disp, sim_frame))
    cv2.putText(combined, "REC", (10, 20), 0, 0.5, (0, 0, 255), 2)
    cv2.imshow("Imitation Training", combined)
    cv2.waitKey(1)
    
    # Sync Timing
    frame_idx += 1
    expected_time = start_time + (frame_idx * target_dt)
    while time.time() < expected_time:
        time.sleep(0.001)

# --- 6. CLEANUP & SAVE CSV ---
cap_demo.release()
out_sim.release()
cv2.destroyAllWindows()

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["joint_angles"])
    for row in all_joint_data:
        writer.writerow([row])

print(f"\nSaved Video: {video_filename}")
print(f"Saved Data: {csv_filename}")
p.disconnect()