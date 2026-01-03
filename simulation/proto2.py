import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
import serial
import csv

# --- 1. CONFIGURATION ---
VIDEO_PATH = "../demo_vids/demo4.webm"
FINAL_LOG = "training_data.csv"
RENDER_W, RENDER_H = 500, 500
KEYPOINT_NAMES = ["Marker", "Elbow", "Wrist", "Gripper", "Prong1", "Prong2", "Object"]

# State storage
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

# --- 3. STAGE 1: DEMO VIDEO LABELING & TRACKING ---
demo_data = [] # Stores keypoints for every frame
demo_points = []

def on_mouse_demo(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(demo_points) < len(KEYPOINT_NAMES):
        demo_points.append([float(x), float(y)])
        print(f"Demo Marked: {KEYPOINT_NAMES[len(demo_points)-1]}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 30 # Fallback
target_period = 1.0 / fps

ret, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (RENDER_W, RENDER_H))
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("STAGE 1: Demo Video")
cv2.setMouseCallback("STAGE 1: Demo Video", on_mouse_demo)

print(f"STEP 1: Label {len(KEYPOINT_NAMES)} points. Press ENTER when done.")
while len(demo_points) < len(KEYPOINT_NAMES):
    img = first_frame.copy()
    for i, pt in enumerate(demo_points):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    cv2.imshow("STAGE 1: Demo Video", img)
    if cv2.waitKey(1) == ord('q'): exit()

cv2.waitKey(0) # Wait for Enter to begin tracking

print("Tracking Demo...")
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (RENDER_W, RENDER_H))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p0 = np.array(demo_points, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    demo_points = p1.reshape(-1, 2).tolist()
    demo_data.append(list(demo_points)) 
    
    old_gray = frame_gray.copy()
    cv2.imshow("STAGE 1: Demo Video", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print(f"Stage 1 Complete. Captured {len(demo_data)} frames.")

# --- 4. STAGE 2: SIMULATION LABELING & RECORDING ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1, rgbaColor=[0.5, 0.5, 0.5, 1.0])
robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

sim_points = []
def on_mouse_sim(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(sim_points) < len(KEYPOINT_NAMES):
        sim_points.append([float(x), float(y)])
        print(f"Sim Marked: {KEYPOINT_NAMES[len(sim_points)-1]}")

def get_sim_frame():
    view = p.computeViewMatrix([0.75, -0.75, 0.75], [0.15, 0.0, 0.15], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(45, 1.0, 0.05, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, 
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=0)
    return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3], cv2.COLOR_RGB2BGR)

cv2.namedWindow("STAGE 2: Simulation")
cv2.setMouseCallback("STAGE 2: Simulation", on_mouse_sim)

print("STEP 2: Label Simulation points. Press ENTER to start countdown.")
while len(sim_points) < len(KEYPOINT_NAMES):
    # Control arm during labeling so you can see where parts are
    pos = [math.radians(serial_state["yaw"]), math.radians(serial_state["shoulder"]), 
           -math.radians(serial_state["elbow"]), math.radians(serial_state["end"]),
           0.5 if serial_state["button"] == 1 else 0.0, -0.5 if serial_state["button"] == 1 else 0.0]
    p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=pos)
    p.stepSimulation()
    
    sim_img = get_sim_frame()
    for i, pt in enumerate(sim_points):
        cv2.circle(sim_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    cv2.imshow("STAGE 2: Simulation", sim_img)
    if cv2.waitKey(1) == ord('q'): exit()

cv2.waitKey(0) # Wait for Enter

# 3-Second Countdown
for i in range(3, 0, -1):
    temp_img = get_sim_frame()
    cv2.putText(temp_img, f"STARTING IN {i}...", (100, 250), 0, 1.2, (0,0,255), 3)
    cv2.imshow("STAGE 2: Simulation", temp_img)
    cv2.waitKey(1000)

# --- 5. DATA COLLECTION LOOP (REAL-TIME SYNC) ---
final_data = []
print("RECORDING...")

for i in range(len(demo_data)):
    loop_start = time.time()
    
    # 1. Physics & Control
    pos = [math.radians(serial_state["yaw"]), math.radians(serial_state["shoulder"]), 
           -math.radians(serial_state["elbow"]), math.radians(serial_state["end"]),
           0.5 if serial_state["button"] == 1 else 0.0, -0.5 if serial_state["button"] == 1 else 0.0]
    
    p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=pos)
    p.stepSimulation()
    
    # 2. Data Capture
    actual_joints = [p.getJointState(robot, j)[0] for j in range(6)]
    final_data.append([demo_data[i], sim_points, actual_joints])
    
    # 3. Visual Feedback (Throttled for performance)
    if i % 2 == 0:
        sim_img = get_sim_frame()
        cv2.putText(sim_img, f"REC: {i}/{len(demo_data)}", (20, 40), 0, 0.7, (0, 255, 0), 2)
        cv2.imshow("STAGE 2: Simulation", sim_img)
        cv2.waitKey(1)

    # 4. SLEEP TO PREVENT "SPED UP" LOOK
    # Ensures the simulation matches the demo video's frame rate
    elapsed = time.time() - loop_start
    if elapsed < target_period:
        time.sleep(target_period - elapsed)

# --- 6. SAVE & EXIT ---
with open(FINAL_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["demo_points", "sim_static_points", "joint_angles"])
    writer.writerows(final_data)

print(f"Data Collection Finished. {len(final_data)} frames saved to {FINAL_LOG}")
p.disconnect()
cv2.destroyAllWindows()