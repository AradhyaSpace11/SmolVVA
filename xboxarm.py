import time
import pybullet as p
import pybullet_data
import numpy as np
from inputs import get_gamepad
import threading

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
robot = p.loadURDF("gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)

# --- 2. CONTROLLER STATE HANDLER ---
# We use a dictionary to keep track of stick positions in the background
state = {
    "LX": 0, "LY": 0, "RX": 0, "RY": 0, 
    "LT": 0, "RT": 0, "BTN_B": 0
}

def update_controller():
    """Background thread to read Xbox inputs without freezing the sim"""
    global state
    while True:
        events = get_gamepad()
        for event in events:
            if event.code == "ABS_X": state["LX"] = event.state
            if event.code == "ABS_Y": state["LY"] = event.state
            if event.code == "ABS_RX": state["RX"] = event.state
            if event.code == "ABS_RY": state["RY"] = event.state
            if event.code == "ABS_Z": state["LT"] = event.state
            if event.code == "ABS_RZ": state["RT"] = event.state
            if event.code == "BTN_EAST": state["BTN_B"] = event.state

# Start the background controller thread
threading.Thread(target=update_controller, daemon=True).start()

# --- 3. ARM CONTROL LOGIC ---
JOINT_IDS = [0, 1, 2, 3, 4, 5]
pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.8), (-0.8, 0)]

def norm(val): 
    """Deadzone and normalization"""
    if abs(val) < 4000: return 0
    return val / 32768.0

print("--- Controller Ready! ---")

while True:
    # 1. Update positions based on current stick states
    pos[0] += norm(state["LX"]) * 0.01   # Base Yaw
    pos[1] -= norm(state["LY"]) * 0.01   # Shoulder
    pos[2] -= norm(state["RY"]) * 0.01   # Elbow
    pos[3] += norm(state["RX"]) * 0.01   # Wrist

    # 2. Gripper Triggers
    grip_speed = 0.02
    if state["RT"] > 0: # Close
        pos[4] += grip_speed
        pos[5] -= grip_speed
    if state["LT"] > 0: # Open
        pos[4] -= grip_speed
        pos[5] += grip_speed

    # 3. Reset
    if state["BTN_B"]:
        pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 4. Apply limits and move
    for i in range(6):
        pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

    p.setJointMotorControlArray(robot, JOINT_IDS, p.POSITION_CONTROL, targetPositions=pos)
    
    p.stepSimulation()
    time.sleep(1./120.)