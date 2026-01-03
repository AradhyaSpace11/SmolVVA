import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
from inputs import get_gamepad

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1)
p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

robot = p.loadURDF("gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

p.changeDynamics(robot, 4, lateralFriction=2.0)
p.changeDynamics(robot, 5, lateralFriction=2.0)

# --- 2. XBOX CONTROLLER THREADING ---
# Shared state between controller thread and simulation loop
controller_state = {
    "LX": 0, "LY": 0, "RX": 0, "RY": 0, 
    "LT": 0, "RT": 0, "BTN_B": 0
}

def xbox_listener():
    global controller_state
    while True:
        try:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X": controller_state["LX"] = event.state
                if event.code == "ABS_Y": controller_state["LY"] = event.state
                if event.code == "ABS_RX": controller_state["RX"] = event.state
                if event.code == "ABS_RY": controller_state["RY"] = event.state
                if event.code == "ABS_Z": controller_state["LT"] = event.state
                if event.code == "ABS_RZ": controller_state["RT"] = event.state
                if event.code == "BTN_EAST": controller_state["BTN_B"] = event.state
        except Exception:
            pass

# Start the listener in the background
threading.Thread(target=xbox_listener, daemon=True).start()

# --- 3. CONTROL & CAMERA SETUP ---
joints = [0, 1, 2, 3, 4, 5] 
pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

RENDER_W, RENDER_H = 400, 400
SKIP_FRAMES = 5
frame_counter = 0

def get_frame(eye, target, up):
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:,:,:3], cv2.COLOR_RGB2BGR)

def norm(val, deadzone=4000):
    if abs(val) < deadzone: return 0
    return val / 32768.0

# --- 4. MAIN LOOP ---
print("Controller Active! Use Sticks for Arm, Triggers for Gripper, 'B' to Reset.")

while True:
    # 1. Process Controller Inputs
    move_speed = 0.03
    pos[0] += norm(controller_state["LX"]) * move_speed  # Base
    pos[1] -= norm(controller_state["LY"]) * move_speed  # Shoulder
    pos[2] -= norm(controller_state["RY"]) * move_speed  # Elbow
    pos[3] += norm(controller_state["RX"]) * move_speed  # Wrist

    # Gripper (Triggers are 0-255 or 0-1024 depending on driver)
    if controller_state["RT"] > 10:
        pos[4] = min(pos[4] + 0.02, 0.5)
        pos[5] = max(pos[5] - 0.02, -0.5)
    if controller_state["LT"] > 10:
        pos[4] = max(pos[4] - 0.02, 0.0)
        pos[5] = min(pos[5] + 0.02, 0.0)

    # Reset
    if controller_state["BTN_B"]:
        pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Clamp all positions to URDF limits
    for i in range(6):
        pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

    # 2. Physics Step
    p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos, forces=[100,100,100,100,40,40])
    p.stepSimulation()
    frame_counter += 1

    # 3. Camera Updates (Every 5 frames)
    if frame_counter % SKIP_FRAMES == 0:
        link_state = p.getLinkState(robot, 2)
        h = link_state[0][2]
        ang = pos[0] + (math.pi / 2)
        
        side_img = get_frame([1.2*math.cos(ang), 1.2*math.sin(ang), h], [0,0,h], [0,0,1])
        top_img = get_frame([0.1, 0, 1.2], [0.1, 0, 0], [0, 1, 0])
        
        combined_view = np.vstack((side_img, top_img))
        cv2.imshow("Robot Monitoring", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()