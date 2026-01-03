import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import threading
import serial

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Clean grey floor
plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1)
p.changeVisualShape(plane_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1.0])

robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

# High friction for better gripping
p.changeDynamics(robot, 4, lateralFriction=2.0)
p.changeDynamics(robot, 5, lateralFriction=2.0)

# --- 2. SERIAL INPUT THREAD (ESP32) ---
serial_state = {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "end": 0.0, "button": 0}

def serial_listener(port="/dev/ttyUSB0", baud=115200):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(1.0)
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                try:
                    y, s, e, en, b = line.split(",")
                    serial_state["yaw"] = float(y)
                    serial_state["shoulder"] = float(s)
                    serial_state["elbow"] = float(e)
                    serial_state["end"] = float(en)
                    serial_state["button"] = int(b)
                except Exception:
                    pass
        except Exception:
            time.sleep(1.0)

threading.Thread(target=serial_listener, daemon=True).start()

# --- 3. CONTROL & CAMERA SETUP ---
joints = [0, 1, 2, 3, 4, 5]
pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

# Single diagonal top camera settings
RENDER_W, RENDER_H = 500, 500
SKIP_FRAMES = 5
frame_counter = 0

EYE = [0.75, -0.75, 0.75]   # Camera position
TARGET = [0.15, 0.0, 0.15]  # Camera focus point
UP = [0, 0, 1]
FOV = 45

def get_frame():
    view = p.computeViewMatrix(EYE, TARGET, UP)
    proj = p.computeProjectionMatrixFOV(FOV, 1.0, 0.05, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(
        RENDER_W, RENDER_H, view, proj, 
        shadow=0, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    img = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def deg_to_rad(deg):
    return np.clip(math.radians(deg), -math.pi, math.pi)

# --- 4. MAIN LOOP ---
print("System Active! ESP32 Pots controlling arm.")
print("Single diagonal-top camera view. Press 'q' in CV window to exit.")

while True:
    # 1. Update positions from Serial Data
    pos[0] = deg_to_rad(serial_state["yaw"])
    pos[1] = deg_to_rad(serial_state["shoulder"])
    pos[2] = -deg_to_rad(serial_state["elbow"])
    pos[3] = deg_to_rad(serial_state["end"])

    # 2. Gripper Logic: Close when button pressed
    if serial_state["button"] == 1:
        pos[4], pos[5] = 0.5, -0.5  # Closed
    else:
        pos[4], pos[5] = 0.0, 0.0   # Open

    # 3. Clamp for safety
    for i in range(6):
        pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

    # 4. Physics Step
    p.setJointMotorControlArray(
        robot, joints, p.POSITION_CONTROL, 
        targetPositions=pos, 
        forces=[100, 100, 100, 100, 40, 40]
    )
    p.stepSimulation()
    frame_counter += 1

    # 5. Render Single Camera View
    if frame_counter % SKIP_FRAMES == 0:
        img = get_frame()
        cv2.imshow("Robot Monitoring (Diagonal Top)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()