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

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1, rgbaColor=[0.5, 0.5, 0.5, 1.0])

robot = p.loadURDF("../3D_models/gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

# --- 2. SERIAL INPUT THREAD ---
serial_state = {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "end": 0.0, "button": 0}

def serial_listener(port="/dev/ttyUSB0", baud=115200):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    try:
                        y, s, e, en, b = line.split(",")
                        serial_state["yaw"] = float(y)
                        serial_state["shoulder"] = float(s)
                        serial_state["elbow"] = float(e)
                        serial_state["end"] = float(en)
                        serial_state["button"] = int(b)
                    except: pass
        except: time.sleep(1.0)

threading.Thread(target=serial_listener, daemon=True).start()

# --- 3. LOGGING & CAMERA SETUP ---
joints = [0, 1, 2, 3, 4, 5]
pos = [0.0] * 6
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

EYE, TARGET, UP = [0.75, -0.75, 0.75], [0.15, 0.0, 0.15], [0, 0, 1]
last_log_time = time.time()

def get_frame():
    view = p.computeViewMatrix(EYE, TARGET, UP)
    proj = p.computeProjectionMatrixFOV(45, 1.0, 0.05, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(500, 500, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (500, 500, 4))[:, :, :3], cv2.COLOR_RGB2BGR)

# --- 4. MAIN LOOP ---
while True:
    # 1. Map Serial to Radians
    pos[0] = math.radians(serial_state["yaw"])
    pos[1] = math.radians(serial_state["shoulder"])
    pos[2] = -math.radians(serial_state["elbow"])
    pos[3] = math.radians(serial_state["end"])
    pos[4], pos[5] = (0.5, -0.5) if serial_state["button"] == 1 else (0.0, 0.0)

    for i in range(6):
        pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

    p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos)
    p.stepSimulation()

    # 2. LOGGING EVERY 0.5 SECONDS
    current_time = time.time()
    if current_time - last_log_time >= 0.5:
        # We add 90 to the degree conversion so that 'Reset' (0 rad) displays as 90 deg
        log_angles = [round(math.degrees(p_val) + 90, 1) for p_val in pos[:4]]
        grip_val = 1 if serial_state["button"] == 1 else 0
        print(f"Angles: Yaw={log_angles[0]}°, Shld={log_angles[1]}°, Elb={log_angles[2]}°, Wrist={log_angles[3]}° | Grip={grip_val}")
        last_log_time = current_time

    # 3. Visuals
    cv2.imshow("Monitoring", get_frame())
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()