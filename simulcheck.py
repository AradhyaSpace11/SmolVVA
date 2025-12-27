import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Set up the plain grey world
plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, textureUniqueId=-1) # Removes chess board
p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1]) # Sets grey color

# Load Robot and Cube
robot = p.loadURDF("gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1]) # Blue Cube

# Physics Tweaks: Higher friction makes it MUCH easier to pick things up
p.changeDynamics(robot, 4, lateralFriction=2.0) # Left Finger
p.changeDynamics(robot, 5, lateralFriction=2.0) # Right Finger
p.changeDynamics(cube_id, -1, lateralFriction=1.0)

# --- 2. CONTROL SETUP ---
joints = [0, 1, 2, 3, 4, 5] 
pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Speed Optimizations
RENDER_W, RENDER_H = 400, 400
SKIP_FRAMES = 5
frame_counter = 0

def get_frame(eye, target, up):
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:,:,:3], cv2.COLOR_RGB2BGR)

# --- 3. MAIN LOOP ---
while True:
    keys = p.getKeyboardEvents()
    if ord('z') in keys and (keys[ord('z')] & p.KEY_WAS_TRIGGERED): break

    step = 0.02
    # Arm Controls (WASD / QERF)
    if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN): pos[0] -= step
    if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN): pos[0] += step
    if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN): pos[1] += step
    if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN): pos[1] -= step
    if ord('q') in keys and (keys[ord('q')] & p.KEY_IS_DOWN): pos[2] -= step
    if ord('e') in keys and (keys[ord('e')] & p.KEY_IS_DOWN): pos[2] += step
    if ord('r') in keys and (keys[ord('r')] & p.KEY_IS_DOWN): pos[3] += step
    if ord('f') in keys and (keys[ord('f')] & p.KEY_IS_DOWN): pos[3] -= step

    # Gripper Controls (T to Close / G to Open)
    if ord('t') in keys and (keys[ord('t')] & p.KEY_IS_DOWN):
        pos[4] = min(pos[4] + 0.05, 0.5)
        pos[5] = max(pos[5] - 0.05, -0.5)
    if ord('g') in keys and (keys[ord('g')] & p.KEY_IS_DOWN):
        pos[4] = max(pos[4] - 0.05, 0.0)
        pos[5] = min(pos[5] + 0.05, 0.0)

    p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos, forces=[100,100,100,100,40,40])
    p.stepSimulation()
    frame_counter += 1

    # Camera Updates
    if frame_counter % SKIP_FRAMES == 0:
        link_state = p.getLinkState(robot, 2)
        h = link_state[0][2]
        ang = pos[0] + (math.pi / 2) # North-facing orbit
        
        side_img = get_frame([1.2*math.cos(ang), 1.2*math.sin(ang), h], [0,0,h], [0,0,1])
        top_img = get_frame([0.1, 0, 1.2], [0.1, 0, 0], [0, 1, 0])
        
        combined_view = np.vstack((side_img, top_img))
        cv2.imshow("Robot Monitoring", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()