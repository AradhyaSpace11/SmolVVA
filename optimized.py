import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

# --- Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
robot = p.loadURDF("simple_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)

joints = [0, 1, 2, 3] 
pos = [0.0, 0.0, 0.0, 0.0]

# --- PERFORMANCE TWEAKS ---
RENDER_WIDTH = 480   # Lowering internal resolution for speed
RENDER_HEIGHT = 480
SKIP_FRAMES = 6      # Only update camera every 6th physics step
frame_counter = 0

def get_fast_camera(eye_pos, target_pos, up_vector):
    view_matrix = p.computeViewMatrix(eye_pos, target_pos, up_vector)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=5.0)

    # The shadowMap=0 and renderer flag here are optimized for speed
    _, _, rgb_px, _, _ = p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        shadow=0, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    frame = np.reshape(rgb_px, (RENDER_HEIGHT, RENDER_WIDTH, 4))[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Pre-create windows (makes them faster)
cv2.namedWindow("North Orbit View", cv2.WINDOW_NORMAL)
cv2.namedWindow("Top View", cv2.WINDOW_NORMAL)

while True:
    keys = p.getKeyboardEvents()
    if ord('z') in keys and (keys[ord('z')] & p.KEY_WAS_TRIGGERED): break

    # Movement 
    step = 0.015
    if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN): pos[0] -= step
    if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN): pos[0] += step
    if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN): pos[1] += step
    if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN): pos[1] -= step
    if ord('q') in keys and (keys[ord('q')] & p.KEY_IS_DOWN): pos[2] -= step
    if ord('e') in keys and (keys[ord('e')] & p.KEY_IS_DOWN): pos[2] += step
    if ord('r') in keys and (keys[ord('r')] & p.KEY_IS_DOWN): pos[3] += step
    if ord('f') in keys and (keys[ord('f')] & p.KEY_IS_DOWN): pos[3] -= step

    p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos)
    
    # Physics is always 240Hz
    p.stepSimulation()
    frame_counter += 1

    # --- Only update Cameras every N frames ---
    if frame_counter % SKIP_FRAMES == 0:
        # 1. North Orbit View
        link_state = p.getLinkState(robot, 2)
        h = link_state[0][2]
        # North offset (pi/2)
        orbit_angle = pos[0] + (math.pi / 2)
        eye_north = [1.0 * math.cos(orbit_angle), 1.0 * math.sin(orbit_angle), h]
        
        north_img = get_fast_camera(eye_north, [0, 0, h], [0, 0, 1])
        cv2.imshow("North Orbit View", north_img)

        # 2. Top View
        top_img = get_fast_camera([0.001, 0, 1.5], [0, 0, 0], [0, 1, 0])
        cv2.imshow("Top View", top_img)

    # OpenCV waitKey is slow; only call it when we render
    if frame_counter % SKIP_FRAMES == 0:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()