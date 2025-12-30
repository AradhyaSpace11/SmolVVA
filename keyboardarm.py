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

plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

robot = p.loadURDF("gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

# High friction for fingers to help with gripping
p.changeDynamics(robot, 4, lateralFriction=2.0)
p.changeDynamics(robot, 5, lateralFriction=2.0)

# --- 2. CONTROL & CAMERA SETUP ---
joints = [0, 1, 2, 3, 4, 5] 
# Start positions: Last two indices are gripper fingers, set to gripped (0.5, -0.5)
pos = [0.0, 0.0, 0.0, 0.0, 0.5, -0.5]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

RENDER_W, RENDER_H = 400, 400
SKIP_FRAMES = 5
frame_counter = 0

def get_frame(eye, target, up):
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 5.0)
    _, _, rgb, _, _ = p.getCameraImage(RENDER_W, RENDER_H, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return cv2.cvtColor(np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:,:,:3], cv2.COLOR_RGB2BGR)

# --- 3. MAIN LOOP ---
print("Keyboard Control Active!")
print("Z/X: Base | Q/A: Shoulder | E/D: Elbow | R/F: Wrist | C: Ungrip")

while True:
    # Get all keyboard events
    keys = p.getKeyboardEvents()
    
    move_speed = 0.02
    grip_speed = 0.05

    # 1. Handle Arm Movement (Check if key is held down)
    if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: pos[0] -= move_speed
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: pos[0] += move_speed
    
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: pos[1] += move_speed
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN: pos[1] -= move_speed
    
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: pos[2] += move_speed
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN: pos[2] -= move_speed
    
    if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN: pos[3] += move_speed
    if ord('f') in keys and keys[ord('f')] & p.KEY_IS_DOWN: pos[3] -= move_speed

    # 2. Gripper Logic (Default is Gripped)
    if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
        # Ungrip (Open)
        pos[4] = max(pos[4] - grip_speed, 0.0)
        pos[5] = min(pos[5] + grip_speed, 0.0)
    else:
        # Default State: Gripped (Closed)
        pos[4] = min(pos[4] + grip_speed, 0.5)
        pos[5] = max(pos[5] - grip_speed, -0.5)

    # Clamp all positions to URDF limits
    for i in range(6):
        pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

    # 3. Physics Step
    p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=pos, forces=[100,100,100,100,40,40])
    p.stepSimulation()
    frame_counter += 1

    # 4. Camera Updates (Every 5 frames)
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