import time
import pybullet as p
import pybullet_data
import numpy as np
import cv2

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
#p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(
    plane_id,
    -1,
    rgbaColor=[0.5, 0.5, 0.5, 1.0],   # pure grey
    textureUniqueId=-1               # remove texture
)

robot = p.loadURDF("gripper_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0.5, 1, 1])

p.changeDynamics(robot, 4, lateralFriction=2.0)
p.changeDynamics(robot, 5, lateralFriction=2.0)

# --- 2. CONTROL SETUP ---
joints = [0, 1, 2, 3, 4, 5]
pos = [0.0, 0.0, 0.0, 0.0, 0.5, -0.5]
LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]

# --- 3. CAMERA SETUP (single diagonal top view) ---
RENDER_W, RENDER_H = 500, 500
SKIP_FRAMES = 5
frame_counter = 0

# Diagonal top camera (zoomed in)
EYE = [0.75, -0.75, 0.75]   # move closer/farther for zoom
TARGET = [0.15, 0.0, 0.15]  # focus near the arm + cube
UP = [0, 0, 1]

FOV = 45  # smaller = more zoom
ASPECT = 1.0
NEAR = 0.05
FAR = 5.0

def get_frame():
  view = p.computeViewMatrix(EYE, TARGET, UP)
  proj = p.computeProjectionMatrixFOV(FOV, ASPECT, NEAR, FAR)
  _, _, rgb, _, _ = p.getCameraImage(
    RENDER_W, RENDER_H, view, proj,
    shadow=0,
    renderer=p.ER_BULLET_HARDWARE_OPENGL
  )
  img = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3]
  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

print("Keyboard Control Active!")
print("Z/X: Base | Q/A: Shoulder | E/D: Elbow | R/F: Wrist | C: Ungrip")
print("Single diagonal-top camera view. Press q (opencv window) to exit.")

while True:
  keys = p.getKeyboardEvents()
  move_speed = 0.02
  grip_speed = 0.05

  if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN: pos[0] -= move_speed
  if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN: pos[0] += move_speed

  if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN: pos[1] += move_speed
  if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN: pos[1] -= move_speed

  if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN: pos[2] += move_speed
  if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN: pos[2] -= move_speed

  if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN: pos[3] += move_speed
  if ord('f') in keys and keys[ord('f')] & p.KEY_IS_DOWN: pos[3] -= move_speed

  if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
    pos[4] = max(pos[4] - grip_speed, 0.0)
    pos[5] = min(pos[5] + grip_speed, 0.0)
  else:
    pos[4] = min(pos[4] + grip_speed, 0.5)
    pos[5] = max(pos[5] - grip_speed, -0.5)

  for i in range(6):
    pos[i] = np.clip(pos[i], LIMITS[i][0], LIMITS[i][1])

  p.setJointMotorControlArray(
    robot, joints, p.POSITION_CONTROL,
    targetPositions=pos,
    forces=[100, 100, 100, 100, 40, 40]
  )
  p.stepSimulation()
  frame_counter += 1

  if frame_counter % SKIP_FRAMES == 0:
    img = get_frame()
    cv2.imshow("Robot Monitoring (Diagonal Top)", img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()
