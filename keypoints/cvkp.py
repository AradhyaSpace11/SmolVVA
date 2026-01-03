import cv2
import numpy as np
import os

# --- 1. SETTINGS ---
video_path = '../demo_vids/demo6.mp4'
DISPLAY_WIDTH = 640  # The window size you actually see

if not os.path.exists(video_path):
    print(f"Error: Could not find {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video")
    exit()

# Get original aspect ratio to resize window correctly
h_orig, w_orig = first_frame.shape[:2]
aspect_ratio = h_orig / w_orig
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * aspect_ratio)

# Resize first frame ONLY for the selection window
selection_frame = cv2.resize(first_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

# --- 2. INTERACTIVE SELECTION ---
selected_points = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        img_copy = selection_frame.copy()
        for i, p in enumerate(selected_points):
            cv2.circle(img_copy, (p[0], p[1]), 5, (0, 255, 0), -1)
            cv2.putText(img_copy, str(i), (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Select Joints (Small View)', img_copy)

print(f"Tracking on full {w_orig}x{h_orig} | Displaying at {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print("INSTRUCTIONS: Click Base, Marker, Shoulder, Elbow, Wrist, Gripper. Press ENTER.")

cv2.imshow('Select Joints (Small View)', selection_frame)
cv2.setMouseCallback('Select Joints (Small View)', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- 3. TRACKING SETUP ---
# Map the points from the small display window back to the REAL video coordinates
scale_x = w_orig / DISPLAY_WIDTH
scale_y = h_orig / DISPLAY_HEIGHT
real_points = [[p[0] * scale_x, p[1] * scale_y] for p in selected_points]

p0 = np.array(real_points, dtype=np.float32).reshape(-1, 1, 2)
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame) # Keep mask full size for precision
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0, 255, (len(selected_points), 3))

# --- 4. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow on FULL RESOLUTION
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 10, colors[i].tolist(), -1)

        # Merge tracking mask with frame
        combined = cv2.add(frame, mask)
        
        # --- RESIZE ONLY FOR DISPLAY ---
        display_img = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Keypoint Tracking (Full Data, Smol Window)', display_img)
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()