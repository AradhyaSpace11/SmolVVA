import cv2
import numpy as np
import json
import os

# --- CONFIG ---
VIDEO_PATH = 'demo_vids/demo3.webm'
DATA_FILE = 'training_data.json'
POINTS_ORDER = ["Marker", "Elbow", "Wrist", "Gripper", "Prong1", "Prong2", "Object"]
SKIP_FRAMES = 30 

cap = cv2.VideoCapture(VIDEO_PATH)
all_labels = []
current_pts = []

def click_event(event, x, y, flags, param):
    global current_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_pts) < len(POINTS_ORDER):
            current_pts.append([int(x), int(y)])

cv2.namedWindow("Labeler")
cv2.setMouseCallback("Labeler", click_event)

frame_idx = 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break

    img_display = frame.copy()

    # Draw existing points
    for i, pt in enumerate(current_pts):
        cv2.circle(img_display, (pt[0], pt[1]), 5, (0, 255, 0), -1)
        # Subtle text label
        cv2.putText(img_display, f"{i+1}", (pt[0] + 8, pt[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- DISCREET TEXT OVERLAY (No Black Box) ---
    # We use a simple helper to draw text with a border so it's visible on grey/white
    def draw_text_with_border(img, text, pos):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3) # Border
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1) # Text

    draw_text_with_border(img_display, f"Frame: {frame_idx}", (20, 30))
    
    if len(current_pts) < len(POINTS_ORDER):
        draw_text_with_border(img_display, f"NEXT: {POINTS_ORDER[len(current_pts)]}", (20, 60))
    else:
        draw_text_with_border(img_display, "DONE! Press 'n'", (20, 60))

    cv2.imshow("Labeler", img_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n') and len(current_pts) == len(POINTS_ORDER):
        all_labels.append({"frame": int(frame_idx), "points": current_pts.copy()})
        with open(DATA_FILE, 'w') as f:
            json.dump(all_labels, f, indent=4)
        current_pts = []
        frame_idx += SKIP_FRAMES
        print(f"Saved frame {frame_idx}")

    elif key == ord('c'):
        current_pts = []

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()