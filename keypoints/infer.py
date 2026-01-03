import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.models as models

# --- 1. MODEL DEFINITION ---
class KeypointDetector(nn.Module):
    def __init__(self):
        super(KeypointDetector, self).__init__()
        self.backbone = models.resnet18(weights=None) 
        self.backbone.fc = nn.Linear(512, 14) 
    def forward(self, x):
        return self.backbone(x)

# --- 2. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointDetector()
model.load_state_dict(torch.load('keypoint_model.pth', map_location=device))
model.to(device)
model.eval()

VIDEO_PATH = '../demo_vids/demo4.webm'
cap = cv2.VideoCapture(VIDEO_PATH)
POINTS_ORDER = ["Marker", "Elbow", "Wrist", "Gripper", "Prong1", "Prong2", "Object"]

# --- 3. INITIALIZE WITH AI ---
ret, first_frame = cap.read()
h_orig, w_orig = first_frame.shape[:2]

# Run AI on first frame to get starting positions
img = cv2.resize(first_frame, (224, 224))
img_tensor = torch.FloatTensor(img.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(img_tensor).cpu().numpy()[0]

# Convert normalized AI output to real pixel coordinates
p0 = output.reshape(7, 1, 2)
p0[:, :, 0] *= w_orig
p0[:, :, 1] *= h_orig
p0 = p0.astype(np.float32)

# --- 4. OPTICAL FLOW PARAMS (For Smoothness) ---
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
lk_params = dict(winSize=(31, 31), # Large window for stability
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

print("Starting Hybrid Tracking (AI Init + Optical Flow)...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate movement from last frame (The smooth part)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        # Drawing
        for i in range(7):
            x, y = p1[i].ravel()
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(frame, POINTS_ORDER[i], (int(x)+10, int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Update for next frame
        old_gray = frame_gray.copy()
        p0 = p1
    
    # Optional: Every 100 frames, you could re-run the AI to "reset" any drift
    # but for demo3.webm, pure flow should stay locked on.

    cv2.imshow('AI-Initialized Smooth Tracking', cv2.resize(frame, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()