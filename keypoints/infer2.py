import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.models as models

# --- 1. MODEL DEFINITION (Must match train.py) ---
class KeypointHeatmapNet(nn.Module):
    def __init__(self):
        super(KeypointHeatmapNet, self).__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 7, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.decoder(x)

# --- 2. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointHeatmapNet().to(device)

try:
    model.load_state_dict(torch.load('heatmap_model.pth', map_location=device))
    model.eval()
    print("Heatmap model loaded successfully!")
except FileNotFoundError:
    print("Error: heatmap_model.pth not found. Run train.py first.")
    exit()

VIDEO_PATH = '../demo_vids/demo4.webm'
cap = cv2.VideoCapture(VIDEO_PATH)
POINTS_ORDER = ["Marker", "Elbow", "Wrist", "Gripper", "Prong1", "Prong2", "Object"]

# Confidence Threshold: Only draw if the heatmap peak is "bright" enough
CONFIDENCE_THRESH = 0.09

while True:
    ret, frame = cap.read()
    if not ret: break
    
    orig_h, orig_w = frame.shape[:2]
    
    # Pre-process
    img_input = cv2.resize(frame, (224, 224))
    img_tensor = torch.FloatTensor(img_input.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    
    # Run Model
    with torch.no_grad():
        heatmaps = model(img_tensor).cpu().numpy()[0] # Shape (7, 224, 224)

    # --- 3. PEAK EXTRACTION (The Classification Logic) ---
    for i in range(7):
        heatmap = heatmaps[i]
        
        # Find the global maximum in the 224x224 heatmap
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        
        if max_val > CONFIDENCE_THRESH:
            # Map 224x224 back to original video resolution
            x = int(max_loc[0] * (orig_w / 224))
            y = int(max_loc[1] * (orig_h / 224))
            
            # Draw results
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"{POINTS_ORDER[i]} ({max_val:.2f})", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # If the peak is too dim, the model is "unconfident"
            pass 

    # --- 4. OPTIONAL: SHOW THE HEATMAPS ---
    # Combine all 7 heatmaps into one for a "debug" view
    combined_heatmap = np.max(heatmaps, axis=0)
    combined_heatmap = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
    cv2.imshow('Model Confidence Heatmap', cv2.resize(heatmap_img, (400, 300)))

    # Show Final Video
    cv2.imshow('Robust Keypoint Detection', cv2.resize(frame, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()