import cv2
import time
import serial
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"  # <-- Match your Arduino Uno COM port
BAUD_RATE = 115200
MODEL_PATH = "hand_landmarker.task"
ALPHA = 0.35

# Load pure-NumPy neural policy weights
weights = np.load("kinebase_policy.npz")
mean, std = weights["mean"], weights["std"]
W1, b1 = weights["W1"], weights["b1"]
W2, b2 = weights["W2"], weights["b2"]
W3, b3 = weights["W3"], weights["b3"]
print("[OK] Loaded trained neural policy: kinebase_policy.npz")

def predict_angle(pose_vec):
    x = (np.array(pose_vec, dtype=np.float32).reshape(1, -1) - mean) / std
    a1 = np.maximum(0, x @ W1 + b1)
    a2 = np.maximum(0, a1 @ W2 + b2)
    out = float((a2 @ W3 + b3)[0, 0]) * 180.0
    out = float(np.nan_to_num(out, nan=90.0))
    return float(np.clip(out, 0, 180))

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    time.sleep(2)
    print(f"[OK] Connected to Arduino Uno on {SERIAL_PORT}")
except Exception as e:
    arduino = None
    print(f"[WARN] Serial error: {e}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_tracking_confidence=0.65
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
smoothed_angle = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int(time.time() * 1000)

    res = detector.detect_for_video(mp_img, ts)

    if res.hand_world_landmarks:
        lms = res.hand_world_landmarks[0]
        wx, wy, wz = lms[0].x, lms[0].y, lms[0].z

        pose_vec = []
        for lm in lms:
            pose_vec.extend([lm.x - wx, lm.y - wy, lm.z - wz])

        raw_pred = predict_angle(pose_vec)
        smoothed_angle = (ALPHA * raw_pred) + ((1 - ALPHA) * smoothed_angle)
        target_servo = int(round(smoothed_angle))

        if arduino:
            arduino.write(f"{target_servo}\n".encode("utf-8"))

        cv2.putText(frame, f"AI Policy Prediction: {target_servo} deg", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("KineBase Autonomous Policy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()