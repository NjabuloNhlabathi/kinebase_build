import os
import cv2
import json
import time
import math
import serial
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"  # <-- Match your Arduino Uno COM port
BAUD_RATE = 115200
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Speed settings for automated loops
FAST_LOOP_STEP = 18   # Degrees moved per frame during Peace Sign fast loop
STEP_INTERVAL_S = 0.4 # Seconds between 45-deg steps during Rock/Horns mode

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

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

def classify_gesture(lms):
    """Detects discrete hand gestures from 21 MediaPipe landmarks."""
    # Check if Index (8), Middle (12), Ring (16), Pinky (20) tips are above their PIP joints
    index_up = lms[8].y < lms[6].y
    middle_up = lms[12].y < lms[10].y
    ring_up = lms[16].y < lms[14].y
    pinky_up = lms[20].y < lms[18].y

    fingers = [index_up, middle_up, ring_up, pinky_up]

    if fingers == [True, True, True, True]:
        return "OPEN_PALM"
    elif fingers == [False, False, False, False]:
        return "FIST"
    elif fingers == [True, False, False, False]:
        return "POINT"
    elif fingers == [True, True, False, False]:
        return "PEACE_LOOP"
    elif fingers == [True, True, True, False]:
        return "TILT_STEER"
    elif fingers == [True, False, False, True]:
        return "STEP_INDEX"
    else:
        return "HOLD"

def compute_hand_tilt_angle(lms):
    """Maps left/right hand tilt (wrist 0 to middle MCP 9) into 0-180 degrees."""
    dx = lms[9].x - lms[0].x
    dy = lms[0].y - lms[9].y  # Invert Y since screen Y increases downward
    tilt_deg = math.degrees(math.atan2(dx, dy))  # -45 (left) to +45 (right)
    servo_angle = np.interp(tilt_deg, [-40, 40], [0, 180])
    return int(np.clip(round(servo_angle), 0, 180))

cap = cv2.VideoCapture(0)

# State variables
active_mode = "OPEN_PALM"
candidate_gesture = "OPEN_PALM"
gesture_streak = 0
target_servo = 0
loop_direction = 1
last_step_time = time.time()

is_recording = False
episode_frames = []
episode_count = 1

print("\n--- KINEBASE GESTURE WHEEL CONTROLLER ---")
print("Controls: [R] Start Recording | [S] Save JSON | [Q] Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        img_lms = result.hand_landmarks[0]
        world_lms = result.hand_world_landmarks[0]

        # Draw fingertip dots for visual feedback
        for tip_id in [4, 8, 12, 16, 20]:
            cx, cy = int(img_lms[tip_id].x * w), int(img_lms[tip_id].y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        # Debounce gesture switching (4 matching frames required)
        detected = classify_gesture(img_lms)
        if detected == candidate_gesture:
            gesture_streak += 1
        else:
            candidate_gesture = detected
            gesture_streak = 1

        if gesture_streak >= 4 and detected != "HOLD":
            active_mode = detected

        # Execute active mode behavior
        if active_mode == "OPEN_PALM":
            target_servo = 0
        elif active_mode == "FIST":
            target_servo = 180
        elif active_mode == "POINT":
            target_servo = 90
        elif active_mode == "PEACE_LOOP":
            target_servo += FAST_LOOP_STEP * loop_direction
            if target_servo >= 180:
                target_servo = 180
                loop_direction = -1
            elif target_servo <= 0:
                target_servo = 0
                loop_direction = 1
        elif active_mode == "TILT_STEER":
            target_servo = compute_hand_tilt_angle(img_lms)
        elif active_mode == "STEP_INDEX":
            if time.time() - last_step_time >= STEP_INTERVAL_S:
                target_servo = (target_servo + 45) if target_servo < 180 else 0
                last_step_time = time.time()

        # Stream angle to Arduino Uno
        if arduino:
            arduino.write(f"{target_servo}\n".encode("utf-8"))

        if is_recording:
            episode_frames.append({
                "timestamp_ms": timestamp_ms,
                "gesture_mode": active_mode,
                "action_servo_deg": target_servo,
                "world_landmarks_m": [
                    {"id": i, "x": round(lm.x, 5), "y": round(lm.y, 5), "z": round(lm.z, 5)}
                    for i, lm in enumerate(world_lms)
                ]
            })

    # On-screen HUD
    cv2.putText(frame, f"Mode: {active_mode}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
    cv2.putText(frame, f"Wheel Angle: {target_servo} deg", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    rec_text = "REC [HARVESTING]" if is_recording else "IDLE [READY]"
    rec_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(frame, rec_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)

    cv2.imshow("KineBase Gesture Wheel", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        is_recording = True
        episode_frames = []
        print(f"[REC] Harvesting gesture episode_{episode_count:04d}...")
    elif key == ord('s') and is_recording:
        is_recording = False
        filename = f"kinebase_gesture_{episode_count:04d}.json"
        payload = {
            "schema_version": "kinebase-lkm-v0.1",
            "task_description": "gesture_wheel_actuation",
            "total_frames": len(episode_frames),
            "frames": episode_frames
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVED] {filename} ({len(episode_frames)} frames)")
        episode_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()