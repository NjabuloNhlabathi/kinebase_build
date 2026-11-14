import json
import time
import serial

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"  # <-- Match your Arduino Uno COM port
BAUD_RATE = 115200
EPISODE_FILE = "kinebase_gesture_0001.json"

with open(EPISODE_FILE, "r") as f:
    episode = json.load(f)

frames = episode["frames"]
print(f"[LOADED] {EPISODE_FILE} | Task: {episode['task_description']} | Frames: {len(frames)}")

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
time.sleep(2)  # Wait for Arduino Uno reset
print(f"[REPLAYING] Streaming trajectory to {SERIAL_PORT}...")

for i, frame in enumerate(frames):
    angle = frame["action_servo_deg"]
    mode = frame.get("gesture_mode", "UNKNOWN")

    arduino.write(f"{angle}\n".encode("utf-8"))
    print(f"Frame {i+1:03d}/{len(frames):03d} | Mode: {mode:<10} | Servo: {angle:3d} deg")

    # Match original recording timing between frames
    if i + 1 < len(frames):
        dt_sec = (frames[i + 1]["timestamp_ms"] - frame["timestamp_ms"]) / 1000.0
        time.sleep(max(0.01, min(dt_sec, 0.2)))

print("[DONE] Hardware replay complete.")
arduino.close()