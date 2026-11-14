import glob
import json
import csv
import numpy as np

episode_files = sorted(glob.glob("kinebase_gesture_*.json"))
if not episode_files:
    raise FileNotFoundError("No kinebase_gesture_*.json files found in this folder.")

X_features = []  # Normalized 63-value hand pose vectors (21 landmarks * 3 axes)
y_angles = []    # Target servo angles (0-180)
y_modes = []     # Gesture mode labels

for ep_idx, filepath in enumerate(episode_files):
    with open(filepath, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        lms = frame["world_landmarks_m"]
        # Landmark 0 is the wrist origin
        wx, wy, wz = lms[0]["x"], lms[0]["y"], lms[0]["z"]

        # Subtract wrist coordinates so hand pose is position-invariant
        normalized_pose = []
        for lm in lms:
            normalized_pose.extend([
                round(lm["x"] - wx, 6),
                round(lm["y"] - wy, 6),
                round(lm["z"] - wz, 6)
            ])

        X_features.append(normalized_pose)
        y_angles.append(frame["action_servo_deg"])
        y_modes.append(frame.get("gesture_mode", "UNKNOWN"))

X = np.array(X_features, dtype=np.float32)
y = np.array(y_angles, dtype=np.float32)

# Save compressed NumPy dataset for PyTorch / scikit-learn training
np.savez("kinebase_training_data.npz", X=X, y=y, modes=np.array(y_modes))

# Save CSV for inspection
with open("kinebase_training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")] + ["gesture_mode", "target_servo_deg"]
    writer.writerow(header)
    for row_x, mode, angle in zip(X_features, y_modes, y_angles):
        writer.writerow(row_x + [mode, angle])

print(f"[SUCCESS] Compiled {len(episode_files)} episodes ({len(X)} total frames).")
print(f"  -> Tensor shape: X={X.shape}, y={y.shape}")
print("  -> Saved: kinebase_training_data.npz & kinebase_training_data.csv")