import numpy as np

# Load compiled dataset from build_dataset.py
data = np.load("kinebase_training_data.npz")
X = data["X"].astype(np.float32)  # Shape: (N, 63)
y = data["y"].astype(np.float32).reshape(-1, 1)  # Shape: (N, 1)

if len(X) < 30:
    raise ValueError("Record at least 30+ frames across multiple gestures before training.")

# Clean any accidental NaNs in raw data and normalize inputs
X = np.nan_to_num(X, nan=0.0)
y = np.nan_to_num(y, nan=0.0)

mean = X.mean(axis=0, keepdims=True)
std = np.where(X.std(axis=0, keepdims=True) < 1e-6, 1.0, X.std(axis=0, keepdims=True))
X_norm = (X - mean) / std

# Scale target angles from [0, 180] -> [0.0, 1.0] to prevent exploding gradients
y_scaled = y / 180.0

# 80/20 Train/Validation split
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_norm))
split = int(0.8 * len(X_norm))
X_train, X_val = X_norm[indices[:split]], X_norm[indices[split:]]
y_train, y_val = y_scaled[indices[:split]], y_scaled[indices[split:]]

# Initialize 2-hidden-layer neural network (63 -> 64 -> 32 -> 1)
W1 = rng.normal(0, np.sqrt(2.0 / 63), (63, 64)).astype(np.float32)
b1 = np.zeros((1, 64), dtype=np.float32)
W2 = rng.normal(0, np.sqrt(2.0 / 64), (64, 32)).astype(np.float32)
b2 = np.zeros((1, 32), dtype=np.float32)
W3 = rng.normal(0, np.sqrt(2.0 / 32), (32, 1)).astype(np.float32)
b3 = np.zeros((1, 1), dtype=np.float32)

lr = 0.01
epochs = 600
batch_size = 32

print(f"[TRAINING] Fitting stabilized neural policy on {len(X_train)} frames...")

for epoch in range(epochs):
    perm = rng.permutation(len(X_train))
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        xb, yb = X_train[idx], y_train[idx]

        # Forward pass (ReLU)
        z1 = xb @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(0, z2)
        pred = a2 @ W3 + b3

        # Backward pass with gradient clipping
        d_pred = np.clip(2.0 * (pred - yb) / len(xb), -1.0, 1.0)
        dW3 = np.clip(a2.T @ d_pred, -1.0, 1.0)
        db3 = np.clip(d_pred.sum(axis=0, keepdims=True), -1.0, 1.0)

        da2 = d_pred @ W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = np.clip(a1.T @ dz2, -1.0, 1.0)
        db2 = np.clip(dz2.sum(axis=0, keepdims=True), -1.0, 1.0)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = np.clip(xb.T @ dz1, -1.0, 1.0)
        db1 = np.clip(dz1.sum(axis=0, keepdims=True), -1.0, 1.0)

        W3 -= lr * dW3; b3 -= lr * db3
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

# Verify no NaNs exist before saving
if np.isnan(W1).any() or np.isnan(W2).any() or np.isnan(W3).any():
    raise RuntimeError("Training diverged to NaN—check dataset values.")

# Evaluate on validation split (rescale back to 0-180 degrees)
val_a1 = np.maximum(0, X_val @ W1 + b1)
val_a2 = np.maximum(0, val_a1 @ W2 + b2)
val_preds = np.clip((val_a2 @ W3 + b3) * 180.0, 0, 180)
mae = float(np.mean(np.abs((y_val * 180.0) - val_preds)))

print(f"[EVAL] Validation Mean Absolute Error: {mae:.2f} degrees")
np.savez("kinebase_policy.npz", mean=mean, std=std, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("[SAVED] Exported clean policy to kinebase_policy.npz")