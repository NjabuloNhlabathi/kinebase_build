import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# Load compiled dataset from build_dataset.py
data = np.load("kinebase_training_data.npz")
X = data["X"]  # Shape: (N, 63) wrist-centered 3D landmarks
y = data["y"]  # Shape: (N,) target servo angles (0-180)

if len(X) < 30:
    raise ValueError("Record at least 30+ frames across multiple gestures before training.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a 2-hidden-layer neural network policy (63 -> 128 -> 64 -> 1)
policy_model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42
    )
)

print(f"[TRAINING] Fitting neural policy on {len(X_train)} frames...")
policy_model.fit(X_train, y_train)

preds = np.clip(policy_model.predict(X_test), 0, 180)
mae = mean_absolute_error(y_test, preds)
print(f"[EVAL] Validation Mean Absolute Error: {mae:.2f} degrees")

joblib.dump(policy_model, "kinebase_policy.pkl")
print("[SAVED] Exported trained policy to kinebase_policy.pkl")