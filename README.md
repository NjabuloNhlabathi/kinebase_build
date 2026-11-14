# KineBase

**Edge-AI Kinematic Data Harvesting & Imitation Learning Pipeline**

KineBase is a hardware-to-AI robotics pipeline that captures 3D human hand kinematics via monocular computer vision, translates gestures into real-time physical servo actuation, records structured state-action trajectories (`JSON`), and trains neural imitation-learning policies to drive robotic actuators without hard-coded rules.

---

## System Architecture

```text
[Webcam Feed] ──► [MediaPipe 3D Hand Landmarker] ──► [Gesture / Policy Engine]
                                                            │
                        ┌───────────────────────────────────┴───────────────────────────────────┐
                        ▼                                                                       ▼
         [USB Serial (115200 baud)]                                              [JSON Episode Recorder]
                        │                                                                       │
                        ▼                                                                       ▼
         [Arduino Uno + FS90 Servo]                                              [Wrist-Centered Normalization]
                                                                                                │
                                                                                                ▼
                                                                                 [MLP Neural Policy (.pkl)]
```

Real-Time Teleoperation (kinebase_uno.py): Tracks 21 3D hand landmarks (world_landmarks_m) at video frame rates, classifies discrete and continuous gestures, streams target angles (0°–180°) over serial to an Arduino Uno, and logs frame-by-frame trajectories.Embedded Actuation (kinebase_uno_servo.ino): Low-latency C++ firmware running on an Arduino Uno that parses incoming degree commands and drives an FS90 servo wheel.Standalone Hardware Replay (replay_episode.py): Streams recorded JSON trajectories back to the microcontroller with original inter-frame timing to validate physical reproducibility without a camera.Kinematic Dataset Compiler (build_dataset.py): Normalizes all 21 3D landmarks relative to the wrist origin (id: 0) for position invariance and compiles episodes into ML-ready NumPy tensors (.npz) and .csv tables.Imitation Learning Policy (train_policy.py & run_ai_policy.py): Trains a 2-hidden-layer Multi-Layer Perceptron (63 -> 64 -> 32 -> 1) using pure NumPy with target scaling and gradient clipping to predict servo commands autonomously in real time.Hardware Requirements & WiringHost Compute: Laptop/PC with webcam (Python 3.9+)Microcontroller: Arduino UnoActuator: FS90 9g Micro Servo (with wheel/horn attachment)Wiring: 3x Male-to-Male jumper wiresFS90 Servo WireArduino Uno PinFunctionBrown / BlackGNDCommon GroundRed5VServo PowerOrange / Yellow~9 (Digital Pin 9)PWM Control SignalGesture Control MapHand GestureFinger StateServo Wheel ActionOpen PalmAll 4 fingers extendedHome / Stop (0°): Halts active loops and returns to 0°.PointIndex extendedCenter (90°): Moves wheel to midpoint and holds.Peace SignIndex + Middle extendedFast Sweep (0° ↔ 180°): Continuously oscillates between limits.Three FingersIndex + Middle + Ring extendedTilt Steering (0°–180°): Maps left/right hand tilt proportionally.Rock / HornsIndex + Pinky extendedStep Indexer: Advances in 45° increments (0° → 45° → 90° → 135° → 180°).Closed FistAll 4 fingers curledMax Lock (180°): Rotates wheel to 180° and holds.Repository StructurePlaintextkinebase/
├── kinebase_uno_servo.ino      # Arduino Uno PWM servo firmware
├── kinebase_uno.py             # Live MediaPipe tracker, gesture controller & JSON harvester
├── replay_episode.py           # Standalone JSON-to-hardware trajectory player
├── build_dataset.py            # Wrist-origin normalizer & .npz/.csv compiler
├── train_policy.py             # Pure-NumPy MLP training pipeline (exports kinebase_policy.npz)
├── run_ai_policy.py            # Autonomous neural policy inference controller
└── hand_landmarker.task        # MediaPipe Hand Landmarker model (auto-downloaded on first run)
Quickstart1. Flash the Arduino UnoOpen kinebase_uno_servo.ino in the Arduino IDE.Select Tools > Board > Arduino Uno and your active Port (e.g., COM3 or /dev/ttyACM0).Click Upload, then close the Serial Monitor before running Python scripts.2. Install Python DependenciesBashpip install opencv-python mediapipe pyserial numpy
3. Harvest Kinematic DemonstrationsUpdate SERIAL_PORT in kinebase_uno.py to match your Arduino port, then run:Bashpython kinebase_uno.py
R — Start recording a demonstration episode (REC [HARVESTING])S — Stop recording and export kinebase_gesture_XXXX.jsonQ — Quit and release serial/camera resources4. Replay, Compile & Train the AI PolicyBash# 1. Replay a recorded episode directly on the servo
python replay_episode.py

# 2. Normalize all JSON episodes into kinebase_training_data.npz
python build_dataset.py

# 3. Train the pure-NumPy neural network policy (outputs kinebase_policy.npz)
python train_policy.py

# 4. Run live autonomous inference without hard-coded gesture rules
python run_ai_policy.py
Dataset Schema (kinebase-lkm-v0.1)Each recorded episode exports a structured JSON trajectory containing frame-level timestamps, active control modes, target actuator states, and metric 3D hand landmarks:JSON{
  "schema_version": "kinebase-lkm-v0.1",
  "task_description": "gesture_wheel_actuation",
  "total_frames": 142,
  "frames": [
    {
      "timestamp_ms": 1731584120450,
      "gesture_mode": "PEACE_LOOP",
      "action_servo_deg": 108,
      "world_landmarks_m": [
        { "id": 0, "x": -0.00124, "y": 0.04112, "z": 0.01503 }
      ]
    }
  ]
}
Roadmap[ ] Temporal Frame Stacking: Feed sliding windows of 3–5 consecutive frames so the neural policy learns dynamic trajectories and continuous oscillation speed.[ ] Multi-DOF Hand Actuation: Scale from single-axis wheel validation to 5-finger tendon actuation via PCA9685 PWM drivers.[ ] Wireless Edge Telemetry: Offload actuation to ESP32 nodes over UDP/WebSockets.[ ] Multi-Angle Capture: Synchronize distributed Raspberry Pi camera pods to eliminate single-view occlusion.[ ] Holonomic Mobile Base: Map wrist pitch/roll kinematics to a 4-wheel omni-directional rover chassis.
