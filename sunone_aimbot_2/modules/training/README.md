# PID Governor Training

This folder is offline-only. It is for creating a small MLP that learns how to
scale PID terms and final speed; it does not send mouse movement and it is not
part of runtime until the exported ONNX model is wired into C++.

## Generate a Dataset

```powershell
python training/generate_pid_dataset.py `
  --config x64/DML/config.ini `
  --output training/data/pid_governor_dataset.csv `
  --episodes-per-profile 64 `
  --steps-per-episode 180 `
  --max-speed-multiple 5
```

The generator covers stopped targets, offset stationary targets, linear motion,
direction changes, moving-away/shrinking targets, and noisy/missed detections.
Synthetic target speed is capped at:

```text
pid_max_pixel_step * pid_actuator_hz * pid_output_scale * max_speed_multiple
```

You can add `--max-speed-px-s 3000` to impose an absolute cap if the live config
is temporarily set much higher than the speed range you want to learn.

## Train

```powershell
python training/train_pid_governor.py `
  --dataset training/data/pid_governor_dataset.csv `
  --output training/models/pid_governor.pt `
  --metadata training/models/pid_governor.json `
  --epochs 25
```

The model inputs are raw controller state features. The outputs are:

```text
label_kp_scale, label_ki_scale, label_kd_scale, label_speed_scale
```

## Export ONNX

```powershell
python training/export_pid_governor_onnx.py `
  --model training/models/pid_governor.pt `
  --output training/models/pid_governor.onnx
```

The ONNX graph includes feature normalization, so C++ can feed raw feature
values in the order listed in the generated metadata JSON.

## Enable Runtime Governor

After exporting, set these in `config.ini`:

```ini
pid_governor_enabled = true
pid_governor_model_path = training/models/pid_governor.onnx
pid_governor_blend = 1.000
pid_governor_max_speed_multiple = 5.000
```

If the file is missing or inference fails, runtime falls back to pure PID.

## AI_Training Runtime Capture

The `AI_Training` ImGui tab manages the optional neural PID tuner, PID data
recording, scoring, and one-step train/export launch.

Recorded PID rows are written to:

```ini
ai_training_log_path = training/logs/ai_training_pid.csv
```

Each row includes target-size-normalized error, convergence score, PID terms,
governor scales, emitted mouse deltas, closing rate, and overshoot risk. Score
the current log with:

```powershell
python training/score_ai_training_log.py `
  --input training/logs/ai_training_pid.csv `
  --status training/status/ai_training_status.json
```

The train/export button launches:

```powershell
python train_pid_governor_auto.py `
  --config x64/DML/config.ini `
  --dml-dir x64/DML `
  --status training/status/ai_training_status.json
```

The status JSON is used by the GUI progress bars.

## Evaluate

```powershell
python training/evaluate_pid_governor.py `
  --dataset training/data/pid_governor_dataset.csv `
  --model training/models/pid_governor.pt
```

# Neural Tracker Training

The neural tracker model is optional. It scores tracker/detection association
candidates and blends with the classical tracker score. Runtime falls back to
the classical tracker if the feature is disabled or if
`training/models/neural_tracker.onnx` is missing.

## One-Step Neural Tracker Build

```powershell
python train_neural_tracker_auto.py `
  --config x64/DML/config.ini `
  --dml-dir x64/DML
```

This synthetic-only run creates training outputs but skips runtime deployment by
default. Use it as a bootstrap check, not as the live tracker.

To deploy into another copied runtime folder, point `--dml-dir` at that folder:

```powershell
python train_neural_tracker_auto.py `
  --config x64/good/DML/config.ini `
  --dml-dir x64/good/DML
```

## Neural Tracker Logs

For real data, enable:

```ini
neural_tracker_log_enabled = true
neural_tracker_log_path = training/logs/neural_tracker_association.csv
```

Then merge the log into training:

```powershell
python train_neural_tracker_auto.py `
  --merge-log training/logs/neural_tracker_association.csv
```

Runs with `--merge-log` deploy by default. To force deployment of a
synthetic-only model for experiments, pass `--deploy-synthetic`.

## YOLO Workspace Assistant

Use the Tkinter assistant to turn an image folder into an Ultralytics YOLO
dataset with pseudo-labels from an existing detector:

```powershell
python training/yolo_workspace_assistant.py
```

The assistant creates `images/train`, `images/val`, `images/test`,
`labels/train`, `labels/val`, `labels/test`, a `data.yaml`, and a
`pseudo_label_manifest.csv`. Leave class names blank to preserve the detector's
class names, or enable single-class output if you intentionally want all
selected detections collapsed into one class.

## Manual Neural Tracker Steps

```powershell
python training/generate_neural_tracker_dataset.py `
  --output training/data/neural_tracker_dataset.csv `
  --samples 20000

python training/train_neural_tracker.py `
  --dataset training/data/neural_tracker_dataset.csv `
  --output training/models/neural_tracker.pt

python training/export_neural_tracker_onnx.py `
  --model training/models/neural_tracker.pt `
  --output training/models/neural_tracker.onnx
```
