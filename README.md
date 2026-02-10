Edge-AI Semiconductor Defect Detection
IESA DeepTech Hackathon Phase 1 Submission

Project Overview
This project provides an automated microscopic inspection system designed to detect integrated circuit defects. The solution is optimized for NXP i.MX RT hardware using a lightweight MobileNetV2 architecture.

Technical Specifications
Model Backbone: MobileNetV2 (Transfer Learning).

Input Size: 224x224 (Grayscale).

Format: ONNX (Opset 13) for NXP eIQ compatibility.

Target Classes: 8 (Bridge, Clean, Crack, Malformed_Via, Open, Other, Scratch, Short).

Results
Training Accuracy: 66.34%.

Validation Accuracy: ~65.35%.

Model Size: ~27MB.

How to Run
Install dependencies: pip install tensorflow tf2onnx numpy<2.0.0.

Run python scripts/split_dataset.py to organize data into Train/Val/Test.

Run python scripts/augment_and_train.py to reproduce the training.

//Dataset Access :

https://drive.google.com/drive/folders/12kAE-yvK-GfXXj6fx-alhKfu3KD-6IJK?usp=sharing
