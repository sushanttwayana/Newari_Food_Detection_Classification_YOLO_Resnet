# Newari Food Detection using ResNet50, YOLOv8, and Hybrid Models

### 📌 Project Overview

This project implements and compares three different models for detecting and classifying 16 types of traditional Newari food items from images:

* YOLOv8-only – Real-time object detection.

* ResNet50-only – High-accuracy food classification.

* Hybrid (YOLOv8 + ResNet50) – Combines the speed of YOLO with the accuracy of ResNet for improved results, especially in complex or overlapping food scenes.

### 🧠 Objective

* Detect multiple food items in an image.

* Accurately classify food categories: e.g., Yomari, Bara, Chatamari, Achar, etc.

* Evaluate and compare models using standard metrics: Precision, Recall, F1-score, mAP, and IoU.

* Handle challenges like overlapping objects, small food items, and similar-looking dishes.

🧱 Model Architectures

🔹 YOLOv8-only (Object Detection)
Fast and efficient object detection in a single forward pass.

Performs well in localizing objects but struggles with fine-grained classification in overlapping objects.

🔹 ResNet50-only (Image Classification)
Deep CNN trained to classify an entire image into one food category.

Excels in identifying subtle differences between food items but lacks localization.

🔹 Hybrid (ResNet50 + YOLOv8)
ResNet50: Used as a high-level feature extractor or classifier to guide YOLO.

YOLOv8: Performs object localization, with added context from ResNet to handle overlapping or ambiguous scenes.

Best performance for real-world complex food images.


### ⚙️ Implementation

Framework: PyTorch

Backbone: ResNet50 (pretrained on ImageNet)

Detection Head: YOLOv8 (custom loss functions)

Data Augmentation: Resize (512x512), normalization, Albumentations, Gaussian noise, etc.

Loss Functions:

Localization Loss: MSE

Classification Loss: Cross-Entropy

Confidence Loss: BCE


### Evaluation Metrics

| **Metric**       | **Description**                                                                 |
|------------------|---------------------------------------------------------------------------------|
| **IoU**          | Measures overlap between predicted and ground-truth bounding boxes              |
| **mAP@0.5**      | Average precision at IoU ≥ 0.5 threshold                                         |
| **mAP@0.5:0.95** | Mean Average Precision across multiple IoU thresholds (stricter evaluation)     |
| **Precision**    | TP / (TP + FP) – Correct detections out of total predicted                      |
| **Recall**       | TP / (TP + FN) – Correct detections out of total actual objects                 |
| **F1-Score**     | Harmonic mean of precision and recall                                           |


### Performance Metrices

| Model              | mAP@0.5 | Precision | Recall | F1-Score |
|--------------------|---------|-----------|--------|----------|
| YOLOv8 Only        | 0.89    | 0.87      | 0.86   | 0.865    |
| ResNet50 Only      | 0.91*   | 0.90      | 0.89   | 0.895    |
| Hybrid YOLO+ResNet | 0.963   | 0.94      | 0.92   | 0.94     |


### Visual Analysis & Graphs

* Confusion Matrix: Highlights strong class-wise accuracy (Bara, Yomari, etc.) and minor confusion in similar classes (e.g., Dhau vs. Saag).

* F1 vs Confidence Curve: Peak F1 at 0.432 threshold (F1 = 0.94).

* Precision vs Confidence Curve: 100% precision at 0.983 threshold.

* Recall vs Confidence Curve: 98% recall at 0.000 threshold.

* PR Curve: mAP@0.5 = 0.963, showcasing model strength across all food categories.


## Why Hybrid YOLO + ResNet?

🧠 CNN filters overlapping noise and improves object focus.

🎯 ResNet helps YOLO tune anchor boxes & classification context.

💡 Reduces false positives and improves small object detection.

📈 Highest accuracy and real-world performance among all models





