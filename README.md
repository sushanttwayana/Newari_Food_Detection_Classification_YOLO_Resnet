# Newari Food Detection using ResNet50, YOLOv8, and Hybrid Models

📌 Project Overview

This project implements and compares three different models for detecting and classifying 16 types of traditional Newari food items from images:

YOLOv8-only – Real-time object detection.

ResNet50-only – High-accuracy food classification.

Hybrid (YOLOv8 + ResNet50) – Combines the speed of YOLO with the accuracy of ResNet for improved results, especially in complex or overlapping food scenes.

🧠 Objective

Detect multiple food items in an image.

Accurately classify food categories: e.g., Yomari, Bara, Chatamari, Achar, etc.

Evaluate and compare models using standard metrics: Precision, Recall, F1-score, mAP, and IoU.

Handle challenges like overlapping objects, small food items, and similar-looking dishes.

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
