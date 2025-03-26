# Newari Food Detection using ResNet50, YOLOv8, and Hybrid Models

📌 Project Overview
This project implements and compares three different models for detecting and classifying 16 types of traditional Newari food items from images:

✅ YOLOv8-only – Real-time object detection.

✅ ResNet50-only – High-accuracy food classification.

✅ Hybrid (YOLOv8 + ResNet50) – Combines YOLO's speed with ResNet's accuracy for improved results, especially in overlapping or complex food scenes.

🧠 Objective
🍽 Detect multiple food items in a single image.

🍛 Accurately classify food categories: e.g., Yomari, Bara, Chatamari, Achar, etc.

📊 Evaluate models using:

Precision

Recall

F1-score

mAP

IoU

🧩 Handle challenges like:

Overlapping objects

Small food items

Visually similar categories

🧱 Model Architectures
🔹 YOLOv8-only (Object Detection)
⚡ Fast & efficient object detection in a single forward pass.

📍 Great for localization, but struggles with fine-grained classification in overlapping scenes.

🔹 ResNet50-only (Image Classification)
🧠 Deep CNN trained to classify the entire image into one food category.

🧬 Excels at identifying subtle differences between dishes, but cannot localize objects (no bounding boxes).

🔹 Hybrid (YOLOv8 + ResNet50)
🧩 ResNet50: High-level feature extractor or classifier.

📦 YOLOv8: Object localization.

🤝 Works together to:

Reduce false positives

Improve detection in overlapping scenes

Achieve best overall accuracy

