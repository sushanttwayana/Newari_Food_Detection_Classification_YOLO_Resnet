import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Load YOLO Only Model
@st.cache_resource
def load_yolo_only_model():
    return YOLO("best_100_yolo_only.pt")

# Load YOLO + ResNet Model
@st.cache_resource
def load_yolo_resnet_model():
    return YOLO("best_100_yolo_resnet.pt")

# Load ResNet Model
@st.cache_resource
def load_resnet_model():
    return tf.keras.models.load_model("trained_model_70.h5")

# Preprocess image for ResNet
def preprocess_image(image_file, target_size=(128, 128)):
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch
    return input_arr, image

# Predict using ResNet
def predict_with_resnet(model, input_arr):
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = predictions[0][result_index]
    return result_index, confidence

# Predict using YOLO
def yolo_detect(model, image_path):
    results = model.predict(source=image_path, imgsz=512, conf=0.5)
    return results

# Draw YOLO bounding boxes
def draw_yolo_boxes(image_path, results):
    image = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = "detected_image.jpg"
    cv2.imwrite(output_path, image)
    return output_path

# Load models
yolo_only_model = load_yolo_only_model()
yolo_resnet_model = load_yolo_resnet_model()
resnet_model = load_resnet_model()

# Streamlit UI
st.sidebar.title("Image Classifier")
st.title("Food Classification and Detection")
app_mode = st.sidebar.radio("Choose a page:", ["Home", "About the Project", "Predict Image"])

if app_mode == "Home":
    st.image("images/n1.jpg", caption="AI-powered Food Classifier", use_container_width=True)
    st.write("Upload an image, and the trained model will predict the category of the image with confidence.")

elif app_mode == "About the Project":
    st.write("""
    - This app classifies Newari food using **ResNet** and detects food items using **YOLOv8**.
    - Trained on 16 different food categories.
    """)

elif app_mode == "Predict Image":
    st.write("Upload an image, choose a method, and get predictions.")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_image_path = "temp_uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        prediction_method = st.radio("Choose Prediction Method:", ["ResNet Only", "YOLO Only", "YOLO + ResNet"])

        if st.button("Predict"):
            if prediction_method == "ResNet Only":
                input_arr, display_image = preprocess_image(temp_image_path)
                result_index, confidence = predict_with_resnet(resnet_model, input_arr)
                
                try:
                    with open("labels.txt", "r") as f:
                        class_names = [line.strip() for line in f.readlines()]
                    predicted_label = class_names[result_index]
                    st.success(f"Prediction: {predicted_label} ({confidence*100:.2f}% Confidence)")
                    st.image(display_image, caption=f"Predicted as: {predicted_label}", use_container_width=True)
                except FileNotFoundError:
                    st.error("Error: The `labels.txt` file is missing.")

            elif prediction_method == "YOLO Only":
                results = yolo_detect(yolo_only_model, temp_image_path)
                detected_image_path = draw_yolo_boxes(temp_image_path, results)
                st.image(detected_image_path, caption="Detected Image with YOLO", use_container_width=True)
                detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
                st.write(f"YOLO Detected Classes: {', '.join(set(detected_classes))}")

            elif prediction_method == "YOLO + ResNet":
                results = yolo_detect(yolo_resnet_model, temp_image_path)
                detected_image_path = draw_yolo_boxes(temp_image_path, results)
                st.image(detected_image_path, caption="Detected Image with YOLO + ResNet", use_container_width=True)
                detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
                st.write(f"YOLO + ResNet Detected Classes: {', '.join(set(detected_classes))}")
                
                input_arr, display_image = preprocess_image(temp_image_path)
                result_index, confidence = predict_with_resnet(resnet_model, input_arr)
                try:
                    with open("labels.txt", "r") as f:
                        class_names = [line.strip() for line in f.readlines()]
                    predicted_label = class_names[result_index]
                    st.success(f"ResNet Classification: {predicted_label} ({confidence*100:.2f}% Confidence)")
                except FileNotFoundError:
                    st.error("Error: The `labels.txt` file is missing.")
