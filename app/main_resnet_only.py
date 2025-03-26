import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function for loading and preprocessing the image
def preprocess_image(image_file, target_size=(128, 128)):
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    return input_arr, image

# Function for making predictions
def predict_image(model, input_arr):
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Index of max element
    confidence = predictions[0][result_index]  # Confidence score of prediction
    return result_index, confidence

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model_65.h5")

model = load_model()

# Sidebar for navigation
st.sidebar.title("Image Classifier")
app_mode = st.sidebar.radio("Choose a page:", ["Home", "About the Project", "Predict Image"])

# run main.py not app.py as i have made the modifications there love you

# Main app pages
if app_mode == "Home":
    st.title("Welcome to the Image Classifier App!")
    st.image("images/n1.jpg", caption="AI-powered Food Classifier", use_container_width=True)
    st.write("""
        Upload an image, and the trained model will predict the category of the image with confidence.
    """)
elif app_mode == "About the Project":
    st.title("About the Project")
    st.write("""
        - This is a deep learning-based food classifier application built with TensorFlow and Streamlit.
        - The model is trained on images of Newari food items and uses a convolutional neural network (CNN) for predictions.
        - Here, for now, we have trained the model over 16 different classes of the data items: Baji, Bara, Bhutan, Bhuti, Buffalo Masu (Dakulaa), Channa, Dhau, Kachilaa, Lain Tarkari, Lainachar, Saag, Thoo, Aloo, Egg, Sapumicha, and Yomari.
        - We have trained the data over almost 500 images of each class, split into training, validation, and testing sets, where 80% of the images were split for training and 10% for validation and testing each.
    """)

    # Define the classes and their sample image paths
    classes_with_images = [
        ("Baji", "images/baji.jpg"),
        ("Bara", "images/bara.jpg"),
        ("Bhutan", "images/bhutan.png"),
        ("Bhuti", "images/bhuti.jpg"),
        ("Buffalo Masu (Dakulaa)", "images/dakala.jpg"),
        ("Channa", "images/chana.jpg"),
        ("Dhau", "images/dhau.jpg"),
        ("Kachilaa", "images/kachila.jpg"),
        ("Lain Tarkari", "images/laintarkari.jpg"),
        ("Lainachar", "images/lainachar.jpg"),
        ("Saag", "images/saag.png"),
        ("Thoo", "images/thoo.png"),
        ("Aloo", "images/aloo.jpg"),
        ("Egg", "images/egg1.jpg"),
        ("Sapumicha", "images/sapumicha1.jpg"),
        ("Yomari", "images/yomari.jpg"),
    ]

    # Display images in a 4-column grid
    st.write("### Sample Images from Each Class")
    num_columns = 4
    columns = st.columns(num_columns)

    for idx, (class_name, image_path) in enumerate(classes_with_images):
        with columns[idx % num_columns]:
            try:
                st.image(image_path, caption=class_name, use_container_width=True)
            except FileNotFoundError:
                st.error(f"Image for {class_name} not found.")

    # st.write("# About the model")
    # st.write("""
    #     The model uses multiple convolutional layers followed by dense layers for classification.
    #     The input image size is (64, 64). The architecture is as follows:
    #     - Convolutional Layers: 3 layers with ReLU activations and MaxPooling.
    #     - Flatten Layer: Converts 2D feature maps to 1D.
    #     - Fully Connected Layers: 2 dense layers with softmax activation at the final layer.
    # """)
elif app_mode == "Predict Image":
    st.title("Image Prediction")
    st.write("Upload an image, and the model will predict the category with a confidence score.")

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        if st.button("Predict"):
            input_arr, display_image = preprocess_image(uploaded_file)
            result_index, confidence = predict_image(model, input_arr)

            # Read class names from a file (labels.txt)
            try:
                with open("labels.txt", "r") as f:
                    class_names = [line.strip() for line in f.readlines()]

                predicted_label = class_names[result_index]

                # Display the result
                st.success(f"Prediction: {predicted_label} ({confidence*100:.2f}% Confidence)")
                st.image(display_image, caption=f"Predicted as: {predicted_label}", use_container_width=True)
            except FileNotFoundError:
                st.error("Error: The `labels.txt` file is missing. Please ensure it is present in the correct directory.")
