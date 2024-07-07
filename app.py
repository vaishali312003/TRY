import streamlit as st
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to match model's expected sizing
    resized_image = cv2.resize(image, (100, 100))
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype('float32') / 255.0
    # Expand dimensions to create a batch of size 1
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Load the trained model
model = load_model('/workspaces/TRY/model.h5')  # Adjust the path as per your model location

# Streamlit app
st.title("Image Classification App")
st.markdown("-------------------")

operation = st.selectbox("Choose an operation", ["None", "Upload a Picture"])

if operation == "Take a Picture":
    st.text("Please capture a picture")
    st.markdown("-------")
    image_capture = st.camera_input("Capture Image", key="first", use_video_port=True)
    
    if image_capture is not None:
        st.image(image_capture, caption="Captured Image", use_column_width=True)
        
        # Preprocess the captured image
        processed_image = preprocess_image(image_capture)
        
        # Predict the class probabilities
        probabilities = model.predict(processed_image)
        threshold = 0.5  # Adjust the threshold as needed
        
        # Classify the image based on thresholding
        if probabilities[0][0] > threshold:
            st.write("Predicted class: Recyclable")
        else:
            st.write("Predicted class: Organic")

if operation == "Upload a Picture":
    st.text("Please upload a picture")
    st.markdown("-------")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_upload = cv2.imdecode(file_bytes, 1)
        
        # Preprocess the uploaded image
        processed_image = preprocess_image(image_upload)
        
        # Predict the class probabilities
        probabilities = model.predict(processed_image)
        threshold = 0.5  # Adjust the threshold as needed
        
        # Classify the image based on thresholding
        if probabilities[0][0] > threshold:
            st.write("Predicted class: Recyclable")
        else:
            st.write("Predicted class: Organic")

