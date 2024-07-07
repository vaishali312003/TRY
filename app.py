import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))
    normalized_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Load the trained model
model = load_model('model.h5')  # Make sure the model file is in the same directory

# Streamlit app
st.title("Image Classification App")
st.markdown("-------------------")

operation = st.selectbox("Choose an operation", ["None", "Take a Picture", "Upload a Picture"])

if operation == "Take a Picture":
    st.text("Please capture a picture")
    st.markdown("-------")
    image_capture = st.camera_input("Capture Image", key="first", use_video_port=True)
    
    if image_capture is not None:
        st.image(image_capture, caption="Captured Image", use_column_width=True)
        
        # Preprocess the captured image
        bytes_data = image_capture.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)
        
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
        
        # Display the uploaded image
        st.image(image_upload, caption="Uploaded Image", use_column_width=True)
        
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
