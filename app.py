import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((100, 100))
    image = np.array(image)
    normalized_image = image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Load the trained model
model = load_model('model.h5')  # Make sure the model file is in the same directory

# Streamlit app
st.title("Image Classification App")
st.markdown("-------------------")

st.text("Please upload a picture")
st.markdown("-------")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_upload = Image.open(uploaded_file)
    
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
