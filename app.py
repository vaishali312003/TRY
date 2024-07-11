import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import hashlib
from datetime import datetime
import json
import os

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Dummy user database file
user_db_file = "user_db.json"

# Load or initialize user database
def load_user_db():
    if os.path.exists(user_db_file):
        with open(user_db_file, "r") as file:
            return json.load(file)
    else:
        return {}

def save_user_db(users):
    with open(user_db_file, "w") as file:
        json.dump(users, file)

users = load_user_db()

# File to store user history
history_file = "user_history.json"

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Function to load user history
def load_history(username):
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            data = json.load(file)
            return data.get(username, [])
    return []

# Function to save user history
def save_history(username, history):
    data = {}
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            data = json.load(file)
    data[username] = history
    with open(history_file, "w") as file:
        json.dump(data, file)

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))
    normalized_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Load model
model = load_model('model.h5')

# Streamlit app layout
st.set_page_config(page_title="EcoTrack", page_icon="🌍", layout="wide")


def login():
    st.session_state.username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if st.session_state.username in users and users[st.session_state.username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.history = load_history(st.session_state.username)
            st.success(f"Welcome, {st.session_state.username}!")
        else:
            st.error("Invalid username or password")


def signup():
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Signup"):
        if new_username in users:
            st.error("Username already exists")
        else:
            users[new_username] = hash_password(new_password)
            save_user_db(users)
            st.success("User created successfully. Please log in.")

def logout():
    save_history(st.session_state.username, st.session_state.history)
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.history = []
    st.success("Logged out successfully")


def classify_image(image):
    processed_image = preprocess_image(image)
    probabilities = model.predict(processed_image)
    threshold = 0.5
    return "Recyclable" if probabilities[0][0] > threshold else "Organic"


def main_app():
    st.markdown(
        """
        <style>
        .main {
            background-color: black;
            padding: 10px;
            border-radius: 10px;
        }
        .stTextInput > div > div > input {
            background-color: #F1F8E9;
            color: #000;
        }
        .stButton > button {
            background-color: #388E3C;
            color: white;
        }
        .stFileUploader > div > button {
            background-color: #388E3C;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("EcoTrack: Waste Classification and Disposal Guidance")
    st.markdown("## Upload an Image to Classify")
    st.markdown("### Identify whether the waste is recyclable or organic")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_upload = cv2.imdecode(file_bytes, 1)
        prediction = classify_image(image_upload)
        st.image(image_upload, caption=f"Uploaded Image - Predicted: {prediction}", use_column_width=True)
        
        st.session_state.history.append({
            "timestamp": datetime.now().isoformat(),
            "image": uploaded_file.name,
            "prediction": prediction
        })

        if prediction == "Recyclable":
            st.success("This item should be placed in the blue recycling bin. Examples include paper, cardboard, plastic bottles, and metal cans.")
        else:
            st.error("This item should be placed in the green waste bucket. Examples include food scraps, yard trimmings, and soiled paper.")
        
    st.markdown("## History")
    if st.session_state.history:
        for record in st.session_state.history:
            st.markdown(f"**Time:** {record['timestamp']}")
            st.markdown(f"**Image:** {record['image']}")
            st.markdown(f"**Prediction:** {record['prediction']}")
    else:
        st.markdown("No history available.")

    st.markdown("## FAQ")
    faq_section()

    st.markdown("## Chatbot")
    chatbot_section()


def faq_section():
    st.markdown("""
    ### Frequently Asked Questions
    **Q: What items are considered recyclable?**
    - A: Recyclable items include paper, cardboard, plastic bottles, and metal cans.

    **Q: What items are considered organic waste?**
    - A: Organic waste includes food scraps, yard trimmings, and soiled paper.

    **Q: How do I dispose of electronic waste?**
    - A: Electronic waste should be taken to designated e-waste recycling centers.

    **Q: Can glass be recycled?**
    - A: Yes, glass bottles and jars can be recycled.
    """)


def chatbot_section():
    st.markdown("### Ask the Chatbot")
    user_input = st.text_input("Enter your question:")
    if st.button("Submit"):
        response = generate_response(user_input)
        st.write(response)


def generate_response(user_input):
    # Simple hardcoded responses for demonstration purposes
    if "recycle" in user_input.lower():
        return "Recyclable items include paper, cardboard, plastic bottles, and metal cans."
    elif "organic" in user_input.lower():
        return "Organic waste includes food scraps, yard trimmings, and soiled paper."
    else:
        return "Sorry, I don't have an answer to that question."


# App layout
if not st.session_state.authenticated:
    option = st.sidebar.selectbox("Choose an option", ["Login", "Signup"])
    if option == "Login":
        login()
    elif option == "Signup":
        signup()
else:
    st.sidebar.button("Logout", on_click=logout)
    main_app()
