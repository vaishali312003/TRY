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

model = load_model('model.h5')  


st.set_page_config(page_title="EcoTrack", page_icon="🌍", layout="wide")


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

def classify_image(image):
    processed_image = preprocess_image(image)
    probabilities = model.predict(processed_image)
    threshold = 0.5  
    return "Recyclable" if probabilities[0][0] > threshold else "Organic"

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_upload = cv2.imdecode(file_bytes, 1)
    prediction = classify_image(image_upload)
    st.image(image_upload, caption=f"Uploaded Image - Predicted: {prediction}", use_column_width=True)
    
    if prediction == "Recyclable":
        st.success("This item should be placed in the blue recycling bin. Examples include paper, cardboard, plastic bottles, and metal cans.")
    else:
        st.error("This item should be placed in the green waste bucket. Examples include food scraps, yard trimmings, and soiled paper.")


def chatbot_response(user_input):
    suggestions = {
        "organic": {
            "description": "Organic waste includes items that can be composted.",
            "examples": [
                ("Food scraps", "DATASET/DATASET/TEST/O/O_13963.jpg")
            ],
            "disposal": "You should place organic waste in the green waste bucket.",
            "image_path": "DATASET/DATASET/TEST/O/O_13963.jpg"
        },
        "recyclable": {
            "description": "Recyclable waste includes items that can be processed and reused.",
            "examples": [
              
                ("waste cloths and all can be recycable ", "DATASET/DATASET/TEST/R/R_10000.jpg")
            ],
            "disposal": "You should place recyclable waste in the blue recycling bin.",
            "image_path": "DATASET/DATASET/TEST/R/R_10000.jpg"
        }
    }
    response = ""
    if "organic" in user_input.lower():
        response = suggestions["organic"]["description"]
        response += "\n\n**Examples:**"
        for item, img in suggestions["organic"]["examples"]:
            response += f"\n- {item}"
            st.image(img, caption=item, width=100)
        response += f"\n\n{suggestions['organic']['disposal']}"
        st.image(suggestions["organic"]["image_path"], caption="Organic Waste Example", use_column_width=True)
    elif "recyclable" in user_input.lower():
        response = suggestions["recyclable"]["description"]
        response += "\n\n**Examples:**"
        for item, img in suggestions["recyclable"]["examples"]:
            response += f"\n- {item}"
            st.image(img, caption=item, width=100)
        response += f"\n\n{suggestions['recyclable']['disposal']}"
        st.image(suggestions["recyclable"]["image_path"], caption="Recyclable Waste Example", use_column_width=True)
    else:
        response = "Please ask about either organic or recyclable waste."
    return response

st.markdown("## Help and Query")
st.markdown("### Ask EcoTrack about waste disposal")

user_query = st.text_input("Ask me about waste disposal (e.g., 'What is organic waste?' or 'Where do I put recyclables?')")

if user_query:
    response = chatbot_response(user_query)
    st.write(response)

# Additional Features
st.markdown("## Waste Disposal Tips")
st.markdown("""
- **Reduce**: Minimize waste by choosing reusable items and avoiding single-use products.
- **Reuse**: Find ways to reuse items before discarding them. For example, use jars as storage containers.
- **Recycle**: Properly sort your recyclables to ensure they are processed correctly.
- **Compost**: Compost organic waste to create nutrient-rich soil for gardening.
""")

st.markdown("## Frequently Asked Questions (FAQs)")
faq_expander = st.expander("Click to see FAQs")

with faq_expander:
    st.markdown("""
    **Q: What items can be recycled?**
    - A: Common recyclable items include paper, cardboard, plastic bottles, metal cans, and glass containers.
    
    **Q: What is considered organic waste?**
    - A: Organic waste includes food scraps, yard trimmings, and soiled paper products.
    
    **Q: How can I reduce waste at home?**
    - A: Reduce waste by choosing reusable products, buying in bulk, and composting organic waste.
    
    **Q: Where can I dispose of hazardous waste?**
    - A: Hazardous waste should be taken to designated disposal facilities. Check with your local waste management for details.
    """)

st.markdown("## Contact Us")
st.markdown("""
If you have any questions or need further assistance, please contact us at [support@ecotrack.com](mailto:support@ecotrack.com).
""")
