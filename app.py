import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image
import cv2
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FarmFlick", layout="centered")

# --- CONFIG ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
model_path = current_dir / "model" / "crop_model.h5"

class_names = [
    "Pepper_Bell_Bacterial_Spot",
    "Pepper_Bell_Healthy",
    "PlantVillage",
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites",
    "Tomato_Target_Spot",
    "Tomato_Mosaic_Virus",
    "Tomato_YellowLeaf_Curl_Virus"
]

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# --- SIMPLIFIED UI STYLE ---
st.markdown("""
    <style>
    .report {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #2e7d32;
    }
    .healthy {
        border-left-color: #4caf50 !important;
    }
    .disease {
        border-left-color: #f44336 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🌿 FarmFlick - Plant Disease Detection")
st.write("Upload a leaf image to check for diseases")

# --- FUNCTIONS ---
def preprocess_and_predict(image):
    try:
        img = image.resize((224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        label = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return label, confidence
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def display_report(label, confidence):
    # Determine report style based on health status
    report_class = "healthy" if "healthy" in label.lower() else "disease"
    
    st.markdown(f"""
    <div class="report {report_class}">
        <h3>Detection Report</h3>
        <p><strong>Status:</strong> {label.replace('_', ' ').title()}</p>
        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disease-specific recommendations
    if "bacterial_spot" in label.lower():
        st.warning("**Recommended Actions:**")
        st.write("- Remove infected leaves immediately")
        st.write("- Apply copper-based fungicides")
        st.write("- Avoid overhead watering")
        st.write("- Ensure proper plant spacing")
    elif "blight" in label.lower():
        st.warning("**Recommended Actions:**")
        st.write("- Remove affected foliage")
        st.write("- Apply fungicide containing chlorothalonil")
        st.write("- Water plants at the base")
        st.write("- Improve air circulation")
    elif "healthy" not in label.lower():
        st.warning("**General Recommendations:**")
        st.write("- Isolate affected plants")
        st.write("- Consult with local agricultural extension")
        st.write("- Monitor plants regularly")
    else:
        st.success("**Plant Care Tips:**")
        st.write("- Continue regular monitoring")
        st.write("- Maintain proper watering schedule")
        st.write("- Ensure adequate sunlight and nutrients")

# --- IMAGE UPLOAD AND PROCESSING ---
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        # Display original image
        st.image(image, caption="Uploaded Image", width=300)
        
        # Process and show results
        with st.spinner("Analyzing image..."):
            label, confidence = preprocess_and_predict(image)
        
        if label and confidence:
            display_report(label, confidence)
            
    except Exception as e:
        st.error(f"Error processing your image: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.caption("FarmFlick - AI-powered plant health analysis")