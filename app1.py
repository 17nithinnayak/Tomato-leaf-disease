import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import datetime
import matplotlib.pyplot as plt
import io

# Page Config
st.set_page_config(page_title="Tomato Disease Classifier - JSSSTU", layout="wide")

# --- HEADER SECTION ---
col1, col2 = st.columns([1, 2])
with col1:
    st.image("jssstu_logo.png", width=300)
with col2:
    st.markdown(f"""
        ## JSS Science and Technology University  
        ### Department of Information Science and Engineering  
        #### *Tomato Leaf Disease Detection Using Deep Learning*  
        **Open Day Project Showcase — May 31, 2025**
    """)

st.markdown("""
**Mentor:** Prof. Sindhu A S  
**Students:** Nithin G, Mohammed Shoaib, Tarun P and Raksha R
---
""")

# --- SIDEBAR ABOUT ME ---


# --- CLASS LABELS ---
class_labels = ['Bacterial-spot', 'Early-blight', 'Healthy', 'Late-blight',
                'Leaf-mold', 'Mosaic-virus', 'Septoria-leaf-spot', 'Yellow-leaf-curl-virus']

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato-disease-detection-model.h5")

model = load_model()

# --- PREDICTION FUNCTION ---
def predict(img):
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0]  # Get 1D array of predictions

    # Show each class and its probability
    st.subheader("🔍 Prediction Probabilities")
    for label, prob in zip(class_labels, pred):
        st.write(f"**{label}**: {prob:.4f}")

    pred_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred)
    return pred_class, confidence, pred


# --- UPLOAD SECTION ---
st.header("📸 Upload a Tomato Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    pred_class, confidence, pred_vector = predict(image)

    st.success(f"### 🧠 Prediction: `{pred_class}`")
    st.info(f"Confidence: `{confidence*100:.2f}%`")

    # Optional: plot class probabilities as bar chart
    fig, ax = plt.subplots()
    ax.bar(class_labels, pred_vector, color="#007ACC")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    ax.set_title("Class-wise Confidence")
    st.pyplot(fig)

    # --- Download Report ---
    buffer = io.StringIO()
    buffer.write(f"Prediction Report\n\nDate: {datetime.date.today()}\n")
    buffer.write(f"Predicted Class: {pred_class}\nConfidence: {confidence*100:.2f}%\n")
    st.download_button("📅 Download Report", buffer.getvalue(), file_name="prediction_report.txt")

# --- ACCORDIONS FOR DISEASE INFO ---
st.subheader("📘 Tomato Leaf Disease Info")
with st.expander("Bacterial-spot"):
    st.write("Small, dark spots with yellow halos. Caused by bacteria. Affects leaves and fruit.")
with st.expander("Early-blight"):
    st.write("Dark concentric rings on leaves and stems. Caused by fungus *Alternaria solani*.")
with st.expander("Healthy"):
    st.write("Leaf is green and undamaged. No disease symptoms visible.")
with st.expander("Late-blight"):
    st.write("Water-soaked spots that turn brown, affecting large parts. Very destructive.")
with st.expander("Leaf-mold"):
    st.write("Yellow spots on upper leaf surface, fuzzy mold underneath. Caused by fungus.")
with st.expander("Mosaic-virus"):
    st.write("Mottled, yellow-green pattern. Stunted growth. Spread by insects.")
with st.expander("Septoria-leaf-spot"):
    st.write("Small circular spots with gray centers. One of the most common tomato diseases.")
with st.expander("Yellow-leaf-curl-virus"):
    st.write("Upward curling leaves, yellowing, and stunted plants. Spread by whiteflies.")




# --- ABOUT US SECTION ---
st.title("About Us")
st.markdown("""
Team Members: Nithin G, Mohammed Shoaib, Tarun P and Raksha R

We are a team of four 2nd-year Information Science students from JSS Science and Technology University, Mysuru.  
Our shared interest in Artificial Intelligence and its applications in agriculture led us to develop this project —  
**"Tomato Leaf Disease Detection Using Deep Learning"**.

This project was built collaboratively, where team members contributed to model development, frontend design, backend integration, and data handling.  
Our goal is to leverage AI to support farmers and researchers in identifying tomato plant diseases early, effectively, and efficiently.
""")
