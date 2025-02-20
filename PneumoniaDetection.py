import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("xray_model.h5")

# Function to predict X-ray
def predict_xray(img_path):
    # Load the image with the correct size and ensure it is in RGB mode
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')  # Ensure RGB
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image to the same range as the training data

    # Make the prediction
    prediction = model.predict(img_array)

    # Check the result (since it's binary classification, check if output > 0.5)
    return "Pneumonia" if prediction[0] > 0.5 else "Normal"

# Streamlit UI

st.title("X-ray Image Analysis")
st.write("Upload an X-ray image to analyze it for Pneumonia.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224), color_mode='rgb')
    st.image(img, caption="Uploaded X-ray Image", use_container_width=True)
    #st.image(image, use_container_width=True)


    # Make prediction
    prediction = predict_xray(uploaded_file)
    
    # Display the result
    st.write(f"Prediction: {prediction}")
