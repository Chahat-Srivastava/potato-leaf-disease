import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the trained model
MODEL_PATH = "my_model1.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names based on the model's training data
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Set up the title and description in Streamlit
st.title("Potato Leaf Disease Classification")
st.write("Upload a potato leaf image, and the model will classify it as Early Blight, Healthy, or Late Blight.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to make it suitable for the model.
    """
    img = Image.open(image_file)
    img = img.resize((256, 256))  # Resize the image to the target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

if uploaded_file is not None:
    # Preprocess the image
    image_data = preprocess_image(uploaded_file)
    
    # Make a prediction
    prediction = model.predict(image_data)
    
    # Get the class with the highest probability
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Get the confidence percentage
    
    # Display the uploaded image and prediction results
    st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{predicted_class}** with **{confidence:.2f}%** confidence.")

    # Show a plot of class probabilities
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    st.pyplot(fig)


    # Optionally show the raw prediction values
    st.write(f"Prediction probabilities: {prediction}")

else:
    st.write("Please upload an image to classify.")
