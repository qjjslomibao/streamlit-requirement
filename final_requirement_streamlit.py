# Importing necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Title of the Streamlit app
st.title("Streamlit App with Image Classification")

# File uploader widget to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Function to perform image classification using TensorFlow
def classify_image(image):
    # Load the trained model (replace with your own model)
    model = tf.keras.models.load_model("your_model_path")

    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize the image to match model's expected sizing
    img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize the input image

    # Make predictions
    predictions = model.predict(img_array)

    return predictions

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Call the classification function
    predictions = classify_image(image)

    # Display the classification results
    st.write("Prediction Results:")
    for i, pred in enumerate(predictions[0]):
        st.write(f"Class {i}: {pred * 100:.2f}% confidence")

# Link to open the app in Colab
colab_link = "<a href=\"https://colab.research.google.com/github/qjjslomibao/streamlit-requirement/blob/main/final_requirement_streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
st.markdown(colab_link, unsafe_allow_html=True)
