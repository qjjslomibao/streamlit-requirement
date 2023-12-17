import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import subprocess

# Set page layout
st.set_page_config(
    page_title="Happy or Sad Classification",
    page_icon=":smiley:",
    layout="wide"
)

# Add a title with a custom font and color
st.title("Happy or Sad Classification")
st.markdown(
    """
    <style>
        .big-font {
            font-size: 36px !important;
            color: #3498db !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# File uploader widget to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Function to perform image classification using TensorFlow
def classify_image(image):
    try:
        # Load the trained model (replace with your own model)
        model = tf.keras.models.load_model("best_model.h5")

        # Compile the model
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Preprocess the image
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (64, 64))  # Resize the image to match model's expected input size
        img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension
        img_array = img_array / 255.0  # Normalize the input image

        # Make predictions
        predictions = model.predict(img_array)

        return predictions

    except Exception as e:
        st.error(f"Error during image classification: {e}")
        return None

# Display the uploaded image and perform classification
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Add a header with a custom font and color
    st.markdown(
        """
        <style>
            .big-font {
                font-size: 24px !important;
                color: #3498db !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<p class="big-font">Uploaded Image</p>', unsafe_allow_html=True)

    # Display the image
    st.image(image, use_column_width=True)

    # Call the classification function
    predictions = classify_image(image)

    if predictions is not None:
        # Display the classification results
        st.markdown('<p class="big-font">Prediction Results</p>', unsafe_allow_html=True)

        # Extracting class labels
        class_labels = ["Sad", "Happy"]

        # Finding the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]

        # Display the result
        st.write(f"The model predicts: {predicted_class_label} with confidence {predictions[0][predicted_class_index] * 100:.2f}%")

# Link to open the app in Colab
colab_link = "<a href=\"https://colab.research.google.com/github/qjjslomibao/streamlit-requirement/blob/main/final_requirement_streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
st.markdown(colab_link, unsafe_allow_html=True)

# Run Streamlit app in the background
streamlit_command = "streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py"
subprocess.Popen(streamlit_command, shell=True)
