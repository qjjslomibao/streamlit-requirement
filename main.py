{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMW0xpA0bk0oq7uz76gOHib",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/qjjslomibao/streamlit-requirement/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install --upgrade pip\n",
        "!pip show streamlit\n",
        "!pip install streamlit\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zkNLBxbC0h7C",
        "outputId": "a622c82d-d97f-4695-aa98-69210a017d2c"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.1.2)\n",
            "Collecting pip\n",
            "  Downloading pip-23.3.1-py3-none-any.whl (2.1 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.1/2.1 MB\u001b[0m \u001b[31m14.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pip\n",
            "  Attempting uninstall: pip\n",
            "    Found existing installation: pip 23.1.2\n",
            "    Uninstalling pip-23.1.2:\n",
            "      Successfully uninstalled pip-23.1.2\n",
            "Successfully installed pip-23.3.1\n",
            "\u001b[33mWARNING: Package(s) not found: streamlit\u001b[0m\u001b[33m\n",
            "\u001b[0mCollecting streamlit\n",
            "  Downloading streamlit-1.29.0-py2.py3-none-any.whl.metadata (8.2 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Collecting importlib-metadata<7,>=1.4 (from streamlit)\n",
            "  Downloading importlib_metadata-6.11.0-py3-none-any.whl.metadata (4.9 kB)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.23.5)\n",
            "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (23.2)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=6.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (10.0.1)\n",
            "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.0)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.2.3)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.5.0)\n",
            "Requirement already satisfied: tzlocal<6,>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.2)\n",
            "Collecting validators<1,>=0.2 (from streamlit)\n",
            "  Downloading validators-0.22.0-py3-none-any.whl.metadata (4.7 kB)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.40-py3-none-any.whl.metadata (12 kB)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.8.1b0-py2.py3-none-any.whl (4.8 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m35.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.2)\n",
            "Collecting watchdog>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-3.0.0-py3-none-manylinux2014_x86_64.whl (82 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.1/82.1 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<7,>=1.4->streamlit) (3.17.0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2023.11.17)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.11.2)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.32.0)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.13.2)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Downloading streamlit-1.29.0-py2.py3-none-any.whl (8.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.4/8.4 MB\u001b[0m \u001b[31m91.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading GitPython-3.1.40-py3-none-any.whl (190 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m190.6/190.6 kB\u001b[0m \u001b[31m12.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading importlib_metadata-6.11.0-py3-none-any.whl (23 kB)\n",
            "Downloading validators-0.22.0-py3-none-any.whl (26 kB)\n",
            "Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m4.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Installing collected packages: watchdog, validators, smmap, importlib-metadata, pydeck, gitdb, gitpython, streamlit\n",
            "  Attempting uninstall: importlib-metadata\n",
            "    Found existing installation: importlib-metadata 7.0.0\n",
            "    Uninstalling importlib-metadata-7.0.0:\n",
            "      Successfully uninstalled importlib-metadata-7.0.0\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.40 importlib-metadata-6.11.0 pydeck-0.8.1b0 smmap-5.0.1 streamlit-1.29.0 validators-0.22.0 watchdog-3.0.0\n",
            "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv\u001b[0m\u001b[33m\n",
            "\u001b[0m"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MTfV6wDi0eLF",
        "outputId": "e3003af3-5848-4d46-9451-20c2ff8addcc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2023-12-17 03:30:35.206 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ],
      "source": [
        "# Importing necessary libraries\n",
        "import streamlit as st\n",
        "from PIL import Image\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "\n",
        "# Title of the Streamlit app\n",
        "st.title(\"Streamlit App with Image Classification\")\n",
        "\n",
        "# File uploader widget to upload an image file\n",
        "uploaded_file = st.file_uploader(\"Choose an image...\", type=\"jpg\")\n",
        "\n",
        "# Function to perform image classification using TensorFlow\n",
        "def classify_image(image):\n",
        "    try:\n",
        "        # Load the trained model (replace with your own model)\n",
        "        model = tf.keras.models.load_model(\"best_model.h5\")\n",
        "\n",
        "        # Create an optimizer with weight decay\n",
        "        optimizer = tf.optimizers.Adam(learning_rate=0.001)\n",
        "\n",
        "        # Manually apply weight decay to the optimizer's variables\n",
        "        for var in model.trainable_variables:\n",
        "            if \"kernel\" in var.name:\n",
        "                optimizer.add_weight_decay(0.0001, var)\n",
        "\n",
        "        # Compile the model with the custom optimizer\n",
        "        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "        # Preprocess the image\n",
        "        img_array = np.array(image)\n",
        "        img_array = tf.image.resize(img_array, (224, 224))  # Resize the image to match model's expected sizing\n",
        "        img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension\n",
        "        img_array = img_array / 255.0  # Normalize the input image\n",
        "\n",
        "        # Make predictions\n",
        "        predictions = model.predict(img_array)\n",
        "\n",
        "        return predictions\n",
        "\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error during image classification: {e}\")\n",
        "        return None\n",
        "\n",
        "# Display the uploaded image and perform classification\n",
        "if uploaded_file is not None:\n",
        "    image = Image.open(uploaded_file)\n",
        "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
        "    st.write(\"\")\n",
        "    st.write(\"Classifying...\")\n",
        "\n",
        "    # Call the classification function\n",
        "    predictions = classify_image(image)\n",
        "\n",
        "    if predictions is not None:\n",
        "        # Display the classification results\n",
        "        st.write(\"Prediction Results:\")\n",
        "        for i, pred in enumerate(predictions[0]):\n",
        "            st.write(f\"Class {i}: {pred * 100:.2f}% confidence\")\n",
        "\n",
        "# Link to open the app in Colab\n",
        "colab_link = \"<a href=\\\"https://colab.research.google.com/github/qjjslomibao/streamlit-requirement/blob/main/final_requirement_streamlit.py\\\" target=\\\"_parent\\\"><img src=\\\"https://colab.research.google.com/assets/colab-badge.svg\\\" alt=\\\"Open In Colab\\\"/></a>\"\n",
        "st.markdown(colab_link, unsafe_allow_html=True)\n"
      ]
    }
  ]
}