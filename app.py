import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

model = load_model()

# Load and preprocess image
def model_predict(image):
    H, W, C = 224, 224, 3  # Set height, width, and color channels
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, H, W, 3)
    
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Class name mapping (reverse dictionary)
class_name = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Streamlit UI
st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

img = Image.open('Disease.jpg')
st.image(img)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

elif app_mode == 'Disease Recognition':
    st.header("Plant Disease Prediction System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)

        if st.button("Predict"):
            st.write("Our prediction:")
            result_index = model_predict(test_image)
            st.success(f"Model is predicting that it is {class_name[result_index]}")
