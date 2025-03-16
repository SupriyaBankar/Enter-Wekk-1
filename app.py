import streamlit as st
import tensorflow as tf
import numpy as np 
import os #access folder
import cv2 #n for img resize

# Load model once (not inside function)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

#load and preprocess image
def model_predict(image_path):
    model = tf.keras.models.load_model('plant_disease_cnn_model.keras')
    img = cv2.imread(image_path)
    H, W, C=224, 224, 3  #set heigh widthand color
    img =cv2.resize(img, (H, W))
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =np.array(img)
    img =img.astype('float32')
    img =img/255.0
    img = img.reshape(1,H, W, C)
    
    prediction =np.argmax(model.predict(img), axis=-1)[0]
    return prediction

st.sidebar.title('Plant Disease Prediction System for Sustainablr Agriculture')
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognitiom'])

from PIL import Image
img = Image.open('Disease.jpg')
st.image(img)

if (app_mode =='Home'):
    st.markdown("<h1 style='text-align: center;'>Plant prediction System for Sustainable Agriculture</h1>")
elif(app_mode== 'Disease Recognition'):
    st.header("Plant prediction System for Sustainable Agriculture")
    test_image =st.file_uploader("choose an image:")
    
    if test_image is not None:
        save_path =os.path.join(os.getcwd(), test_image.name)
        print (save_path)
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())
            
    if(st.button("show img")):
        st.image(test_image, width=400, use_container_width=True)
        
    if(st.button("predict")):
        st.write("Our prediction")
        result_index = model_predict(save_path)
        print(result_index)
        
        class_name = { 'Apple___Apple_scab': 0,
            'Apple___Black_rot': 1,
            'Apple___Cedar_apple_rust': 2,
            'Apple___healthy': 3,
            'Blueberry___healthy': 4,
            'Cherry_(including_sour)___Powdery_mildew': 5,
            'Cherry_(including_sour)___healthy': 6,
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
            'Corn_(maize)___Common_rust_': 8,
            'Corn_(maize)___Northern_Leaf_Blight': 9,
            'Corn_(maize)___healthy': 10,
            'Grape___Black_rot': 11,
            'Grape___Esca_(Black_Measles)': 12,
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
            'Grape___healthy': 14,
            'Orange___Haunglongbing_(Citrus_greening)': 15,
            'Peach___Bacterial_spot': 16,
            'Peach___healthy': 17,
            'Pepper,_bell___Bacterial_spot': 18,
            'Pepper,_bell___healthy': 19,
            'Potato___Early_blight': 20,
            'Potato___Late_blight': 21,
            'Potato___healthy': 22,
            'Raspberry___healthy': 23,
            'Soybean___healthy': 24,
            'Squash___Powdery_mildew': 25,
            'Strawberry___Leaf_scorch': 26,
            'Strawberry___healthy': 27,
            'Tomato___Bacterial_spot': 28,
            'Tomato___Early_blight': 29,
            'Tomato___Late_blight': 30,
            'Tomato___Leaf_Mold': 31,
            'Tomato___Septoria_leaf_spot': 32,
            'Tomato___Spider_mites Two-spotted_spider_mite': 33,
            'Tomato___Target_Spot': 34,
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
            'Tomato___Tomato_mosaic_virus': 36,
            'Tomato___healthy': 37 
        }
        
        st.success("model is predicting that it is {}".format(class_name[result_index]))