import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
classnames = ['Mild_Demented','Moderate_Demented','Non_Demented','Very_Mild_Demented']

# Load the model
model = tf.keras.models.load_model('./Alzheimer.h5')

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit app
def main():
    st.title("Alzheimers Prediction")
    uploaded_file = st.file_uploader("Upload your Brain MRI ")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,caption='Uploaded Image.',use_column_width=True)
        image_array = preprocess_image(image)
        prediction = model.predict(image_array)
        ind = np.argmax(prediction[0])
        st.write("Predicted Class:", classnames[ind])

if __name__ == '__main__':
    main()
