import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
classnames = ['Mild_Demented','Moderate_Demented','Non_Demented','Very_Mild_Demented']

# Load the model


# Streamlit app
def main():
    st.title("Alzheimers Prediction")
    uploaded_file = st.file_uploader("Upload your Brain MRI ")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img,caption='Uploaded Image.',use_column_width=True)
        model = tf.keras.models.load_model('./Alzheimer.h5')
        st.write(model.input_shape)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.repeat(img_array, 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        st.write(img_array.shape)
        prediction = model.predict(img_array)
        ind = np.argmax(prediction[0])
        st.write("Predicted Class:", classnames[ind])

if __name__ == '__main__':
    main()
