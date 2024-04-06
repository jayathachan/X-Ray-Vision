import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import model
from model import create_model 

st.title("X-Ray Vision")
st.markdown("Extracing diagnostic insights from chest images")
col1, col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1", type=['png', 'jpg', 'jpeg'])
image_2 = None
if image_1:
    image_2 = col2.file_uploader("X-ray 2", type=['png', 'jpg', 'jpeg'])

predict_button = st.button('Predict on uploaded files')

@st.cache_resource
def create_model():
    model_tokenizer = model.create_model()
    return model_tokenizer

def predict(image_1, image_2, model_tokenizer):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB")  # converting to 3 channels
            image_1 = image_1.resize((300, 300))  # Resize image to 300x300
            image_1 = np.array(image_1) / 255
            if image_2 is None:
                image_2 = image_1
            else:
                image_2 = Image.open(image_2).convert("RGB")  # converting to 3 channels
                image_2 = image_2.resize((300, 300))  # Resize image to 300x300
                image_2 = np.array(image_2) / 255
            col1, col2 = st.columns(2)
            col1.image(image_1, width=300)
            col2.image(image_2, width=300)
            st.write("")  # Add empty line for spacing
            caption = model.function1([image_1], [image_2], model_tokenizer)
            st.markdown(" ### **Impression:**")
            impression = st.empty()
            impression.write(caption[0])
            time_taken = "Time Taken for prediction: %i seconds" % (time.process_time() - start)
            st.write(time_taken)
            del image_1, image_2
        else:
            st.markdown("## Upload an Image")

model_tokenizer = create_model()

predict(image_1, image_2, model_tokenizer)






