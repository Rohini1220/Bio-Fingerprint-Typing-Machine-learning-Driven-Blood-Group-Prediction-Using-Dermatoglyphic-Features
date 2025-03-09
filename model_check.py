import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
try:
    model = load_model(r'C:\Users\vinna\Desktop\Projects\Blood Group Data\blood_group_fingerprint_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise
