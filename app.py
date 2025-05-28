import os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Debug info (remove after testing)
st.write("## Debug Info")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files present: {os.listdir()}")

@st.cache_resource
def load_artifacts():
    try:
        pipe = load('pipe.pkl')
        df = load('df.pkl')
        return pipe, df
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {str(e)}")
        st.stop()

pipe, df = load_artifacts()

st.title("💻 Laptop Price Predictor")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    laptop_type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', sorted([2, 4, 6, 8, 12, 16, 24, 32, 64]))
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5)
    touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
    
with col2:
    ips = st.radio('IPS Display', ['No', 'Yes'], horizontal=True)
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 15.6)
    resolution = st.selectbox('Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', 
        '3200x1800', '2880x1800', '2560x1600', '2560x1440'
    ])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# Storage inputs
hdd = st.select_slider('HDD (GB)', options=[0, 128, 256, 512, 1024, 2048], value=0)
ssd = st.select_slider('SSD (GB)', options=[0, 8, 128, 256, 512, 1024], value=256)

# Bottom section
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os_type = st.selectbox('Operating System', df['os'].unique())

if st.button('🚀 Predict Price', type='primary'):
    # Preprocess inputs
    try:
        ppi = ((int(resolution.split('x')[0])**2 + int(resolution.split('x')[1])**2)**0.5)/screen_size
        query = np.array([company, laptop_type, ram, weight, 
                         int(touchscreen == 'Yes'), int(ips == 'Yes'),
                         ppi, cpu, hdd, ssd, gpu, os_type]).reshape(1, -1)
        
        prediction = np.exp(pipe.predict(query)[0])
        st.success(f"### Predicted Price: ₹{int(prediction):,}")
        
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
