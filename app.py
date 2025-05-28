import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load the model and data
try:
    pipe = joblib.load('pipe.pkl')
    df = joblib.load('df.pkl')
except Exception as e:
    st.error(f"Error loading model/data files: {e}")
    st.stop()

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (in kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
     '2560x1600', '2560x1440', '2304x1440']
)

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('Operating System', df['os'].unique())

# Predict Button
if st.button('Predict Price'):
    # Convert categorical to binary
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size
    except Exception as e:
        st.error(f"Error calculating PPI: {e}")
        st.stop()

    # Create query array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    try:
        prediction = pipe.predict(query)[0]
        st.success(f"The predicted price of this configuration is ₹{int(np.exp(prediction))}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
