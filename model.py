import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import locale

# Set locale to Indian
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# Load the trained model
model = joblib.load('best_catboost_regression_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you have saved the scaler too

# Function to make predictions
def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit app
st.title('Car Price Prediction')

# Create selection Arrays
ownerArr = ['2ND' , '1ST' , '3RD' , '4TH']
fuelArr = ['DIESEL' , 'PETROL' , 'CNG']
transmissionArr = ['Manual' , 'Automatic']
body_typeArr = ['luxury%20suv' , 'hatchback' , 'sedan' , 'suv' , 'luxury%20sedan']
companyArr = ['Mahindra' , 'Jeep' , 'Honda' , 'Toyota' , 'Hyundai' , 'Ford' , 'Maruti' , 'Renault',
 'Nissan', 'Volkswagen', 'Datsun', 'Tata', 'Skoda', 'KIA', 'MG',]

col1, col2 = st.columns(2)

# Create form inputs
with col1:
    owner = ownerArr.index(st.selectbox('Owner', ownerArr)) + 1
    fuel = fuelArr.index(st.selectbox('Fuel', fuelArr)) + 1
    transmission = transmissionArr.index(st.selectbox('Transmission', transmissionArr)) + 1
    body_type = body_typeArr.index(st.selectbox('Body Type', body_typeArr)) + 1
    company = companyArr.index(st.selectbox('Company' , companyArr)) + 1
    kms_driven = st.number_input('Distance Driven (Km)', min_value=0, step=5000)

with col2:
    airbags = st.selectbox('Airbags', [True, False])
    alloy_wheels = st.selectbox('Alloy Wheels', [True, False])
    cruise_control = st.selectbox('Cruise Control', [True, False])
    steering_mounted_controls = st.selectbox('Steering Mounted Controls', [True, False])
    sunroof_moonroof = st.selectbox('Sunroof/Moonroof', [True, False])
    year = st.number_input('Year', min_value=2012, max_value=2022)

# Prepare input features
features = [owner, fuel, transmission, body_type, kms_driven,
            airbags, alloy_wheels, cruise_control, steering_mounted_controls, sunroof_moonroof, year, company]

# Make prediction
if st.button('Predict Price'):
    price = predict(features)
    st.write(f'Predicted Price: ₹{locale.currency(price, grouping=True)}')
