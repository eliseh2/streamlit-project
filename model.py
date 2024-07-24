import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_linear_regression_model.pkl')
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
colorArr = ['silver' , 'white' , 'red' , 'blue' , 'brown' , 'black' , 'purple' , 'orange' , 'yellow'
 ,'green']
ownerArr = ['2ND' , '1ST' , '3RD' , '4TH']
fuelArr = ['DIESEL' , 'PETROL' , 'CNG']
transmissionArr = ['Manual' , 'Automatic']
body_typeArr = ['luxury%20suv' , 'hatchback' , 'sedan' , 'suv' , 'luxury%20sedan']


# Create form inputs
color = colorArr.index(st.selectbox('Color', colorArr)) + 1
owner = st.selectbox('Owner', [1, 2, 3, 4])
fuel = st.selectbox('Fuel', [1, 2, 3])
transmission = st.selectbox('Transmission', [1, 2])
body_type = st.selectbox('Body Type', [1, 2, 3, 4, 5])
kms_driven = st.number_input('Kms Driven', min_value=0)

airbags = st.selectbox('Airbags', [True, False])
alloy_wheels = st.selectbox('Alloy Wheels', [True, False])
cruise_control = st.selectbox('Cruise Control', [True, False])
steering_mounted_controls = st.selectbox('Steering Mounted Controls', [True, False])
sunroof_moonroof = st.selectbox('Sunroof/Moonroof', [True, False])

# Prepare input features
features = [color, owner, fuel, transmission, body_type, kms_driven,
            airbags, alloy_wheels, cruise_control, steering_mounted_controls, sunroof_moonroof]

# Make prediction
if st.button('Predict Price'):
    price = predict(features)
    st.write(f'Predicted Price: ₹{price:,.2f}')
