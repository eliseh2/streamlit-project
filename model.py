import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved models
cb_model = joblib.load('best_cb_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Define the input features in the correct order
input_features = ['Year', 'Age', 'Total_Purchases', 'Amount', 'Ratings']

# Define a function to preprocess input data
def preprocess_data(year, age, total_purchases, amount, ratings):
    input_data = pd.DataFrame({
        'Year': [year],
        'Age': [age],
        'Total_Purchases': [total_purchases],
        'Amount': [amount],
        'Ratings': [ratings]
    })
    
    # Ensure the columns are in the same order as during training
    input_data = input_data[input_features]
    
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    return input_pca

# Streamlit user interface
st.title('CatBoost Model Prediction App')
st.header('Enter the details for prediction')

year = st.number_input('Year', min_value=2000, max_value=2024, value=2020)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
total_purchases = st.number_input('Total Purchases', min_value=0, max_value=1000, value=10)
amount = st.number_input('Amount', min_value=0.0, value=100.0)
ratings = st.number_input('Ratings', min_value=0.0, max_value=5.0, value=4.0)

if st.button('Predict'):
    input_data = preprocess_data(year, age, total_purchases, amount, ratings)
    prediction = cb_model.predict(input_data)
    st.write(f'Predicted Total Purchases: {prediction[0]}')

# Display the raw data
st.header('Raw Data')
data = pd.read_csv('C:\\Users\\Elise\\Downloads\\processed_data.csv')
st.write(data)
