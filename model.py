import streamlit as st
import joblib
import pandas as pd

# Load the trained CatBoost Regressor model
model = joblib.load('best_cb_model.pkl')

# Title of the web app
st.title("CatBoost Regressor Deployment")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    Transaction_ID = st.sidebar.number_input("Transaction_ID", value=0)
    Customer_ID = st.sidebar.number_input("Customer_ID", value=0)
    Phone = st.sidebar.number_input("Phone", value=0)
    Zipcode = st.sidebar.number_input("Zipcode", value=0)
    Age = st.sidebar.slider("Age", 0, 100, 25)
    Year = st.sidebar.number_input("Year", value=2023)
    Total_Purchases = st.sidebar.number_input("Total_Purchases", value=0)
    Amount = st.sidebar.number_input("Amount", value=0.0)
    Total_Amount = st.sidebar.number_input("Total_Amount", value=0.0)
    Ratings = st.sidebar.slider("Ratings", 1.0, 5.0, 3.0)
    data = {
        'Transaction_ID': Transaction_ID,
        'Customer_ID': Customer_ID,
        'Phone': Phone,
        'Zipcode': Zipcode,
        'Age': Age,
        'Year': Year,
        'Total_Purchases': Total_Purchases,
        'Amount': Amount,
        'Total_Amount': Total_Amount,
        'Ratings': Ratings,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Store the user input data
input_df = user_input_features()

# Display the input data
st.subheader('User Input parameters')
st.write(input_df)

# Make prediction using the model
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(prediction)

st.write(f'Predicted Value: {prediction[0]}')
