import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd



# Load the trained model, encoders and scalers
model = tf.keras.models.load_model('regression_model.h5')

# Loading Gender encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Loading Geography encoder
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

# Loading Scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Choose the Nationality', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Enter Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Enter your Credit Score')
exited = st.selectbox('User Exited or not', [0,1])
tenure = st.slider('Select Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Do you have a credit card', [0,1])
is_active_member = st.selectbox('Are you an active member', [0,1])

# Preparing the data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited], 
})

# One hot encoding Geography Column
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())


# Combining both
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

# Scaling
input_data_scaled = scaler.transform(input_df)


# Making the predictions
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'The predicted salary of the customer is : {prediction_prob}')