import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title('Diabetes Prediction')

# Create input fields for user data
pregnancies = st.number_input('Number of Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0)
insulin = st.number_input('Insulin', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

# Define valid ranges for each input
valid_ranges = {
    'Number of Pregnancies': (0, 20),
    'Glucose Level': (0.0, 200.0),
    'Blood Pressure': (0.0, 200.0),
    'Skin Thickness': (0.0, 100.0),
    'Insulin': (0.0, 1000.0),
    'BMI': (0.0, 60.0),
    'Diabetes Pedigree Function': (0.0, 3.0),
    'Age': (0, 120)
}

# Validate inputs
errors = []
if not (valid_ranges['Number of Pregnancies'][0] <= pregnancies <= valid_ranges['Number of Pregnancies'][1]):
    errors.append('Number of Pregnancies is out of range.')
if not (valid_ranges['Glucose Level'][0] <= glucose <= valid_ranges['Glucose Level'][1]):
    errors.append('Glucose Level is out of range.')
if not (valid_ranges['Blood Pressure'][0] <= blood_pressure <= valid_ranges['Blood Pressure'][1]):
    errors.append('Blood Pressure is out of range.')
if not (valid_ranges['Skin Thickness'][0] <= skin_thickness <= valid_ranges['Skin Thickness'][1]):
    errors.append('Skin Thickness is out of range.')
if not (valid_ranges['Insulin'][0] <= insulin <= valid_ranges['Insulin'][1]):
    errors.append('Insulin is out of range.')
if not (valid_ranges['BMI'][0] <= bmi <= valid_ranges['BMI'][1]):
    errors.append('BMI is out of range.')
if not (valid_ranges['Diabetes Pedigree Function'][0] <= diabetes_pedigree_function <= valid_ranges['Diabetes Pedigree Function'][1]):
    errors.append('Diabetes Pedigree Function is out of range.')
if not (valid_ranges['Age'][0] <= age <= valid_ranges['Age'][1]):
    errors.append('Age is out of range.')

# Button for making predictions
if st.button('Predict'):
    if errors:
        for error in errors:
            st.error(error)
    else:
        features = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin,
            bmi, diabetes_pedigree_function, age
        ]])

        # Standardize the data
        standardized_data = scaler.transform(features)

        # Make prediction
        prediction = model.predict(standardized_data)

        # Display the result
        result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
        color = 'red' if result == 'Diabetic' else 'green'
        st.markdown(f'<div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">Prediction: {result}</div>', unsafe_allow_html=True)
