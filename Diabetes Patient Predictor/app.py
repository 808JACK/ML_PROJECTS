import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Function to check if the patient's table exists
def check_patient_table_exists(patient_id):
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    table_name = f"patient_{patient_id}"
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    table_exists = cursor.fetchone()
    conn.close()
    return table_exists


# Create a patient-specific table if it doesn't exist
def create_patient_table(patient_id):
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    table_name = f"patient_{patient_id}"
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        date TEXT,
        pregnancies INTEGER,
        glucose_level REAL,
        blood_pressure REAL,
        skin_thickness REAL,
        insulin REAL,
        bmi REAL,
        diabetes_pedigree_function REAL,
        age INTEGER,
        prediction TEXT
    );
    """)
    conn.commit()
    conn.close()


# Function to validate the inputs
def validate_inputs(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function,
                    age):
    # Define valid ranges for each input
    if not (0 <= pregnancies <= 20):
        st.error("Invalid value for Number of Pregnancies. It should be between 0 and 20.")
        return False
    if not (0 <= glucose <= 300):
        st.error("Invalid value for Glucose Level. It should be between 0 and 300.")
        return False
    if not (0 <= blood_pressure <= 200):
        st.error("Invalid value for Blood Pressure. It should be between 0 and 200.")
        return False
    if not (0 <= skin_thickness <= 100):
        st.error("Invalid value for Skin Thickness. It should be between 0 and 100.")
        return False
    if not (0 <= insulin <= 1000):
        st.error("Invalid value for Insulin. It should be between 0 and 1000.")
        return False
    if not (10 <= bmi <= 100):
        st.error("Invalid value for BMI. It should be between 10 and 100.")
        return False
    if not (0 <= diabetes_pedigree_function <= 2):
        st.error("Invalid value for Diabetes Pedigree Function. It should be between 0 and 2.")
        return False
    if not (0 <= age <= 120):
        st.error("Invalid value for Age. It should be between 0 and 120.")
        return False

    return True


# Function to visualize patient data
def visualize_data(patient_id):
    conn = sqlite3.connect('patients.db')
    query = f"SELECT prediction, COUNT(*) FROM patient_{patient_id} GROUP BY prediction"
    data = pd.read_sql(query, conn)
    conn.close()

    # Bar chart visualization
    st.subheader("Diabetic vs Non-diabetic History")
    fig, ax = plt.subplots()
    ax.bar(data['prediction'], data['COUNT(*)'], color=['red', 'green'])
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')
    ax.set_title('Count of Diabetic vs Non-diabetic Predictions')
    st.pyplot(fig)


# Define the Streamlit app
st.title('Diabetes Prediction')

# Add login functionality
st.sidebar.subheader("Login")
patient_id = st.sidebar.text_input("Enter Patient ID")

if patient_id:
    if not check_patient_table_exists(patient_id):
        st.sidebar.warning("Patient ID not found. A new record will be created for this ID.")
        create_patient_table(patient_id)
    else:
        st.sidebar.success("Logged in successfully")

        # Show the previous data for this patient
        conn = sqlite3.connect('patients.db')
        query = f"SELECT * FROM patient_{patient_id}"
        patient_df = pd.read_sql(query, conn)
        conn.close()

        st.subheader(f"Patient Data for ID: {patient_id}")
        st.dataframe(patient_df)

else:
    st.sidebar.warning("Please enter your patient ID to log in.")

# Create input fields for user data
pregnancies = st.number_input('Number of Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0)
insulin = st.number_input('Insulin', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)


# Function to save patient data to the database
def save_patient_data(patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                      diabetes_pedigree_function, age, prediction):
    try:
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()

        table_name = f"patient_{patient_id}"
        cursor.execute(f"""
        INSERT INTO {table_name} (date, pregnancies, glucose_level, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction))

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        st.error(f"An error occurred while saving the data: {e}")


# Button for making predictions
if st.button('Predict'):
    # Validate inputs
    if validate_inputs(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function,
                       age):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                              bmi, diabetes_pedigree_function, age]])

        # Standardize the data
        standardized_data = scaler.transform(features)

        # Make prediction
        prediction = model.predict(standardized_data)

        # Display the result
        result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
        color = 'red' if result == 'Diabetic' else 'green'
        st.markdown(
            f'<div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">Prediction: {result}</div>',
            unsafe_allow_html=True)

        # Save patient data with prediction to the database
        save_patient_data(patient_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                          diabetes_pedigree_function, age, result)
        st.success("Data saved successfully!")

        # Heading for personalized analysis and graph
        st.subheader("Personalized Analysis")

        # Show visualizations if more than two rows exist
        if len(patient_df) >= 2:
            visualize_data(patient_id)
