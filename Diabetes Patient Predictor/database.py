import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st

# Connect to the SQLite database with context manager
def get_db_connection():
    return sqlite3.connect("patients.db")

# Function to check if a table exists
def check_table_exists(cursor, table_name):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

# Function to create a new patient table
def create_patient_table(cursor, patient_id):
    table_name = f"patient_{patient_id}"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT PRIMARY KEY,
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

# Function to check if the input is valid
def is_valid_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    if pregnancies < 0 or pregnancies > 20:
        return False
    if glucose < 0 or glucose > 300:
        return False
    if blood_pressure < 0 or blood_pressure > 200:
        return False
    if skin_thickness < 0 or skin_thickness > 100:
        return False
    if insulin < 0 or insulin > 800:
        return False
    if bmi < 10 or bmi > 100:
        return False
    if diabetes_pedigree_function < 0 or diabetes_pedigree_function > 2:
        return False
    if age < 0 or age > 120:
        return False
    return True

# Function to insert data into a patient table with validation
def insert_patient_data(cursor, patient_id, data):
    date, pregnancies, glucose_level, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction = data

    # Validate inputs
    if not is_valid_input(pregnancies, glucose_level, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
        st.error("Invalid input values. Please ensure that the values are within humanly possible ranges.")
        return

    table_name = f"patient_{patient_id}"
    cursor.execute(f"""
        INSERT INTO {table_name} (date, pregnancies, glucose_level, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, data)

# Simulate time-series data if not available
def simulate_time_series_data(patient_id, num_days=30):
    simulated_data = []
    for i in range(num_days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
        glucose_level = 100 + (i % 10) * 5  # Simulating glucose level variations
        bmi = 25 + (i % 5) * 0.2  # Simulating BMI variations
        data = (date, 3, glucose_level, 80, 25, 130, bmi, 0.627, 45, "Positive")
        simulated_data.append(data)
    return simulated_data

# Function to analyze patient data and generate visualizations
def analyze_patient_data(cursor, patient_id, show_new=False):
    table_name = f"patient_{patient_id}"
    # Fetch all data
    data = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
    if not data:
        print(f"No data available for patient {patient_id}.")
        return

    df = pd.DataFrame(data,
                      columns=["date", "pregnancies", "glucose_level", "blood_pressure", "skin_thickness", "insulin",
                               "bmi", "diabetes_pedigree_function", "age", "prediction"])

    # Streamlit slider for selecting date range
    start_date = st.date_input("Start Date", min_value=min(pd.to_datetime(df["date"])))
    end_date = st.date_input("End Date", max_value=max(pd.to_datetime(df["date"])))

    # Filter data based on selected date range
    df_filtered = df[(pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(df["date"]) <= pd.to_datetime(end_date))]

    # Plotly visualization 1: Glucose levels over time
    fig_glucose = px.line(df_filtered, x='date', y='glucose_level', title=f"Glucose Levels Over Time for Patient {patient_id}",
                          labels={'date': 'Date', 'glucose_level': 'Glucose Level'})
    st.plotly_chart(fig_glucose)

    # Plotly visualization 2: BMI over time
    fig_bmi = px.line(df_filtered, x='date', y='bmi', title=f"BMI Over Time for Patient {patient_id}",
                      labels={'date': 'Date', 'bmi': 'BMI'})
    st.plotly_chart(fig_bmi)

    # Plotly visualization 3: Prediction trends (positive/negative)
    prediction_counts = df_filtered["prediction"].value_counts()
    fig_prediction = px.bar(prediction_counts, x=prediction_counts.index, y=prediction_counts.values,
                            title=f"Prediction Trend for Patient {patient_id}",
                            labels={'x': 'Prediction', 'y': 'Count'})
    st.plotly_chart(fig_prediction)

    if show_new:
        # Update visualization after new prediction (simulate new prediction)
        new_data = (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Current timestamp
            3,  # Pregnancies
            110.5,  # Updated glucose level
            82,  # Updated blood pressure
            26,  # Updated skin thickness
            135,  # Updated insulin
            33.0,  # Updated BMI
            0.630,  # Updated Diabetes Pedigree Function
            45,  # Age
            "Positive"  # Updated Prediction
        )
        # Insert new prediction data into the database
        insert_patient_data(cursor, patient_id, new_data)
        st.write("New prediction data inserted and visualizations updated.")

# Function to handle patient data and database operations
def handle_patient_data(patient_id, data):
    try:
        # Establish connection using context manager
        with get_db_connection() as conn:
            cursor = conn.cursor()

            table_name = f"patient_{patient_id}"

            # Create table and insert data
            if check_table_exists(cursor, table_name):
                # Existing patient: Insert data and analyze if more than 2 rows exist
                insert_patient_data(cursor, patient_id, data)
                row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                show_new = False
                if row_count > 2:
                    show_new = st.checkbox("Show updated data with new prediction", value=False)
                    analyze_patient_data(cursor, patient_id, show_new=show_new)
            else:
                # New patient: Create table and insert data
                create_patient_table(cursor, patient_id)
                # Simulate time-series data for new patient
                simulated_data = simulate_time_series_data(patient_id)
                for row in simulated_data:
                    insert_patient_data(cursor, patient_id, row)
                print(f"New table created and simulated data added for patient {patient_id}.")
                analyze_patient_data(cursor, patient_id, show_new=False)

            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

