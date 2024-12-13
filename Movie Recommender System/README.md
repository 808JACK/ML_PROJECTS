# Diabetes Patient Predictor

## What's This About?
Have you ever wondered if you're at risk for diabetes? This project uses machine learning to predict whether a patient is likely to develop diabetes based on medical data. 
It analyzes various features such as age, blood pressure, BMI, insulin levels, and more to make an informed prediction.

## The Dataset
I used the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains medical information and diabetes test results from patients. It’s perfect for training a predictive model to determine the likelihood of diabetes.

## How It Works:
- **Data Collection:** The system uses medical data to train a machine learning model.
- **Data Preprocessing:** The raw data is cleaned and preprocessed to handle missing values, normalize data, and convert categorical variables.
- **Model Training:** Various machine learning algorithms, such as logistic regression, decision trees, or SVM, are trained on the data to predict diabetes risk.
- **Prediction:** Based on a patient’s data, the system predicts whether they are likely to develop diabetes.

## Tools:
1. **Python** for coding and implementing machine learning algorithms.
2. **Streamlit** for creating a user-friendly web interface for diabetes prediction.
3. **Pickle** for loading pre-trained models (the diabetes prediction model).
4. **NumPy** for handling numerical data and making predictions.
5. **Scikit-learn** for building and training the machine learning model and scaling the data.
6. **Pandas** for data manipulation and preprocessing.
