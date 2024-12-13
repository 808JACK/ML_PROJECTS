# Diabetes Patient Predictor

## What's This About?
Ever wonder if you are at risk for diabetes? This project uses machine learning to predict whether a patient is likely to develop diabetes based on medical data. 
It analyzes various features such as age, blood pressure, BMI, insulin levels, and more to make an informed prediction.

## The Dataset
I used the Kaggele diabetes patient dataset (given in repo too), which contains medical information and diabetes test results from patients. 
It's perfect for training a predictive model to determine the likelihood of diabetes.

## How It Works:

- **Data Collection:** The system uses medical data to train a machine learning model.
- **Data Preprocessing:** The raw data is cleaned and preprocessed to handle missing values, normalize data, and convert categorical variables.
- **Model Training:** Different machine learning algorithms, such as logistic regression, decision trees, or SVM, are trained on the data to predict diabetes risk.
- **Prediction:** Given a patient's data, the system predicts whether he or she is likely to develop diabetes.

## Tools:
1. **Python** for coding and implementing machine learning algorithms.
2. **Streamlit** for creating a user-friendly web interface for diabetes prediction.
3. **Pickle** for loading pre-trained models (the diabetes prediction model).
4. **NumPy** for dealing with the numerical data and actually doing the prediction.
5. **Scikit-learn** for building and training the machine learning model and rescaling the data.
6. **Pandas**: to manipulate and pre-process our data.
7. **HTML** to add custom styles, and even to print outputs with background colors in streamlit.
