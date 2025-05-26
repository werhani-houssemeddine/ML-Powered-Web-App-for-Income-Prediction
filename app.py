import streamlit as st
import pandas as pd
import joblib
import json

# Load model, scaler, and label encoder
model = joblib.load('income_model.joblib')
scaler = joblib.load('scaler.joblib')
le_income = joblib.load('label_encoder.joblib')
with open('columns.json', 'r') as f:
    model_columns = json.load(f)

st.title("Income Prediction App")

# Define the input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=0)
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
        'State-gov', 'Without-pay', 'Never-worked'
    ])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0)
    education = st.selectbox("Education", [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
        'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th',
        '10th', 'Doctorate', '5th-6th', 'Preschool'
    ])
    education_num = st.number_input("Education Num", min_value=0)
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces'
    ])
    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", min_value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=0)
    native_country = st.selectbox("Native Country", [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
        'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
        'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
        'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
        'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands'
    ])
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data
        data = {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": native_country
        }

        input_df = pd.DataFrame([data])
        categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                            'relationship', 'race', 'sex', 'native_country']
        numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                          'capital_loss', 'hours_per_week']

        for col in numerical_cols:
            input_df[col] = input_df[col].astype(float)

        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_columns]

        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted Income: {'>=50k' if int(prediction) == 0 else '<50k' }")
