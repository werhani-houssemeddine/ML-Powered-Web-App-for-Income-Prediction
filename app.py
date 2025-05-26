from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import json

app = Flask(__name__)

model = joblib.load('income_model.joblib')
scaler = joblib.load('scaler.joblib')
le_income = joblib.load('label_encoder.joblib')
with open('columns.json', 'r') as f:
  model_columns = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
  try:
      # Get input data from request
      data = request.get_json()
      
      # Create DataFrame from input
      input_data = pd.DataFrame([data])
      
      # Identify categorical and numerical columns
      categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']
      numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                        'capital_loss', 'hours_per_week']
      
      # Convert numerical columns to float
      for col in numerical_cols:
          input_data[col] = input_data[col].astype(float)
      
      # One-hot encode categorical variables
      input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
      
      # Align columns with training data
      for col in model_columns:
          if col not in input_encoded.columns:
              input_encoded[col] = 0
      input_encoded = input_encoded[model_columns]
      
      # Scale the features
      input_scaled = scaler.transform(input_encoded)
      
      # Make prediction
      prediction = model.predict(input_scaled)[0]
      
      return jsonify({'prediction': int(prediction)})
  
  except Exception as e:
      return jsonify({'error': str(e)}), 400
  
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path.startswith('static/'):
        return send_from_directory('static', path[7:])
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
  app.run()