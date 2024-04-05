from flask import Flask, request, jsonify 
from flask_cors import CORS 
from tensorflow.keras.models import load_model 
import numpy as np
import joblib 
import pandas as pd 
from datetime import datetime 

app = Flask(__name__) 
CORS(app) 

# Load the LSTM model and scaler 
model = load_model('models/lstm_model.h5') 
scaler = joblib.load('models/scaler.pkl') 
labelEncoders = joblib.load('models/encoders.pkl') 

# Define endpoint to handle form submission and forecasting 
@app.route('/forecast', methods=['POST']) 
def forecast(): 
    data = request.json 
    product_code = labelEncoders['Product_Code'].transform([data.get('Product_Code')])[0] 
    warehouse = labelEncoders['Warehouse'].transform([data.get('Warehouse')])[0] 
    product_category = labelEncoders['Product_Category'].transform([data.get('Product_Category')])[0] 
    date = data.get('Date') # Assuming the date is in 'YYYY-MM-DD' format 
    # Convert the date to UNIX timestamp 
    date = (datetime.strptime(date, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds() 
   
    try: 
        # Preprocessing 
        input_data = [product_code, warehouse, product_category, date]   
        # Reshape the data and scale it 
        input_data = scaler.transform([input_data]) 
        # Generate forecast using the LSTM model 
        forecast = model.predict(input_data) 
        # Convert forecasted data to JSON 
        result = {'forecast': float(forecast[0][0])} 
        return jsonify(result) 
    except Exception as e: 
        print('Error generating forecast:', str(e)) 
        return jsonify({'error': 'Failed to generate forecast'}), 500 

if __name__ == '__main__': 
    app.run(debug=True)