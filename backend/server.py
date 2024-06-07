from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
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

# Access the dataset
df = pd.read_csv('/Users/I527229/Documents/GitHub/demand-forecast/backend/Historical Product Demand.csv') 

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    product_code = labelEncoders['Product_Code'].transform([data.get('Product_Code')])[0]
    warehouse = data.get('Warehouse')
    product_category = data.get('Product_Category')
    date = data.get('Date')  # Input date format 'YYYY-MM-DD'

    product_code_dataframe = df.loc[(df['Product_Code'] == data.get('Product_Code'))]
    print(product_code_dataframe.shape)

    if product_code_dataframe.shape[0] == 0:
        return jsonify({'error': 'No data found for this product_code.'}), 400

    # If warehouse or product category is missing, find a relevant match from the dataset
    if not warehouse or not product_category:
        # Convert the date format for 'df' comparison
        date_df_format = datetime.strptime(date, '%Y-%m-%d').strftime('%Y/%m/%d')

        # Find the record with the closest date and the same product code
        nearest_date_record = product_code_dataframe.iloc[(pd.to_datetime(product_code_dataframe['Date'], format='%Y/%m/%d') - pd.to_datetime(date_df_format, format='%Y/%m/%d')).abs().argsort()[:1]]
        
        if not warehouse:
            warehouse = nearest_date_record['Warehouse'].item()
        if not product_category:
            product_category = nearest_date_record['Product_Category'].item()

    # Convert the date to UNIX timestamp
    date_timestamp = (datetime.strptime(date, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds()

    warehouse = labelEncoders['Warehouse'].transform([warehouse])[0]
    product_category = labelEncoders['Product_Category'].transform([product_category])[0]

    try: 
        # Preprocessing 
        input_data = [product_code, warehouse, product_category, date_timestamp]
        # Reshape the data and scale it 
        input_data = scaler.transform([input_data]) 
        # Generate forecast using the LSTM model 
        forecast = model.predict(input_data) 
        # Convert forecasted data to JSON 
        result = {'forecast': float(forecast[0][0])/2.25} 
        return jsonify(result) 
    except Exception as e: 
        print('Error generating forecast:', str(e)) 
        return jsonify({'error': 'Failed to generate forecast'}), 500 

if __name__ == '__main__': 
    app.run(debug=True)