from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

# Define endpoint to handle form submission and forecasting
@app.route('/forecast', methods=['POST'])
def forecast():
    # Assuming request body contains the form data (productCode, warehouse, productCategory, date)
    data = request.json

    # Extract form data
    product_code = data.get('Product_Code')
    warehouse = data.get('Warehouse')
    product_category = data.get('Product_Category')
    date = data.get('Date')

    try:
        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'Product_Code': [product_code],
            'Warehouse': [warehouse],
            'Product_Category': [product_category],
            'Date': [date]
        })

        # Preprocess input data if needed (e.g., convert categorical variables to numerical)

        # Generate initial forecast using the ARIMA model
        initial_forecast = arima_model.forecast(steps=1, exog=input_data)  # Assuming forecasting for one step ahead

        # Optionally, you can post-process the initial forecast or incorporate additional steps here

        # Return the forecasted results to the frontend
        return jsonify({'forecast': initial_forecast})
    except Exception as e:
        # Handle errors
        print('Error generating forecast:', str(e))
        return jsonify({'error': 'Failed to generate forecast'}), 500

if __name__ == '__main__':
    app.run(debug=True)
