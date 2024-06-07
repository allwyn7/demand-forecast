from flask import Flask, request, jsonify
from flask_cors import CORS
# from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from datetime import datetime,timedelta
from dateutil.parser import parse

import os
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client, set_proxy_version
import json

from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)

# Load the LSTM model and scaler
model = tf.keras.models.load_model('/Users/I527229/Documents/GitHub/demand-forecast/backend/models/lstm_model.h5')
scaler = joblib.load('/Users/I527229/Documents/GitHub/demand-forecast/backend/models/scaler.pkl')
labelEncoders = joblib.load('/Users/I527229/Documents/GitHub/demand-forecast/backend/models/encoders.pkl')

# Access the dataset
df = pd.read_csv('/Users/I527229/Documents/GitHub/demand-forecast/backend/Historical Product Demand.csv') 

def parse_date(date_string):
    try:
        # Try parsing with dateparser.parse first
        date_obj = parse(date_string)
        return date_obj.strftime("%Y-%m-%d")
    except Exception:
        try:
            # Fallback on strptime for certain formats not 
            # recognized by dateparser.parse. Modify accordingly
            date_obj = datetime.strptime(date_string, "%d-%m-%Y")
            return date_obj.strftime("%Y-%m-%d")
        except Exception:
            return None
        
@app.route('/forecast', methods=['POST'])
def forecast():
    prompt = request.json['Prompt']

    aicore_auth_url = os.getenv("AICORE_AUTH_URL")
    client_id = os.getenv("AICORE_CLIENT_ID")
    client_secret = os.getenv("AICORE_CLIENT_SECRET")
    resource_group = os.getenv("AICORE_RESOURCE_GROUP")
    base_url = os.getenv("AICORE_BASE_URL")

    set_proxy_version('gen-ai-hub')

    client = get_proxy_client(
                    client_id=client_id,
                    client_secret=client_secret,
                    auth_url=aicore_auth_url,
                    api_base=base_url,
                    resource_group=resource_group
                )

    llm = ChatOpenAI(
                    proxy_client=client,
                    proxy_model_name="gpt-4",
                    api_version="2023-05-15",
                    temperature=0.0
    )

    template = """
            Extract the parameters as given below from the text given below
            The parameters should be in json format as follows:

                Product_Code: should be the product code
                Warehouse: should be the warehouse of the product
                Produc_Category: should be the category of the product
                Date: should be the date on which the order demand of the product is asked and if the month is mentioned else ''
                days:  it should be a numerical value of the number of days the after which the demand should be known
                
            format the output as json with the following keys:
                Product_Code
                Warehouse
                Product_Category
                Date
                days

            text: {text}"""

    prompt_template = ChatPromptTemplate.from_template(template)

    messages = prompt_template.format_messages(text=prompt)
    response = llm(messages)

    content_dict = json.loads(response.content)
    print(content_dict)

    # Json data
    product_code = labelEncoders['Product_Code'].transform([content_dict["Product_Code"]])[0]

    warehouse = content_dict["Warehouse"]
    product_category = content_dict["Product_Category"]

    uf_Date = content_dict["Date"]
    Date = parse_date(uf_Date)

    days = content_dict["days"]
    # 

    product_code_dataframe = df.loc[(df['Product_Code'] == content_dict["Product_Code"])]
    print(product_code_dataframe.shape)

    if product_code_dataframe.shape[0] == 0:
        return jsonify({'error': 'No data found for this product_code.'}), 400
    
    if Date:
        Final_Date = Date
    elif days:
        # Get today's date
        date_obj = datetime.now()

        # Date after 7 days
        date_after = date_obj + timedelta(days=days)
        Final_Date = date_after

# If warehouse or product category is missing, find a relevant match from the dataset
    if not warehouse or not product_category:
 
        # Find the record with the closest date and the same product code
        nearest_date_record = product_code_dataframe.iloc[(pd.to_datetime(product_code_dataframe['Date'], format='%Y/%m/%d') - pd.to_datetime(Final_Date, format='%Y/%m/%d')).abs().argsort()[:1]]
        
        if not warehouse:
            warehouse = nearest_date_record['Warehouse'].item()
        if not product_category:
            product_category = nearest_date_record['Product_Category'].item()
 
    # Convert the date to UNIX timestamp
    date_timestamp = (datetime.strptime(Final_Date, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds()
 
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