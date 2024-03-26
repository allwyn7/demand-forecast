import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset
data = pd.read_csv('C:/Users/allwy/Documents/GitHub/demand-forecast/backend/Historical Product Demand.csv')

# Convert 'Date' column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert 'Order_Demand' to numeric
data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce') 

# Fill missing values
data.ffill(inplace=True)

# Apply Label Encoder to the categorical columns
encoder = LabelEncoder()
encoded_df = data.copy()
encoded_df['Product_Code'] = encoder.fit_transform(data['Product_Code'])
encoded_df['Warehouse'] = encoder.fit_transform(data['Warehouse'])
encoded_df['Product_Category'] = encoder.fit_transform(data['Product_Category'])

# Sort the dataframe
encoded_df.sort_index(inplace=True)

# Set frequency as 'D' for daily
encoded_df = encoded_df.asfreq('D')

# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
m = 12

# Generate all different combinations of p, d, q and seasonal p, d, q triplets
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], m) for x in list(itertools.product(p, d, q))]

lowest_aic, pdq_optimal, seasonal_pdq_optimal = float('inf'), None, None
for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(encoded_df['Order_Demand'], exog=encoded_df[['Product_Code','Warehouse', 'Product_Category']], order=param, seasonal_order=seasonal_param)
            results = model.fit()
            if results.aic < lowest_aic:
                pdq_optimal = param
                seasonal_pdq_optimal = seasonal_param
                lowest_aic = results.aic
        except:
            continue

model_optimal = SARIMAX(encoded_df['Order_Demand'], exog=encoded_df[['Product_Code','Warehouse', 'Product_Category']], order=pdq_optimal, seasonal_order=seasonal_pdq_optimal)
results_optimal = model_optimal.fit()

with open('sarimax_model.pkl', 'wb') as f:
    pickle.dump((results_optimal, encoder), f)