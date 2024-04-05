import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load Data
data = pd.read_csv('/Users/I527229/Documents/GitHub/demand-forecast/backend/Historical Product Demand.csv')

# Handle missing values
data.dropna(inplace=True)

# Convert 'Date' to a numerical representation
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Remove parentheses and convert 'Order_Demand' to numeric
data['Order_Demand'] = data['Order_Demand'].str.replace('[\(\),]', '')
data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')

# Drop rows with NaN values in 'Order_Demand'
data.dropna(subset=['Order_Demand'], inplace=True)

# Apply Label Encoding
label_encoders = {}
for col in ['Product_Code', 'Warehouse', 'Product_Category']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save the encoders for future use
joblib.dump(label_encoders, 'encoders.pkl')

# Scale 'Order_Demand' using StandardScaler
scaler = StandardScaler()
data['Order_Demand'] = scaler.fit_transform(data[['Order_Demand']])

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Define X and y
X = data.drop(columns=['Order_Demand']).values
y = data['Order_Demand'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Define batch_sizes and epochs for grid search
batch_sizes = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]

# Start grid search
best_val_loss = float('inf')
best_hps = None

# Define cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for batch_size in batch_sizes:
    for epoch in epochs:
        print(f"Trying batch_size = {batch_size}, epochs = {epoch}")

        # Cross validate
        cv_scores = []
        for train_indices, val_indices in kfold.split(X_train):
            # Create a new model for each fold
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Make checkpoint for the best model
            checkpoint = tf_ModelCheckpoint('lstm_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # Fit the model and evaluate
            X_train_fold = X_train[train_indices].reshape(X_train[train_indices].shape[0], 1, X_train[train_indices].shape[1])
            X_val_fold = X_train[val_indices].reshape(X_train[val_indices].shape[0], 1, X_train[val_indices].shape[1])
            
            history = model.fit(X_train_fold, y_train[train_indices],
                                validation_data=(X_val_fold, y_train[val_indices]), epochs=epoch, batch_size=batch_size, callbacks=[early_stopping, checkpoint])
            score = model.evaluate(X_val_fold, y_train[val_indices])
            cv_scores.append(score)

        # Compute mean cross-validation score
        mean_val_loss = np.mean(cv_scores)
        print(f"Mean validation loss = {mean_val_loss}")

        # Update best score and hyperparameters
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_hps = (batch_size, epoch)
            best_model = model

print(f"Best hyperparameters found: batch_size = {best_hps[0]}, epochs = {best_hps[1]}")

# Save the model with best hyperparameters
best_model.save('lstm_best_model.keras')

# Save the history for future use
joblib.dump(history.history, 'model_history.pkl')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')

plt.title('Model loss progress during training/validation')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch number')
plt.legend(loc='upper right')

plt.show()
