import torch
import torch.nn as nn
import optuna
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
    
def create_seq(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)    

data = pd.read_csv('C:/Users/allwy/Documents/GitHub/demand-forecast/backend/Historical Product Demand.csv')
data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')
data = data.dropna()
data = data['Order_Demand'].values.astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data .reshape(-1, 1))

seq_length = 5
x, y = create_seq(data, seq_length)

train_size = int(len(y) * 0.67)
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
val_data_X = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
val_data_Y = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial):
    input_dim = 1
    output_dim = 1
    hidden_dim = int(trial.suggest_float("hidden_dim", 16, 256, log=True))
    num_layers = int(trial.suggest_int("num_layers", 1, 3))
    num_epochs = int(trial.suggest_int("num_epochs", 50, 200))
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 64
    train_dataset = TensorDataset(trainX, trainY)
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_iterator):
            current_batch_size = x_batch.size(0)
            x_batch = x_batch.view([current_batch_size, -1, 1]).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.view(-1,1))
  
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

    model.eval()
    valid_outputs = model(val_data_X.view([-1, seq_length, 1]).to(device))
    val_loss = criterion(valid_outputs, val_data_Y.view([-1,1]).to(device))

    return val_loss.item()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_trial = study.best_trial
print('Loss: {}'.format(best_trial.value))
print("Best hyperparameters: {}".format(best_trial.params))

model = LSTMModel(input_dim=1, hidden_dim=int(best_trial.params["hidden_dim"]), num_layers=int(best_trial.params["num_layers"]), output_dim=1)
model = model.to(device)
criterion = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params["learning_rate"])

num_epochs = best_trial.params["num_epochs"]
batch_size = 64
train_dataset = TensorDataset(trainX, trainY)
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
for epoch in range(num_epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_iterator):
        current_batch_size = x_batch.size(0)
        x_batch = x_batch.view([current_batch_size, -1, 1]).to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.view(-1,1))
  
        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'lstm_model.ckpt')