import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/data/PycharmProjects/pythonProject/Demodatalstmsc.csv')
data = data.rolling(window=10).mean()
data = data[10:]
# data['conv_cases'] = data['cases'].rolling(window=5).sum()
# data = data[['Weather', 'Testing Rate', 'conv_cases', 'R0Delta']]
# data = data[5:]
data.to_csv('/data/PycharmProjects/test.csv')
x = data.iloc[:, 2:3]
y = data.iloc[:, 3:4]
test_data = pd.read_csv('/data/PycharmProjects/Demodatalstmtestsc.csv')
test_data['conv_cases'] = test_data['cases'].rolling(window=7).sum()
test_data = test_data[['Weather', 'Testing Rate', 'conv_cases', 'R0Delta']]
test_data = test_data[5:]


xt = test_data.iloc[:, 2:3]
print(xt)

sc = StandardScaler()
#feature_range=(0, 1)
x = sc.fit_transform(x)
x = x[~np.isnan(x).any(axis=1)]
sc1 = StandardScaler()
y = sc1.fit_transform(y)
y = y[~np.isnan(data).any(axis=1)]
sc2 = StandardScaler()
xt = sc2.fit_transform(xt)
print(xt)
# Define train size and test size.
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, test_size=0.15)
x_train, y_train, x_test = np.array(x), np.array(y), np.array(xt)
# Reshape the data to allow for training LSTM.
X_train = x.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
# Set LSTM network's hyper parameters.
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units=25, return_sequences=True))
regressor.add(LSTM(units=15))
regressor.add(Dropout(0.1))
# Adding the output layer.
regressor.add(Dense(units=1))
# Compiling the RNN.
regressor.compile(loss='mean_squared_error', optimizer='adam')
regressor.fit(X_train, y_train, epochs=50, batch_size=20, verbose=2, shuffle=False)
predictions = regressor.predict(X_test)
predictions1 = sc1.inverse_transform(predictions)
prediction_output = pd.DataFrame(predictions1,columns=['predicted_R0']).to_csv('/data/PycharmProjects/lstmoutputsc.csv')
plt.plot(predictions, color='green', label='Predicted R0')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('R0')
plt.legend()
plt.show()
