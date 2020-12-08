import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from pandas import read_csv
from pandas.core.strings import str_extract
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/data/PycharmProjects/SCdemodata.csv')
mobility_data = pd.read_csv('./data/2020_US_Region_Mobility_Report.csv')
mobility_data = mobility_data[mobility_data['sub_region_2'] == 'San Diego County']
mobility_data = mobility_data.iloc[48:]
mobility_data = mobility_data[
    ['date', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
     'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
     'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
# mobility_data.to_csv('./data/mobility_dump.csv')
data = pd.merge(mobility_data, data, on=['date'], how='inner')
data.to_csv('./data/mobility_dump.csv')
data = data.drop(columns=['date'])
# data.to_csv('./data/mobility_dump.csv')

data = data.astype('float32')
x = data.iloc[:, 0:7]
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
print(x)

x = x[~np.isnan(x).any(axis=1)]
y = data.iloc[:, 7:8]
scaler1 = MinMaxScaler(feature_range=(0, 1))
y = scaler1.fit_transform(y)
y = y[~np.isnan(y).any(axis=1)]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test = x_test.reshape(x_test.shape[0], 1, x_train.shape[1])

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units=25, return_sequences=True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units=15))
regressor.add(Dropout(0.1))
# # Adding the output layer
regressor.add(Dense(units=1))
# # Compiling the RNN
regressor.compile(loss='mean_squared_error', optimizer='adam')
regressor.fit(X_train, y_train, epochs=50, batch_size=20, verbose=2, shuffle=False)
predictions = regressor.predict(X_test)
predictions = scaler1.inverse_transform(predictions)
prediction_output = pd.DataFrame(predictions).to_csv('/home/pravalli/PycharmProjects/sccases.csv')
# # # scores = regressor.evaluate(X_test, y_test,verbose=0, return_dict=True)
# # # print(scores)plt.plot(y_test, color = 'black', label = 'Actual R0')
plt.plot(predictions, color='green', label='Predicted Cases')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Cases')
plt.legend()
plt.show()
