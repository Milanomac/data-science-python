# Description:  this program uses an artificial recurrent neural
#               network called LSTM (long short term memory) to 
#               predict the closing stock price of a corporation 
#               (Apple Inc.) using past 60 days stock price.

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import joblib

# Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2022-07-17')

# show the data
# print(df)

# Get the number of rows and columns in the data set
# print(df.shape)

# Visualize the closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# Conver the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model (80% of dataset)
training_data_len = math.ceil(len(dataset) * 0.8) # math.ceil rounds that up
# print(training_data_len)

# scale the data (normalization so all the data points are between 0 and 1)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
# -----------------------------------------------------
# creating training data set
# -----------------------------------------------------
# creating scaled training data set
train_data = scaled_data[0:training_data_len , :]
# train_data contains all values of scaled_data from index 0 to training_data_len
# , :] ---> get back all the columns

# split data into x_train and y_train data sets
x_train = [] # independent training variables
y_train = [] # target variables

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0]) # never reaches i and we get 0th column
    y_train.append(train_data[i, 0])
    # x_train will contain 60 values (position 0 to 59) and y_train will contain 61st value at position 60
    # y_train is the one value we are going to predict
    # if i <= 60:
        # print(x_train)
        # print(y_train) # this we want to predict

# convert the x_train and y_train to numpy arrays
x_train = np.array(x_train) 
y_train = np.array(y_train)
# reshape the data because the LSTM network expects the input to be 3-d in form of
# 1. number of samples; 
# 2. number of time steps;
# 3. number of features
# and right now our x_train data set is 2-d

# print(x_train.shape) # (1543, 60) are rows (left), columns (right)
    # # x_train = np.reshape(x_train, (1543, 60, 1)) where 1 is the closing price (1 feature)
    # # to be more robust we can write it as:
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# -----------------------------------------------------
# build the LSTM model
# -----------------------------------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False)) # false because we are not gonna be using LSTM in following layers
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Save the training data set

filename = 'model.sav'
joblib.dump(model,filename)

# -----------------------------------------------------
# Create the testing data set
# -----------------------------------------------------
# Create a bew array containing scaked values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0]) # x_test data set contains past 60 values and y_test contains the actual 61 values

# Convert the data to a numpy array so that we can use it in LSTM model
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)

# inverse transforming: we want predictions to contain the same values as y_test
predictions = scaler.inverse_transform(predictions)

# Evaluate the model: get the root mean squared error (RMSE): lower RMSE is better fitted
# rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
# print(rmse)

# plot the data
train = data[:training_data_len] # Contains data from index 0 to training_data_len
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='12-01-01', end='2022-07-17')

# Create new dataframe
new_df = apple_quote.filter(['Close'])

# Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values

# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list
X_test = []

# Append the past 60 days
X_test.append(last_60_days_scaled)

# Convert the X_test to np array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test)

# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price) # that is the predicted price for July 18th, 2022

# Actual price 
apple_quote_2 = web.DataReader('AAPL', data_source='yahoo', start='2022-07-18', end='2022-07-18')
print(apple_quote_2['Close'])
