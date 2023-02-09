#Install if you dont have:
#python -m pip install pandas
#python -m pip install numpy
#python -m pip install keras

from smtpd import DebuggingServer
import numpy as np
import pandas as pd

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
# print(concrete_data.head())
concrete_data.head()

# see how many data points there are (samples to train our model)
# print(concrete_data.shape)
concrete_data.shape

# checking the dataset for missing values
concrete_data.describe()
concrete_data.isnull().sum

# Split data into predictors and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']
# sanity check (first few rows of predictors: see that it does not have column for the target)
print(predictors.head())

# normalizing the the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
# print(predictors_norm.head())

# Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors_norm.shape[1] # number of predictors
# ------------------------------------------------------------------------------------------------------------

# BUILDING A NEURAL NETWORK

import keras

from keras.models import Sequential
from keras.layers import Dense

def regression_model():
    model = Sequential()
    # Dense adds another subsequent hidden layers and an output layer where 50 is the number of neurons 
    model.add(Dense(45, activation ='relu' , input_shape=(n_cols,)))
    model.add(Dense(30, activation = 'relu'))
    model.add(Dense(1))
    
    #compile model
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
    return model

# TRAINING AND TESTING THE NETWORK

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=200, verbose=2)

# link to learn more about other functions used for prediction or evaluation https://keras.io/api/models/sequential/

