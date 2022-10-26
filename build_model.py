import numpy as np
import pandas as pd
from tensorflow import keras
from keras import Sequential 
from keras.layers import Dense,Dropout,LSTM


class model_builder:
    def __init__(self):
        self.model = None
    
    def preprocess_data(self,dataset,time_step):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    def create_model(self,X_train):
        model = Sequential()
        model.add(LSTM(150,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dropout(0.2))

        model.add(LSTM(150,return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(150))
        model.add(Dropout(0.2))

        model.add(Dense(1))
        model.compile(loss='mean_squared_error' , metrics = ['mse', 'mae'],optimizer='adam')
        return model
    
    def fit(self,X_train,y_train):
        model = self.create_model(X_train)
        model.fit(X_train,y_train,epochs=300,batch_size=64,verbose=1)
        self.model = model
        return model
        