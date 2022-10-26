import numpy as np
import pandas as pd
from Scheduler import isscheduled
from build_model import model_builder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import yfinance 
import requests
import streamlit as st
from datetime import datetime
import os
# Getting Current Date

def main():

    if 'Season' not in st.session_state:
        st.session_state['Season'] = True
    else:
        st.session_state['Season'] = False    
    with open("prevdays.txt",'r') as file:
        previous_date = file.readline()
        file.close()
    with open("isdone.txt",'r') as file:
        isdone = file.readline()
        file.close()
    isExist = os.path.exists('LSTM_build.h5')
    data = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata")
    current_date = data.json()['datetime'].split('T')[0]  
    if isExist == False and isdone == 'Done':
        with open("isdone.txt",'w') as file:
            file.write("Not Done")
            file.close() 
    
    with st.spinner(f'Wait for Model Updation It will Retrain Every 10th day from the retraining day which was on {previous_date} It Will take only 15-20 mins Please Wait **Do not close The window**...'):
        builder = model_builder()
        # We will Schedule Our APP to re build model after 10 days
        with open('isdone.txt') as file:
            isDone = file.readline()
            file.close()

            
        isDone = isDone.strip()
        if isscheduled()==True or isExist==False and isDone != 'Done':
        # Getting The data
            with open('started.txt','w') as file:
                file.write("True")
                file.close()
            X = yfinance.download('BTC-USD',start = '2017-01-01')
            X = X['Close']
            data = X
            # Preprocessing data

            X = X[:len(X)-30]
            X_test = data[len(X):len(data)]
            # Scaling
            scaled = MinMaxScaler()
            values = scaled.fit_transform(np.array(X).reshape(-1,1))
            X_train = values
            X_test = scaled.fit_transform(np.array(X_test).reshape(-1,1))

            # Creating Time Series data to Feed To LSTM
            X_train,y_train = builder.preprocess_data(X_train,10)
            X_test,y_test = builder.preprocess_data(X_test,10)
            X_train = X_train[...,np.newaxis] 
            X_test = X_test[...,np.newaxis]

            model = builder.fit(X_train,y_train)
            model.save('LSTM_build.h5',save_format = 'h5',overwrite = True)
            with open('isdone.txt','w') as file:
                file.write('Done')
                file.close()
            with open('prevdays.txt','w') as file:
                data = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata")
                date = data.json()['datetime'].split('T')[0]      
                file.write(date)
                file.close()
            
        if previous_date != current_date and isscheduled()==False and isExist==True:
            with open('isdone.txt','w') as file:      
                file.write('NotDone')
                file.close()
            with open("started.txt",'w') as file:
                file.write("False")
                file.close()
            
            model = keras.models.load_model('LSTM_build.h5')
        elif previous_date == current_date and isdone ==  'Done' and isExist == True:
            model = keras.models.load_model('LSTM_build.h5')
        elif isExist == False and isdone =='Done':
            st.write("There Might Be some issues or the model is rebuilding in background please try and reload application after 5mins.")
    if st.session_state['Season'] == True:
        if int(current_date.split("-")[1])>=10 or int(current_date.split("-")[1])<=2:
            st.snow()
            st.session_state['season'] = 'Winter'
        else:
            st.balloons()
            st.session_state['season']= 'Summer'
    
    with open("prevdays.txt",'r') as file:
        previous_date = file.readline()
        file.close()
    st.success(f"Model Is updated On {previous_date} and will next Update after 10 days ")

    # Creating Future Outcomes
    def forecast_day():
        df = yfinance.download('BTC-USD',start = '2017-01-01')
        df = df.reset_index()
        X_future = df['Close'].shift(-2)
        X_dates = df['Date'].shift(-2)
        X_dates = X_dates[len(X_dates)-12:len(X_dates)]
        X_dates = X_dates[...,np.newaxis]
        X_dates = builder.preprocess_data(X_dates,10)

        X_future = X_future[len(X_future)-12:len(X_future)]
        scaled = MinMaxScaler()
        X_future = scaled.fit_transform(np.array(X_future).reshape(-1,1))
        X_future,y_future = builder.preprocess_data(X_future,10)

        X_future = X_future[...,np.newaxis]
        result = model.predict(X_future)
        ans = scaled.inverse_transform(result)
        if ans>df['Close'].iloc[-1]:
            return 'Positive 🟢',ans
        else:
            return 'Negative 🔴 ',ans

    def create_sample_data(df,days):
        store_index = []
        for day in range(days):
            # Creating Temporary DataFrame
            dt = df.index + pd.Timedelta(days = 1)
            next_data = pd.DataFrame({'Close':[1]},index =[dt[-1]])
            df = pd.concat([df,next_data])
            store_index.append(dt[-1])
        return df,store_index

    # This function Forecast Prices For 10 Days or less.
    def forecast_timeline(X,days):
        if days>10:
            return False
        final_values = []
        temp_data = X.iloc[-1]
        for day in range(days):
            X = X.shift(-2)
            X_future = X[len(X)-12:len(X)]
            X = X.dropna()
            scaled = MinMaxScaler()
            X_future = scaled.fit_transform(np.array(X_future).reshape(-1,1))
            X_future,y_future = builder.preprocess_data(X_future,10)
            result = model.predict(X_future)
            X = X.to_list()
            X.append(scaled.inverse_transform(result).reshape(1)[0])
            X = pd.Series(X)
            final_values.append(scaled.inverse_transform(result).reshape(1)[0])
        final_values.insert(0,temp_data)
        return final_values
    
    def predict_future(days):
        df = yfinance.download('BTC-USD',start = '2017-01-01')
        df.reset_index(inplace = True)
        X = df['Close']
        future_Values = forecast_timeline(X,days)
        df = df[['Close','Date']]
        df.set_index('Date',inplace = True)
        final_df,store_index = create_sample_data(df,days)
        for i,index in enumerate(store_index):
            final_df['Close'].loc[index] = future_Values[i]
        
        return final_df,future_Values

    
        
    st.title('Bitcoin Price Prediction ₿')
    st.write(f"HOWDY! Wonderfull ***{st.session_state['season']}*** Season. Welcome to Bitcoin ( ₿ ) Price Prediction APP It will Predict Closing Price For Bitcoin. These Predictions are based on **LSTM** Model Trained Over Historical Bitcoin Data From **2017 till {previous_date}** . the Model retratins every 10th day, The Prediction are totally based on previous Closing Values so do not invest money based on Such Predictions. Its only for Educational Purposes and should not be used for finacial purpose.")
    st.write("Why LSTM ? Because it Performed well on the data, I used LSTM,ArimaMax,SariMax,Temporal Fusion transformer,FbProphet, NeuralProphet many different time series model for predictions in which lstm performed the best so I Selected LSTM. If you want To check how we came to conclusion. Check out https://shorturl.at/cwHM4 For Code.")
    one_day = False
    days = int(st.number_input('Enter no of Days for Prediction'))
    st.write("or you can Select one day prediction from below ***IMPORTANT*** After **Checkbox is Clicked** Do Not Press Submit It Will automatically Run")
    if st.checkbox(label = 'One Day Prediction'):
        one_day = True
    if one_day == True:
        days = 1
    if st.button('Submit') and days<=10 and days>0:
        dataframe,values = predict_future(days)
        data = requests.get("http://worldtimeapi.org/api/timezone/Asia/Kolkata")
        date = data.json()['datetime'].split('T')[0]
        st.line_chart(data=dataframe)
        for i in range(len(values)-1):
            if values[i+1]>values[i]:
                st.write(f'Day{i+1}:Positive Growth 🟢')
            else:
                st.write(f'Day{i+1}:Negative Growth 🔴')
    if days == 1:
        result,ans = forecast_day()
        st.markdown(f"Tommorow **Bitcoin** Can Show {result} Movement.")
        st.markdown(f"And Price Can Be Around : $ {ans.reshape(1)[0]}.")
    elif days>10 or days<0:
        st.write("Please Renter Days As you have exceeded 10 days limit or the input is too small, If you think everything is correct still it's showing wront output please check if you are entering any spaces while input or send us feedback at info@tusharnautiyal.ml")
if __name__ == '__main__':
    main()