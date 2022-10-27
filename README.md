# Bitcoin Price Prediction Using Continously Trained LSTM and Analysis With ARIMAX, SARIMAX, FBprophet, LSTM, NeuralProphet and Temporal-Fusion-Transformer
<a href="https://www.linkedin.com/in/tusharnautiyal/"> <img src = "https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a> <img src = "https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src = "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"/> 
<a href = 'https://huggingface.co/spaces/TusharNautiyal/BTC-Prediction' target = '_blank'>**Check Deployment**</a>
Perdicting **Bitcoin** Closing Prices using differnet Timeseries based model. Deployed using ***Streamlit and Hugging Face Spaces with LSTM Model*** That automatcally retrains every **10th day** For eg if last trained was **2022-10-16** then it will train automatically with all the new data on 10th day that is **2022-10-26**. Check out the **notebook** to understand why we chose **LSTM** Over others for deployment.
Check Deployment At hugging Face Spaces <a href = 'https://huggingface.co/spaces/TusharNautiyal/BTC-Prediction' target = '_blank'>**CLick Here**</a> To Visit

# Understanding Model
I Tried Multiple Time Series Model But after analysis each one of them i stick with **Stacked LSTM Base Model**. Our model has 3 ***LSTM*** Layers each with 150 units in which two have **return sequences** as true the last one is false. Return Sequence are true when stacking **LSTM** layers so that the second LSTM layer has sequence input.

![MC-LSTM](https://user-images.githubusercontent.com/74553737/198223324-24ee5118-e044-401c-91bc-2d0ca01ecec1.jpg)

Data was given in **10 timesteps** and was not multivariate so this data is basically not going to consider ***sentimental*** or ***inflatory values***. And this project is totally for educational purposes and not for any finacial purposes should not be used for financial gain.

# Understanding Auto Training.

The model auto-trains itself every 10th day of month this is done using **scheduler.py** to check wether it's time, if the time is right for retraining it will check wether on that day its done or not if not it will retrain the whole model using **build_model.py** with new data from **yahoo finance** for **Bitcoin** till the current date.
***build_model.py*** creates the model and also The data is ***preprocessed*** into timeseries format with 10 timesteps and process the inputs as 3d inputs from 1dimensional array and divided into X_train,y_train. Where X_train is having 10timesteps and y_train is result after 10timesteps of data which will be predicted by our model.

# Video Demonstration

https://user-images.githubusercontent.com/74553737/198229864-473fa5eb-797d-4b1a-a9fa-bf449e19a318.mp4


# Future Aspects
I will be creating one more model in future but this time we will be using sentimental and inflatory values to get as close as possible to real world scenarios. Currently we are not using any external factors for Bitcoin Closing price prediction and not feeding the external factors data to the model.
Secondly i will also like to make these prediction for each and every Crypto currency out there so the data collection procedure will be more better next time.
