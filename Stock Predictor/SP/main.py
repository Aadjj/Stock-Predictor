import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math

START = "2017-07-27" # start date of data
TODAY = date.today().strftime("%Y-%m-%d")# today's date

st.title('Predict Stock Trends ')# Web App Title

stocks = ('TTM', 'RELI', 'IBM', 'SBIN.NS','MSFT')# selecting stocks that you want to predict, names are from yahoo finance
selected_stock = st.selectbox('Select dataset for prediction', stocks)# selection menu for selecting stocks, passing stocks tuple as argument
title = selected_stock # storing selected stock name in titles variable
num_prediction  = st.slider('Days of prediction:', 10, 30) # slider for selecting number of days of prediction and storing in variable

@st.cache # (memorize function executions) check 1. Name of func 2. Code of Func 3. Input parameters of function
def load_data(ticker): # calling function with selected_stock
    data = yf.download(ticker, START, TODAY) # download stock data from yf
    data.reset_index(inplace=True) # puts dates in first column
    return data

data_load_state = st.text('Loading data...') # display text
df = load_data(selected_stock) # load data
data_load_state.text('Loading data... done!') # display this text after data loads

st.subheader('Raw data') # subheading
st.write(df.tail()) # displays last 5 rows of df


close_data = df['Close'].values # returns only values of closing price from df
close_data = close_data.reshape((-1,1)) # tells numpy to rearrange the multi-dimensional array into one column and value number of rows

split_percent = 0.80 # defining data split percentage
split = int(split_percent*len(close_data)) # stores number of rows till 80% of data

close_train = close_data[:split] # 80% close data for testing
close_test = close_data[split:] # rest 20% close data for training

date_train = df['Date'][:split] # 80% date data for training
date_test = df['Date'][split:]  # 20% date data for testing

look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)



model = Sequential()

model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 15
model.fit(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))


trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground truth'
)
layout = go.Layout(
    title= selected_stock +" Stocks",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
st.plotly_chart(fig)
close_data = close_data.reshape((-1))

num_prediction=math.floor(num_prediction/10)*10
def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
trace1 = go.Scatter(
    x = df['Date'].tolist(),
    y = close_data,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Prediction'
)
layout = go.Layout(
    title= selected_stock +" Stocks",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
st.plotly_chart(fig)
