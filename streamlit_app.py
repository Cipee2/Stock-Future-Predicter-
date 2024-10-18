import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2023-07-30'

st.title('Stock Future Predictor')

use_input = st.text_input('Enter stock Ticker', 'AAPL')

if st.button('Predict'):
    df = yf.download(use_input, start, end)

    # Describing data 
    st.subheader('Data From 2010-2023')
    st.write(df.describe())

    # Closing Price vs Time Chart 
    st.subheader('Closing Price VS Time Chart')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], color='yellow')
    plt.title('Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

    # Closing Price vs Time Chart with 100 moving average 
    st.subheader('Closing Price VS Time Chart with 100 Moving Average')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red', label='100 Moving Average')
    plt.plot(df['Close'], color='yellow', label='Closing Price')
    plt.title('Closing Price with 100 MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    # Closing Price vs Time Chart with 100 & 200 moving average 
    st.subheader('Closing Price VS Time Chart with 100 & 200 Moving Averages')
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red', label='100 Moving Average')
    plt.plot(ma200, color='green', label='200 Moving Average')
    plt.plot(df['Close'], color='yellow', label='Closing Price')
    plt.title('Closing Price with 100 & 200 MAs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    # Splitting data into train and test 
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    st.write('Training data shape:', data_training.shape)
    st.write('Testing data shape:', data_testing.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load Model 
    model = load_model('model.h5')

    # Testing past 
    pass_100_days = data_training.tail(100)
    final_df = pd.concat([pass_100_days, data_testing], ignore_index=True)

    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
        
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Prediction
    y_predicted = model.predict(x_test)

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph 
    def plot_transparent_graph():
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.title('Prediction vs Original Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

    # Main function to call the plotting function
    def main():
        plot_transparent_graph()

    if __name__ == "__main__":
        main()
