import os
import pandas as pd
import numpy as np

def get_training_data(symbol,start_time,n):
    #read in all data
    df = pd.read_csv(os.path.join('data',f"{symbol}-USD.csv"))
    #convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    #get index of start time in time column
    try:
        stop_index = df[df['time'] == start_time].index[0]
    except:
        #get closest time prior to start time
        stop_index = df[df['time'] < start_time].index[0]
    start_index = stop_index - n
    #get n rows of data after start time
    df = df.iloc[start_index:stop_index]
    #convert close and volume to numpy arrays
    close = df['close'].to_numpy()
    volume = df['volume'].to_numpy()
    #normalize close and volume
    close = close/close[0]
    volume = volume/volume[0]
    #convert close and volume to single 2 x n array
    training_data = np.array([close,volume])
    #return training data
    return training_data

if __name__ == "__main__":
    training_data = get_training_data("BTC", "2023-12-27 11:46:00", 100)
    print(training_data)