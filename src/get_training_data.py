import os
import pandas as pd
import numpy as np

def get_training_data(symbol,start_time,n):
    #read in all data
    #df = pd.read_csv(os.path.join('data',f"{symbol}-USD.csv"),usecols=['time','close','volume'])
    df = pd.read_parquet(os.path.join('data', f"{symbol}-USD.parquet"))
    #convert time to datetime
    #df['time'] = pd.to_datetime(df['time'])
    #get index of start time in time column
    try:
        stop_index = df[df['time'] == start_time].index[0]
        #stop_index = df['time'].searchsorted(start_time, side='right') - 1
        #start_index = max(stop_index - n, 0)
    except:
        #get closest time prior to start time
        stop_index = df[df['time'] < start_time].index[0]
    start_index = stop_index - n
    #get n rows of data after start time
    df = df.iloc[start_index:stop_index]
    #convert close and volume to numpy arrays
    close = df['close'].to_numpy()
    volume = df['volume'].to_numpy()
    current_price = close[-1]
    #normalize close and volume
    close = close/current_price
    volume = volume/volume[0]
    #convert close and volume to single 2 x n array
    training_data = np.array([close,volume]).astype(np.float32)
    #return training data
    return training_data, current_price

if __name__ == "__main__":
    from datetime import datetime
    stopwatch = datetime.now()
    training_data, current_price = get_training_data("BTC", "2023-12-28 11:45:00", 100)
    runtime = datetime.now() - stopwatch
    print(training_data)
    print(f"Current price: {current_price}")
    print(f"Runtime: {runtime}")