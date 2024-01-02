import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_training_data(symbol,start_time,n):
    #read in all data
    df = pd.read_csv(os.path.join('data',f"{symbol}-USD.csv"),usecols=['time','close','volume'])
    #df = pd.read_parquet(os.path.join('data', f"{symbol}-USD.parquet"))
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

def get_all_training_data(symbol, start_time, n, sim_length):
    all_training_data = []
    current_prices = []
    #df = pd.read_csv(os.path.join('data',f"{symbol}-USD.csv"),usecols=['time','close','volume'])
    df = pd.read_parquet(os.path.join('data', f"{symbol}-USD.parquet"))
    d_start_time = pd.to_datetime(start_time)
    for i in range(sim_length):
        try:
            stop_index = df[df['time'] == start_time].index[0]
        except:
            #get closest time prior to start time
            stop_index = df[df['time'] < start_time].index[0]
        start_index = stop_index - n
        #get n rows of data after start time
        df_slice = df.iloc[start_index:stop_index]
        #convert close and volume to numpy arrays
        close = df_slice['close'].to_numpy()
        volume = df_slice['volume'].to_numpy()
        current_price = close[-1]
        #normalize close and volume
        close = close/current_price
        volume = volume/volume[-1]
        #convert close and volume to single 2 x n array
        training_data = np.array([close,volume])
        all_training_data.append(training_data)
        current_prices.append(current_price)
        d_start_time = d_start_time + timedelta(minutes=10)
        start_time = d_start_time.strftime("%Y-%m-%d %H:%M:%S")
    return all_training_data, current_prices


if __name__ == "__main__":
    from datetime import datetime
    stopwatch = datetime.now()
    #training_data, current_price = get_training_data("BTC", "2023-12-28 11:45:00", 100)
    training_data, current_price = get_all_training_data("BTC", "2023-12-28 11:45:00", 100, 500)
    runtime = datetime.now() - stopwatch
    print(training_data)
    print(f"Current price: {current_price}")
    print(f"Runtime: {runtime}")