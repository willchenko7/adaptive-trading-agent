import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

'''
goal: get training data for model
-training data is a 2 x n array where n is the number of minutes of data we want to use representing previous price

input:
-symbol: string representing symbol
-start_time: string representing start time
-n: int representing number of minutes of data to use
-sim_length: int representing number of minutes to simulate

output:
-training_data: list of 2 x n arrays representing training data
-current_price: float representing current price

'''

def get_all_training_data(symbol, start_time, n, sim_length):
    all_training_data = []
    current_prices = []
    df = pd.read_csv(os.path.join('data',f"{symbol}-USD.csv"),usecols=['time','close','volume'])
    d_start_time = pd.to_datetime(start_time)
    #for each 10 minute interval, get n rows of previous data
    for i in range(sim_length):
        #get index of start time, if start time is not in df, get closest time prior to start time
        try:
            start_index = df[df['time'] == start_time].index[0]
        except:
            #get closest time prior to start time
            start_index = df[df['time'] < start_time].index[0]
        #get index of stop time
        stop_index = start_index + n
        #get n rows of data after start time
        df_slice = df.iloc[start_index:stop_index]
        #convert close and volume to numpy arrays
        close = df_slice['close'].to_numpy()
        volume = df_slice['volume'].to_numpy()
        current_price = close[0]
        #normalize close and volume
        close = close/current_price
        volume = volume/volume[0]
        #convert to numpy array
        training_data = np.array(close)
        #add to all data
        all_training_data.append(training_data)
        current_prices.append(current_price)
        #increment start time by 10 minutes
        d_start_time = d_start_time + timedelta(minutes=10)
        start_time = d_start_time.strftime("%Y-%m-%d %H:%M:%S")
    return all_training_data, current_prices

if __name__ == "__main__":
    from datetime import datetime
    stopwatch = datetime.now()
    training_data, current_price = get_all_training_data("BTC", "2023-12-28 11:45:00", 100, 500)
    runtime = datetime.now() - stopwatch
    print(training_data)
    print(f"Current price: {current_price}")
    print(f"Runtime: {runtime}")
