
'''
goal: given weights, find fitness level
'''

import numpy as np
from forward import forward, sigmoid
from get_training_data import get_training_data
import os
import pandas as pd
from datetime import datetime, timedelta

def sim(w,b,n_layers,input_size,layer_sizes):
    stopwatch = datetime.now()
    running_total = 1000
    current_coin = 'USDC'
    sim_length = 500
    s_initial_start_time = "2023-12-27 11:46:00"
    #convert start time to datetime object
    d_initial_start_time = pd.to_datetime(s_initial_start_time)
    n_prev = 1000
    symbols = [i.replace('-USD.csv','') for i in os.listdir('data') if i.endswith('.csv') and not i.startswith('tickers')]
    for i in range(0,sim_length):
        outputs = []
        for symbol in symbols:
            #print(symbol)
            training_data = get_training_data(symbol, s_initial_start_time, n_prev)
            output = forward(training_data, w, b, n_layers)
            outputs.append(output)
        if current_coin == 'USDC':
            #get best coin to buy (highest output)
            best_coin = symbols[np.argmax(outputs)]
            if outputs[symbols.index(best_coin)] < 0.5:
                #hold USDC
                best_coin = 'USDC'
        else:
            if outputs[symbols.index(current_coin)] > 0.5:
                #hold current coin
                best_coin = current_coin
            else:
                best_coin = symbols[np.argmax(outputs)]
                if outputs[symbols.index(best_coin)] < 0.5:
                    #sell current coin for best coin
                    best_coin = current_coin
        current_coin, running_total,current_price = transact(best_coin, current_coin, running_total, s_initial_start_time)
        print(f"Current coin: {current_coin}, Running total: {running_total}")
        print(f"start time: {s_initial_start_time}, dollar value: {running_total*current_price}")
        #update start time by adding 10 minutes to previous start time
        d_initial_start_time = d_initial_start_time + timedelta(minutes=10)
        s_initial_start_time = d_initial_start_time.strftime("%Y-%m-%d %H:%M:%S")
    final_price = running_total * current_price
    runtime = datetime.now() - stopwatch
    runtime = runtime.total_seconds()
    print(f"Final price: {final_price}")
    print(f"Runtime: {runtime}")
    return final_price

def transact(best_coin, current_coin, running_total, s_initial_start_time):
    if best_coin == current_coin:
        current_price = get_current_price(current_coin, s_initial_start_time)
        return current_coin, running_total,current_price
    best_coin_price = get_current_price(best_coin, s_initial_start_time)
    current_coin_price = get_current_price(current_coin, s_initial_start_time)
    #multiply running total by current coin price (this will now be in USDC)
    running_total = running_total * current_coin_price
    #multiple running total by .97 (to simulate 3% fee for each trade. gas fees fluctuate, so this is a rough estimate)
    running_total = running_total * .97
    #now convert running total to amount of best coin
    running_total = running_total / best_coin_price
    return best_coin, running_total, best_coin_price

def get_current_price(symbol, start_time):
    if symbol == 'USDC':
        return 1.0
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
    #get current price
    current_price = df.iloc[stop_index]['close']
    return float(current_price)

if __name__ == "__main__":
    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size) for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size) for size in layer_sizes]
    sim(w,b,n_layers,input_size,layer_sizes)