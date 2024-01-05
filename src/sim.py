
'''
goal: given weights, find fitness level
'''

import numpy as np
from forward import forward, sigmoid, forward_with_attention
from get_training_data import get_training_data, get_all_training_data
import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

def process_symbol(args):
    symbol, s_initial_start_time, n_prev, w, b, n_layers = args
    training_data, current_price = get_training_data(symbol, s_initial_start_time, n_prev)
    output = forward(training_data, w, b, n_layers)
    return output, current_price

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
        '''
        outputs = []
        for symbol in symbols:
            #print(symbol)
            training_data = get_training_data(symbol, s_initial_start_time, n_prev)
            output = forward(training_data, w, b, n_layers)
            outputs.append(output)
        '''
        tasks = [(symbol, s_initial_start_time, n_prev, w, b, n_layers) for symbol in symbols]
        #parallelize
        outputs = []
        current_prices = []
        with ProcessPoolExecutor() as executor:
            results = executor.map(process_symbol, tasks)
            #outputs = list(results)
            for output, current_price in results:
                outputs.append(output)
                current_prices.append(current_price)
        
        print(current_prices)
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
        current_coin, running_total,current_price = transact(best_coin, current_coin, running_total, s_initial_start_time,symbols,current_prices)
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
    #indicate potential next steps to user
    #1. what is current status of current coin? (hold, buy, sell)
    next_thoughts = []
    next_thoughts.append(f"Current coin: {current_coin}")
    #2. what is the best coin to buy next?
    best_coin = symbols[np.argmax(outputs)]
    next_thoughts.append(f"Best coin to buy next: {best_coin}")
    return final_price , ";" .join(next_thoughts)

def transact(best_coin, current_coin, running_total, s_initial_start_time,symbols,current_prices):
    if best_coin == current_coin:
        current_price = get_current_price(current_coin, s_initial_start_time,symbols,current_prices)
        return current_coin, running_total,current_price
    best_coin_price = get_current_price(best_coin, s_initial_start_time,symbols,current_prices)
    current_coin_price = get_current_price(current_coin, s_initial_start_time,symbols,current_prices)
    #multiply running total by current coin price (this will now be in USDC)
    running_total = running_total * current_coin_price
    #multiple running total by .97 (to simulate 3% fee for each trade. gas fees fluctuate, so this is a rough estimate)
    running_total = running_total * .97
    #now convert running total to amount of best coin
    running_total = running_total / best_coin_price
    return best_coin, running_total, best_coin_price

def get_current_price(symbol, start_time,symbols,current_prices):
    if symbol == 'USDC':
        return 1.0
    '''
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
    '''
    current_price = current_prices[symbols.index(symbol)]
    return float(current_price)

def get_symbol_data(args):
    symbol, s_initial_start_time, n_prev, sim_length = args
    symbol_training_data, symbol_current_prices = get_all_training_data(symbol, s_initial_start_time, n_prev, sim_length)
    return symbol_training_data, symbol_current_prices

def get_symbol_results(args):
    all_output = []
    symbol_training_data, n_prev, w, b, n_layers,attn_weights, attn_query, attn_keys, attn_values = args
    for training_data in symbol_training_data:
        #output = forward(training_data, w, b, n_layers)
        output = forward_with_attention(training_data, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values)
        all_output.append(output)
    return all_output


def acc_sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values,s_initial_start_time="2023-12-27 11:46:00"):
    stopwatch = datetime.now()
    running_total = 1000
    current_coin = 'USDC'
    sim_length = 500
    #s_initial_start_time = "2023-12-27 11:46:00"
    d_initial_start_time = pd.to_datetime(s_initial_start_time)
    n_prev = 1000
    sell_mark = 0.1
    symbols = [i.replace('-USD.csv','') for i in os.listdir('data') if i.endswith('.csv') and not i.startswith('tickers')]
    tasks = [(symbol, s_initial_start_time, n_prev, sim_length) for symbol in symbols]
    all_training_data = []
    all_current_prices = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(get_symbol_data, tasks)
        for training_data, current_price in results:
            all_training_data.append(training_data)
            all_current_prices.append(current_price)
    all_output = []
    tasks = [(training_data, n_prev, w, b, n_layers,attn_weights, attn_query, attn_keys, attn_values) for training_data in all_training_data]
    with ProcessPoolExecutor() as executor:
        results = executor.map(get_symbol_results, tasks)
        for output in results:
            all_output.append(output)
    for i in range(0,sim_length):
        outputs = [output[i] for output in all_output]
        #normalize outputs to 0-1 (0 for thw lowest output, 1 for the highest)
        outputs = [(output - min(outputs))/(max(outputs) - min(outputs)) for output in outputs]
        #outputs = [output/max(outputs) for output in outputs]
        #normalize outputs to a even distribution between 0 and 1
        #print(outputs)
        current_prices = [current_price[i] for current_price in all_current_prices]
        if current_coin == 'USDC':
            #get best coin to buy (highest output)
            best_coin = symbols[np.argmax(outputs)]
            if outputs[symbols.index(best_coin)] < sell_mark:
                #hold USDC
                best_coin = 'USDC'
        else:
            if outputs[symbols.index(current_coin)] > sell_mark:
                #hold current coin
                best_coin = current_coin
            else:
                best_coin = symbols[np.argmax(outputs)]
                if outputs[symbols.index(best_coin)] < sell_mark:
                    #sell current coin for best coin
                    best_coin = current_coin
        current_coin, running_total,current_price = transact(best_coin, current_coin, running_total, '',symbols,current_prices)
        print(f"Current coin: {current_coin}, Running total: {running_total}")
        print(f"i: {i}, dollar value: {running_total*current_price}")
    final_price = running_total * current_price
    runtime = datetime.now() - stopwatch
    runtime = runtime.total_seconds()
    print(f"Final price: {final_price}")
    #print(f"Runtime: {runtime}")
    #indicate potential next steps to user
    #1. what is current status of current coin? (hold, buy, sell)
    next_thoughts = []
    next_thoughts.append(f"Current coin: {current_coin}")
    #2. what is the best coin to buy next?
    best_coin = symbols[np.argmax(outputs)]
    next_thoughts.append(f"Best coin to buy next: {best_coin}")
    return final_price, ";" .join(next_thoughts)

if __name__ == "__main__":
    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    buffer = 1
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size)*buffer for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size)*buffer for size in layer_sizes]
    attention_layer_index = 0 
    layer_output_dim = layer_sizes[attention_layer_index]
    attn_dim = layer_output_dim 
    attn_query = np.random.rand(attn_dim).astype(np.float64)
    attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_weights = np.random.rand(attn_dim).astype(np.float64)

    final_price, next_thoughts = acc_sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values)
    #print(final_price)