import numpy as np
from forward import forward_with_attention
from get_training_data import get_all_training_data
import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

'''
goal: simulate trading based on model
- simulation will start at a given time and run for a given length of time (sim_length)

input:
- w: weights
- b: biases
- n_layers: int representing number of layers
- input_size: int representing input size
- layer_sizes: list of ints representing layer sizes
- attn_weights: attention weights
- attn_query: attention query
- attn_keys: attention keys
- attn_values: attention values
- s_initial_start_time: string representing start time
- sim_length: int representing number of minutes to simulate

output:
- final_price: float representing final price
- next_thoughts: string representing next thoughts

'''

def transact(best_coin, current_coin, running_total, s_initial_start_time,symbols,current_prices):
    '''
    goal: simulate a transaction
    '''
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
    '''
    goal: get current price of symbol
    '''
    if symbol == 'USDC':
        return 1.0
    current_price = current_prices[symbols.index(symbol)]
    return float(current_price)

def get_symbol_data(args):
    '''
    wrapper function for get_all_training_data to allow for multiprocessing
    '''
    symbol, s_initial_start_time, n_prev, sim_length = args
    symbol_training_data, symbol_current_prices = get_all_training_data(symbol, s_initial_start_time, n_prev, sim_length)
    return symbol_training_data, symbol_current_prices

def get_symbol_results(args):
    '''
    wrapper function for forward_with_attention to allow for multiprocessing
    -runs forward_with_attention on each symbol's training data for each iteration of the simulation
    '''
    all_output = []
    symbol_training_data, n_prev, w, b, n_layers,attn_weights, attn_query, attn_keys, attn_values = args
    for training_data in symbol_training_data:
        output = forward_with_attention(training_data, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values)
        all_output.append(output)
    return all_output


def sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values,s_initial_start_time="2023-12-27 11:46:00"):
    stopwatch = datetime.now()
    running_total = 1000
    current_coin = 'USDC'
    sim_length = 500
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
        #print(f"Current coin: {current_coin}, Running total: {running_total}")
        #print(f"i: {i}, dollar value: {running_total*current_price}")
    final_price = running_total * current_price
    runtime = datetime.now() - stopwatch
    runtime = runtime.total_seconds()
    #print(f"Final price: {final_price}")
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

    final_price, next_thoughts = sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values)
    #print(final_price)