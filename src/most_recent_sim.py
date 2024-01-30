import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sim import sim, transact

'''
goal: run a simulation on the most recent data, using the best model
-returns fitness and next thoughts
'''

def most_recent_sim():
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]
    n_layers = len(layer_sizes)
    sim_length = 500
    #get the latest start time from btc
    btc_df = pd.read_csv(os.path.join('data','BTC-USD.csv'))
    latest_start_time = btc_df.iloc[0]['time']
    latest_start_time = pd.to_datetime(latest_start_time)
    model_name = "best_model.pkl"
    w, b, attn_weights, attn_query, attn_keys, attn_values = pickle.load(open(os.path.join('models',model_name),'rb'))
    #start_time is sim_length*10 minutes before latest_start_time
    start_time = latest_start_time - timedelta(minutes=sim_length*10)
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    #print(f"Testing model: {model_name}")
    #print(f"Start time: {start_time}")
    fitness, next_thoughts, outputs, symbols, current_prices = sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values,start_time)
    #print(f"Fitness: {fitness}")
    #print(f"Next thoughts: {next_thoughts}")
    #get actual results
    #GET RUNNING RESULTS
    #read in running results
    running_results = pd.read_csv('running_results.csv')
    #get the last coin and amount
    current_coin = running_results.iloc[-1]['coin']
    old_current_coin = current_coin
    current_amount = running_results.iloc[-1]['amount']
    sell_mark = 0.1
    #transact logic
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
    current_coin, running_total,current_price = transact(best_coin, current_coin, current_amount, '',symbols,current_prices)
    b_executed_trade = False
    just_bought = ''
    if old_current_coin != current_coin:
        #append to running_results
        running_results = running_results._append({'timestamp':latest_start_time,'coin':current_coin,'amount':running_total},ignore_index=True)
        running_results.to_csv('running_results.csv',index=False)
        b_executed_trade = True
        just_bought = f'Just sold {old_current_coin} and bought {current_coin}'
        #print(just_bought)
    return just_bought

if __name__ == "__main__":
    jb = most_recent_sim()
    print(jb)
