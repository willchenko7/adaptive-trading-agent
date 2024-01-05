import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sim import acc_sim

input_size = 1000
layer_sizes = [500, 200, 100, 50, 1]
n_layers = len(layer_sizes)

#get the latest start time from btc
btc_df = pd.read_csv(os.path.join('data','BTC-USD.csv'))
latest_start_time = btc_df.iloc[0]['time']
latest_start_time = pd.to_datetime(latest_start_time)

model_name = "best_model.pkl"
w, b, attn_weights, attn_query, attn_keys, attn_values = pickle.load(open(os.path.join('models',model_name),'rb'))

#start_time is sim_length*10 minutes before latest_start_time

start_time = latest_start_time - timedelta(minutes=5000)
start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

print(f"Testing model: {model_name}")
print(f"Start time: {start_time}")

fitness, next_thoughts = acc_sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values,start_time)

print(f"Fitness: {fitness}")
print(f"Next thoughts: {next_thoughts}")
