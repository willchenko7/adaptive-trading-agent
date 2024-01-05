import os
import pickle
import shutil
import numpy as np
from sim import acc_sim

input_size = 1000
layer_sizes = [500, 200, 100, 50, 1]
n_layers = len(layer_sizes)

start_times = ["2023-11-27 11:46:00","2023-12-01 11:46:00","2023-12-05 11:46:00","2023-12-09 11:46:00"]
models = [f for f in os.listdir('models') if f.endswith('.pkl')]

all_model_results = []
for model in models:
    model_results = []
    for start_time in start_times:
        w, b, attn_weights, attn_query, attn_keys, attn_values = pickle.load(open(os.path.join('models',model),'rb'))
        model_results.append(acc_sim(w,b,n_layers,input_size,layer_sizes,attn_weights, attn_query, attn_keys, attn_values,start_time))
    all_model_results.append(model_results)

#select model with best results
model_results = np.array(all_model_results)
model_results = np.mean(model_results,axis=1)
best_model_index = np.argmax(model_results)
best_model = models[best_model_index]
print(f"Best model: {best_model}")
#copy best model to best_model.pkl
shutil.copy(os.path.join('models',best_model),os.path.join('models','best_model.pkl'))