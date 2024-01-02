import os

def get_symbols():
    symbols = [i.replace('-USD.csv','') for i in os.listdir('data') if i.endswith('.csv') and not i.startswith('tickers')]
    return symbols