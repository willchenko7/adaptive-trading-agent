import pandas as pd
from get_symbols import get_symbols

def convert_csv_to_parquet(csv_filepath, parquet_filepath):
    df = pd.read_csv(csv_filepath)
    df.to_parquet(parquet_filepath)
    return

if __name__ == "__main__":
    symbols = get_symbols()
    for symbol in symbols:
        convert_csv_to_parquet(f'data/{symbol}-USD.csv', f'data/{symbol}-USD.parquet')
