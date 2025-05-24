import pandas as pd

for i, chunk in enumerate(pd.read_csv('data_timeseries_columns.csv', chunksize=100000)):
    chunk.to_csv(f"{i+1}.csv", index=False)