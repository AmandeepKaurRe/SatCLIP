import pathlib

import pandas as pd

folder_path = pathlib.Path('/scratch/akaur64/GeoSSL/satclip/stra_con/')
csv_paths = [p for p in folder_path.iterdir() if p.suffix =='.csv']

df = pd.concat([
    pd.read_csv(path)
    for path in csv_paths
])

import os
for path in csv_paths:
    os.remove(path)

print(df.shape[0])
df.to_csv('/scratch/akaur64/GeoSSL/satclip/stra_con/index.csv', index=False)
