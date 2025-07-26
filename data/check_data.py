import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

HISTORICAL_DATA_FILE = Path("outputs/all_symbols_history.parquet")
df = pd.read_parquet(HISTORICAL_DATA_FILE)
groups = df.groupby('symbol')

group_shapes = []

for symbol, group_df in groups:
    shape = np.shape(group_df)
    date_min = group_df['date'].min()
    date_max = group_df['date'].max()
    group_shapes.append({'symbol': symbol, 'rows': shape[0], 'columns': shape[1], 'date_min':date_min, 'date_max':date_max})

output_file = Path("outputs/group_shapes.csv")
pd.DataFrame(group_shapes).to_csv(output_file, index=False)

print(f"Group shapes saved to {output_file}")


# 1497402
# 1497384
# 1497366