import pandas as pd
import numpy as np

df = pd.read_csv("release_sites.csv")
idx = np.linspace(0, len(df) - 1, 30).astype(int)
df.iloc[idx].to_csv("release_sites_test30.csv", index=False)