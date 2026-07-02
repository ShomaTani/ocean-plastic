import parcels
import plasticparcels
import xarray
import torch

print("parcels:", parcels.__version__)
print("plasticparcels:", plasticparcels.__version__)
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

import xarray as xr

ds = xr.open_dataset("./data/raw/test_glorys_2020_week1.nc")
print(ds)
# uo, voが time, latitude, longitude 次元で入ってるはず