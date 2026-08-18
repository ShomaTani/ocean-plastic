"""
ERA5 10m風速(u10, v10)のダウンロード。
GLORYSと同じ領域(東経120-150, 北緯25-50)・期間(2018-2022)に合わせる。

事前準備:
  1. https://cds.climate.copernicus.eu/ でアカウント作成
  2. Personal Access Token を取得
  3. ~/.cdsapirc に以下を保存:
       url: https://cds.climate.copernicus.eu/api
       key: <あなたのトークン>
  4. pip install cdsapi --break-system-packages
  5. reanalysis-era5-single-levels のライセンスに同意
     (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences)

使い方: python download_era5_wind.py
"""
import zipfile
import shutil
from pathlib import Path
import cdsapi
import xarray as xr

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOWNLOAD = OUT_DIR / "era5_download_raw"   # 拡張子なし。zipでもncでもここに来る
FINAL_FILE = OUT_DIR / "era5_2018_2022_wind_120-150E_25-50N.nc"

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "year": ["2018", "2019", "2020", "2021", "2022"],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": ["00:00", "06:00", "12:00", "18:00"],  # 6時間毎
    "data_format": "netcdf",
    "download_format": "unarchived",   # これでzip化を回避
    "area": [50, 120, 25, 150],        # [North, West, South, East]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(str(RAW_DOWNLOAD))

# ---- 保険: 万一zipで返ってきた場合だけ解凍してマージする ----
if zipfile.is_zipfile(RAW_DOWNLOAD):
    print("(想定外) zip形式で受信 → 解凍してマージします")
    extract_dir = OUT_DIR / "_era5_extract_tmp"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(RAW_DOWNLOAD) as zf:
        zf.extractall(extract_dir)
    nc_files = sorted(extract_dir.glob("*.nc"))
    print(f"  中身: {[f.name for f in nc_files]}")
    if len(nc_files) == 0:
        raise RuntimeError("zipの中に.ncファイルが見つからない。中身を手動確認して")
    datasets = [xr.open_dataset(f) for f in nc_files]
    merged = xr.merge(datasets)  # u10, v10 を1つのDatasetにまとめる
    merged.to_netcdf(FINAL_FILE)
    for ds in datasets:
        ds.close()
    shutil.rmtree(extract_dir)
    RAW_DOWNLOAD.unlink()
else:
    print("単一.nc形式で受信 (想定通り)")
    RAW_DOWNLOAD.rename(FINAL_FILE)

print(f"done: {FINAL_FILE}")
# 中身の簡易確認
check = xr.open_dataset(FINAL_FILE)
print(f"variables: {list(check.data_vars)}")
print(f"dims: {dict(check.sizes)}")