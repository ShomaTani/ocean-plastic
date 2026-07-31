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

使い方: python download_era5_wind.py
"""
import cdsapi

c = cdsapi.Client()

# GLORYSと解像度感を揃えるため、まずは日次相当(6時間毎)を取得。
# windageの効果を見るだけなら日次平均でも十分だが、季節風のような
# 数日スケールの現象を捉えるには6時間毎の方が無難。
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": ["2018", "2019", "2020", "2021", "2022"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],  # 6時間毎
        # area: [North, West, South, East]  ※CMEMSとは順序が違うので注意
        "area": [50, 120, 25, 150],
        "data_format": "netcdf",
    },
    "data/raw/era5_2018_2022_wind_120-150E_25-50N.nc",
)
print("done")