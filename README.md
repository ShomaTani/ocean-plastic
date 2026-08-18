TRACE実装

このプロジェクトのディレクトリ構成は：
data：データ分析
forward：順モデル
backward：逆モデル
となっている

Streamlitのアプリをローカルで動かすには

1. 依存パッケージをインストール(`requirements.txt`参照)
   ```
   conda create -n plastic python=3.11
   conda activate plastic
   pip install -r requirements.txt
   ```

2. 以下のファイルがリポジトリに存在することを確認する(無ければ後述の生成手順を実行):
   - `data/current_field.npz`, `data/current_field_2018-01-01.npz` 〜 `2022-01-01.npz`
   - `forward/best_model.pt`, `backward/best_model.pt`

   これらは`.gitignore`で除外されているため、GLORYS/ERA5の生データダウンロードや
   シミュレーション・学習を一切やり直さずに再現したい場合は、force-addして
   リポジトリに含めておく必要がある(現状は`forward/best_model.pt`のみ追跡済み)。

3. 起動
   ```
   streamlit run streamlit_app.py
   ```
   ブラウザで `http://localhost:8501` が開く。

上記2のファイルさえあれば、GLORYS/ERA5の生データダウンロードやCMEMS/CDSの
認証情報は一切不要(Streamlitの推論は学習済みモデルと海流場ファイルだけで完結する)。
無い場合は `data/` 以下のパイプラインを先頭から実行して再生成する必要がある
(`sim_main.py` → `build_current_field.py` → `inputdata_dim.py`/`build_backward_pairs.py`
→ `make_split.py`/`make_split_backward.py` → `forward/train.py`/`backward/train.py`)。
