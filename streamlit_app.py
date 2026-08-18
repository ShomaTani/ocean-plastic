"""
TRACE — Streamlit デモアプリ

順モデル(放出点→漂着分布)・逆モデル(観測点→責任マップ)を、地図をクリックして
その場で推論・可視化できる形にしたもの。forward/predict_point.py,
backward/predict_point.py と同じ推論・配色ロジックを、Streamlit用に関数化して使う。

起動: streamlit run streamlit_app.py
"""

import importlib
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GRID_N = 128
SIGMA_PX = 1.0
UPSCALE = 6  # クリック用の表示画像の拡大率(128 -> 768px)。wideレイアウトに合わせて拡大

# =====================================================================
# forward/ と backward/ は同じファイル名(main.py, dataset.py, losses.py,
# train.py)を持つので、素朴に import すると片方しか読めない(後から読んだ
# 方でsys.modulesが上書きされる)。1つのStreamlitプロセス内で両方を同時に
# 使うために、読み込むたびにsys.modulesから該当名を追い出してから読み直す
# ヘルパーを使い、戻り値の辞書に直接参照を保持しておく。
# =====================================================================
_SHARED_NAMES = ["main", "dataset", "losses", "train"]


def import_fresh(dir_path):
    for name in _SHARED_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(dir_path))
    try:
        return {name: importlib.import_module(name) for name in _SHARED_NAMES}
    finally:
        sys.path.remove(str(dir_path))


@st.cache_resource
def load_side(dir_name):
    mods = import_fresh(ROOT / dir_name)
    model = mods["main"].AttentionUNet(
        in_ch=mods["train"].IN_CH, base_ch=mods["train"].BASE_CH
    ).to(mods["train"].DEVICE)
    model.load_state_dict(torch.load(mods["train"].CKPT_PATH, map_location=mods["train"].DEVICE))
    model.eval()
    return {
        "model": model,
        "device": mods["train"].DEVICE,
        "to_prob_map": mods["losses"].to_prob_map,
        # forward(dilation=3, exp10)とbackward(dilation=1, bexp07)で許容範囲が
        # 違うので、それぞれのtrain.pyが実際に訓練で使ったマスクをそのまま流用する
        # (ここで独自に作り直すと訓練時と推論時のマスクがズレる事故につながる)
        "coastal_mask": mods["train"].COASTAL_MASK,
    }


SEASON_BASELINE_LABEL = "5年平均(季節性を無視した旧基準)"


def discover_current_dates():
    """data/ にある current_field_{date}.npz を全て探して日付リストを返す(新しい順)。
    季節データ拡張(sim_main.py <date> → build_current_field.py <date>)で放出日を
    追加したら、ここは変更不要でそのまま選択肢に増える。"""
    files = sorted(DATA_DIR.glob("current_field_*.npz"))
    dates = [f.stem.replace("current_field_", "") for f in files]
    return sorted(dates, reverse=True)


@st.cache_resource
def load_grid(current_file_name):
    """GLORYS生データは開かず、current_field*.npz(既に128x128に整形済み)と
    確定済みのドメイン範囲(東経120-150, 北緯25-50)だけで完結させる。
    current_file_nameで「どの放出日(年)の海流場を使うか」を切り替えられる。"""
    c = np.load(DATA_DIR / current_file_name)
    land_mask = c["land_mask"]
    coastal_mask = binary_dilation(land_mask) & ~land_mask
    gx = np.linspace(120.0, 150.0, GRID_N + 1)
    gy = np.linspace(25.0, 50.0, GRID_N + 1)

    # クリックが陸/沖合すぎる場合に「最寄りの沿岸セル」へ補正するためのKDTree。
    # 一度だけ構築してキャッシュしておく(毎クリック作り直すのは無駄なので)。
    from scipy.spatial import cKDTree
    coastal_rc = np.column_stack(np.where(coastal_mask))  # (n_coastal, 2) = [row, col]
    coastal_tree = cKDTree(coastal_rc)

    return {
        "u": c["u"], "v": c["v"],
        "land_mask": land_mask, "coastal_mask": coastal_mask,
        "gx": gx, "gy": gy,
        "coastal_rc": coastal_rc, "coastal_tree": coastal_tree,
    }


MAX_SNAP_RADIUS_PX = 15  # これより遠いと「最寄りの沿岸」への補正を諦めてエラーにする


def snap_to_coastal(row, col, grid):
    """(row, col)が沿岸セルでなければ、最寄りの沿岸セルの(row, col)を返す。
    近すぎる沿岸が無い(内陸奥深く・外洋のど真ん中)場合はNoneを返す。"""
    if grid["coastal_mask"][row, col]:
        return row, col, 0.0
    dist, idx = grid["coastal_tree"].query([row, col])
    if dist > MAX_SNAP_RADIUS_PX:
        return None
    r, c = grid["coastal_rc"][idx]
    return int(r), int(c), float(dist)


def lonlat_to_pixel(lon, lat, grid):
    gx, gy = grid["gx"], grid["gy"]
    col = int(np.clip(np.floor((lon - gx[0]) / (gx[1] - gx[0])), 0, GRID_N - 1))
    row = int(np.clip(np.floor((lat - gy[0]) / (gy[1] - gy[0])), 0, GRID_N - 1))
    return row, col


def make_gaussian(row, col):
    onehot = np.zeros((GRID_N, GRID_N), dtype=np.float64)
    onehot[row, col] = 1.0
    g = gaussian_filter(onehot, sigma=SIGMA_PX, mode="constant")
    return (g / g.sum()).astype(np.float32)


def predict(mode, row, col, grid):
    side = load_side("forward" if mode == "forward" else "backward")
    gaussian = make_gaussian(row, col)
    x = torch.from_numpy(np.stack([gaussian, grid["u"], grid["v"]])).unsqueeze(0).float()
    with torch.no_grad():
        logits = side["model"](x.to(side["device"]))
        prob = side["to_prob_map"](logits, mask=side["coastal_mask"]).cpu().numpy()[0, 0]
    return gaussian, prob


# =====================================================================
# 可視化(forward/backward predict_point.pyと同じ配色ロジック)
# =====================================================================
LAND_COLOR = (0.87, 0.83, 0.72, 1)
CMAP_FLOOR = 0.35


def to_rgba(values, land_mask, threshold, cmap_name="Purples", cmap_floor=CMAP_FLOOR):
    below = values <= threshold
    show = ~below & ~land_mask
    cmap = plt.get_cmap(cmap_name)
    vmax = values[show].max() if show.any() else 1.0
    norm = plt.Normalize(vmin=threshold, vmax=vmax)
    t = norm(values)
    rgba = cmap(cmap_floor + (1 - cmap_floor) * t)
    rgba[below] = (1, 1, 1, 1)
    rgba[land_mask] = LAND_COLOR
    return rgba, norm, cmap


def base_map_image(grid):
    """クリック用のベタ塗り画像。matplotlibの軸・余白を一切含めない
    (ピクセル座標がそのままrow/colに対応するようにするため)。
    選択済みの地点は下の結果カード(prediction_figureのinput point)で
    確認できるので、ここではマーカーを重ねない。"""
    land_mask = grid["land_mask"]
    rgba = np.where(land_mask[..., None], np.array(LAND_COLOR), np.array((1, 1, 1, 1)))
    rgba = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(rgba), mode="RGBA")  # row0=南なので画像的には下→上下反転
    img = img.resize((GRID_N * UPSCALE, GRID_N * UPSCALE), Image.NEAREST)
    return img


THRESHOLD_MULTIPLIER = 10  # softmaxの背景ノイズを除去するため一様分布の10倍を閾値にする


def prediction_figure(gaussian, prob, land_mask, title):
    gaussian_rgba, _, _ = to_rgba(gaussian, land_mask, threshold=0.0)
    uniform_baseline = 1.0 / (GRID_N * GRID_N)
    prob_log = np.log1p(prob * 1000)
    prob_rgba, prob_norm, prob_cmap = to_rgba(
        prob_log, land_mask, threshold=np.log1p(uniform_baseline * THRESHOLD_MULTIPLIER * 1000)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    axes[0].imshow(gaussian_rgba, origin="lower")
    axes[0].set_title("input point")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(prob_rgba, origin="lower")
    axes[1].set_title(title)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
    plt.colorbar(sm, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    return fig


# =====================================================================
# UI
# =====================================================================
st.set_page_config(page_title="TRACE", layout="wide")

# 画面上部にタイトル+モード切替を固定表示するためのCSS。
# st.container(key=...)がこのstreamlitバージョン(1.38.0)に無く、構造的な
# セレクタ(nth-child等)はStreamlitの内部構造をたまたま拾ってページ全体を
# 覆ってしまったので、確実に一意な要素だけを直接狙う方式にする:
#   - タイトルは st.title() をやめて、自分で書いたHTMLの<div>を固定表示
#   - モード切替は div[data-testid="stRadio"] がページ内に1つしか無いことを
#     利用して、それ自体に直接 position:fixed をかける
# Streamlit標準のヘッダー(Deployボタン等の帯)は自作ヘッダーの上に被って
# TRACEの文字を隠してしまうため非表示にする。
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }

    .trace-header {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 730px;
        z-index: 9999;
        background: rgba(250, 250, 250, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 16px 24px 0 24px;
        box-sizing: border-box;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: rgb(49, 51, 63);
    }

    div[data-testid="stRadio"] {
        position: fixed;
        top: 56px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 730px;
        z-index: 9999;
        background: rgba(250, 250, 250, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 8px 24px 14px 24px;
        box-sizing: border-box;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }
    /* モード切替をガラス風セグメントコントロールに */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        background: rgba(120, 120, 128, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 999px;
        padding: 3px;
        gap: 2px;
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        justify-content: center;
        margin: 0;
        padding: 7px 10px;
        border-radius: 999px;
        transition: background 0.15s ease, box-shadow 0.15s ease;
        cursor: pointer;
    }
    /* ネイティブのラジオ丸印は非表示にして、テキストだけのピルにする */
    div[data-testid="stRadio"] label > div:first-child {
        display: none;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stRadio"] label p {
        font-size: 0.85rem;
        font-weight: 500;
        color: rgb(49, 51, 63) !important;
    }
    /* 固定ヘッダーの下に隠れる分だけ本文側に空白を確保する */
    .header-spacer {
        height: 120px;
    }
    /* wideレイアウトの余白を切り詰めつつ、地図・結果画像は中央寄せにする */
    div[data-testid="stAppViewBlockContainer"] {
        max-width: 1100px;
        padding-left: 3rem;
        padding-right: 3rem;
        margin: 0 auto;
    }
    /* streamlit_image_coordinatesはiframeの中に画像を描画するが、iframe自体は
    親要素の全幅を取ってしまい、画像はその中で左寄せになる。iframeの方を
    画像の実サイズ(UPSCALE*128px)に縮めてから中央寄せする。 */
    iframe {
        width: 768px !important;
        display: block !important;
        margin: 0 auto !important;
    }
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    /* 「地図をクリックして...」の説明文を、下の地図(768px, 中央寄せ)と
    左端が揃うように、同じ幅で中央寄せする */
    div[data-testid="stCaptionContainer"] {
        max-width: 768px;
        margin: 0 auto;
    }
    </style>
    <div class="trace-header">TRACE</div>
    """,
    unsafe_allow_html=True,
)

mode_label = st.radio(
    "モード", ["順モデル(放出点 → 漂着分布)", "逆モデル(観測点 → 責任マップ)"],
    horizontal=True, label_visibility="collapsed",
)
mode = "forward" if mode_label.startswith("順") else "backward"
st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)

available_dates = discover_current_dates()
date_options = available_dates + [SEASON_BASELINE_LABEL]
selected_date = st.selectbox(
    "海流条件(放出日)",
    date_options,
    index=0,
    help="どの年の冬(1月1日起点、90日間)の海流場で推論するか選べます。"
         f"「{SEASON_BASELINE_LABEL}」は季節データ拡張前の2018-2022年5年間の静的平均です。"
         "sim_main.py <date> と build_current_field.py <date> を実行すれば選択肢を追加できます。",
)
current_file = (
    "current_field.npz" if selected_date == SEASON_BASELINE_LABEL
    else f"current_field_{selected_date}.npz"
)
grid = load_grid(current_file)

st.caption("地図をクリックして沿岸の地点を選択")

if "selected" not in st.session_state:
    st.session_state.selected = None

base_img = base_map_image(grid)
coords = streamlit_image_coordinates(base_img, key="map_click")

if coords is not None:
    px, py = coords["x"], coords["y"]
    col = px // UPSCALE
    row = GRID_N - 1 - py // UPSCALE  # 画像は上端がrow大、グリッドはrow0が南(下)
    row = int(np.clip(row, 0, GRID_N - 1))
    col = int(np.clip(col, 0, GRID_N - 1))

    snapped = snap_to_coastal(row, col, grid)
    if snapped is None:
        st.warning("近くに沿岸が見つかりませんでした。海岸に近い場所を選んでください。")
    else:
        s_row, s_col, dist_px = snapped
        st.session_state.selected = (s_row, s_col)

if st.session_state.selected is not None:
    row, col = st.session_state.selected
    with st.spinner("推論中..."):
        gaussian, prob = predict(mode, row, col, grid)
    title = "predicted beaching distribution" if mode == "forward" else "predicted responsibility"
    fig = prediction_figure(gaussian, prob, grid["land_mask"], title)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    st.image(buf)
