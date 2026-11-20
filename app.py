import gc
import os
import textwrap

import folium
from folium.plugins import HeatMap
import gdown
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="GIS・商圏分析システム MVP", layout="wide")
st.title("🗺️ GIS・商圏分析システム (MVP)")

REGIONS = [
    ("北海道", ["北海道"]),
    ("東北", ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"]),
    ("関東", ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"]),
    ("甲信越・北陸", ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県"]),
    ("東海", ["岐阜県", "静岡県", "愛知県", "三重県"]),
    ("近畿", ["滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"]),
    ("中国", ["鳥取県", "島根県", "岡山県", "広島県", "山口県"]),
    ("四国", ["徳島県", "香川県", "愛媛県", "高知県"]),
    ("九州・沖縄", ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]),
]

PREF_LIST = [pref for _, prefs in REGIONS for pref in prefs]

PREF_CENTERS = {
    "北海道": (43.064, 141.347),
    "青森県": (40.824, 140.740),
    "岩手県": (39.704, 141.153),
    "宮城県": (38.269, 140.872),
    "秋田県": (39.719, 140.102),
    "山形県": (38.240, 140.363),
    "福島県": (37.750, 140.468),
    "茨城県": (36.342, 140.447),
    "栃木県": (36.566, 139.884),
    "群馬県": (36.391, 139.060),
    "埼玉県": (35.857, 139.649),
    "千葉県": (35.605, 140.123),
    "東京都": (35.689, 139.692),
    "神奈川県": (35.448, 139.643),
    "新潟県": (37.902, 139.023),
    "富山県": (36.695, 137.211),
    "石川県": (36.595, 136.626),
    "福井県": (36.065, 136.222),
    "山梨県": (35.664, 138.568),
    "長野県": (36.651, 138.181),
    "岐阜県": (35.391, 136.722),
    "静岡県": (34.977, 138.383),
    "愛知県": (35.180, 136.907),
    "三重県": (34.730, 136.509),
    "滋賀県": (35.004, 135.869),
    "京都府": (35.021, 135.756),
    "大阪府": (34.686, 135.520),
    "兵庫県": (34.691, 135.183),
    "奈良県": (34.685, 135.833),
    "和歌山県": (34.226, 135.168),
    "鳥取県": (35.504, 134.238),
    "島根県": (35.472, 133.051),
    "岡山県": (34.662, 133.935),
    "広島県": (34.397, 132.460),
    "山口県": (34.186, 131.471),
    "徳島県": (34.066, 134.559),
    "香川県": (34.340, 134.043),
    "愛媛県": (33.842, 132.766),
    "高知県": (33.560, 133.531),
    "福岡県": (33.606, 130.418),
    "佐賀県": (33.249, 130.299),
    "長崎県": (32.745, 129.874),
    "熊本県": (32.790, 130.742),
    "大分県": (33.238, 131.613),
    "宮崎県": (31.911, 131.424),
    "鹿児島県": (31.560, 130.558),
    "沖縄県": (26.212, 127.681),
}

PREF_CENTERS_ARR = np.array(
    [[PREF_CENTERS[p][0], PREF_CENTERS[p][1]] for p in PREF_LIST],
    dtype=np.float32,
)

SHORT_PREF_LABELS = {
    "北海道": "北海道",
    "青森県": "青森",
    "岩手県": "岩手",
    "宮城県": "宮城",
    "秋田県": "秋田",
    "山形県": "山形",
    "福島県": "福島",
    "茨城県": "茨城",
    "栃木県": "栃木",
    "群馬県": "群馬",
    "埼玉県": "埼玉",
    "千葉県": "千葉",
    "東京都": "東京",
    "神奈川県": "神奈川",
    "新潟県": "新潟",
    "富山県": "富山",
    "石川県": "石川",
    "福井県": "福井",
    "山梨県": "山梨",
    "長野県": "長野",
    "岐阜県": "岐阜",
    "静岡県": "静岡",
    "愛知県": "愛知",
    "三重県": "三重",
    "滋賀県": "滋賀",
    "京都府": "京都",
    "大阪府": "大阪",
    "兵庫県": "兵庫",
    "奈良県": "奈良",
    "和歌山県": "和歌山",
    "鳥取県": "鳥取",
    "島根県": "島根",
    "岡山県": "岡山",
    "広島県": "広島",
    "山口県": "山口",
    "徳島県": "徳島",
    "香川県": "香川",
    "愛媛県": "愛媛",
    "高知県": "高知",
    "福岡県": "福岡",
    "佐賀県": "佐賀",
    "長崎県": "長崎",
    "熊本県": "熊本",
    "大分県": "大分",
    "宮崎県": "宮崎",
    "鹿児島県": "鹿児島",
    "沖縄県": "沖縄",
}

PARQUET_FILE_ID = "1Cog9sWsi9vh9HKSNYKK2wf2q8HG-kOXj"

MAX_HEATMAP_POINTS = 4000
MAX_GRID_POINTS = 350
MAX_CITIES_PER_VIEW = 20
BATCH_ROWS = 100000


class ZoomHeatGridSwitcher(folium.MacroElement):
    def __init__(self, heat_layer, grid_layer, zoom_threshold=13):
        super().__init__()
        self._name = "ZoomHeatGridSwitcher"
        map_var = "{{ this._parent.get_name() }}"
        heat_var = heat_layer.get_name()
        grid_var = grid_layer.get_name()
        raw_js = textwrap.dedent("""
        {% macro script(this, kwargs) %}
        var map = __MAP_VAR__;
        var heatLayer = __HEAT_VAR__;
        var gridLayer = __GRID_VAR__;

        function toggleByZoom() {
            var z = map.getZoom();
            if (z >= __ZOOM_THRESHOLD__) {
                if (map.hasLayer(heatLayer)) { map.removeLayer(heatLayer); }
                if (!map.hasLayer(gridLayer)) { map.addLayer(gridLayer); }
            } else {
                if (map.hasLayer(gridLayer)) { map.removeLayer(gridLayer); }
                if (!map.hasLayer(heatLayer)) { map.addLayer(heatLayer); }
            }
        }

        map.on('zoomend', toggleByZoom);
        toggleByZoom();
        {% endmacro %}
        """).strip()
        js_body = (
            raw_js
            .replace("__MAP_VAR__", map_var)
            .replace("__HEAT_VAR__", heat_var)
            .replace("__GRID_VAR__", grid_var)
            .replace("__ZOOM_THRESHOLD__", str(zoom_threshold))
        )
        self._template = folium.Template(js_body)


def _ensure_parquet_local(parquet_file_id: str, output_path: str = "small_area_slim.parquet") -> bool:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    url = f"https://drive.google.com/uc?id={parquet_file_id}"
    gdown.download(url, output_path, quiet=True)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _iter_parquet_batches(path: str, cols_wanted: list[str], batch_size: int = BATCH_ROWS):
    """PyArrowでバッチ単位に読み出し、Python/Pandasオブジェクトを最小化して返す"""
    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema_arrow.names)
    use_cols = [c for c in cols_wanted if c in schema_cols]
    for record_batch in pf.iter_batches(batch_size=batch_size, columns=use_cols):
        batch_df = record_batch.to_pandas(types_mapper=None)
        del record_batch
        yield batch_df


def _extract_pref_series(df: pd.DataFrame) -> pd.Series:
    code_col = next((c for c in ["PREF_CODE", "CITY_CODE", "KEY_CODE"] if c in df.columns), None)
    if code_col:
        codes2 = (
            df[code_col]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .fillna("")
            .str.zfill(5)
            .str[:2]
        )
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        pref_series = codes2.map(code_map).fillna("")
        del codes2
    else:
        pref_series = pd.Series("", index=df.index, dtype=str)

    if "PREF_NAME" in df.columns:
        raw_pref = df["PREF_NAME"].fillna("").astype(str).str.strip()
        pref_series = pd.Series(
            np.where(pref_series != "", pref_series, raw_pref),
            index=df.index,
            dtype=str,
        )
        del raw_pref

    if (pref_series == "").any() and {"CITY_NAME", "X_CODE", "Y_CODE"}.issubset(df.columns):
        city_col_str = df["CITY_NAME"].fillna("").astype(str)
        valid_mask = (city_col_str != "") & df["X_CODE"].notna() & df["Y_CODE"].notna()
        if valid_mask.any():
            valid_df = pd.DataFrame({
                "CITY_NAME": city_col_str.loc[valid_mask],
                "Y_CODE": pd.to_numeric(df.loc[valid_mask, "Y_CODE"], errors="coerce"),
                "X_CODE": pd.to_numeric(df.loc[valid_mask, "X_CODE"], errors="coerce"),
            }).dropna()
            if len(valid_df) > 0:
                city_coords = valid_df.groupby("CITY_NAME", observed=True)[["Y_CODE", "X_CODE"]].mean()
                coords = city_coords.values[:, None, :]
                dists = ((coords - PREF_CENTERS_ARR[None, :, :]) ** 2).sum(axis=2)
                nearest_idx = dists.argmin(axis=1)
                city_to_pref = dict(zip(city_coords.index, [PREF_LIST[i] for i in nearest_idx]))
                mask_empty = (pref_series == "") & (city_col_str != "")
                pref_series = pref_series.copy()
                pref_series.loc[mask_empty] = city_col_str.loc[mask_empty].map(city_to_pref).fillna("")
                del valid_df, city_coords, coords, dists, nearest_idx, city_to_pref, mask_empty
        del city_col_str

    return pref_series


@st.cache_data(show_spinner=False)
def load_pref_summary(parquet_file_id: str) -> pd.DataFrame:
    """バッチ読込で都道府県別サマリー（集計値と都道府県一覧のみ）をキャッシュする"""
    output = "small_area_slim.parquet"
    if not _ensure_parquet_local(parquet_file_id, output):
        return pd.DataFrame(columns=["PREF_NAME", "JINKO", "SETAI", "AREA_COUNT"])

    cols_wanted = ["PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE", "CITY_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"]
    parts = []

    for batch_df in _iter_parquet_batches(output, cols_wanted, batch_size=BATCH_ROWS):
        if len(batch_df) == 0:
            del batch_df
            continue
        jinko = pd.to_numeric(batch_df.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int64)
        setai = pd.to_numeric(batch_df.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int64)
        pref_series = _extract_pref_series(batch_df)

        part_df = pd.DataFrame({
            "PREF_NAME": pref_series,
            "JINKO": jinko,
            "SETAI": setai,
            "AREA_COUNT": np.ones(len(batch_df), dtype=np.int64),
        })
        part_df = part_df[part_df["PREF_NAME"] != ""].groupby("PREF_NAME", as_index=False).sum()
        parts.append(part_df)
        del batch_df, pref_series, jinko, setai, part_df
        gc.collect()

    if not parts:
        return pd.DataFrame(columns=["PREF_NAME", "JINKO", "SETAI", "AREA_COUNT"])

    summary_df = pd.concat(parts, ignore_index=True).groupby("PREF_NAME", as_index=False).sum()
    del parts
    gc.collect()
    return summary_df


@st.cache_data(show_spinner=False)
def load_city_index(parquet_file_id: str, target_pref: str) -> pd.DataFrame:
    """バッチ読込で選択都道府県の市区町村一覧（件数・人口・世帯数）を軽量に取得する"""
    output = "small_area_slim.parquet"
    if not target_pref or target_pref == "（すべて）" or not _ensure_parquet_local(parquet_file_id, output):
        return pd.DataFrame(columns=["CITY_NAME", "JINKO", "SETAI", "AREA_COUNT"])

    cols_wanted = ["PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE", "CITY_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"]
    parts = []

    for batch_df in _iter_parquet_batches(output, cols_wanted, batch_size=BATCH_ROWS):
        if len(batch_df) == 0:
            del batch_df
            continue
        pref_series = _extract_pref_series(batch_df)
        mask = (pref_series == target_pref).to_numpy()
        del pref_series
        if not mask.any():
            del batch_df, mask
            continue

        sub = batch_df.loc[mask].copy()
        del batch_df, mask
        sub["CITY_NAME"] = sub.get("CITY_NAME", "不明").fillna("不明").astype(str)
        sub["JINKO"] = pd.to_numeric(sub.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int64)
        sub["SETAI"] = pd.to_numeric(sub.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int64)

        part_city = sub.groupby("CITY_NAME", as_index=False).agg(
            JINKO=("JINKO", "sum"),
            SETAI=("SETAI", "sum"),
            AREA_COUNT=("CITY_NAME", "count"),
        )
        parts.append(part_city)
        del sub, part_city
        gc.collect()

    if not parts:
        return pd.DataFrame(columns=["CITY_NAME", "JINKO", "SETAI", "AREA_COUNT"])

    city_df = (
        pd.concat(parts, ignore_index=True)
        .groupby("CITY_NAME", as_index=False)
        .agg(
            JINKO=("JINKO", "sum"),
            SETAI=("SETAI", "sum"),
            AREA_COUNT=("AREA_COUNT", "sum"),
        )
        .sort_values(["JINKO", "CITY_NAME"], ascending=[False, True])
        .reset_index(drop=True)
    )
    del parts
    gc.collect()
    return city_df


@st.cache_data(show_spinner=False)
def load_city_chunk_data(parquet_file_id: str, target_pref: str, selected_cities: tuple[str, ...]) -> pd.DataFrame:
    """バッチ読込で選択市区町村チャンク分の小地域データのみを抽出する"""
    output = "small_area_slim.parquet"
    if not target_pref or target_pref == "（すべて）" or not selected_cities or not _ensure_parquet_local(parquet_file_id, output):
        return pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])

    cols_wanted = ["PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"]
    city_set = set(selected_cities)
    parts = []

    for batch_df in _iter_parquet_batches(output, cols_wanted, batch_size=BATCH_ROWS):
        if len(batch_df) == 0:
            del batch_df
            continue
        pref_series = _extract_pref_series(batch_df)
        city_series = batch_df.get("CITY_NAME", pd.Series("不明", index=batch_df.index)).fillna("不明").astype(str)
        mask = ((pref_series == target_pref) & city_series.isin(city_set)).to_numpy()
        del pref_series, city_series
        if not mask.any():
            del batch_df, mask
            continue

        sub = batch_df.loc[mask].copy()
        del batch_df, mask
        sub["PREF_NAME"] = target_pref
        sub["JINKO"] = pd.to_numeric(sub.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int32)
        sub["SETAI"] = pd.to_numeric(sub.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int32)
        sub["X_CODE"] = pd.to_numeric(sub["X_CODE"], errors="coerce").astype(np.float32)
        sub["Y_CODE"] = pd.to_numeric(sub["Y_CODE"], errors="coerce").astype(np.float32)
        sub = sub.dropna(subset=["X_CODE", "Y_CODE"])
        sub["CITY_NAME"] = sub.get("CITY_NAME", "").fillna("").astype(str)
        sub["S_NAME"] = sub["S_NAME"].fillna("不明").astype(str) if "S_NAME" in sub.columns else "不明"

        keep_out = ["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"]
        sub = sub[[c for c in keep_out if c in sub.columns]]
        parts.append(sub)
        del sub
        gc.collect()

    if not parts:
        return pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])

    out_df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out_df


def add_base_tiles(m):
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        attr="地理院タイル（淡色地図）",
        name="国土地理院 淡色地図",
        overlay=False,
        control=True,
        max_zoom=18,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        attr="地理院タイル（標準地図）",
        name="国土地理院 標準地図",
        overlay=False,
        control=True,
        max_zoom=18,
    ).add_to(m)


def set_selected_pref(pref_name: str):
    st.session_state["selected_pref"] = pref_name
    st.session_state["pref_selectbox"] = pref_name
    st.session_state["city_page"] = 0
    st.session_state["selected_cities"] = []


if "selected_pref" not in st.session_state:
    st.session_state["selected_pref"] = "（すべて）"
if "city_page" not in st.session_state:
    st.session_state["city_page"] = 0
if "selected_cities" not in st.session_state:
    st.session_state["selected_cities"] = []
if "load_summary_requested" not in st.session_state:
    st.session_state["load_summary_requested"] = False

st.sidebar.header("🔍 エリア検索・選択")

# 初期起動時の自動バッチ走査をオフにできるようにし、メモリ超過（Oh no.）を防止
auto_scan_summary = st.sidebar.checkbox(
    "起動時に全国サマリーを自動集計する",
    value=False,
    help="オフにすると初回起動時のメモリ消費を最小限に抑えます（都道府県一覧は全47都道府県から選択可能）。",
)

if auto_scan_summary or st.session_state["load_summary_requested"]:
    with st.spinner("都道府県別サマリーを読み込んでいます..."):
        summary_df = load_pref_summary(PARQUET_FILE_ID)
else:
    summary_df = pd.DataFrame(columns=["PREF_NAME", "JINKO", "SETAI", "AREA_COUNT"])
    if st.sidebar.button("全国サマリーを集計する", use_container_width=True):
        st.session_state["load_summary_requested"] = True
        st.rerun()

available_prefs_set = set(summary_df["PREF_NAME"].dropna().astype(str).unique()) if len(summary_df) > 0 else set()
available_prefs = [p for p in PREF_LIST if p in available_prefs_set] if available_prefs_set else PREF_LIST.copy()

pref_options = ["（すべて）"] + available_prefs

if st.session_state["selected_pref"] not in pref_options:
    st.session_state["selected_pref"] = "（すべて）"
if "pref_selectbox" not in st.session_state or st.session_state["pref_selectbox"] not in pref_options:
    st.session_state["pref_selectbox"] = st.session_state["selected_pref"]

search_query = st.sidebar.text_input(
    "都道府県を検索（例: 新潟、東京、大阪）",
    placeholder="都道府県名を入力",
    key="pref_search_input",
)

if search_query:
    matched_prefs = [
        p for p in available_prefs
        if search_query.strip() in p or search_query.strip() in SHORT_PREF_LABELS.get(p, "")
    ]
    if matched_prefs:
        st.sidebar.caption(f"候補: {', '.join(matched_prefs[:6])}")
        if st.sidebar.button(f"「{matched_prefs[0]}」を選択", key="apply_search_pref"):
            set_selected_pref(matched_prefs[0])
            st.rerun()
    else:
        st.sidebar.caption("該当する都道府県が見つかりません。")

st.sidebar.selectbox(
    "プルダウンから都道府県を選択",
    pref_options,
    key="pref_selectbox",
    on_change=lambda: (
        st.session_state.update({
            "selected_pref": st.session_state["pref_selectbox"],
            "city_page": 0,
            "selected_cities": [],
        })
    ),
)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("選択解除", use_container_width=True):
        set_selected_pref("（すべて）")
        st.rerun()
with col_sb2:
    if st.button("キャッシュ再取得", use_container_width=True):
        if os.path.exists("small_area_slim.parquet"):
            os.remove("small_area_slim.parquet")
        st.cache_data.clear()
        st.session_state["load_summary_requested"] = False
        set_selected_pref("（すべて）")
        st.rerun()

selected_pref = st.session_state["selected_pref"]
is_pref_selected = selected_pref != "（すべて）"

with st.expander("🗺️ 地方・都道府県ボタンから選ぶ", expanded=not is_pref_selected):
    st.caption("1) 都道府県を選ぶ → 2) 市区町村チャンク（1〜3件推奨）を選ぶと地図を描画します（メモリ節約モード）。")
    for region_name, region_prefs in REGIONS:
        present_prefs = [p for p in region_prefs if p in available_prefs]
        if not present_prefs:
            continue
        st.markdown(f"**{region_name}**")
        btn_cols = st.columns(min(len(present_prefs), 8))
        for idx, pref in enumerate(present_prefs):
            col = btn_cols[idx % len(btn_cols)]
            is_current = (pref == selected_pref)
            btn_label = f"✅ {SHORT_PREF_LABELS.get(pref, pref)}" if is_current else SHORT_PREF_LABELS.get(pref, pref)
            if col.button(
                btn_label,
                key=f"btn_pref_{pref}",
                type="primary" if is_current else "secondary",
                use_container_width=True,
            ):
                set_selected_pref(pref)
                st.rerun()

active_cities_tuple: tuple[str, ...] = tuple()

if not is_pref_selected:
    area_label = "未選択"
    total_pop = 0
    total_households = 0
    area_count = 0
else:
    area_label = selected_pref
    pref_row = summary_df[summary_df["PREF_NAME"] == selected_pref] if len(summary_df) > 0 else pd.DataFrame()
    if len(pref_row) > 0:
        total_pop = int(pref_row["JINKO"].iloc[0])
        total_households = int(pref_row["SETAI"].iloc[0])
        area_count = int(pref_row["AREA_COUNT"].iloc[0])
    else:
        total_pop = 0
        total_households = 0
        area_count = 0

    with st.spinner(f"{selected_pref} の市区町村一覧を取得しています..."):
        city_index_df = load_city_index(PARQUET_FILE_ID, selected_pref)

    if len(city_index_df) > 0 and (total_pop == 0 and total_households == 0 and area_count == 0):
        total_pop = int(city_index_df["JINKO"].sum())
        total_households = int(city_index_df["SETAI"].sum())
        area_count = int(city_index_df["AREA_COUNT"].sum())

    all_city_names = city_index_df["CITY_NAME"].tolist()
    total_cities = len(all_city_names)
    total_pages = max(1, (total_cities + MAX_CITIES_PER_VIEW - 1) // MAX_CITIES_PER_VIEW)
    current_page = min(max(0, st.session_state["city_page"]), total_pages - 1)
    st.session_state["city_page"] = current_page

    page_start = current_page * MAX_CITIES_PER_VIEW
    page_end = min(total_cities, page_start + MAX_CITIES_PER_VIEW)
    page_cities = all_city_names[page_start:page_end]

    st.markdown("### 🧩 市区町村チャンク分割（メモリ節約モード）")
    c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([1, 1, 2, 2])
    with c_nav1:
        if st.button("◀ 前の市区町村群", disabled=(current_page <= 0), use_container_width=True):
            st.session_state["city_page"] = current_page - 1
            st.session_state["selected_cities"] = []
            st.rerun()
    with c_nav2:
        if st.button("次の市区町村群 ▶", disabled=(current_page >= total_pages - 1), use_container_width=True):
            st.session_state["city_page"] = current_page + 1
            st.session_state["selected_cities"] = []
            st.rerun()
    with c_nav3:
        if st.button("先頭2市区町村を読込", disabled=(len(page_cities) == 0), use_container_width=True):
            st.session_state["selected_cities"] = page_cities[:2]
            st.rerun()
    with c_nav4:
        if st.button("地図描画をクリア", use_container_width=True):
            st.session_state["selected_cities"] = []
            st.rerun()

    st.caption(
        f"全 {total_cities} 市区町村のうち {page_start + 1}〜{page_end} 件目を表示中 "
        f"（ページ {current_page + 1} / {total_pages}）。1度に読み込む市区町村を1〜2件に絞ると安定しやすくなります。"
    )

    valid_selected = [c for c in st.session_state["selected_cities"] if c in set(all_city_names)]
    chosen_cities = st.multiselect(
        "描画する市区町村を選択（複数可・推奨 1〜2 市区町村）",
        options=page_cities if page_cities else all_city_names,
        default=[c for c in valid_selected if c in set(page_cities)] if page_cities else valid_selected,
        key="city_chunk_multiselect",
    )
    st.session_state["selected_cities"] = chosen_cities
    active_cities_tuple = tuple(chosen_cities)

filtered_df = (
    load_city_chunk_data(PARQUET_FILE_ID, selected_pref, active_cities_tuple)
    if (is_pref_selected and len(active_cities_tuple) > 0)
    else pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("対象エリア", area_label)
m2.metric("総人口", f"{total_pop:,} 人")
m3.metric("総世帯数", f"{total_households:,} 世帯")
m4.metric("対象エリア数", f"{area_count:,} 件")

if is_pref_selected and len(filtered_df) > 0:
    center_lat = float(filtered_df["Y_CODE"].mean())
    center_lon = float(filtered_df["X_CODE"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None, control_scale=True)
    add_base_tiles(m)

    if len(filtered_df) > MAX_HEATMAP_POINTS:
        st.caption(f"※メモリ・描画負荷抑制のため、ヒートマップは {len(filtered_df):,} 件から {MAX_HEATMAP_POINTS:,} 件を抽出して表示しています。")
        heat_df = filtered_df.sample(n=MAX_HEATMAP_POINTS, random_state=42)
    else:
        heat_df = filtered_df

    heat_group = folium.FeatureGroup(name="人口ヒートマップ", show=True)
    heat_points = [
        [float(row.Y_CODE), float(row.X_CODE), max(float(row.JINKO), 1.0)]
        for row in heat_df.itertuples(index=False)
    ]
    if heat_points:
        HeatMap(heat_points, radius=16, blur=18, max_zoom=13).add_to(heat_group)
    heat_group.add_to(m)

    grid_group = folium.FeatureGroup(name="詳細グリッド（ズーム13以上）", show=False)
    half_lat = 0.0006
    half_lon = 0.00075
    for row in filtered_df.iloc[:MAX_GRID_POINTS].itertuples(index=False):
        lat = float(row.Y_CODE)
        lon = float(row.X_CODE)
        s_name = getattr(row, "S_NAME", "不明")
        c_name = getattr(row, "CITY_NAME", "")
        jinko_val = int(getattr(row, "JINKO", 0))
        setai_val = int(getattr(row, "SETAI", 0))
        popup_html = (
            f"<b>{c_name} {s_name}</b><br>"
            f"人口: {jinko_val:,} 人<br>"
            f"世帯数: {setai_val:,} 世帯"
        )
        folium.Rectangle(
            bounds=[[lat - half_lat, lon - half_lon], [lat + half_lat, lon + half_lon]],
            color="#d95f02",
            weight=1,
            fill=True,
            fill_opacity=0.25,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(grid_group)
    grid_group.add_to(m)

    ZoomHeatGridSwitcher(heat_group, grid_group, zoom_threshold=13).add_to(m)

    del heat_df, heat_points
    gc.collect()

    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(
        m,
        width="100%",
        height=620,
        returned_objects=[],
    )
elif is_pref_selected:
    st.info("「🧩 市区町村チャンク分割」から描画したい市区町村を選択（または「先頭2市区町村を読込」をクリック）してください。")
else:
    st.info("地方ブロックの都道府県ボタン、サイドバーの検索、またはプルダウンから都道府県を選択してください。")
