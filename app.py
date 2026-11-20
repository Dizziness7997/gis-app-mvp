import gc
import json
import os
import re
import textwrap

import folium
from folium.plugins import HeatMap
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="GIS・商圏分析システム MVP", layout="wide")
st.title("🗺️ GIS・商圏分析システム (MVP)")

PREF_LIST = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

PREF_CENTERS = np.array([
    [43.064, 141.347], [40.824, 140.740], [39.704, 141.153], [38.269, 140.872],
    [39.719, 140.102], [38.240, 140.363], [37.750, 140.468], [36.342, 140.447],
    [36.566, 139.884], [36.391, 139.060], [35.857, 139.649], [35.605, 140.123],
    [35.689, 139.692], [35.448, 139.643], [37.902, 139.023], [36.695, 137.211],
    [36.595, 136.626], [36.065, 136.222], [35.664, 138.568], [36.651, 138.181],
    [35.391, 136.722], [34.977, 138.383], [35.180, 136.907], [34.730, 136.509],
    [35.004, 135.869], [35.021, 135.756], [34.686, 135.520], [34.691, 135.183],
    [34.685, 135.833], [34.226, 135.168], [35.504, 134.238], [35.472, 133.051],
    [34.662, 133.935], [34.397, 132.460], [34.186, 131.471], [34.066, 134.559],
    [34.340, 134.043], [33.842, 132.766], [33.560, 133.531], [33.606, 130.418],
    [33.249, 130.299], [32.745, 129.874], [32.790, 130.742], [33.238, 131.613],
    [31.911, 131.424], [31.560, 130.558], [26.212, 127.681]
], dtype=np.float32)

PARQUET_FILE_ID = "1Cog9sWsi9vh9HKSNYKK2wf2q8HG-kOXj"
GEOJSON_FILE_ID = "13LAo42gRdBajjeaT95Xx0x8krjlfINFj"

MAX_HEATMAP_POINTS = 15000
MAX_GRID_POINTS = 1500


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


@st.cache_data(show_spinner=False)
def load_data(parquet_file_id: str):
    output = "small_area_slim.parquet"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={parquet_file_id}"
        gdown.download(url, output, quiet=True)

    if not os.path.exists(output):
        return pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])

    cols_wanted = [
        "PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE",
        "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"
    ]
    df = pd.read_parquet(output)
    keep_cols = [c for c in cols_wanted if c in df.columns]
    df = df[keep_cols]

    df["JINKO"] = pd.to_numeric(df.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int32)
    df["SETAI"] = pd.to_numeric(df.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int32)
    df["X_CODE"] = pd.to_numeric(df["X_CODE"], errors="coerce").astype(np.float32)
    df["Y_CODE"] = pd.to_numeric(df["Y_CODE"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["X_CODE", "Y_CODE"]).reset_index(drop=True)
    df["CITY_NAME"] = df.get("CITY_NAME", "").fillna("").astype(str)

    code_candidates = ["PREF_CODE", "CITY_CODE", "KEY_CODE"]
    found_code_col = next((c for c in code_candidates if c in df.columns), None)

    pref_from_code = pd.Series("", index=df.index, dtype=str)
    if found_code_col:
        codes2 = (
            df[found_code_col]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .fillna("")
            .str.zfill(5)
            .str[:2]
        )
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        pref_from_code = codes2.map(code_map).fillna("")

    if "PREF_NAME" in df.columns:
        raw_pref = df["PREF_NAME"].fillna("").astype(str).str.strip()
        pref_series = pd.Series(
            np.where(pref_from_code != "", pref_from_code, raw_pref),
            index=df.index,
            dtype=str,
        )
    else:
        pref_series = pref_from_code

    if (pref_series == "").any():
        valid_city = df[df["CITY_NAME"] != ""]
        if len(valid_city) > 0:
            city_coords = valid_city.groupby("CITY_NAME", observed=True)[["Y_CODE", "X_CODE"]].mean()
            coords = city_coords.values[:, None, :]
            dists = ((coords - PREF_CENTERS[None, :, :]) ** 2).sum(axis=2)
            nearest_idx = dists.argmin(axis=1)
            city_to_pref = dict(zip(city_coords.index, [PREF_LIST[i] for i in nearest_idx]))
            mask_empty = (pref_series == "") & (df["CITY_NAME"] != "")
            pref_series.loc[mask_empty] = df.loc[mask_empty, "CITY_NAME"].map(city_to_pref).fillna("")

    df["PREF_NAME"] = pref_series.astype("category")
    df["CITY_NAME"] = df["CITY_NAME"].astype("category")
    if "S_NAME" in df.columns:
        df["S_NAME"] = df["S_NAME"].fillna("不明").astype("category")
    else:
        df["S_NAME"] = pd.Series("不明", index=df.index, dtype="category")

    for drop_c in ["PREF_CODE", "CITY_CODE", "KEY_CODE"]:
        if drop_c in df.columns:
            df.drop(columns=[drop_c], inplace=True)

    del pref_series, pref_from_code
    gc.collect()
    return df


@st.cache_data(show_spinner=False)
def load_boundaries(geojson_file_id: str):
    output = "admin_boundaries_slim.geojson"
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={geojson_file_id}"
        gdown.download(url, output, quiet=True)

    if not os.path.exists(output):
        return {"type": "FeatureCollection", "features": []}

    with open(output, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not (isinstance(data, dict) and isinstance(data.get("features"), list)):
        return {"type": "FeatureCollection", "features": []}

    normalized_features = []
    for feat in data["features"]:
        props = dict(feat.get("properties") or {})
        p_val = str(props.get("PREF_NAME") or props.get("N03_001") or "").strip()
        if p_val:
            props["PREF_NAME"] = p_val
            feat_copy = dict(feat)
            feat_copy["properties"] = props
            normalized_features.append(feat_copy)

    return {"type": "FeatureCollection", "features": normalized_features}


def extract_pref_from_tooltip(tooltip_text: str | None) -> str | None:
    if not tooltip_text or not isinstance(tooltip_text, str):
        return None
    cleaned = re.sub(r"^都道府県\s*[:：]\s*", "", tooltip_text.strip()).strip()
    if cleaned in PREF_LIST:
        return cleaned
    for pref in PREF_LIST:
        if pref in cleaned:
            return pref
    return None


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


with st.spinner("軽量データと行政界データを読み込んでいます..."):
    df = load_data(PARQUET_FILE_ID)
    boundaries_data = load_boundaries(GEOJSON_FILE_ID)

if "selected_pref" not in st.session_state:
    st.session_state["selected_pref"] = "（すべて）"
if "prev_clicked_tooltip" not in st.session_state:
    st.session_state["prev_clicked_tooltip"] = None

st.sidebar.header("🔍 エリア選択")

if st.sidebar.button("キャッシュ再取得（データ更新時）"):
    for fname in ["small_area_slim.parquet", "admin_boundaries_slim.geojson"]:
        if os.path.exists(fname):
            os.remove(fname)
    st.cache_data.clear()
    st.session_state["selected_pref"] = "（すべて）"
    st.session_state["prev_clicked_tooltip"] = None
    st.rerun()

available_prefs = [p for p in PREF_LIST if p in set(df["PREF_NAME"].dropna().astype(str).unique())]
if not available_prefs:
    available_prefs = sorted([p for p in df["PREF_NAME"].dropna().astype(str).unique() if p != ""])

pref_options = ["（すべて）"] + available_prefs

if st.session_state["selected_pref"] not in pref_options:
    st.session_state["selected_pref"] = "（すべて）"

selected_pref = st.sidebar.selectbox(
    "都道府県を選択（地図クリックでも選択可能）",
    pref_options,
    key="selected_pref",
)

is_pref_selected = selected_pref != "（すべて）"

if not is_pref_selected:
    filtered_df = df.iloc[0:0]
    area_label = "未選択（地図クリックまたは選択肢から都道府県を選択）"
    total_pop = 0
    total_households = 0
    area_count = 0
else:
    filtered_df = df[df["PREF_NAME"].astype(str) == selected_pref]
    area_label = selected_pref
    total_pop = int(filtered_df["JINKO"].sum())
    total_households = int(filtered_df["SETAI"].sum())
    area_count = int(len(filtered_df))

m1, m2, m3, m4 = st.columns(4)
m1.metric("対象エリア", area_label)
m2.metric("総人口", f"{total_pop:,} 人")
m3.metric("総世帯数", f"{total_households:,} 世帯")
m4.metric("対象エリア数", f"{area_count:,} 件")

if is_pref_selected and len(filtered_df) > 0:
    pref_idx = PREF_LIST.index(selected_pref) if selected_pref in PREF_LIST else None
    if pref_idx is not None:
        center_lat = float(PREF_CENTERS[pref_idx, 0])
        center_lon = float(PREF_CENTERS[pref_idx, 1])
    else:
        center_lat = float(filtered_df["Y_CODE"].mean())
        center_lon = float(filtered_df["X_CODE"].mean())
    zoom_start = 9
else:
    center_lat, center_lon = 36.5, 138.0
    zoom_start = 5

m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None, control_scale=True)
add_base_tiles(m)

if not is_pref_selected and boundaries_data["features"]:
    folium.GeoJson(
        boundaries_data,
        name="都道府県境界線（クリックで選択）",
        style_function=lambda _: {
            "color": "#1f4e78",
            "weight": 1.5,
            "fillColor": "#4a90e2",
            "fillOpacity": 0.12,
        },
        highlight_function=lambda _: {
            "color": "#d95f02",
            "weight": 2.5,
            "fillColor": "#fdae61",
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["PREF_NAME"],
            aliases=["都道府県"],
            localize=True,
        ),
    ).add_to(m)

if is_pref_selected and len(filtered_df) > 0:
    if len(filtered_df) > MAX_HEATMAP_POINTS:
        st.caption(f"※描画サイズ抑制のため、ヒートマップは {len(filtered_df):,} 件から {MAX_HEATMAP_POINTS:,} 件を抽出して表示しています。")
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

map_state = st_folium(
    m,
    width="100%",
    height=620,
    returned_objects=["last_object_clicked_tooltip"],
)

clicked_tooltip = (map_state or {}).get("last_object_clicked_tooltip")
if clicked_tooltip and clicked_tooltip != st.session_state["prev_clicked_tooltip"]:
    st.session_state["prev_clicked_tooltip"] = clicked_tooltip
    clicked_pref = extract_pref_from_tooltip(clicked_tooltip)
    if clicked_pref and clicked_pref != st.session_state["selected_pref"]:
        st.session_state["selected_pref"] = clicked_pref
        st.rerun()
