import os
import gc
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from branca.element import MacroElement, Template
import gdown

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


def filter_spatial_outliers(df_sub: pd.DataFrame, max_deg_from_median: float = 2.0) -> pd.DataFrame:
    # Remove far spatial outliers from area bounds/hull calculations
    if len(df_sub) < 10:
        return df_sub
    med_lat = float(df_sub["Y_CODE"].median())
    med_lon = float(df_sub["X_CODE"].median())
    d_lat = (df_sub["Y_CODE"].to_numpy(dtype=np.float64) - med_lat) ** 2
    d_lon = (df_sub["X_CODE"].to_numpy(dtype=np.float64) - med_lon) ** 2
    dist = np.sqrt(d_lat + d_lon)
    keep_mask = dist <= max_deg_from_median
    if int(keep_mask.sum()) >= 3:
        return df_sub.loc[keep_mask]
    return df_sub


def compute_convex_hull(points_yx: np.ndarray) -> list:
    # Compute convex hull polygon coordinates from [lat, lon] array without external libs
    if len(points_yx) == 0:
        return []
    unique_pts = np.unique(np.round(points_yx.astype(np.float64), 5), axis=0)
    if len(unique_pts) <= 2:
        pts = unique_pts
    else:
        order = np.lexsort((unique_pts[:, 0], unique_pts[:, 1]))
        pts_sorted = unique_pts[order]

        def cross(o, a, b):
            return (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1])

        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        pts = np.array(hull) if len(hull) >= 3 else unique_pts

    center = pts.mean(axis=0)
    expanded = []
    for y, x in pts:
        dy = y - center[0]
        dx = x - center[1]
        expanded.append([float(y + dy * 0.02), float(x + dx * 0.02)])
    return expanded


class ZoomHeatGridSwitcher(MacroElement):
    def __init__(self, heat_layer, grid_layer, zoom_threshold=13):
        super().__init__()
        self._name = "ZoomHeatGridSwitcher"
        map_var = "{{ this._parent.get_name() }}"
        heat_var = heat_layer.get_name()
        grid_var = grid_layer.get_name()
        js_body = textwrap.dedent(f'''
        {{% macro script(this, kwargs) %}}
        var map = {map_var};
        var heatLayer = {heat_var};
        var gridLayer = {grid_var};

        function toggleByZoom() {{
            var z = map.getZoom();
            if (z >= {zoom_threshold}) {{
                if (map.hasLayer(heatLayer)) {{ map.removeLayer(heatLayer); }}
                if (!map.hasLayer(gridLayer)) {{ map.addLayer(gridLayer); }}
            }} else {{
                if (map.hasLayer(gridLayer)) {{ map.removeLayer(gridLayer); }}
                if (!map.hasLayer(heatLayer)) {{ map.addLayer(heatLayer); }}
            }}
        }}

        map.on('zoomend', toggleByZoom);
        toggleByZoom();
        {{% endmacro %}}
        ''').strip()
        self._template = Template(js_body)


@st.cache_data(show_spinner=False)
def load_data():
    output = "small_area_slim.parquet"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={PARQUET_FILE_ID}"
        gdown.download(url, output, quiet=True)

    df = pd.read_parquet(output)
    df["JINKO"] = pd.to_numeric(df["JINKO"], errors="coerce").fillna(0).astype(np.int32)
    df["SETAI"] = pd.to_numeric(df["SETAI"], errors="coerce").fillna(0).astype(np.int32)
    df["X_CODE"] = pd.to_numeric(df["X_CODE"], errors="coerce").astype(np.float32)
    df["Y_CODE"] = pd.to_numeric(df["Y_CODE"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["X_CODE", "Y_CODE"]).reset_index(drop=True)
    df["CITY_NAME"] = df["CITY_NAME"].astype(str)

    pref_candidates = ["PREF_NAME", "KEN_NAME", "都道府県名", "都道府県"]
    found_pref_col = next((c for c in pref_candidates if c in df.columns), None)
    code_candidates = ["PREF", "KEN", "JIS_CODE", "KEY_CODE", "CITY_CODE"]
    found_code_col = next((c for c in code_candidates if c in df.columns), None)

    if found_pref_col and df[found_pref_col].notna().any():
        pref_series = df[found_pref_col].fillna("").astype(str)
    elif found_code_col:
        codes = df[found_code_col].astype(str).str.zfill(5).str[:2]
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        pref_series = codes.map(code_map).fillna("")
    else:
        pref_series = pd.Series("", index=df.index, dtype=str)

    if (pref_series == "").any():
        city_coords = df.groupby("CITY_NAME", observed=True)[["Y_CODE", "X_CODE"]].mean()
        coords = city_coords.values[:, None, :]
        dists = ((coords - PREF_CENTERS[None, :, :]) ** 2).sum(axis=2)
        nearest_idx = dists.argmin(axis=1)
        city_to_pref = dict(zip(city_coords.index, [PREF_LIST[i] for i in nearest_idx]))
        mask_empty = pref_series == ""
        pref_series.loc[mask_empty] = df.loc[mask_empty, "CITY_NAME"].map(city_to_pref)

    df["PREF_NAME"] = pref_series.astype("category")
    df["CITY_NAME"] = df["CITY_NAME"].astype("category")
    if "S_NAME" in df.columns:
        df["S_NAME"] = df["S_NAME"].fillna("不明").astype("category")

    del pref_series
    gc.collect()
    return df


def add_base_tiles(m):
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        attr="地理院タイル（淡色地図）",
        name="国土地理院 淡色地図",
        overlay=False,
        control=True,
        max_zoom=18
    ).add_to(m)
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        attr="地理院タイル（標準地図）",
        name="国土地理院 標準地図",
        overlay=False,
        control=True,
        max_zoom=18
    ).add_to(m)


with st.spinner("軽量データを読み込んでいます..."):
    df = load_data()

st.sidebar.header("🔍 エリア絞り込み")

pref_keyword = st.sidebar.text_input(
    "都道府県を文字で絞り込み",
    placeholder="例：北海道、東京都、新潟県"
)

existing_prefs = set(df["PREF_NAME"].cat.categories)
available_prefs = [p for p in PREF_LIST if p in existing_prefs]
filtered_prefs = [p for p in available_prefs if pref_keyword.strip() in p] if pref_keyword.strip() else available_prefs
selected_pref = st.sidebar.selectbox("都道府県を選択", ["（すべて）"] + filtered_prefs, index=0)

city_keyword = st.sidebar.text_input(
    "市区町村を文字で絞り込み（都道府県未選択でも可）",
    placeholder="例：札幌市、新宿区、新潟市"
)

if selected_pref != "（すべて）":
    city_candidates = sorted(df.loc[df["PREF_NAME"] == selected_pref, "CITY_NAME"].unique().tolist())
else:
    city_candidates = sorted(df["CITY_NAME"].cat.categories.tolist())

filtered_cities = [c for c in city_candidates if city_keyword.strip() in c] if city_keyword.strip() else city_candidates
selected_city = st.sidebar.selectbox("市区町村を選択", ["（すべて）"] + filtered_cities, index=0)

st.sidebar.caption("表示：広域はヒートマップ＋外枠 / 拡大（ズーム13以上）はグリッド（クリックで地図内ポップアップ）")

has_area_filter = (
    selected_pref != "（すべて）"
    or selected_city != "（すべて）"
    or bool(pref_keyword.strip())
    or bool(city_keyword.strip())
)

is_city_selected = (selected_city != "（すべて）") or bool(city_keyword.strip())

mask = pd.Series(True, index=df.index)
if selected_pref != "（すべて）":
    mask &= (df["PREF_NAME"] == selected_pref)
elif pref_keyword.strip():
    matched_p = [p for p in available_prefs if pref_keyword.strip() in p]
    mask &= df["PREF_NAME"].isin(matched_p)

if selected_city != "（すべて）":
    mask &= (df["CITY_NAME"] == selected_city)
elif city_keyword.strip():
    matched_c = [c for c in city_candidates if city_keyword.strip() in c]
    mask &= df["CITY_NAME"].isin(matched_c)

target_df = df.loc[mask]

total_jinko = int(target_df["JINKO"].sum())
total_setai = int(target_df["SETAI"].sum())
area_label = (
    f"{selected_pref} / {selected_city}"
    if selected_pref != "（すべて）" and selected_city != "（すべて）"
    else (selected_city if selected_city != "（すべて）" else (selected_pref if selected_pref != "（すべて）" else "全国"))
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("対象エリア", area_label)
col2.metric("総人口", f"{total_jinko:,} 人")
col3.metric("総世帯数", f"{total_setai:,} 世帯")
col4.metric("対象エリア数", f"{len(target_df):,} 件")

st.markdown("---")

if not has_area_filter:
    st.info("左のサイドバーから「都道府県」または「市区町村」を選択・入力すると、対象エリアのヒートマップと外枠が表示されます。")
    m = folium.Map(location=[36.5, 138.0], zoom_start=5, tiles=None)
    add_base_tiles(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=650, returned_objects=[])
    del m
    gc.collect()
elif not target_df.empty:
    clean_df = filter_spatial_outliers(target_df, max_deg_from_median=2.0)
    coords_yx = clean_df[["Y_CODE", "X_CODE"]].to_numpy(dtype=np.float32)
    center_lat = float(np.median(coords_yx[:, 0]))
    center_lon = float(np.median(coords_yx[:, 1]))
    zoom_start = 13 if is_city_selected else 8

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)
    add_base_tiles(m)

    # 1. 都道府県・市区町村の分布に沿った多角形外枠
    hull_coords = compute_convex_hull(coords_yx)
    if len(hull_coords) >= 3:
        folium.Polygon(
            locations=hull_coords,
            color="#d9381e",
            weight=2.5,
            dash_array="6, 4",
            fill=True,
            fill_color="#d9381e",
            fill_opacity=0.04,
            tooltip=f"{area_label} 外枠"
        ).add_to(m)

    # 2. ヒートマップレイヤー（広域ズーム用）
    fg_heat = folium.FeatureGroup(name="広域ヒートマップ", show=True)
    max_hm = 10000
    hm_df = clean_df if len(clean_df) <= max_hm else clean_df.sample(max_hm, random_state=42)
    heat_data = list(zip(
        hm_df["Y_CODE"].astype(float).tolist(),
        hm_df["X_CODE"].astype(float).tolist(),
        np.maximum(hm_df["JINKO"].astype(float).values, 1.0).tolist()
    ))
    HeatMap(
        heat_data,
        radius=18 if len(hm_df) > 3000 else 22,
        blur=15,
        max_zoom=18,
        gradient={0.2: "cyan", 0.4: "lime", 0.65: "yellow", 0.85: "orange", 1.0: "red"}
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # 3. グリッドレイヤー（拡大ズーム13以上用・クリックで地図内ポップアップ表示）
    fg_grid = folium.FeatureGroup(name="詳細グリッド", show=False)
    max_grid = 2500
    if len(clean_df) <= max_grid:
        grid_df = clean_df
    else:
        grid_df = clean_df.nlargest(max_grid, "JINKO")

    dlat, dlon = 0.0006, 0.0008
    for row in grid_df.itertuples():
        lat, lon = float(row.Y_CODE), float(row.X_CODE)
        jinko, setai = int(row.JINKO), int(row.SETAI)
        s_name = str(row.S_NAME) if hasattr(row, "S_NAME") else "不明"
        city_n = str(row.CITY_NAME)

        fill_col = "#e31a1c" if jinko >= 500 else ("#fd8d3c" if jinko >= 200 else ("#fecc5c" if jinko >= 50 else "#ffffb2"))
        popup_html = f"<b>{city_n} {s_name}</b><br>人口: {jinko:,}人<br>世帯: {setai:,}世帯"

        folium.Rectangle(
            bounds=[[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
            color="#e67e22",
            weight=0.6,
            fill=True,
            fill_color=fill_col,
            fill_opacity=0.65,
            tooltip=f"{city_n} {s_name}（人口: {jinko:,}人）",
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_grid)
    fg_grid.add_to(m)

    m.get_root().add_child(ZoomHeatGridSwitcher(fg_heat, fg_grid, zoom_threshold=13))

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=650, returned_objects=[])
    del m
    gc.collect()
else:
    st.warning("条件に一致するエリアデータがありません。絞り込み条件を変更してください。")
