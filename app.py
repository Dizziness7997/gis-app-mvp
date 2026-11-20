import gc
import os
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
def load_pref_summary(parquet_file_id: str) -> pd.DataFrame:
    """軽量な都道府県別サマリー（集計値と都道府県一覧のみ）をキャッシュする"""
    output = "small_area_slim.parquet"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={parquet_file_id}"
        gdown.download(url, output, quiet=True)

    if not os.path.exists(output):
        return pd.DataFrame(columns=["PREF_NAME", "JINKO", "SETAI", "AREA_COUNT"])

    cols_available = ["PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE", "CITY_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"]
    df = pd.read_parquet(output)
    keep_cols = [c for c in cols_available if c in df.columns]
    df = df[keep_cols]

    jinko = pd.to_numeric(df.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int64)
    setai = pd.to_numeric(df.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int64)

    code_col = next((c for c in ["PREF_CODE", "CITY_CODE", "KEY_CODE"] if c in df.columns), None)
    if code_col:
        codes2 = df[code_col].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(5).str[:2]
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        pref_series = codes2.map(code_map).fillna("")
    else:
        pref_series = pd.Series("", index=df.index, dtype=str)

    if "PREF_NAME" in df.columns:
        raw_pref = df["PREF_NAME"].fillna("").astype(str).str.strip()
        pref_series = pd.Series(np.where(pref_series != "", pref_series, raw_pref), index=df.index, dtype=str)

    if (pref_series == "").any() and "CITY_NAME" in df.columns and "X_CODE" in df.columns and "Y_CODE" in df.columns:
        valid_mask = (df["CITY_NAME"].fillna("").astype(str) != "") & df["X_CODE"].notna() & df["Y_CODE"].notna()
        if valid_mask.any():
            valid_df = pd.DataFrame({
                "CITY_NAME": df.loc[valid_mask, "CITY_NAME"].astype(str),
                "Y_CODE": pd.to_numeric(df.loc[valid_mask, "Y_CODE"], errors="coerce"),
                "X_CODE": pd.to_numeric(df.loc[valid_mask, "X_CODE"], errors="coerce"),
            }).dropna()
            if len(valid_df) > 0:
                city_coords = valid_df.groupby("CITY_NAME", observed=True)[["Y_CODE", "X_CODE"]].mean()
                coords = city_coords.values[:, None, :]
                dists = ((coords - PREF_CENTERS_ARR[None, :, :]) ** 2).sum(axis=2)
                nearest_idx = dists.argmin(axis=1)
                city_to_pref = dict(zip(city_coords.index, [PREF_LIST[i] for i in nearest_idx]))
                mask_empty = (pref_series == "") & (df["CITY_NAME"].fillna("").astype(str) != "")
                pref_series = pref_series.copy()
                pref_series.loc[mask_empty] = df.loc[mask_empty, "CITY_NAME"].astype(str).map(city_to_pref).fillna("")
                del valid_df, city_coords, coords, dists

    summary_df = pd.DataFrame({
        "PREF_NAME": pref_series,
        "JINKO": jinko,
        "SETAI": setai,
        "AREA_COUNT": np.ones(len(df), dtype=np.int64),
    })
    summary_df = summary_df[summary_df["PREF_NAME"] != ""].groupby("PREF_NAME", as_index=False).sum()

    del df, pref_series, jinko, setai
    gc.collect()
    return summary_df


@st.cache_data(show_spinner=False)
def load_pref_data(parquet_file_id: str, target_pref: str) -> pd.DataFrame:
    """選択された1都道府県分のデータのみを抽出してキャッシュする"""
    output = "small_area_slim.parquet"
    if not target_pref or target_pref == "（すべて）" or not os.path.exists(output):
        return pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])

    cols_wanted = [
        "PREF_NAME", "PREF_CODE", "CITY_CODE", "KEY_CODE",
        "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"
    ]
    df = pd.read_parquet(output)
    keep_cols = [c for c in cols_wanted if c in df.columns]
    df = df[keep_cols]

    code_col = next((c for c in ["PREF_CODE", "CITY_CODE", "KEY_CODE"] if c in df.columns), None)
    if code_col:
        codes2 = df[code_col].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(5).str[:2]
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        pref_series = codes2.map(code_map).fillna("")
    else:
        pref_series = pd.Series("", index=df.index, dtype=str)

    if "PREF_NAME" in df.columns:
        raw_pref = df["PREF_NAME"].fillna("").astype(str).str.strip()
        pref_series = pd.Series(np.where(pref_series != "", pref_series, raw_pref), index=df.index, dtype=str)

    mask = (pref_series == target_pref).to_numpy()
    df = df.loc[mask].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])

    df["PREF_NAME"] = target_pref
    df["JINKO"] = pd.to_numeric(df.get("JINKO", 0), errors="coerce").fillna(0).astype(np.int32)
    df["SETAI"] = pd.to_numeric(df.get("SETAI", 0), errors="coerce").fillna(0).astype(np.int32)
    df["X_CODE"] = pd.to_numeric(df["X_CODE"], errors="coerce").astype(np.float32)
    df["Y_CODE"] = pd.to_numeric(df["Y_CODE"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["X_CODE", "Y_CODE"]).reset_index(drop=True)
    df["CITY_NAME"] = df.get("CITY_NAME", "").fillna("").astype(str)
    df["S_NAME"] = df["S_NAME"].fillna("不明").astype(str) if "S_NAME" in df.columns else "不明"

    drop_cols = [c for c in ["PREF_CODE", "CITY_CODE", "KEY_CODE"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

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


with st.spinner("都道府県別サマリーを読み込んでいます..."):
    summary_df = load_pref_summary(PARQUET_FILE_ID)

if "selected_pref" not in st.session_state:
    st.session_state["selected_pref"] = "（すべて）"

available_prefs_set = set(summary_df["PREF_NAME"].dropna().astype(str).unique())
available_prefs = [p for p in PREF_LIST if p in available_prefs_set]
if not available_prefs:
    available_prefs = PREF_LIST.copy()

pref_options = ["（すべて）"] + available_prefs

if st.session_state["selected_pref"] not in pref_options:
    st.session_state["selected_pref"] = "（すべて）"
if "pref_selectbox" not in st.session_state or st.session_state["pref_selectbox"] not in pref_options:
    st.session_state["pref_selectbox"] = st.session_state["selected_pref"]

st.sidebar.header("🔍 エリア検索・選択")

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
    on_change=lambda: st.session_state.update({"selected_pref": st.session_state["pref_selectbox"]}),
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
        set_selected_pref("（すべて）")
        st.rerun()

selected_pref = st.session_state["selected_pref"]
is_pref_selected = selected_pref != "（すべて）"

with st.expander("🗾 地方・都道府県ボタンから選ぶ", expanded=not is_pref_selected):
    st.caption("地方ブロックごとのボタンから都道府県を選ぶと、対象エリアのヒートマップ・グリッド地図を表示します（初期表示は軽量化のためヒートマップ未描画）。")
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

if not is_pref_selected:
    filtered_df = pd.DataFrame(columns=["PREF_NAME", "CITY_NAME", "S_NAME", "X_CODE", "Y_CODE", "JINKO", "SETAI"])
    area_label = "未選択（地方ボタン・検索・プルダウンから選択）"
    total_pop = 0
    total_households = 0
    area_count = 0
else:
    pref_row = summary_df[summary_df["PREF_NAME"] == selected_pref]
    area_label = selected_pref
    if len(pref_row) > 0:
        total_pop = int(pref_row["JINKO"].iloc[0])
        total_households = int(pref_row["SETAI"].iloc[0])
        area_count = int(pref_row["AREA_COUNT"].iloc[0])
    else:
        total_pop = 0
        total_households = 0
        area_count = 0

    with st.spinner(f"{selected_pref} の小地域データを読み込んでいます..."):
        filtered_df = load_pref_data(PARQUET_FILE_ID, selected_pref)

m1, m2, m3, m4 = st.columns(4)
m1.metric("対象エリア", area_label)
m2.metric("総人口", f"{total_pop:,} 人")
m3.metric("総世帯数", f"{total_households:,} 世帯")
m4.metric("対象エリア数", f"{area_count:,} 件")

if is_pref_selected and len(filtered_df) > 0:
    if selected_pref in PREF_CENTERS:
        center_lat, center_lon = PREF_CENTERS[selected_pref]
    else:
        center_lat = float(filtered_df["Y_CODE"].mean())
        center_lon = float(filtered_df["X_CODE"].mean())

    m = folium.Map(location=[float(center_lat), float(center_lon)], zoom_start=9, tiles=None, control_scale=True)
    add_base_tiles(m)

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

    st_folium(
        m,
        width="100%",
        height=620,
        returned_objects=[],
    )
else:
    st.info("地方ブロックの都道府県ボタン、サイドバーの検索、またはプルダウンから都道府県を選択すると、地図（ヒートマップ・ズーム切替グリッド）が表示されます。")
