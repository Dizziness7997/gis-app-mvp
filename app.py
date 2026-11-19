import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
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

# 47都道府県庁の代表座標（都道府県コード/列がない場合の最寄り判定用）
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

@st.cache_data(show_spinner=False)
def load_data():
    file_id = "1mx8pOhlG4kv3VelpZ-sSqzphLh-5l0uN"
    output = "small_area_centroids.csv"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=True)

    # メモリ節約のため必要な列のみ確認して読み込み
    header = pd.read_csv(output, nrows=0).columns.tolist()
    pref_col = next((c for c in ["PREF_NAME", "KEN_NAME", "都道府県名", "都道府県"] if c in header), None)
    code_col = next((c for c in ["PREF", "KEN", "JIS_CODE", "KEY_CODE", "CITY_CODE"] if c in header), None)

    use_cols = [c for c in ["CITY_NAME", "S_NAME", "JINKO", "SETAI", "X_CODE", "Y_CODE", pref_col, code_col] if c and c in header]
    df = pd.read_csv(output, usecols=use_cols, low_memory=False)

    df["JINKO"] = pd.to_numeric(df["JINKO"], errors="coerce").fillna(0).astype(np.int32)
    df["SETAI"] = pd.to_numeric(df["SETAI"], errors="coerce").fillna(0).astype(np.int32)
    df["X_CODE"] = pd.to_numeric(df["X_CODE"], errors="coerce").astype(np.float32)
    df["Y_CODE"] = pd.to_numeric(df["Y_CODE"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["X_CODE", "Y_CODE"]).reset_index(drop=True)
    df["CITY_NAME"] = df["CITY_NAME"].fillna("不明").astype(str)

    # 都道府県の割り当て（列 → JISコード先頭2桁 → 座標最近傍の順で確実に47都道府県を付与）
    if pref_col and df[pref_col].notna().any():
        df["PREF_NAME"] = df[pref_col].fillna("").astype(str)
    elif code_col:
        codes = df[code_col].astype(str).str.zfill(5).str[:2]
        code_map = {f"{i+1:02d}": PREF_LIST[i] for i in range(47)}
        df["PREF_NAME"] = codes.map(code_map).fillna("")
    else:
        df["PREF_NAME"] = ""

    # 未割り当て行があれば市区町村ごとの代表座標から47都道府県を高速判定
    if (df["PREF_NAME"] == "").any():
        city_coords = df.groupby("CITY_NAME")[["Y_CODE", "X_CODE"]].mean()
        coords = city_coords.values[:, None, :]  # (N_city, 1, 2)
        dists = ((coords - PREF_CENTERS[None, :, :]) ** 2).sum(axis=2)
        nearest_idx = dists.argmin(axis=1)
        city_to_pref = dict(zip(city_coords.index, [PREF_LIST[i] for i in nearest_idx]))
        mask = df["PREF_NAME"] == ""
        df.loc[mask, "PREF_NAME"] = df.loc[mask, "CITY_NAME"].map(city_to_pref)

    # カテゴリ型にしてメモリ大幅削減
    df["PREF_NAME"] = df["PREF_NAME"].astype("category")
    df["CITY_NAME"] = df["CITY_NAME"].astype("category")
    return df

with st.spinner("データを読み込んでいます（初回は1〜2分かかります）..."):
    df = load_data()

# ==========================================
# サイドバー：都道府県＆市区町村の絞り込み（文字入力対応）
# ==========================================
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

display_mode = st.sidebar.radio(
    "地図表示スタイル",
    ["自動（広域：ヒートマップ / 詳細：グリッド）", "ヒートマップ重視", "詳細グリッド重視"],
    index=0
)

# ==========================================
# メモリをコピーせずブールマスクで高速抽出
# ==========================================
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

if not target_df.empty:
    center_lat = float(target_df["Y_CODE"].mean())
    center_lon = float(target_df["X_CODE"].mean())

    zoom_start = 13 if selected_city != "（すべて）" else (8 if selected_pref != "（すべて）" else 6)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)
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

    use_heatmap = (
        display_mode == "ヒートマップ重視"
        or (display_mode.startswith("自動") and len(target_df) > 1500)
    )

    if use_heatmap:
        hm_df = target_df if len(target_df) <= 30000 else target_df.sample(30000, random_state=42)
        heat_data = list(zip(
            hm_df["Y_CODE"].astype(float).tolist(),
            hm_df["X_CODE"].astype(float).tolist(),
            np.maximum(hm_df["JINKO"].astype(float).values, 1.0).tolist()
        ))
        HeatMap(
            heat_data,
            radius=18 if len(hm_df) > 5000 else 22,
            blur=15,
            max_zoom=13,
            gradient={0.2: "cyan", 0.4: "lime", 0.65: "yellow", 0.85: "orange", 1.0: "red"}
        ).add_to(m)
    else:
        grid_df = target_df if len(target_df) <= 2500 else target_df.head(2500)
        if len(target_df) > 2500:
            st.info(f"表示件数が多いため上位2,500件を描画しています（全{len(target_df):,}件）。市区町村で絞り込むと全件表示されます。")

        dlat, dlon = 0.0006, 0.0008
        for row in grid_df.itertuples():
            lat, lon = float(row.Y_CODE), float(row.X_CODE)
            jinko, setai = int(row.JINKO), int(row.SETAI)
            s_name = row.S_NAME if hasattr(row, "S_NAME") and pd.notna(row.S_NAME) else "不明"
            city_n = str(row.CITY_NAME)

            fill_col = "#e31a1c" if jinko >= 500 else ("#fd8d3c" if jinko >= 200 else ("#fecc5c" if jinko >= 50 else "#ffffb2"))
            popup_html = f"<b>{city_n} {s_name}</b><br>人口: {jinko:,}人<br>世帯: {setai:,}世帯"

            folium.Rectangle(
                bounds=[[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
                color="#e67e22",
                weight=0.5,
                fill=True,
                fill_color=fill_col,
                fill_opacity=0.65,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=650, returned_objects=[])
else:
    st.warning("条件に一致するエリアデータがありません。絞り込み条件を変更してください。")
