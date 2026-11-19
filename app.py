import streamlit as st
import pandas as pd
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

PREFIX_TO_PREF = {
    "札幌市": "北海道", "旭川市": "北海道", "函館市": "北海道",
    "仙台市": "宮城県",
    "さいたま市": "埼玉県",
    "千葉市": "千葉県",
    "千代田区": "東京都", "中央区": "東京都", "港区": "東京都", "新宿区": "東京都",
    "文京区": "東京都", "台東区": "東京都", "墨田区": "東京都", "江東区": "東京都",
    "品川区": "東京都", "目黒区": "東京都", "大田区": "東京都", "世田谷区": "東京都",
    "渋谷区": "東京都", "中野区": "東京都", "杉並区": "東京都", "豊島区": "東京都",
    "北区": "東京都", "荒川区": "東京都", "板橋区": "東京都", "練馬区": "東京都",
    "足立区": "東京都", "葛飾区": "東京都", "江戸川区": "東京都", "八王子市": "東京都",
    "横浜市": "神奈川県", "川崎市": "神奈川県", "相模原市": "神奈川県",
    "新潟市": "新潟県",
    "静岡市": "静岡県", "浜松市": "静岡県",
    "名古屋市": "愛知県",
    "京都市": "京都府",
    "大阪市": "大阪府", "堺市": "大阪府",
    "神戸市": "兵庫県",
    "岡山市": "岡山県",
    "広島市": "広島県",
    "北九州市": "福岡県", "福岡市": "福岡県",
    "熊本市": "熊本県",
}

def infer_pref(city_name: str) -> str:
    if not isinstance(city_name, str) or not city_name:
        return ""
    if city_name.startswith("北海道"):
        return "北海道"
    for p in PREF_LIST:
        if city_name.startswith(p):
            return p
    for prefix, pref in PREFIX_TO_PREF.items():
        if city_name.startswith(prefix):
            return pref
    return ""

@st.cache_data
def load_data():
    file_id = "1mx8pOhlG4kv3VelpZ-sSqzphLh-5l0uN"
    output = "small_area_centroids.csv"

    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Googleドライブから214万件のマスターデータをダウンロード中...（初回のみ数分かかります）"):
            gdown.download(url, output, quiet=False)

    df = pd.read_csv(output, low_memory=False)
    df["JINKO"] = pd.to_numeric(df["JINKO"], errors="coerce").fillna(0)
    df["SETAI"] = pd.to_numeric(df["SETAI"], errors="coerce").fillna(0)
    df = df.dropna(subset=["X_CODE", "Y_CODE"])
    df["CITY_NAME"] = df["CITY_NAME"].astype(str)

    pref_candidates = ["PREF_NAME", "KEN_NAME", "都道府県名", "都道府県"]
    found_pref_col = next((c for c in pref_candidates if c in df.columns), None)
    if found_pref_col:
        df["PREF_NAME"] = df[found_pref_col].fillna("").astype(str)
    else:
        df["PREF_NAME"] = df["CITY_NAME"].apply(infer_pref)

    return df

with st.spinner("データを読み込んでいます..."):
    df = load_data()

if df.empty:
    st.error("データの読み込みに失敗しました。ファイルIDや共有設定をご確認ください。")
    st.stop()

st.sidebar.header("🔍 エリア絞り込み")

pref_keyword = st.sidebar.text_input(
    "都道府県を文字で絞り込み",
    placeholder="例：北海道、東京都、新潟県"
)

data_prefs = set(p for p in df["PREF_NAME"].unique() if p)
available_prefs = [p for p in PREF_LIST if p in data_prefs] if data_prefs else PREF_LIST[:]
if not available_prefs:
    available_prefs = PREF_LIST[:]

if pref_keyword.strip():
    filtered_prefs = [p for p in available_prefs if pref_keyword.strip() in p]
else:
    filtered_prefs = available_prefs

pref_options = ["（すべて）"] + filtered_prefs
selected_pref = st.sidebar.selectbox("都道府県を選択", pref_options, index=0)

city_keyword = st.sidebar.text_input(
    "市区町村を文字で絞り込み（都道府県未選択でも可）",
    placeholder="例：札幌市、新宿区、新潟市"
)

if selected_pref != "（すべて）":
    matched_pref_rows = df[df["PREF_NAME"] == selected_pref]
    base_city_df = matched_pref_rows if not matched_pref_rows.empty else df
else:
    base_city_df = df

city_candidates = sorted(base_city_df["CITY_NAME"].dropna().unique().tolist())
if city_keyword.strip():
    filtered_cities = [c for c in city_candidates if city_keyword.strip() in c]
else:
    filtered_cities = city_candidates

city_options = ["（すべて）"] + filtered_cities
selected_city = st.sidebar.selectbox("市区町村を選択", city_options, index=0)

display_mode = st.sidebar.radio(
    "地図表示スタイル",
    ["自動（広域：ヒートマップ / 詳細：グリッド）", "ヒートマップ重視", "詳細グリッド重視"],
    index=0
)

target_df = df.copy()
if selected_pref != "（すべて）":
    pref_rows = target_df[target_df["PREF_NAME"] == selected_pref]
    if not pref_rows.empty:
        target_df = pref_rows
elif pref_keyword.strip():
    kw_rows = target_df[target_df["PREF_NAME"].str.contains(pref_keyword.strip(), na=False)]
    if not kw_rows.empty:
        target_df = kw_rows

if selected_city != "（すべて）":
    target_df = target_df[target_df["CITY_NAME"] == selected_city]
elif city_keyword.strip():
    target_df = target_df[target_df["CITY_NAME"].str.contains(city_keyword.strip(), na=False)]

total_jinko = target_df["JINKO"].sum()
total_setai = target_df["SETAI"].sum()
area_label = (
    f"{selected_pref} / {selected_city}"
    if selected_pref != "（すべて）" and selected_city != "（すべて）"
    else (selected_city if selected_city != "（すべて）" else (selected_pref if selected_pref != "（すべて）" else "全国"))
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("対象エリア", area_label)
col2.metric("総人口", f"{int(total_jinko):,} 人")
col3.metric("総世帯数", f"{int(total_setai):,} 世帯")
col4.metric("対象エリア数", f"{len(target_df):,} 件")

st.markdown("---")

if not target_df.empty:
    center_lat = float(target_df["Y_CODE"].mean())
    center_lon = float(target_df["X_CODE"].mean())

    if selected_city != "（すべて）":
        zoom_start = 13
    elif selected_pref != "（すべて）":
        zoom_start = 8
    else:
        zoom_start = 6

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
        hm_df = target_df if len(target_df) <= 50000 else target_df.sample(50000, random_state=42)
        heat_data = [
            [float(r["Y_CODE"]), float(r["X_CODE"]), float(max(r["JINKO"], 1))]
            for _, r in hm_df.iterrows()
        ]
        HeatMap(
            heat_data,
            radius=18 if len(hm_df) > 5000 else 22,
            blur=15,
            max_zoom=13,
            gradient={0.2: "cyan", 0.4: "lime", 0.65: "yellow", 0.85: "orange", 1.0: "red"}
        ).add_to(m)
    else:
        grid_df = target_df if len(target_df) <= 3000 else target_df.head(3000)
        if len(target_df) > 3000:
            st.info(f"表示件数が多いため上位3,000件を描画しています（全{len(target_df):,}件）。市区町村で絞り込むと全件表示されます。")

        dlat = 0.0006
        dlon = 0.0008

        for _, row in grid_df.iterrows():
            lat = float(row["Y_CODE"])
            lon = float(row["X_CODE"])
            jinko = float(row["JINKO"])
            setai = float(row["SETAI"])
            s_name = row["S_NAME"] if "S_NAME" in row and pd.notna(row["S_NAME"]) else "不明"
            city_n = row["CITY_NAME"]

            if jinko >= 500:
                fill_col = "#e31a1c"
            elif jinko >= 200:
                fill_col = "#fd8d3c"
            elif jinko >= 50:
                fill_col = "#fecc5c"
            else:
                fill_col = "#ffffb2"

            popup_html = f"<b>{city_n} {s_name}</b><br>人口: {int(jinko):,}人<br>世帯: {int(setai):,}世帯"

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
