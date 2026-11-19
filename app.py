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

def infer_pref(city_name: str) -> str:
    if not isinstance(city_name, str):
        return ""
    if city_name.startswith("北海道"):
        return "北海道"
    for p in PREF_LIST:
        if city_name.startswith(p):
            return p
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

    # 都道府県カラムの確保
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

# ==========================================
# サイドバー：都道府県＆市区町村の絞り込み（文字入力対応）
# ==========================================
st.sidebar.header("🔍 エリア絞り込み")

# 1. 都道府県（文字入力検索 + 選択）
pref_keyword = st.sidebar.text_input("都道府県を文字で絞り込み", placeholder="例：北海、東京、新潟")
available_prefs = [p for p in PREF_LIST if p in set(df["PREF_NAME"].unique())]
if not available_prefs:
    available_prefs = sorted([p for p in df["PREF_NAME"].unique() if p])

filtered_prefs = [p for p in available_prefs if pref_keyword.strip() in p] if pref_keyword.strip() else available_prefs
pref_options = ["（すべて）"] + filtered_prefs
selected_pref = st.sidebar.selectbox("都道府県を選択", pref_options, index=0)

# 2. 市区町村（文字入力検索は都道府県選択前から可能／選択肢は都道府県絞り込みに連動）
city_keyword = st.sidebar.text_input("市区町村を文字で絞り込み（都道府県未選択でも可）", placeholder="例：札幌、新宿、新潟市")

if selected_pref != "（すべて）":
    base_city_df = df[df["PREF_NAME"] == selected_pref]
else:
    base_city_df = df

city_candidates = sorted(base_city_df["CITY_NAME"].dropna().unique().tolist())
if city_keyword.strip():
    filtered_cities = [c for c in city_candidates if city_keyword.strip() in c]
else:
    filtered_cities = city_candidates

city_options = ["（すべて）"] + filtered_cities
selected_city = st.sidebar.selectbox("市区町村を選択", city_options, index=0)

# 表示モード（広域ヒートマップ / 詳細メッシュ・グリッド）
display_mode = st.sidebar.radio(
    "地図表示スタイル",
    ["自動（広域：ヒートマップ / 詳細：グリッド）", "ヒートマップ重視", "詳細グリッド重視"],
    index=0
)

# ==========================================
# 対象データの抽出
# ==========================================
target_df = df.copy()
if selected_pref != "（すべて）":
    target_df = target_df[target_df["PREF_NAME"] == selected_pref]
elif pref_keyword.strip():
    target_df = target_df[target_df["PREF_NAME"].str.contains(pref_keyword.strip(), na=False)]

if selected_city != "（すべて）":
    target_df = target_df[target_df["CITY_NAME"] == selected_city]
elif city_keyword.strip():
    target_df = target_df[target_df["CITY_NAME"].str.contains(city_keyword.strip(), na=False)]

# メトリクス表示
total_jinko = target_df["JINKO"].sum()
total_setai = target_df["SETAI"].sum()
area_label = (
    f"{selected_pref} / {selected_city}"
    if selected_pref != "（すべて）" and selected_city != "（すべて）"
    else (selected_city if selected_city != "（すべて）" else (selected_pref if selected_pref != "（すべて）" else "全国（絞り込み中）"))
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("対象エリア", area_label)
col2.metric("総人口", f"{int(total_jinko):,} 人")
col3.metric("総世帯数", f"{int(total_setai):,} 世帯")
col4.metric("対象エリア数", f"{len(target_df):,} 件")

st.markdown("---")

# ==========================================
# Folium地図の描画（国土地理院タイル＋ヒートマップ/詳細グリッド）
# ==========================================
if not target_df.empty:
    center_lat = float(target_df["Y_CODE"].mean())
    center_lon = float(target_df["X_CODE"].mean())

    # 対象件数に応じて初期ズームを調整
    if selected_city != "（すべて）":
        zoom_start = 13
    elif selected_pref != "（すべて）":
        zoom_start = 8
    else:
        zoom_start = 6

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)

    # 国土地理院 淡色地図（画像のような見やすいベース地図）＋標準地図＋OSM
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
        # 描画高速化のため、非常に多い場合はサンプリング
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
        # 詳細表示：125mグリッド風の矩形＋人口に応じたマーカー（最大3,000件で描画保護）
        grid_df = target_df if len(target_df) <= 3000 else target_df.head(3000)
        if len(target_df) > 3000:
            st.info(f"表示件数が多いため上位3,000件を描画しています（全{len(target_df):,}件）。市区町村で絞り込むと全件表示されます。")

        dlat = 0.0006  # 約125mメッシュ相当の半幅
        dlon = 0.0008

        for _, row in grid_df.iterrows():
            lat = float(row["Y_CODE"])
            lon = float(row["X_CODE"])
            jinko = float(row["JINKO"])
            setai = float(row["SETAI"])
            s_name = row["S_NAME"] if "S_NAME" in row and pd.notna(row["S_NAME"]) else "不明"
            city_n = row["CITY_NAME"]

            # 人口に応じた色（黄〜橙〜赤）
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
