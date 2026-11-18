import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import gdown

# ページ設定
st.set_page_config(page_title="GIS・商圏分析システム MVP", layout="wide")
st.title("🗺️ GIS・商圏分析システム (MVP)")

# GoogleドライブからCSVをダウンロードしてキャッシュ化する関数
@st.cache_data
def load_data():
    file_id = "1mx8pOhlG4kv3VelpZ-sSqzphLh-5l0uN"
    output = "small_area_centroids.csv"
    
    # ローカルにまだダウンロードされていなければ取得
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner("Googleドライブから214万件のマスターデータをダウンロード中...（初回のみ数分かかります）"):
            gdown.download(url, output, quiet=False)
            
    # パンダスで読み込み
    df = pd.read_csv(output)
    
    # 欠損値処理と型変換
    df['JINKO'] = pd.to_numeric(df['JINKO'], errors='coerce').fillna(0)
    df['SETAI'] = pd.to_numeric(df['SETAI'], errors='coerce').fillna(0)
    df = df.dropna(subset=['X_CODE', 'Y_CODE'])
    return df

with st.spinner("データを読み込んでいます..."):
    df = load_data()

if df.empty:
    st.error("データの読み込みに失敗しました。ファイルIDや共有設定をご確認ください。")
    st.stop()

# サイドバーUI：市区町村でエリア絞り込み
st.sidebar.header("🔍 エリア絞り込み")
cities = df['CITY_NAME'].dropna().unique().tolist()
cities.sort()
selected_city = st.sidebar.selectbox("市区町村を選択", cities)

# 最終的な対象エリアデータ
target_df = df[df['CITY_NAME'] == selected_city]

# メトリクス（主要指標）の表示
total_jinko = target_df['JINKO'].sum()
total_setai = target_df['SETAI'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("選択エリア", f"{selected_city}")
col2.metric("総人口", f"{int(total_jinko):,} 人")
col3.metric("総世帯数", f"{int(total_setai):,} 世帯")

st.markdown("---")

# Folium地図の描画
if not target_df.empty:
    center_lat = target_df['Y_CODE'].mean()
    center_lon = target_df['X_CODE'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    for idx, row in target_df.iterrows():
        radius_size = min(max(row['JINKO'] / 200, 3), 15)
        s_name = row['S_NAME'] if pd.notna(row['S_NAME']) else "不明"
        popup_text = f"<b>{s_name}</b><br>人口: {int(row['JINKO'])}人<br>世帯: {int(row['SETAI'])}世帯"
        
        folium.CircleMarker(
            location=[row['Y_CODE'], row['X_CODE']],
            radius=radius_size,
            popup=folium.Popup(popup_text, max_width=250),
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.6,
            weight=1
        ).add_to(m)
        
    st_folium(m, width="100%", height=600)
else:
    st.warning("このエリアには表示できるデータがありません。")
