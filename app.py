import os
import streamlit as st

st.set_page_config(page_title="GIS・商圏分析システム MVP（起動診断）", layout="wide")
st.title("🗺️ GIS・商圏分析システム (MVP) — 起動診断モード")

st.success("初期表示（Streamlit本体）は正常に起動しています。")

st.subheader("環境・ファイル確認")
st.write("カレントディレクトリ:", os.getcwd())
st.write("ローカルの small_area_slim.parquet 存在:", os.path.exists("small_area_slim.parquet"))

try:
    import folium
    import gdown
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from streamlit_folium import st_folium

    st.write("主要ライブラリのインポート: OK")
except Exception as e:
    st.error(f"インポートエラー: {e}")

if st.button("gdown ダウンロードテスト（1件のみ確認）"):
    parquet_file_id = "1Cog9sWsi9vh9HKSNYKK2wf2q8HG-kOXj"
    out = "small_area_slim.parquet"
    with st.spinner("ダウンロード中..."):
        url = f"https://drive.google.com/uc?id={parquet_file_id}"
        gdown.download(url, out, quiet=True)
    if os.path.exists(out):
        size_mb = os.path.getsize(out) / (1024 * 1024)
        st.success(f"ダウンロード完了: {size_mb:.2f} MB")
        pf = pq.ParquetFile(out)
        st.write("Parquet 行数:", pf.metadata.num_rows)
        st.write("Parquet カラム:", pf.schema_arrow.names)
    else:
        st.error("ダウンロードに失敗しました。")
