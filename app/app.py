import streamlit as st
from google.cloud import bigquery
from google.cloud import storage
import os
from PIL import Image
import io
from googletrans import Translator
import base64
import embedding_func
# vertex ai sdk
import vertexai
from google.cloud import bigquery
from dotenv import load_dotenv

# .envから環境変数を読み込む
load_dotenv()

# cloud run での build 用
#credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# ローカルでの実行用
credentials_path = "./key/credentials.json"

# Vertex AI の設定
project_id = "suzu-develop-stg"  # Google Cloud Project ID を設定
location = "asia-northeast1"  # Google Cloud Region を設定

# BigQuery クライアントを作成
client = bigquery.Client.from_service_account_json(credentials_path)

# Cloud Storage クライアントを作成
storage_client = storage.Client.from_service_account_json(credentials_path)

vertexai.init(project=project_id, location=location,
              credentials=credentials_path)

def display_search_results(query_job):
    """BigQuery の検索結果を表示する関数

    Args:
      query_job: BigQuery クエリの実行結果
    """
    st.write("検索結果:")
    for row in query_job:
        filename = row.gcs_filename
        distance = row.distance

        # Cloud Storage から画像ファイルを取得
        bucket_name = "coordinate-image-sample"
        blob = storage_client.bucket(bucket_name).blob(filename)
        blob_data = blob.download_as_string()

        # 画像を表示
        st.image(blob_data, caption=f"ファイル名: {filename}", use_container_width=True)
        st.write(f"distance: {distance}")

        # 使用アイテムと詳細リンクを追加
        st.write("使用アイテム:")
        st.markdown("- 商品A")
        st.markdown("- 商品B")
        st.markdown(f"[詳細はこちら](https://example.com/)")  

def image_search_page():
    st.title("コーディネート画像検索")

    # 画像アップロード
    uploaded_image = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    # 検索ボタン
    if st.button("検索", key="search_button_tab3"):
        if uploaded_image is not None:
            # 画像を一時ファイルとして保存
            bytes_data = uploaded_image.read()

            # 画像を表示
            img = io.BytesIO(bytes_data)
            st.image(img, caption="アップロードした画像", use_container_width=True)

            # 画像エンベディングを生成
            with st.spinner('画像エンベディングを生成中...'):
                embeddings = embedding_func.getImageEmbeddingFromBytes(bytes_data)

            #st.write(f"画像エンベディング 要素数: {len(embeddings)}")
            #st.write(f"要素の一部: {embeddings[:10]}")

            # BigQuery からコーディネート情報を取得
            query = f"""   
            WITH uploaded_embedding AS (
                SELECT ARRAY[{','.join(str(x) for x in embeddings)}] AS ml_generate_embedding_result
            ), 
            vector_search AS (
                SELECT
                    REGEXP_EXTRACT(base.uri, r'/([^/]+)$') AS gcs_filename,
                    distance
                FROM
                    VECTOR_SEARCH(
                        TABLE `bqml_tutorial.coordinate-image-sample-embeddings`,
                        "ml_generate_embedding_result",
                        (
                            SELECT
                                ml_generate_embedding_result
                            FROM
                                uploaded_embedding
                        ),
                        "ml_generate_embedding_result",
                        top_k => 10
                    )
            )
            SELECT gcs_filename,
                   distance 
            FROM vector_search
            ORDER BY distance
            """

            query_job = client.query(query)

            # 検索結果を表示
            display_search_results(query_job)

# 画像検索ページを表示
image_search_page()