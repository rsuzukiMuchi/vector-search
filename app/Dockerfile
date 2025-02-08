# ベースイメージを指定
FROM python:3.11-slim

# 作業ディレクトリを作成
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# ポートを公開
EXPOSE 8080

# アプリケーションを実行
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]