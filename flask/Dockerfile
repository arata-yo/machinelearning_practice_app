FROM python:3.11-slim

# 必要なシステムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係をコピー
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# uWSGIで起動
CMD ["uwsgi", "--ini", "uwsgi.ini"]