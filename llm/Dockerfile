FROM python:3.11-slim

# システムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデルディレクトリ
RUN mkdir -p /app/models /app/data

# アプリケーションファイル
COPY . .

# モデルのダウンロードスクリプト
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True); \
    model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', trust_remote_code=True); \
    tokenizer.save_pretrained('/app/models/phi2'); \
    model.save_pretrained('/app/models/phi2')" || echo "Model will be downloaded on first run"

EXPOSE 5001

CMD ["python", "app.py"]