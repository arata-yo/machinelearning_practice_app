FROM python:3.11-slim
ENV PYTHONUNBUFFERED 1
RUN mkdir /code \
    apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/
RUN pip install --no-cache-dir \
    pip install --upgrade pip \
    jupyter \
    notebook \
    jupyterlab \
    pandas \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    lightgbm \
    joblib
RUN pip install -r requirements.txt
COPY . /code
