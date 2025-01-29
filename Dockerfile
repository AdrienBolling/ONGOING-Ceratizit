FROM python:3.10-slim
LABEL authors="guilain"

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir top git+https://github.com/AdrienBolling/ONGOING.git

RUN apt-get remove -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY models /app/models
COPY st_app /app/st_app
COPY models_768 /app/models_768
COPY tmp_embed.pkl /app/tmp_embed.pkl

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "/app/st_app/Main.py", "--server.address=0.0.0.0"]