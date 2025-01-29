FROM python:3.10-slim
LABEL authors="guilain"

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements-docker.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements-docker.txt
RUN pip3 install --no-cache-dir top git+https://github.com/AdrienBolling/ONGOING.git

RUN apt-get remove -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY st_app /app/st_app
COPY src /app/src
COPY config.yaml /app/config.yaml

EXPOSE 8501

ENV PYTHONPATH="/app"

ENTRYPOINT ["streamlit", "run", "/app/src/st_app/new_main.py", "--server.address=0.0.0.0"]