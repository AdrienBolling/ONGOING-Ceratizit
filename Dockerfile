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

ENTRYPOINT ["streamlit", "run", "./st_app/Main.py"]