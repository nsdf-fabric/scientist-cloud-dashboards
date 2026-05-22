FROM python:3.11-slim

RUN \
  apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ornl_chess_strain_lib.py ORNL_CHESS_strain.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8059

CMD ["python3", "-m", "bokeh", "serve", "ORNL_CHESS_strain.py",  "--port=8059", "--address=0.0.0.0", "--allow-websocket-origin=*"]
