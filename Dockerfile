FROM python:3.11-slim

RUN \
  apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ornl_chess_strain_lib.py ORNL_CHESS_strain.py refresh_api.py refresh_bus.py serve_nsdf_dashboard.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8059
EXPOSE 8060

CMD ["python3", "serve_nsdf_dashboard.py"]
