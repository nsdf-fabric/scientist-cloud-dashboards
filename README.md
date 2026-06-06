# ORNL CHESS NSDF Dashboard

This dashboard loads the new NSDF measurement schema directly. 

## Expected Data

`data.json`:

```json
{
  "dataset_x": [[1.0, 2.0], [3.0, 4.0]],
  "dataset_y": [69.1, 69.2],
  "bounds": [[0.0, 10.0], [0.0, 10.0]],
  "backend": "sklearn",
  "kernel": "rbf"
}
```

`dataset_x[i][0]` is the labx coordinate, `dataset_x[i][1]` is the labz coordinate,
and `dataset_y[i]` is the observed value at that point. When valid `bounds` are
present, they are used for grid normalization; otherwise the dashboard falls back
to observed coordinate min/max.

To force a fixed grid at dashboard startup, set either:

```bash
ORNL_NSDF_GRID_SIZE=21x13
```

or:

```bash
ORNL_NSDF_GRID_WIDTH=21
ORNL_NSDF_GRID_HEIGHT=13
```

These can be exported in the shell or placed in `ORNL_S3_ENV_FILE`. When set, the
dashboard keeps the fixed grid across reloads and S3 refreshes.

Optional `surrogate.json`:

```json
{
  "workflow_id": "test-workflow",
  "surrogate": [69.1, 69.2],
  "uncertainty": [0.1, 0.2],
  "raw_uncertainty": [0.01, 0.02]
}
```

`surrogate`, `uncertainty`, and `raw_uncertainty` are used only when they are numeric
1D lists matching `len(dataset_y)`. Variance is visualized as `uncertainty ** 2`.

## Data Sources

Preferred environment variables:

- `ORNL_NSDF_DATA_JSON_PATH`
- `ORNL_NSDF_DATA_JSON_URL`
- `ORNL_SURROGATE_JSON_PATH`
- `ORNL_SURROGATE_JSON_URL`

Preferred URL query parameters:

- `nsdf_data_json_path`
- `nsdf_data_json_url`
- `surrogate_json_path`
- `surrogate_json_url`

Compatibility aliases still work, but now point to NSDF `data.json`, not the old
strain schema:

- `ORNL_STRAIN_JSON_PATH`
- `ORNL_STRAIN_JSON_URL`
- `strain_json_path`
- `strain_json_url`

If the data source is a local path ending in `data.json`, the dashboard also tries
`surrogate.json` in the same directory when no explicit surrogate path/URL is set.
For data URLs ending in `data.json`, it tries the sibling `surrogate.json` URL.
Missing inferred surrogate files are non-fatal.

## S3 Event Refresh

For S3-backed data, put credentials, object locations, and the refresh API key in an env file:

```bash
# /secure/path/nsdf-s3.env
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_SESSION_TOKEN=

ORNL_REFRESH_API_KEY=choose-a-secret-refresh-key
ORNL_NSDF_S3_BUCKET=your-bucket
ORNL_NSDF_S3_DATA_KEY=path/to/data.json
ORNL_NSDF_S3_SURROGATE_KEY=path/to/surrogate.json
ORNL_NSDF_S3_ENDPOINT_URL=https://your-s3-compatible-endpoint.example.com
ORNL_NSDF_S3_REGION=us-east-1
```

`ORNL_NSDF_S3_SURROGATE_KEY` is optional. If omitted and the data key ends in
`data.json`, the dashboard tries the sibling `surrogate.json`. Missing surrogate
objects are non-fatal.

Run with the env file path:

```bash
cp .env.example .env
# edit .env, or point ORNL_S3_ENV_FILE at another env file
export ORNL_S3_ENV_FILE=${PWD}/.env

uv sync
uv run python -m nsdf_dashboard
```

Open the dashboard:

```text
http://localhost:8059/
```

Trigger a refresh from S3:

```bash
curl -X POST http://localhost:8060/refresh \
  -H "X-API-Key: choose-a-secret-refresh-key"
```

The endpoint is an alert only; it does not send data. Every open dashboard session
reloads the configured S3 `data.json` and optional `surrogate.json`.

## Docker Run

```bash
IMAGE_NAME=scientist-cloud-dashboards

docker build --tag ${IMAGE_NAME} .

docker run --rm \
  --env-file .env \
  -p 8059:8059 \
  -p 8060:8060 \
  ${IMAGE_NAME}
```

## Docker Compose

```bash
cp .env.example .env
# edit .env
docker compose up -d --build
```

Open:

```text
http://localhost:8059/
```

Trigger refresh:

```bash
curl -X POST http://localhost:8060/refresh \
  -H "X-API-Key: choose-a-secret-refresh-key"
```

## Apptainer Run

First build the Docker image above.

```bash
apptainer build scientist-cloud-dashboards.sif docker-daemon://scientist-cloud-dashboards:latest

apptainer run \
  --env ORNL_NSDF_DATA_JSON_PATH="/ORNL_strain/data.json" \
  --env ORNL_SURROGATE_JSON_PATH="/ORNL_strain/surrogate.json" \
  --env ORNL_STRAIN_SOURCE_ORDER="env_path,env_url,query_url" \
  --bind ${PWD}/ORNL_strain:/ORNL_strain \
  scientist-cloud-dashboards.sif
```

## Local Run

```bash
cp .env.example .env
# edit .env
export ORNL_S3_ENV_FILE=${PWD}/.env

uv sync
uv run python -m nsdf_dashboard
```

Then open:

```text
http://localhost:8059/
```

## Smoke Checks

```bash
python3 -m compileall .
uv run python tests/test_nsdf_dashboard.py
```
