
# instructions

ORNL_CHESS_strain.py tries to import utils_bokeh_dashboard from SCLib. 
If that import fails, it uses built-in standalone init (no auth, no Mongo, which is what ScientistCloud uses)

How they load data (no UUID)

Standalone uses, in order (see `ornl_chess_strain_lib.py` docstring — CLI mode):

- `ORNL_STRAIN_JSON_PATH` — local file path
- `ORNL_STRAIN_JSON_URL` — full https://… (or gateway URL with keys; needs boto3)
- `URL query` — example "?strain_json_url=…" or "?strain_json_path=…"
- `UI` — path/URL fields + Load / reload JSON

 # Docker run

Example:

```bash

IMAGE_NAME=scientist-cloud-dashboards

docker build --tag  ${IMAGE_NAME} .

docker run --rm  \
   -e ORNL_STRAIN_JSON_PATH="/ORNL_strain/reduced_data.json" \
   -e ORNL_STRAIN_SOURCE_ORDER=env_path,env_url,query_url \
   -v ${PWD}/ORNL_strain:/ORNL_strain \
   -p 8059:8059 \
   ${IMAGE_NAME} 

```

 # Apptainer run

Note: **first build the Docker image above**

```bash

apptainer build ${IMAGE_NAME}.sif docker-daemon://${IMAGE_NAME}:latest

apptainer run \
   --env ORNL_STRAIN_JSON_PATH="/ORNL_strain/reduced_data.json" \
   --env ORNL_STRAIN_SOURCE_ORDER="env_path,env_url,query_url" \
   --bind ${PWD}/ORNL_strain:/ORNL_strain \
   ${IMAGE_NAME}.sif

```

# Local run

```bash

export ORNL_STRAIN_JSON_PATH=/data/reduced_data.json

# optional
export ORNL_STRAIN_SOURCE_ORDER=env_path,env_url,query_url

bokeh serve ORNL_CHESS_strain.py \
--port 8059 \
--address 0.0.0.0 \
--allow-websocket-origin=localhost:8059
```

Then open: http://host:8059/ORNL_CHESS_strain/?strain_json_url=https%3A%2F%2F...

# (OPTIONAL) Get data from S3

```bash
export AWS_ACCESS_KEY_ID="XXXXX"
export AWS_SECRET_ACCESS_KEY="YYYYY"
export AWS_DEFAULT_REGION="us-east-1"

# optional
# sudo snap install aws-cli 

aws s3 sync \
  --endpoint-url "https://us-east-1.gw.future-tech-holdings.com" \
  "s3://scientistcloud/IDX_TEST/ORNL_strain/" \
  ./ORNL_strain/
```


# (OPTIONAL) Changes may want to make

- Rename the header in standalone (already falls back to a simple purple “ScientistCloud | ORNL CHESS Strain” banner if SCLib_Dashboards package isn’t installed).
- Set `ORNL_STRAIN_RESOLVE_MODE=cli` so resolution never assumes /mnt/visus_datasets/… portal paths.
- ORNL_CHESS_strain.json / nginx / portal share links — not needed off ScientistCloud.


