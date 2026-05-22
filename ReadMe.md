
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

docker build --tag scientist-cloud-dashboards .

docker run --rm  -e ORNL_STRAIN_JSON_PATH="/data/reduced_data.json" -e ORNL_STRAIN_SOURCE_ORDER=env_path,env_url,query_url scientist-cloud-dashboards

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


# Optional changes may want to make

- Rename the header in standalone (already falls back to a simple purple “ScientistCloud | ORNL CHESS Strain” banner if SCLib_Dashboards package isn’t installed).
- Set `ORNL_STRAIN_RESOLVE_MODE=cli` so resolution never assumes /mnt/visus_datasets/… portal paths.
- ORNL_CHESS_strain.json / nginx / portal share links — not needed off ScientistCloud.


