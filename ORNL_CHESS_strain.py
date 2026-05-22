"""
ORNL / CHESS strain-field dashboard: sparse or dense JSON exports → three heatmaps
(measurement mask, GP-style estimate, variance) per configurable row.

**Load order (where JSON comes from)**

Controlled by ``scientistCloudLib/SCLib_Dashboards/ornl_chess_strain_lib.resolve_strain_paths_for_session``:

- **ScientistCloud data portal** (``base_dir`` / ``save_dir`` under ``/mnt/visus_datasets``): upload dir →
  converted dir → URL query args (``strain_json_path`` / ``strain_json_url``) →
  ``ORNL_STRAIN_JSON_PATH`` / ``ORNL_STRAIN_JSON_URL``.

- **Command line / local**: ``ORNL_STRAIN_JSON_PATH`` / ``ORNL_STRAIN_JSON_URL`` first, then query args,
  then upload/converted dirs.

**Overrides (no code changes)**

- ``ORNL_STRAIN_RESOLVE_MODE`` — ``auto``, ``portal`` (always server-first), ``cli`` (always env-first).
- ``ORNL_STRAIN_SOURCE_ORDER`` — comma tokens:
  ``upload``, ``converted``, ``query_path``, ``query_url``, ``env_path``, ``env_url``
  (e.g. ``env_path,env_url,query_url`` for CHESS).

If both path and URL text fields are non-empty or partially filled, **Load / reload** uses those values
as a manual override. Clear both fields to apply the automatic order again.

URL query parameters (optional):
    strain_json_path   — override local JSON path for this session
    strain_json_url    — override remote JSON URL (full https://…) for this session
    strain_rows        — initial number of plot rows (default 2 or ORNL_STRAIN_INITIAL_ROWS)
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Div, Select, Spinner, TextInput

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SHARED_UTILS_DIR = os.path.join(PROJECT_ROOT, "scientistCloudLib", "SCLib_Dashboards")

# Same layout as Docker ``shared_utilities``: modules live under ``scientistCloudLib/SCLib_Dashboards``.
if SHARED_UTILS_DIR not in sys.path and os.path.isdir(SHARED_UTILS_DIR):
    sys.path.insert(0, SHARED_UTILS_DIR)


def _initialize_dashboard_standalone(
    request: Any = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for ``initialize_dashboard`` when ``utils_bokeh_dashboard``
    is unavailable (CHESS-only checkout). No MongoDB or portal auth; use
    ``ORNL_STRAIN_JSON_PATH`` / ``ORNL_STRAIN_JSON_URL`` or the path/URL fields in the UI.
    """
    if status_callback:
        status_callback("Standalone mode: ScientistCloud dashboard utils not loaded; JSON/S3 only.")

    local_base_dir = os.getenv("LOCAL_BASE_DIR", "")
    return {
        "success": True,
        "auth_result": {
            "is_authorized": True,
            "user_email": None,
            "access_type": "standalone",
            "error": None,
        },
        "mongodb": None,
        "params": {
            "uuid": "local",
            "server": "false",
            "name": "ORNL_CHESS_strain (standalone)",
            "base_dir": local_base_dir,
            "save_dir": local_base_dir,
            "has_args": False,
        },
        "dataset_path": None,
        "error": None,
    }


initialize_dashboard: Any
try:
    from utils_bokeh_dashboard import initialize_dashboard  # noqa: E402
except Exception as exc:
    print(f"[ORNL_CHESS_strain] Using standalone init (SCLib not available: {exc})")
    initialize_dashboard = _initialize_dashboard_standalone

try:
    from SCLib_Dashboards import create_header_banner
except Exception:

    def create_header_banner(dataset_name: str = "", dashboard_type: str = "Dashboard"):
        sc_blue = "#4E477F"
        title_text = (
            f"ScientistCloud | {dashboard_type}: {dataset_name}"
            if dataset_name
            else f"ScientistCloud | {dashboard_type}"
        )
        return Div(
            text=(
                f'<div style="background-color:{sc_blue};padding:10px 20px;color:white;'
                f'font-family:sans-serif;font-size:1.4em;font-weight:bold;">{title_text}</div>'
            ),
            sizing_mode="stretch_width",
        )


from ornl_chess_strain_lib import (  # noqa: E402
    StrainDashboardPaths,
    StrainFieldPlotConfig,
    build_strain_field_grids,
    default_row_headers,
    enrich_strain_paths_from_dataset_doc,
    find_strain_json_under_dataset_dir,
    list_strain_field_headers,
    load_strain_json,
    make_strain_triplet_figures,
    resolve_strain_paths_for_session,
)


doc = curdoc()
_request = doc.session_context.request if hasattr(doc, "session_context") and doc.session_context else None
_init = initialize_dashboard(_request)
if not _init.get("success"):
    err = Div(
        text=f"<pre>Dashboard init failed:\n{_init.get('error','')}</pre>",
        sizing_mode="stretch_width",
    )
    doc.add_root(column(create_header_banner("", "ORNL CHESS Strain"), err))
else:
    _params: Dict[str, Any] = _init.get("params") or {}

    def _first_arg(name: str) -> str:
        raw = _request.arguments.get(name) if _request and getattr(_request, "arguments", None) else None
        if not raw:
            return ""
        v = raw[0]
        return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)).strip())
        except ValueError:
            return default

    initial_rows = _int_env("ORNL_STRAIN_INITIAL_ROWS", 2)
    if _request and getattr(_request, "arguments", None):
        sr = _first_arg("strain_rows")
        if sr.isdigit():
            initial_rows = max(1, min(12, int(sr)))

    _query_path0 = _first_arg("strain_json_path") or _first_arg("strain_json")
    _query_url0 = (
        _first_arg("strain_json_url")
        or _first_arg("json_url")
        or _first_arg("data_link")
        or ""
    ).strip()
    _bd = str(_params.get("base_dir") or "")
    _sd = str(_params.get("save_dir") or "")
    _mongo_pack = _init.get("mongodb") or {}
    _dataset_collection = _mongo_pack.get("collection")

    def _fetch_dataset_doc() -> Optional[Dict[str, Any]]:
        uid = str(_params.get("uuid") or "").strip()
        coll = _dataset_collection
        if coll is None or not uid or uid == "local":
            return None
        if uid.lower().startswith(("http://", "https://", "s3://", "pelican://")):
            return None
        try:
            doc = coll.find_one({"uuid": uid})
            return doc if isinstance(doc, dict) else None
        except Exception:
            return None

    _dataset_doc = _fetch_dataset_doc()

    def _dataset_s3_auth_override() -> Optional[Dict[str, str]]:
        """Use S3 keys from Mongo when the iframe URL truncated query credentials."""
        doc = _dataset_doc
        if not doc:
            return None
        ak = str(doc.get("s3_access_key_id") or "").strip()
        sk = str(doc.get("s3_secret_access_key") or "").strip()
        if not ak or not sk:
            return None
        out: Dict[str, str] = {"access_key_id": ak, "secret_access_key": sk}
        ep = str(doc.get("s3_endpoint_url") or "").strip()
        if ep:
            out["endpoint_url"] = ep.rstrip("/")
        reg = str(doc.get("s3_region_name") or "").strip()
        if reg:
            out["region_name"] = reg
        return out

    def _prefer_upload_mirror_when_url_only(p: StrainDashboardPaths) -> StrainDashboardPaths:
        """
        Portal often prefills only the HTTPS gateway URL. If a mirrored ``*.json`` exists under
        ``base_dir`` / ``save_dir``, load from disk (avoids truncated or gateway-specific keys).
        """
        loc = (p.local_json_path or "").strip()
        jurl = (p.json_url or "").strip()
        if loc:
            return p
        if not jurl:
            return p
        mirror = find_strain_json_under_dataset_dir(_bd) or find_strain_json_under_dataset_dir(_sd)
        if mirror:
            return StrainDashboardPaths(local_json_path=mirror, json_url=jurl)
        return p

    paths = enrich_strain_paths_from_dataset_doc(
        _prefer_upload_mirror_when_url_only(
            resolve_strain_paths_for_session(
                base_dir=_bd,
                save_dir=_sd,
                query_strain_json_path=_query_path0,
                query_strain_json_url=_query_url0,
                env=StrainDashboardPaths.from_environ(),
            )
        ),
        _dataset_doc,
        base_dir=_bd,
        save_dir=_sd,
    )

    plot_cfg = StrainFieldPlotConfig()

    status_div = Div(text="", width_policy="max", height_policy="fixed", height=60)
    json_path_input = TextInput(
        title="Local JSON path (optional override)",
        value=paths.local_json_path or "",
        width=600,
    )
    json_url_input = TextInput(
        title="JSON https URL (optional — full URL you were given, e.g. presigned S3 or public object URL)",
        value=paths.json_url or "",
        width=900,
    )
    grid_w = Spinner(title="Grid width", low=8, high=256, step=1, value=plot_cfg.grid_size[0], width=100)
    grid_h = Spinner(title="Grid height", low=8, high=256, step=1, value=plot_cfg.grid_size[1], width=100)

    payload: Dict[str, Any] = {}
    headers_list: List[str] = []
    n_rows = max(1, min(12, initial_rows))
    row_headers: List[str] = list(default_row_headers(n_rows))
    figures_column = column(sizing_mode="stretch_width")

    def set_status(msg: str, ok: bool = True) -> None:
        color = "#0a0" if ok else "#a00"
        status_div.text = f'<div style="color:{color};font-family:monospace;">{msg}</div>'

    def load_payload() -> None:
        global payload, headers_list, row_headers  # noqa: PLW0603

        loc_in = (json_path_input.value or "").strip()
        url_in = (json_url_input.value or "").strip()
        if loc_in or url_in:
            p = _prefer_upload_mirror_when_url_only(
                StrainDashboardPaths(local_json_path=loc_in, json_url=url_in)
            )
        else:
            p = enrich_strain_paths_from_dataset_doc(
                _prefer_upload_mirror_when_url_only(
                    resolve_strain_paths_for_session(
                        base_dir=_bd,
                        save_dir=_sd,
                        query_strain_json_path=_query_path0,
                        query_strain_json_url=_query_url0,
                        env=StrainDashboardPaths.from_environ(),
                    )
                ),
                _dataset_doc,
                base_dir=_bd,
                save_dir=_sd,
            )
        try:
            payload = load_strain_json(p, mongo_s3_auth=_dataset_s3_auth_override())
        except Exception as e:
            payload = {}
            headers_list = []
            set_status(f"Load failed: {e}", ok=False)
            traceback.print_exc()
            figures_column.children = [Div(text="<i>No data — load JSON first.</i>")]
            return
        headers_list = list_strain_field_headers(payload, header_regex=plot_cfg.header_regex)
        if not headers_list:
            headers_list = list_strain_field_headers(payload, include_non_matching=True)
        if not headers_list:
            set_status("No plottable numeric keys found in JSON.", ok=False)
            figures_column.children = [Div(text="<i>No data — load JSON first.</i>")]
            return
        for i in range(len(row_headers)):
            if row_headers[i] not in headers_list:
                row_headers[i] = headers_list[0]
        if (p.local_json_path or "").strip():
            json_path_input.value = (p.local_json_path or "").strip()
        set_status(f"Loaded {len(headers_list)} field header(s).", ok=True)

    def apply_grid_size() -> None:
        try:
            w = int(grid_w.value)
            h = int(grid_h.value)
        except Exception:
            return
        plot_cfg.grid_size = (max(8, min(256, w)), max(8, min(256, h)))

    def rebuild_figures() -> None:
        apply_grid_size()
        figures_column.children = []
        if not payload or not headers_list:
            figures_column.children = [Div(text="<i>No data — load JSON first.</i>")]
            return
        rows_out: List[Any] = []
        for i in range(n_rows):
            h = row_headers[i] if i < len(row_headers) else headers_list[0]
            try:
                grids = build_strain_field_grids(payload, h, plot_cfg)
                p0, p1, p2 = make_strain_triplet_figures(grids, plot_cfg, row_subtitle=h)
            except Exception as e:
                err = Div(text=f"<pre>Row {i + 1} ({h}): {e}</pre>", sizing_mode="stretch_width")
                rows_out.append(err)
                traceback.print_exc()
                continue

            sel = Select(title=f"Row {i + 1} field", value=h, options=headers_list, width=420)

            def on_header_change(idx: int, attr: str, old: str, new: str) -> None:
                row_headers[idx] = new
                rebuild_figures()

            def _bind_select(idx: int):
                def _cb(attr: str, old: str, new: str) -> None:
                    on_header_change(idx, attr, old, new)

                return _cb

            sel.on_change("value", _bind_select(i))
            # Two rows per set: field selector (which JSON key to plot), then the three heatmaps.
            rows_out.append(
                column(
                    row(sel, sizing_mode="scale_width"),
                    row(p0, p1, p2, sizing_mode="scale_width"),
                    sizing_mode="stretch_width",
                )
            )
        figures_column.children = rows_out

    def on_reload() -> None:
        load_payload()
        rebuild_figures()

    def on_add_row() -> None:
        global n_rows  # noqa: PLW0603

        if n_rows >= 12:
            return
        n_rows += 1
        row_headers.append(headers_list[0] if headers_list else "0/data/uniform_strain")
        rebuild_figures()

    def on_remove_row() -> None:
        global n_rows  # noqa: PLW0603

        if n_rows <= 1:
            return
        n_rows -= 1
        if len(row_headers) > n_rows:
            row_headers.pop()
        rebuild_figures()

    btn_reload = Button(label="Load / reload JSON", button_type="primary", width=180)
    btn_add = Button(label="Add row", width=100)
    btn_remove = Button(label="Remove row", width=120)
    btn_reload.on_click(on_reload)
    btn_add.on_click(on_add_row)
    btn_remove.on_click(on_remove_row)

    # _standalone_note = (
    #     "<p><b>Standalone mode:</b> ScientistCloud <code>utils_bokeh_dashboard</code> was not loaded; "
    #     "this session only loads JSON via env / https URL / the path field (no portal auth).</p>"
    #     if initialize_dashboard is _initialize_dashboard_standalone
    #     else ""
    # )
    # help_div = Div(
    #     text=(
    #         "<p><b>Automatic load order</b> (clear both fields below to use it): "
    #         "<code>" + strain_resolve_order_summary(_bd, _sd) + "</code>. "
    #         "See module docstring in <code>ornl_chess_strain_lib.py</code> for token meanings "
    #         "(<code>upload</code>, <code>converted</code>, <code>query_*</code>, <code>env_*</code>). "
    #         "On the ScientistCloud portal mount (<code>/mnt/visus_datasets/…</code>), server directories are tried "
    #         "before gateway URLs; from the command line, <code>ORNL_STRAIN_JSON_PATH</code> / "
    #         "<code>ORNL_STRAIN_JSON_URL</code> are tried first. "
    #         "Override globally: <code>ORNL_STRAIN_RESOLVE_MODE</code> = <code>auto</code> | "
    #         "<code>portal</code> | <code>cli</code>, or set <code>ORNL_STRAIN_SOURCE_ORDER</code> "
    #         "(comma-separated tokens). "
    #         "Rows: <code>ORNL_STRAIN_INITIAL_ROWS</code> (default 2).</p>"
    #         f"{_standalone_note}"
    #     ),
    #     sizing_mode="stretch_width",
    # )

    controls = column(
        row(json_path_input, sizing_mode="scale_width"),
        row(json_url_input, sizing_mode="scale_width"),
        row(grid_w, grid_h, btn_reload, btn_add, btn_remove, sizing_mode="scale_width"),
        status_div,
        sizing_mode="stretch_width",
    )

    header = create_header_banner("ORNL CHESS strain (JSON)", "ORNL CHESS Strain")
    root = column(header, controls, figures_column, sizing_mode="stretch_width")
    doc.add_root(root)

    if paths.local_json_path or paths.json_url:
        on_reload()
    else:
        figures_column.children = [
            Div(
                text=(
                    "<p>No JSON resolved yet. Set <code>ORNL_STRAIN_JSON_PATH</code> / "
                    "<code>ORNL_STRAIN_JSON_URL</code>, clear both fields to use the automatic order, "
                    "or enter a path or URL above and click <b>Load / reload JSON</b>.</p>"
                )
            )
        ]


# cd /Users/amygooch/GIT/ScientistCloud2.0/scientistcloud/SC_Dashboards/dashboards
#  ORNL_STRAIN_JSON_PATH=/Users/amygooch/Downloads/ORNL_strain/reduced_data.json \
#  bokeh serve ORNL_CHESS_strain.py --port 50171 --allow-websocket-origin=localhost:50171