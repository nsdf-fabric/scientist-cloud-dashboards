"""
ORNL / CHESS NSDF measurement dashboard.

Loads native NSDF ``data.json`` plus optional ``surrogate.json`` and renders three heatmaps:
measurement locations, estimate, and variance. Legacy ``ORNL_STRAIN_JSON_*`` environment
variables and ``strain_json_*`` query parameters remain aliases for NSDF ``data.json``.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, Dict, Optional

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Div, Spinner, TextInput

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SHARED_UTILS_DIR = os.path.join(PROJECT_ROOT, "scientistCloudLib", "SCLib_Dashboards")

if SHARED_UTILS_DIR not in sys.path and os.path.isdir(SHARED_UTILS_DIR):
    sys.path.insert(0, SHARED_UTILS_DIR)


def _initialize_dashboard_standalone(
    request: Any = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
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
            "name": "ORNL_CHESS_NSDF (standalone)",
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
    NSDFLoadedBundle,
    StrainDashboardPaths,
    StrainFieldPlotConfig,
    build_strain_field_grids,
    enrich_strain_paths_from_dataset_doc,
    find_strain_json_under_dataset_dir,
    infer_nsdf_grid_size,
    list_nsdf_field_headers,
    load_simple_env_file,
    load_nsdf_json_bundle,
    make_strain_triplet_figures,
    resolve_strain_paths_for_session,
    validate_nsdf_measurement_doc,
    validate_nsdf_surrogate_doc,
)


def _float_env(name: str, default: float) -> float:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    raw = os.environ.get(name) or env_file_values.get(name) or str(default)
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _dashboard_env_value(name: str) -> str:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    return (os.environ.get(name) or env_file_values.get(name) or "").strip()


def _int_value(raw: str) -> Optional[int]:
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _fixed_grid_size_from_env() -> Optional[tuple[int, int]]:
    compact = _dashboard_env_value("ORNL_NSDF_GRID_SIZE").lower().replace(" ", "")
    if compact:
        for sep in ("x", ",", ":"):
            if sep not in compact:
                continue
            left, right = compact.split(sep, 1)
            width = _int_value(left)
            height = _int_value(right)
            if width and height:
                return width, height
    width = _int_value(_dashboard_env_value("ORNL_NSDF_GRID_WIDTH"))
    height = _int_value(_dashboard_env_value("ORNL_NSDF_GRID_HEIGHT"))
    if width and height:
        return width, height
    return None


doc = curdoc()
_request = doc.session_context.request if hasattr(doc, "session_context") and doc.session_context else None
_init = initialize_dashboard(_request)
if not _init.get("success"):
    err = Div(
        text=f"<pre>Dashboard init failed:\n{_init.get('error','')}</pre>",
        sizing_mode="stretch_width",
    )
    doc.add_root(column(create_header_banner("", "ORNL CHESS NSDF"), err))
else:
    _params: Dict[str, Any] = _init.get("params") or {}

    def _first_arg(name: str) -> str:
        raw = _request.arguments.get(name) if _request and getattr(_request, "arguments", None) else None
        if not raw:
            return ""
        v = raw[0]
        return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)

    _query_data_path0 = (
        _first_arg("nsdf_data_json_path")
        or _first_arg("strain_json_path")
        or _first_arg("strain_json")
    )
    _query_data_url0 = (
        _first_arg("nsdf_data_json_url")
        or _first_arg("strain_json_url")
        or _first_arg("json_url")
        or _first_arg("data_link")
        or ""
    ).strip()
    _query_surrogate_path0 = _first_arg("surrogate_json_path")
    _query_surrogate_url0 = _first_arg("surrogate_json_url")

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
            found = coll.find_one({"uuid": uid})
            return found if isinstance(found, dict) else None
        except Exception:
            return None

    _dataset_doc = _fetch_dataset_doc()

    def _dataset_s3_auth_override() -> Optional[Dict[str, str]]:
        dataset_doc = _dataset_doc
        if not dataset_doc:
            return None
        ak = str(dataset_doc.get("s3_access_key_id") or "").strip()
        sk = str(dataset_doc.get("s3_secret_access_key") or "").strip()
        if not ak or not sk:
            return None
        out: Dict[str, str] = {"access_key_id": ak, "secret_access_key": sk}
        ep = str(dataset_doc.get("s3_endpoint_url") or "").strip()
        if ep:
            out["endpoint_url"] = ep.rstrip("/")
        reg = str(dataset_doc.get("s3_region_name") or "").strip()
        if reg:
            out["region_name"] = reg
        return out

    def _prefer_upload_mirror_when_url_only(p: StrainDashboardPaths) -> StrainDashboardPaths:
        loc = (p.local_json_path or "").strip()
        jurl = (p.json_url or "").strip()
        if loc or not jurl:
            return p
        mirror = find_strain_json_under_dataset_dir(_bd) or find_strain_json_under_dataset_dir(_sd)
        if mirror:
            return StrainDashboardPaths(
                local_json_path=mirror,
                json_url=jurl,
                surrogate_json_path=p.surrogate_json_path,
                surrogate_json_url=p.surrogate_json_url,
                s3_env_file=p.s3_env_file,
                s3_bucket=p.s3_bucket,
                s3_data_key=p.s3_data_key,
                s3_surrogate_key=p.s3_surrogate_key,
                s3_endpoint_url=p.s3_endpoint_url,
                s3_region=p.s3_region,
            )
        return p

    def _resolve_paths() -> StrainDashboardPaths:
        return enrich_strain_paths_from_dataset_doc(
            _prefer_upload_mirror_when_url_only(
                resolve_strain_paths_for_session(
                    base_dir=_bd,
                    save_dir=_sd,
                    query_strain_json_path=_query_data_path0,
                    query_strain_json_url=_query_data_url0,
                    query_surrogate_json_path=_query_surrogate_path0,
                    query_surrogate_json_url=_query_surrogate_url0,
                    env=StrainDashboardPaths.from_environ(),
                )
            ),
            _dataset_doc,
            base_dir=_bd,
            save_dir=_sd,
        )

    paths = _resolve_paths()
    plot_cfg = StrainFieldPlotConfig()
    fixed_grid_size = _fixed_grid_size_from_env()
    if fixed_grid_size:
        plot_cfg.grid_size = fixed_grid_size

    status_div = Div(text="", width_policy="max", height_policy="fixed", height=86)
    data_path_input = TextInput(
        title="NSDF data JSON path",
        value=paths.local_json_path or "",
        width=600,
    )
    data_url_input = TextInput(
        title="NSDF data JSON URL",
        value=paths.json_url or "",
        width=900,
    )
    surrogate_path_input = TextInput(
        title="Optional surrogate JSON path",
        value=paths.surrogate_json_path or "",
        width=600,
    )
    surrogate_url_input = TextInput(
        title="Optional surrogate JSON URL",
        value=paths.surrogate_json_url or "",
        width=900,
    )
    grid_w = Spinner(title="Grid width", low=1, high=512, step=1, value=plot_cfg.grid_size[0], width=100)
    grid_h = Spinner(title="Grid height", low=1, high=512, step=1, value=plot_cfg.grid_size[1], width=100)

    loaded_bundle: Optional[NSDFLoadedBundle] = None
    figures_column = column(sizing_mode="stretch_width")
    s3_refresh_seconds = _float_env("ORNL_NSDF_REFRESH_SECONDS", 10.0)

    def set_status(msg: str, ok: bool = True) -> None:
        color = "#0a0" if ok else "#a00"
        status_div.text = f'<div style="color:{color};font-family:monospace;white-space:pre-wrap;">{msg}</div>'

    def _paths_from_inputs_or_auto() -> StrainDashboardPaths:
        data_path = (data_path_input.value or "").strip()
        data_url = (data_url_input.value or "").strip()
        surrogate_path = (surrogate_path_input.value or "").strip()
        surrogate_url = (surrogate_url_input.value or "").strip()
        if data_path or data_url or surrogate_path or surrogate_url:
            return _prefer_upload_mirror_when_url_only(
                StrainDashboardPaths(
                    local_json_path=data_path,
                    json_url=data_url,
                    surrogate_json_path=surrogate_path,
                    surrogate_json_url=surrogate_url,
                )
            )
        return _resolve_paths()

    def load_payload() -> None:
        global loaded_bundle  # noqa: PLW0603

        p = _paths_from_inputs_or_auto()
        try:
            bundle = load_nsdf_json_bundle(p, mongo_s3_auth=_dataset_s3_auth_override())
            measurement = validate_nsdf_measurement_doc(bundle.data)
            fields = list_nsdf_field_headers(bundle.data, bundle.surrogate)
            inferred_grid_size = infer_nsdf_grid_size(bundle.data)
            active_grid_size = fixed_grid_size or inferred_grid_size
            surrogate_info = validate_nsdf_surrogate_doc(
                bundle.surrogate,
                measurement.observed_values.shape[0],
            )
        except Exception as e:
            loaded_bundle = None
            set_status(f"NSDF load failed: {e}", ok=False)
            traceback.print_exc()
            figures_column.children = [Div(text="<i>No NSDF data loaded.</i>")]
            return

        loaded_bundle = bundle
        if bundle.paths.local_json_path:
            data_path_input.value = bundle.paths.local_json_path
        if bundle.paths.json_url:
            data_url_input.value = bundle.paths.json_url
        if bundle.paths.surrogate_json_path:
            surrogate_path_input.value = bundle.paths.surrogate_json_path
        if bundle.paths.surrogate_json_url:
            surrogate_url_input.value = bundle.paths.surrogate_json_url
        plot_cfg.grid_size = active_grid_size
        grid_w.value = active_grid_size[0]
        grid_h.value = active_grid_size[1]

        msg_parts = [
            f"Loaded NSDF measurement data: {measurement.observed_values.shape[0]} points.",
            f"Inferred grid size: {inferred_grid_size[0]} x {inferred_grid_size[1]}.",
            f"Active grid size: {active_grid_size[0]} x {active_grid_size[1]}.",
            f"Coordinate normalization: {measurement.bounds_source}.",
            "Compatible fields: " + ", ".join(fields) + ".",
        ]
        if bundle.paths.has_s3_source():
            msg_parts.insert(
                0,
                (
                    f"S3 source: s3://{bundle.paths.s3_bucket}/{bundle.paths.s3_data_key}; "
                    f"refresh every {s3_refresh_seconds:g}s."
                ),
            )
        msg_parts.extend(bundle.messages)
        msg_parts.extend(surrogate_info.warnings)
        set_status("\n".join(msg_parts), ok=True)

    def apply_grid_size() -> None:
        try:
            w = int(grid_w.value)
            h = int(grid_h.value)
        except Exception:
            return
        plot_cfg.grid_size = (max(1, min(512, w)), max(1, min(512, h)))

    def rebuild_figures() -> None:
        apply_grid_size()
        figures_column.children = []
        if loaded_bundle is None:
            figures_column.children = [Div(text="<i>No NSDF data loaded.</i>")]
            return
        try:
            grids = build_strain_field_grids(loaded_bundle.data, plot_cfg, loaded_bundle.surrogate)
            p0, p1, p2 = make_strain_triplet_figures(grids, plot_cfg, row_subtitle="dataset_y")
            figures_column.children = [row(p0, p1, p2, sizing_mode="scale_width")]
        except Exception as e:
            figures_column.children = [Div(text=f"<pre>NSDF grid build failed: {e}</pre>")]
            traceback.print_exc()

    def on_reload() -> None:
        load_payload()
        rebuild_figures()

    btn_reload = Button(label="Load / reload JSON", button_type="primary", width=180)
    btn_reload.on_click(on_reload)

    controls = column(
        row(data_path_input, sizing_mode="scale_width"),
        row(data_url_input, sizing_mode="scale_width"),
        row(surrogate_path_input, sizing_mode="scale_width"),
        row(surrogate_url_input, sizing_mode="scale_width"),
        row(grid_w, grid_h, btn_reload, sizing_mode="scale_width"),
        status_div,
        sizing_mode="stretch_width",
    )

    header = create_header_banner("ORNL CHESS NSDF measurements", "ORNL CHESS NSDF")
    root = column(header, controls, figures_column, sizing_mode="stretch_width")
    doc.add_root(root)

    if paths.has_s3_source() or paths.local_json_path or paths.json_url:
        on_reload()
    else:
        figures_column.children = [
            Div(
                text=(
                    "<p>No NSDF data.json resolved yet. Set <code>ORNL_NSDF_DATA_JSON_PATH</code> / "
                    "<code>ORNL_NSDF_DATA_JSON_URL</code>, use legacy <code>ORNL_STRAIN_JSON_*</code> "
                    "aliases, configure <code>ORNL_NSDF_S3_BUCKET</code> / "
                    "<code>ORNL_NSDF_S3_DATA_KEY</code>, or enter a path or URL above and click "
                    "<b>Load / reload JSON</b>.</p>"
                )
            )
        ]
    if paths.has_s3_source():
        doc.add_periodic_callback(on_reload, int(s3_refresh_seconds * 1000))
