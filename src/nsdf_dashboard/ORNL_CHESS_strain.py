"""
ORNL / CHESS NSDF measurement dashboard.

Loads native NSDF ``data.json``, optional ``surrogate.json``, and optional ``next_x.json``;
renders three heatmaps: measurement locations, estimate, and variance.
Legacy ``ORNL_STRAIN_JSON_*`` environment variables and ``strain_json_*`` query parameters
remain aliases for NSDF ``data.json``.
"""
from __future__ import annotations

import html
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Div, InlineStyleSheet, Select, Spinner, TextInput

# Match Bokeh titled-widget label + gap so plain buttons line up with inputs.
_TITLED_WIDGET_LABEL_HEIGHT = 20
_CONTROL_BTN_HEIGHT = 31
_FIXED_BTN_STYLE = InlineStyleSheet(
    css=f"""
:host {{
  height: {_CONTROL_BTN_HEIGHT}px !important;
  min-height: {_CONTROL_BTN_HEIGHT}px !important;
  max-height: {_CONTROL_BTN_HEIGHT}px !important;
}}
:host button {{
  height: {_CONTROL_BTN_HEIGHT}px !important;
  min-height: {_CONTROL_BTN_HEIGHT}px !important;
  max-height: {_CONTROL_BTN_HEIGHT}px !important;
  white-space: nowrap !important;
  line-height: 1 !important;
  box-sizing: border-box !important;
}}
"""
)


def _toolbar_button(label: str, *, button_type: str, width: int) -> Button:
    btn = Button(label=label, button_type=button_type, width=width)
    btn.stylesheets = [_FIXED_BTN_STYLE]
    return btn


def _bottom_aligned_button(btn: Button) -> column:
    return column(
        Div(
            text="&nbsp;",
            height=_TITLED_WIDGET_LABEL_HEIGHT,
            width=1,
            styles={
                "padding": "0",
                "margin": "0",
                "border": "0",
                "font-size": "13px",
                "line-height": "1.35",
                "overflow": "hidden",
                "box-sizing": "border-box",
            },
        ),
        btn,
        sizing_mode="fixed",
    )


def _control_row(*widgets: Any) -> row:
    return row(
        *(_bottom_aligned_button(w) if isinstance(w, Button) else w for w in widgets),
        sizing_mode="scale_width",
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _shared_candidate in (
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "scientistCloudLib", "SCLib_Dashboards")),
):
    if os.path.isdir(_shared_candidate) and _shared_candidate not in sys.path:
        sys.path.insert(0, _shared_candidate)
        break


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
            f"{dashboard_type}: {dataset_name}"
            if dataset_name
            else f"{dashboard_type}"
        )
        return Div(
            text=(
                f'<div style="background-color:{sc_blue};padding:10px 20px;color:white;'
                f'font-family:sans-serif;font-size:1.4em;font-weight:bold;">{title_text}</div>'
            ),
            sizing_mode="stretch_width",
        )


from nsdf_dashboard.ornl_chess_strain_lib import (  # noqa: E402
    NSDFLoadedBundle,
    StrainDashboardPaths,
    StrainFieldPlotConfig,
    apply_nsdf_version_suffix,
    build_strain_field_grids,
    collect_nsdf_triplet_load_issues,
    discover_nsdf_version_options,
    enrich_strain_paths_from_dataset_doc,
    find_strain_json_under_dataset_dir,
    infer_nsdf_grid_size,
    is_scientistcloud_portal_data_mount_context,
    list_nsdf_field_headers,
    load_simple_env_file,
    load_nsdf_json_bundle,
    promote_gateway_json_url_to_s3_paths,
    make_strain_triplet_figures,
    infer_nsdf_bounds_grid_size,
    format_nsdf_workflow_display,
    resolve_nsdf_workflow_id,
    resolve_strain_paths_for_session,
    resolve_nsdf_grid_size,
    surrogate_doc_defines_grid_size,
    validate_nsdf_measurement_doc,
    validate_nsdf_next_x_doc,
    validate_nsdf_surrogate_doc,
)
from nsdf_dashboard.refresh_bus import register_refresh_callback, unregister_refresh_callback  # noqa: E402


def _dashboard_env_value(*names: str) -> str:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    for name in names:
        value = (os.environ.get(name) or env_file_values.get(name) or "").strip()
        if value:
            return value
    return ""


def _int_value(raw: str) -> Optional[int]:
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _fixed_grid_size_from_env() -> Optional[tuple[int, int]]:
    compact = _dashboard_env_value("GRID_SIZE").lower().replace(" ", "")
    if compact:
        for sep in ("x", ",", ":"):
            if sep not in compact:
                continue
            left, right = compact.split(sep, 1)
            width = _int_value(left)
            height = _int_value(right)
            if width and height:
                return width, height
    width = _int_value(_dashboard_env_value("GRID_WIDTH"))
    height = _int_value(_dashboard_env_value("GRID_HEIGHT"))
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
    _query_next_x_path0 = _first_arg("next_x_json_path")
    _query_next_x_url0 = _first_arg("next_x_json_url")

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
        if promote_gateway_json_url_to_s3_paths(StrainDashboardPaths(json_url=jurl)).has_s3_source():
            return p
        mirror = find_strain_json_under_dataset_dir(_bd) or find_strain_json_under_dataset_dir(_sd)
        if mirror:
            return StrainDashboardPaths(
                local_json_path=mirror,
                json_url=jurl,
                surrogate_json_path=p.surrogate_json_path,
                surrogate_json_url=p.surrogate_json_url,
                next_x_json_path=p.next_x_json_path,
                next_x_json_url=p.next_x_json_url,
                local_data_dir=p.local_data_dir,
                s3_env_file=p.s3_env_file,
                s3_bucket=p.s3_bucket,
                s3_data_key=p.s3_data_key,
                s3_surrogate_key=p.s3_surrogate_key,
                s3_next_x_key=p.s3_next_x_key,
                s3_endpoint_url=p.s3_endpoint_url,
                s3_region=p.s3_region,
            )
        return p

    grid_state: Dict[str, Any] = {
        "manual_grid_size": None,
        "active_source": "environment",
        "updating_controls": False,
        "last_status": {},
        "version_suffix": "latest",
        "playback": {
            "active": False,
            "order": [],
            "index": 0,
            "callback_id": None,
        },
    }

    def _resolve_base_paths() -> StrainDashboardPaths:
        return enrich_strain_paths_from_dataset_doc(
            _prefer_upload_mirror_when_url_only(
                resolve_strain_paths_for_session(
                    base_dir=_bd,
                    save_dir=_sd,
                    query_strain_json_path=_query_data_path0,
                    query_strain_json_url=_query_data_url0,
                    query_surrogate_json_path=_query_surrogate_path0,
                    query_surrogate_json_url=_query_surrogate_url0,
                    query_next_x_json_path=_query_next_x_path0,
                    query_next_x_json_url=_query_next_x_url0,
                    env=StrainDashboardPaths.from_environ(),
                )
            ),
            _dataset_doc,
            base_dir=_bd,
            save_dir=_sd,
        )

    def _resolve_paths() -> StrainDashboardPaths:
        base = _resolve_base_paths()
        suffix = str(grid_state.get("version_suffix") or "").strip()
        if suffix in ("", "latest"):
            return apply_nsdf_version_suffix(base, "")
        return apply_nsdf_version_suffix(base, suffix)

    paths = _resolve_paths()
    plot_cfg = StrainFieldPlotConfig()
    env_grid_size = _fixed_grid_size_from_env()
    if env_grid_size:
        plot_cfg.grid_size = env_grid_size
    grid_state["active_source"] = "environment" if env_grid_size else "dataset_x"

    status_div = Div(text="", sizing_mode="stretch_width", visible=False)
    triplet_alert_div = Div(text="", sizing_mode="stretch_width", visible=False)
    _workflow_placeholder = (
        '<div style="font-family:monospace;font-size:14px;padding:6px 0;'
        'min-height:1.35em;line-height:1.35em;color:#888;">Workflow ID: —</div>'
    )
    workflow_div = Div(text=_workflow_placeholder, sizing_mode="stretch_width")
    version_select = Select(
        title="Snapshot",
        value="latest",
        options=[("latest", "Latest (data.json)")],
        width=240,
    )
    _play_btn_width = 90
    play_interval_input = TextInput(title="Play interval (s)", value="2.0", width=110)
    btn_play_backward = _toolbar_button("< Play", button_type="success", width=_play_btn_width)
    btn_play_forward = _toolbar_button("Play >", button_type="success", width=_play_btn_width)
    btn_play_stop = _toolbar_button("Stop", button_type="warning", width=_play_btn_width)
    grid_w = Spinner(title="Grid width", low=1, high=512, step=1, value=plot_cfg.grid_size[0], width=100)
    grid_h = Spinner(title="Grid height", low=1, high=512, step=1, value=plot_cfg.grid_size[1], width=100)
    btn_reset_grid = _toolbar_button("Reset", button_type="default", width=80)
    btn_reload = _toolbar_button("Reload", button_type="primary", width=90)
    btn_toggle_status = _toolbar_button("Show status", button_type="default", width=110)

    loaded_bundle: Optional[NSDFLoadedBundle] = None
    figures_column = column(sizing_mode="stretch_width")

    def set_status(msg: str, ok: bool = True) -> None:
        color = "#0a0" if ok else "#a00"
        status_div.text = (
            f'<div style="color:{color};font-family:monospace;line-height:1.35;'
            f'white-space:pre-wrap;">{msg}</div>'
        )

    def _grid_size_from_controls() -> tuple[int, int]:
        try:
            width = int(grid_w.value)
            height = int(grid_h.value)
        except Exception:
            return plot_cfg.grid_size
        return max(1, min(512, width)), max(1, min(512, height))

    def _set_grid_controls(size: tuple[int, int]) -> None:
        grid_state["updating_controls"] = True
        try:
            grid_w.value = size[0]
            grid_h.value = size[1]
        finally:
            grid_state["updating_controls"] = False

    def _set_triplet_alert(errors: List[str], warnings: List[str]) -> None:
        if not errors and not warnings:
            triplet_alert_div.visible = False
            triplet_alert_div.text = ""
            return
        parts: List[str] = []
        if errors:
            parts.append(
                '<div style="color:#a00;font-family:monospace;font-weight:bold;'
                'margin:6px 0 4px 0;">NSDF triplet errors</div>'
            )
            parts.extend(
                f'<div style="color:#a00;font-family:monospace;margin-left:8px;">'
                f"• {html.escape(msg)}</div>"
                for msg in errors
            )
        if warnings:
            parts.append(
                '<div style="color:#b8860b;font-family:monospace;font-weight:bold;'
                'margin:8px 0 4px 0;">NSDF triplet warnings</div>'
            )
            parts.extend(
                f'<div style="color:#b8860b;font-family:monospace;margin-left:8px;">'
                f"• {html.escape(msg)}</div>"
                for msg in warnings
            )
        triplet_alert_div.text = "".join(parts)
        triplet_alert_div.visible = True

    def _set_loaded_status() -> None:
        last_status = grid_state["last_status"]
        if not last_status:
            return
        active_grid_size = plot_cfg.grid_size
        triplet_errors = list(last_status.get("triplet_errors") or [])
        triplet_warnings = list(last_status.get("triplet_warnings") or [])
        _set_triplet_alert(triplet_errors, triplet_warnings)
        msg_parts = [
            f"Loaded NSDF measurement data: {last_status['measurement_count']} points.",
            f"Inferred grid size: {last_status['inferred_grid_size'][0]} x {last_status['inferred_grid_size'][1]}.",
            f"Active grid size: {active_grid_size[0]} x {active_grid_size[1]}.",
            f"Grid source: {grid_state['active_source']}.",
            f"Coordinate normalization: {last_status['bounds_source']}.",
            "Compatible fields: " + ", ".join(last_status["fields"]) + ".",
        ]
        if last_status.get("surrogate_grid_line"):
            msg_parts.append(last_status["surrogate_grid_line"])
        if last_status.get("next_x_summary"):
            msg_parts.append(last_status["next_x_summary"])
        if last_status.get("next_x_warnings"):
            msg_parts.extend(last_status["next_x_warnings"])
        if last_status.get("version_label"):
            msg_parts.insert(0, f"Snapshot: {last_status['version_label']}.")
        if last_status.get("workflow_line"):
            msg_parts.insert(0, last_status["workflow_line"])
        if last_status["source_line"]:
            msg_parts.insert(0, last_status["source_line"])
        msg_parts.extend(last_status["messages"])
        msg_parts.extend(last_status["warnings"])
        if triplet_errors:
            msg_parts.insert(0, "See NSDF triplet errors above.")
        set_status("\n".join(msg_parts), ok=not triplet_errors)
        if triplet_errors or triplet_warnings:
            status_div.visible = True
            btn_toggle_status.label = "Hide status"
        workflow_line = last_status.get("workflow_line") or ""
        if workflow_line:
            workflow_div.text = (
                f'<div style="font-family:monospace;font-size:14px;padding:6px 0;'
                f'min-height:1.35em;line-height:1.35em;"><b>{workflow_line}</b></div>'
            )
        else:
            workflow_div.text = _workflow_placeholder

    def _refresh_version_options() -> None:
        options = discover_nsdf_version_options(
            _resolve_base_paths(),
            base_dir=_bd,
            save_dir=_sd,
            mongo_s3_auth=_dataset_s3_auth_override(),
        )
        current = str(grid_state.get("version_suffix") or "latest")
        valid_values = {value for value, _label in options}
        if current not in valid_values:
            current = "latest"
            grid_state["version_suffix"] = current
        grid_state["updating_controls"] = True
        try:
            version_select.options = options
            version_select.value = current
        finally:
            grid_state["updating_controls"] = False

    def _snapshot_values_chronological() -> list[str]:
        """Oldest timestamped snapshot first, ``latest`` last."""
        values = [str(value) for value, _label in version_select.options]
        timestamps = sorted(v for v in values if v != "latest")
        if "latest" in values:
            timestamps.append("latest")
        return timestamps

    def _playback_interval_ms() -> int:
        try:
            seconds = float(str(play_interval_input.value or "").strip())
        except ValueError:
            seconds = 2.0
        seconds = max(0.5, min(120.0, seconds))
        return int(seconds * 1000)

    def _stop_playback() -> None:
        playback = grid_state.get("playback") or {}
        callback_id = playback.get("callback_id")
        if callback_id is not None:
            doc.remove_periodic_callback(callback_id)
        grid_state["playback"] = {
            "active": False,
            "order": [],
            "index": 0,
            "callback_id": None,
        }

    def _apply_snapshot_value(value: str) -> None:
        grid_state["version_suffix"] = str(value or "latest")
        grid_state["updating_controls"] = True
        try:
            version_select.value = grid_state["version_suffix"]
        finally:
            grid_state["updating_controls"] = False
        on_reload()

    def _advance_playback() -> None:
        playback = grid_state.get("playback") or {}
        if not playback.get("active"):
            return
        order = playback.get("order") or []
        if len(order) <= 1:
            _stop_playback()
            return
        next_index = int(playback.get("index", 0)) + 1
        if next_index >= len(order):
            _stop_playback()
            return
        playback["index"] = next_index
        grid_state["playback"] = playback
        _apply_snapshot_value(str(order[next_index]))

    def _start_playback(*, forward: bool) -> None:
        _stop_playback()
        chronological = _snapshot_values_chronological()
        if len(chronological) <= 1:
            set_status("Need at least two snapshots to play acquisition.", ok=False)
            status_div.visible = True
            return
        order = chronological if forward else list(reversed(chronological))
        current = str(version_select.value or "latest")
        start_index = order.index(current) if current in order else 0
        callback_id = doc.add_periodic_callback(_advance_playback, _playback_interval_ms())
        grid_state["playback"] = {
            "active": True,
            "order": order,
            "index": start_index,
            "callback_id": callback_id,
        }

    def on_play_forward() -> None:
        _start_playback(forward=True)

    def on_play_backward() -> None:
        _start_playback(forward=False)

    def on_play_stop() -> None:
        _stop_playback()

    def load_payload() -> None:
        global loaded_bundle  # noqa: PLW0603

        _refresh_version_options()
        p = _resolve_paths()
        try:
            bundle = load_nsdf_json_bundle(p, mongo_s3_auth=_dataset_s3_auth_override())
            measurement = validate_nsdf_measurement_doc(bundle.data)
            fields = list_nsdf_field_headers(bundle.data, bundle.surrogate)
            inferred_grid_size = infer_nsdf_grid_size(bundle.data)
            if infer_nsdf_bounds_grid_size(bundle.data) or surrogate_doc_defines_grid_size(bundle.surrogate):
                grid_state["manual_grid_size"] = None
            active_grid_size, active_grid_source = resolve_nsdf_grid_size(
                bundle.data,
                surrogate_doc=bundle.surrogate,
                env_grid_size=env_grid_size,
                manual_grid_size=grid_state["manual_grid_size"],
            )
            surrogate_info = validate_nsdf_surrogate_doc(bundle.surrogate)
            next_x_info = validate_nsdf_next_x_doc(bundle.next_x)
        except Exception as e:
            loaded_bundle = None
            set_status(f"NSDF load failed: {e}", ok=False)
            figures_column.children = [Div(text="<i>No NSDF data loaded.</i>")]
            return

        loaded_bundle = bundle
        plot_cfg.grid_size = active_grid_size
        grid_state["active_source"] = active_grid_source
        _set_grid_controls(active_grid_size)
        source_line = ""
        if bundle.paths.local_data_dir and bundle.paths.local_json_path:
            source_line = f"Local source: {bundle.paths.local_json_path}"
        elif bundle.paths.has_s3_source():
            source_line = (
                f"S3 source: s3://{bundle.paths.s3_bucket}/{bundle.paths.s3_data_key}; "
                "event-triggered refresh."
            )
        elif (bundle.paths.local_json_path or "").strip() or (bundle.paths.json_url or "").strip():
            loc = (bundle.paths.local_json_path or "").strip()
            source_line = f"Source: {loc or bundle.paths.json_url}"
        next_x_summary = ""
        active_workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
        if next_x_info.entries and active_workflow_id:
            active_points = sum(
                int(entry.coordinates.shape[0])
                for entry in next_x_info.entries
                if entry.workflow_id == active_workflow_id
            )
            if active_points:
                next_x_summary = (
                    f"next_x: {active_points} proposed point(s) for workflow {active_workflow_id}."
                )
        elif next_x_info.entries:
            next_x_summary = "next_x: loaded but no active non-demo workflow entry."
        elif bundle.next_x is not None:
            next_x_summary = "next_x: loaded but no valid workflow entries."
        workflow_line = format_nsdf_workflow_display(surrogate_info, next_x_info)
        surrogate_grid_line = ""
        if surrogate_info.plot_dim:
            surrogate_grid_line = f"Surrogate plot dim: {surrogate_info.plot_dim}."
        bounds_size = infer_nsdf_bounds_grid_size(bundle.surrogate or {})
        if bounds_size:
            bounds_note = f"Grid size from bounds: {bounds_size[0]} x {bounds_size[1]}."
            surrogate_grid_line = (
                f"{surrogate_grid_line} {bounds_note}".strip()
                if surrogate_grid_line
                else bounds_note
            )
        if surrogate_info.points:
            points_note = f"Expected points (surrogate): {surrogate_info.points}."
            surrogate_grid_line = (
                f"{surrogate_grid_line} {points_note}".strip()
                if surrogate_grid_line
                else points_note
            )
        version_label = "Latest (data.json)"
        if (p.version_suffix or "").strip():
            version_label = p.version_suffix
        triplet_errors, triplet_warnings = collect_nsdf_triplet_load_issues(p, bundle)
        grid_state["last_status"] = {
            "measurement_count": measurement.observed_values.shape[0],
            "inferred_grid_size": inferred_grid_size,
            "bounds_source": measurement.bounds_source,
            "fields": fields,
            "source_line": source_line,
            "version_label": version_label,
            "workflow_line": workflow_line,
            "surrogate_grid_line": surrogate_grid_line,
            "messages": list(bundle.messages),
            "warnings": list(surrogate_info.warnings),
            "next_x_summary": next_x_summary,
            "next_x_warnings": list(next_x_info.warnings),
            "triplet_errors": triplet_errors,
            "triplet_warnings": triplet_warnings,
        }

    def apply_grid_size() -> None:
        plot_cfg.grid_size = _grid_size_from_controls()

    def rebuild_figures() -> None:
        apply_grid_size()
        figures_column.children = []
        if loaded_bundle is None:
            figures_column.children = [Div(text="<i>No NSDF data loaded.</i>")]
            return
        try:
            grids = build_strain_field_grids(loaded_bundle.data, plot_cfg, loaded_bundle.surrogate)
            surrogate_info = validate_nsdf_surrogate_doc(loaded_bundle.surrogate)
            next_x_info = validate_nsdf_next_x_doc(loaded_bundle.next_x)
            active_workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
            p0, p1, p2 = make_strain_triplet_figures(
                grids,
                plot_cfg,
                row_subtitle="dataset_y",
                next_x_info=next_x_info,
                active_workflow_id=active_workflow_id,
            )
            figures_column.children = [row(p0, p1, p2, sizing_mode="scale_width")]
            if grid_state.get("last_status") is not None:
                resolved = _resolve_paths()
                triplet_errors, triplet_warnings = collect_nsdf_triplet_load_issues(
                    resolved,
                    loaded_bundle,
                    grid_meta=grids.meta,
                )
                grid_state["last_status"]["triplet_errors"] = triplet_errors
                grid_state["last_status"]["triplet_warnings"] = triplet_warnings
            _set_loaded_status()
        except Exception as e:
            figures_column.children = [Div(text=f"<pre>NSDF grid build failed: {e}</pre>")]
            traceback.print_exc()
            if grid_state.get("last_status") is not None:
                grid_state["last_status"]["triplet_errors"] = [
                    f"Grid build failed: {e}",
                ]
                _set_loaded_status()

    def on_version_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        if (grid_state.get("playback") or {}).get("active"):
            _stop_playback()
        grid_state["version_suffix"] = str(new or "latest")
        on_reload()

    def on_reload() -> None:
        load_payload()
        rebuild_figures()

    def on_external_refresh(_doc: Any = doc, _on_reload: Callable[[], None] = on_reload) -> None:
        _doc.add_next_tick_callback(_on_reload)

    def on_grid_control_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        manual_grid_size = _grid_size_from_controls()
        grid_state["manual_grid_size"] = manual_grid_size
        grid_state["active_source"] = "manual controls"
        plot_cfg.grid_size = manual_grid_size
        rebuild_figures()

    def on_reset_grid() -> None:
        grid_state["manual_grid_size"] = None
        if loaded_bundle is None:
            active_grid_size = env_grid_size or plot_cfg.grid_size
            active_grid_source = "environment" if env_grid_size else "dataset_x"
        else:
            active_grid_size, active_grid_source = resolve_nsdf_grid_size(
                loaded_bundle.data,
                surrogate_doc=loaded_bundle.surrogate,
                env_grid_size=env_grid_size,
                manual_grid_size=None,
            )
        plot_cfg.grid_size = active_grid_size
        grid_state["active_source"] = active_grid_source
        _set_grid_controls(active_grid_size)
        rebuild_figures()

    def on_toggle_status() -> None:
        status_div.visible = not status_div.visible
        btn_toggle_status.label = "Hide status" if status_div.visible else "Show status"

    grid_w.on_change("value", on_grid_control_change)
    grid_h.on_change("value", on_grid_control_change)
    version_select.on_change("value", on_version_change)
    btn_play_forward.on_click(on_play_forward)
    btn_play_backward.on_click(on_play_backward)
    btn_play_stop.on_click(on_play_stop)
    btn_reset_grid.on_click(on_reset_grid)
    btn_reload.on_click(on_reload)
    btn_toggle_status.on_click(on_toggle_status)

    controls = column(
        row(version_select, sizing_mode="scale_width"),
        _control_row(grid_w, grid_h, btn_reload, btn_reset_grid, btn_toggle_status),
        _control_row(
            play_interval_input,
            btn_play_backward,
            btn_play_forward,
            btn_play_stop,
        ),
        status_div,
        sizing_mode="stretch_width",
    )

    if is_scientistcloud_portal_data_mount_context(_bd, _sd):
        header = create_header_banner("ORNL CHESS Strain", "ScientistCloud")
    else:
        header = create_header_banner("ORNL CHESS NSDF measurements", "DIAL Dashboard")
    root = column(header, workflow_div, triplet_alert_div, controls, figures_column, sizing_mode="stretch_width")
    doc.add_root(root)

    if paths.local_data_dir or paths.has_s3_source() or paths.local_json_path or paths.json_url:
        on_reload()
    else:
        figures_column.children = [
            Div(
                text=(
                    "<p>No NSDF data.json resolved yet. Configure <code>LOCAL_DATA_DIR</code> "
                    "with local <code>data.json</code>, <code>surrogate.json</code>, and "
                    "<code>next_x.json</code>, or set portal/S3/env paths.</p>"
                )
            )
        ]
    if paths.local_data_dir or paths.has_s3_source() or paths.local_json_path or paths.json_url:
        _refresh_token = register_refresh_callback(on_external_refresh)

        def _cleanup_refresh_callback(
            session_context: Any,
            token: int = _refresh_token,
            unregister: Callable[[int], None] = unregister_refresh_callback,
            stop_playback: Callable[[], None] = _stop_playback,
        ) -> None:
            stop_playback()
            unregister(token)

        doc.on_session_destroyed(_cleanup_refresh_callback)
