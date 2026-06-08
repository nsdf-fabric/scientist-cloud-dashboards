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

import numpy as np
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Checkbox, Div, InlineStyleSheet, Select, Spinner, TextInput

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
    btn_width = int(btn.width or 0) or None
    return column(
        Div(
            text="&nbsp;",
            height=_TITLED_WIDGET_LABEL_HEIGHT,
            width=btn_width or 1,
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
        width=btn_width,
        height=_TITLED_WIDGET_LABEL_HEIGHT + _CONTROL_BTN_HEIGHT,
    )


def _control_row(*widgets: Any) -> row:
    return row(
        *(_bottom_aligned_button(w) if isinstance(w, Button) else w for w in widgets),
        sizing_mode="scale_width",
    )


def _loading_placeholder_div(message: str) -> Div:
    """Centered spinner shown while S3/catalog/figure work is in flight."""
    safe_message = html.escape(message or "Loading...")
    return Div(
        text=(
            "<style>"
            "@keyframes nsdf-dashboard-spin { to { transform: rotate(360deg); } }"
            ".nsdf-dashboard-loading {"
            "min-height:360px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:14px;color:#444;background:#fafafa;"
            "border:1px solid #e6e6e6;border-radius:6px;"
            "}"
            ".nsdf-dashboard-spinner {"
            "width:40px;height:40px;border:4px solid #ddd;border-top-color:#4E477F;"
            "border-radius:50%;animation:nsdf-dashboard-spin 0.85s linear infinite;"
            "}"
            "</style>"
            f'<div class="nsdf-dashboard-loading">'
            f'<div class="nsdf-dashboard-spinner" aria-hidden="true"></div>'
            f"<div>{safe_message}</div>"
            "</div>"
        ),
        sizing_mode="stretch_width",
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
    NSDFTripletIndex,
    StrainDashboardPaths,
    StrainFieldPlotConfig,
    UncertaintyTrendSeries,
    _snapshot_index_in_series,
    _snapshots_chronological,
    _trend_current_index,
    build_uncertainty_trend_from_surrogate_paths,
    build_uncertainty_trend_series,
    uncertainty_trend_from_surrogate_doc,
    apply_nsdf_file_suffixes,
    discover_nsdf_next_x_version_options,
    discover_nsdf_surrogate_version_options,
    resolve_auxiliary_suffix_for_data_snapshot,
    apply_scientistcloud_storage_policy,
    active_workflow_id_from_grid_state,
    build_strain_field_grids,
    collect_nsdf_triplet_load_issues,
    discover_nsdf_triplet_index,
    enrich_strain_paths_from_dataset_doc,
    format_nsdf_workflow_select_label,
    find_strain_json_under_dataset_dir,
    infer_nsdf_grid_size,
    is_scientistcloud_portal_data_mount_context,
    list_nsdf_field_headers,
    load_simple_env_file,
    load_nsdf_json_bundle,
    NSDF_UNKNOWN_WORKFLOW_ID,
    promote_gateway_json_url_to_s3_paths,
    make_strain_triplet_row,
    infer_nsdf_bounds_grid_size,
    resolve_default_workflow_selection,
    resolve_nsdf_workflow_id,
    resolve_strain_paths_for_session,
    resolve_nsdf_grid_size,
    resolve_estimate_color_limits,
    scientistcloud_dataset_is_remote_linked,
    surrogate_doc_defines_grid_size,
    validate_nsdf_measurement_doc,
    _select_next_x_entry,
    validate_nsdf_next_x_doc,
    validate_nsdf_surrogate_doc,
    workflow_id_from_dataset_doc,
)
from nsdf_dashboard.refresh_bus import register_refresh_callback, unregister_refresh_callback  # noqa: E402


def _dashboard_env_value(*names: str) -> str:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    for name in names:
        value = (os.environ.get(name) or env_file_values.get(name) or "").strip()
        if value:
            return value
    return ""


def _float_value(raw: str) -> Optional[float]:
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def _format_color_limit(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{float(value):.4e}"
    return f"{float(value):.6g}"


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
    _query_workflow_id0 = _first_arg("workflow_id")

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
    _remote_linked = scientistcloud_dataset_is_remote_linked(
        _dataset_doc,
        server_param=str(_params.get("server") or ""),
    )

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
        if _remote_linked:
            return p
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
        "surrogate_version_suffix": "latest",
        "next_x_version_suffix": "latest",
        "surrogate_suffix_options": [("latest", "Latest (surrogate.json)")],
        "next_x_suffix_options": [("latest", "Latest (next_x.json)")],
        "surrogate_manual": False,
        "next_x_manual": False,
        "workflow_id": "",
        "triplet_index": None,
        "uncertainty_trend_points_key": "",
        "uncertainty_trend_points": None,
        "playback": {
            "active": False,
            "order": [],
            "index": 0,
            "callback_id": None,
        },
        "loading_generation": 0,
        "figures_work_tick_scheduled": False,
        "pending_figures_work": None,
        "suppress_selector_reload": False,
        "catalog_ready": False,
        "catalog_loading": False,
        "catalog_callback_id": None,
        "color_range_mode": "dynamic",
        "color_range_lo": "",
        "color_range_hi": "",
        "compute_plot4_trend": False,
    }

    def _resolve_base_paths() -> StrainDashboardPaths:
        return apply_scientistcloud_storage_policy(
            enrich_strain_paths_from_dataset_doc(
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
                        remote_linked=_remote_linked,
                    )
                ),
                _dataset_doc,
                base_dir=_bd,
                save_dir=_sd,
            ),
            _dataset_doc,
            server_param=str(_params.get("server") or ""),
            base_dir=_bd,
            save_dir=_sd,
        )

    def _resolve_paths() -> StrainDashboardPaths:
        base = _resolve_base_paths()
        data_suffix = str(grid_state.get("version_suffix") or "latest").strip()
        surrogate_suffix = str(grid_state.get("surrogate_version_suffix") or "latest").strip()
        next_x_suffix = str(grid_state.get("next_x_version_suffix") or "latest").strip()
        return apply_nsdf_file_suffixes(
            base,
            data_suffix=data_suffix,
            surrogate_suffix=surrogate_suffix,
            next_x_suffix=next_x_suffix,
            strict=True,
        )

    paths = _resolve_paths()
    plot_cfg = StrainFieldPlotConfig()
    env_grid_size = _fixed_grid_size_from_env()
    if env_grid_size:
        plot_cfg.grid_size = env_grid_size
    grid_state["active_source"] = "environment" if env_grid_size else "dataset_x"

    log_panel_div = Div(text="", sizing_mode="stretch_width", visible=False)
    _log_scroll_style = (
        "max-height:220px;overflow-y:auto;overflow-x:hidden;"
        "border:1px solid #ccc;border-radius:4px;padding:8px 10px;"
        "background:#fafafa;"
    )
    _workflow_placeholder = (
        '<div style="font-family:monospace;font-size:14px;padding:6px 0;'
        'min-height:1.35em;line-height:1.35em;color:#888;">Workflow ID: —</div>'
    )
    workflow_div = Div(text=_workflow_placeholder, sizing_mode="stretch_width")
    workflow_select = Select(
        title="Workflow",
        value="",
        options=[],
        width=280,
    )
    version_select = Select(
        title="data.json",
        value="latest",
        options=[("latest", "Latest (data.json)")],
        width=220,
    )
    surrogate_select = Select(
        title="surrogate.json",
        value="latest",
        options=[("latest", "Latest (surrogate.json)")],
        width=220,
    )
    next_x_select = Select(
        title="next_x.json",
        value="latest",
        options=[("latest", "Latest (next_x.json)")],
        width=220,
    )
    _play_btn_width = 90
    play_interval_input = TextInput(title="Play interval (s)", value="2.0", width=110)
    btn_play_backward = _toolbar_button("< Play", button_type="success", width=_play_btn_width)
    btn_play_forward = _toolbar_button("Play >", button_type="success", width=_play_btn_width)
    btn_play_stop = _toolbar_button("Stop", button_type="warning", width=_play_btn_width)
    grid_w = Spinner(title="Grid width", low=1, high=512, step=1, value=plot_cfg.grid_size[0], width=100)
    grid_h = Spinner(title="Grid height", low=1, high=512, step=1, value=plot_cfg.grid_size[1], width=100)
    color_range_select = Select(
        title="Plot 1–2 color range",
        value="dynamic",
        options=[
            ("dynamic", "Auto (data min/max)"),
            ("manual", "Fixed min/max"),
        ],
        width=170,
    )
    color_lo_input = TextInput(title="Color min", value="", width=110, disabled=True)
    color_hi_input = TextInput(title="Color max", value="", width=110, disabled=True)
    btn_reset_color_range = _toolbar_button("Reset range", button_type="default", width=100)
    plot4_trend_checkbox = Checkbox(
        label="Compute Plot 4 trend",
        active=False,
        width=180,
    )
    btn_reset_grid = _toolbar_button("Reset", button_type="default", width=80)
    btn_reload = _toolbar_button("Reload", button_type="primary", width=90)
    btn_index_workflows = _toolbar_button("Index workflows", button_type="default", width=140)
    btn_toggle_status = _toolbar_button("Show status", button_type="default", width=110)

    loaded_bundle: Optional[NSDFLoadedBundle] = None
    figures_column = column(sizing_mode="stretch_width")
    _busy_controls: List[Any] = []

    def _set_controls_busy(busy: bool) -> None:
        for widget in _busy_controls:
            try:
                widget.disabled = busy
            except Exception:
                pass

    def _begin_figures_loading(message: str) -> int:
        grid_state["loading_generation"] = int(grid_state.get("loading_generation") or 0) + 1
        token = int(grid_state["loading_generation"])
        figures_column.children = [_loading_placeholder_div(message)]
        _set_controls_busy(True)
        return token

    def _defer_figures_work(work_fn: Callable[[], None], *, message: str) -> None:
        grid_state["pending_figures_work"] = work_fn
        if grid_state.get("figures_work_tick_scheduled"):
            if figures_column.children:
                figures_column.children = [_loading_placeholder_div(message)]
            return

        token = _begin_figures_loading(message)
        grid_state["figures_work_tick_scheduled"] = True

        def _run() -> None:
            grid_state["figures_work_tick_scheduled"] = False
            if grid_state.get("loading_generation") != token:
                return
            try:
                while True:
                    pending = grid_state.get("pending_figures_work")
                    if pending is None:
                        break
                    grid_state["pending_figures_work"] = None
                    pending()
                    if grid_state.get("loading_generation") != token:
                        return
            except Exception as exc:
                traceback.print_exc()
                if grid_state.get("loading_generation") == token:
                    figures_column.children = [
                        Div(
                            text=(
                                "<pre>Dashboard update failed: "
                                f"{html.escape(str(exc))}</pre>"
                            )
                        )
                    ]
            finally:
                if grid_state.get("loading_generation") == token:
                    _set_controls_busy(False)

        doc.add_next_tick_callback(_run)

    def _reload_figures_view() -> None:
        load_payload()
        rebuild_figures()

    def _set_index_workflows_button_state(*, loading: bool = False) -> None:
        if loading:
            btn_index_workflows.label = "Indexing…"
            btn_index_workflows.disabled = True
        elif grid_state.get("catalog_ready"):
            btn_index_workflows.label = "Re-index workflows"
            btn_index_workflows.disabled = False
        else:
            btn_index_workflows.label = "Index workflows"
            btn_index_workflows.disabled = False

    def _bootstrap_minimal_ui_state() -> None:
        """Latest-only selectors until the user indexes workflows/snapshots."""
        grid_state["catalog_ready"] = False
        grid_state["triplet_index"] = None
        grid_state["uncertainty_trend_points_key"] = ""
        grid_state["uncertainty_trend_points"] = None
        grid_state["version_suffix"] = "latest"
        grid_state["surrogate_version_suffix"] = "latest"
        grid_state["next_x_version_suffix"] = "latest"
        grid_state["surrogate_manual"] = False
        grid_state["next_x_manual"] = False
        grid_state["surrogate_suffix_options"] = [("latest", "Latest (surrogate.json)")]
        grid_state["next_x_suffix_options"] = [("latest", "Latest (next_x.json)")]
        grid_state["updating_controls"] = True
        try:
            workflow_select.options = [
                (NSDF_UNKNOWN_WORKFLOW_ID, "(press Index workflows to browse)"),
            ]
            workflow_select.value = NSDF_UNKNOWN_WORKFLOW_ID
            version_select.options = [("latest", "Latest (data.json)")]
            version_select.value = "latest"
            surrogate_select.options = list(grid_state["surrogate_suffix_options"])
            surrogate_select.value = "latest"
            next_x_select.options = list(grid_state["next_x_suffix_options"])
            next_x_select.value = "latest"
        finally:
            grid_state["updating_controls"] = False
        _set_index_workflows_button_state()

    def _update_workflow_hint_from_bundle() -> None:
        if loaded_bundle is None:
            return
        surrogate_info = validate_nsdf_surrogate_doc(loaded_bundle.surrogate)
        next_x_info = validate_nsdf_next_x_doc(loaded_bundle.next_x)
        workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
        if not workflow_id:
            return
        grid_state["workflow_id"] = workflow_id
        grid_state["updating_controls"] = True
        try:
            workflow_select.options = [
                (workflow_id, format_nsdf_workflow_select_label(workflow_id)),
            ]
            workflow_select.value = workflow_id
        finally:
            grid_state["updating_controls"] = False

    def _uncertainty_trend_from_loaded_surrogate(
        current_snap: str,
    ) -> Optional[UncertaintyTrendSeries]:
        if loaded_bundle is None:
            return None
        return uncertainty_trend_from_surrogate_doc(
            loaded_bundle.surrogate,
            current_snapshot=current_snap,
            grid_size=plot_cfg.grid_size,
        )

    def _fast_latest_triplet_load_data() -> None:
        _bootstrap_minimal_ui_state()
        load_payload()
        _update_workflow_hint_from_bundle()

    def _fast_latest_triplet_load_figures() -> None:
        rebuild_figures()

    def _fast_latest_triplet_load() -> None:
        _fast_latest_triplet_load_data()
        _fast_latest_triplet_load_figures()

    def _cancel_background_catalog() -> None:
        callback_id = grid_state.get("catalog_callback_id")
        if callback_id is not None:
            try:
                doc.remove_next_tick_callback(callback_id)
            except Exception:
                pass
        grid_state["catalog_callback_id"] = None
        grid_state["catalog_loading"] = False

    def _defer_catalog_index() -> None:
        """Scan storage for workflow/snapshot dropdowns; only runs when user requests it."""
        if grid_state.get("catalog_loading"):
            return
        _cancel_background_catalog()
        grid_state["catalog_loading"] = True
        _set_index_workflows_button_state(loading=True)
        _set_controls_busy(True)
        grid_state["updating_controls"] = True
        try:
            current_workflow = str(grid_state.get("workflow_id") or "").strip()
            if current_workflow and current_workflow != NSDF_UNKNOWN_WORKFLOW_ID:
                workflow_select.options = [
                    (
                        current_workflow,
                        f"{format_nsdf_workflow_select_label(current_workflow)} (indexing…)",
                    ),
                ]
                workflow_select.value = current_workflow
            else:
                workflow_select.options = [
                    (NSDF_UNKNOWN_WORKFLOW_ID, "(indexing workflows and snapshots…)"),
                ]
                workflow_select.value = NSDF_UNKNOWN_WORKFLOW_ID
        finally:
            grid_state["updating_controls"] = False

        def _run_catalog_index() -> None:
            grid_state["catalog_callback_id"] = None
            try:
                _rebuild_triplet_catalog(preserve_workflow=True, preserve_snapshot=True)
                rebuild_figures(show_loading_gap=False)
                set_status(
                    "Indexed workflows and snapshots. Selectors are ready for browsing.",
                    ok=True,
                )
            except Exception as exc:
                traceback.print_exc()
                set_status(f"Workflow catalog index failed: {exc}", ok=False)
            finally:
                grid_state["catalog_loading"] = False
                _set_index_workflows_button_state()
                _set_controls_busy(False)

        callback_id = doc.add_next_tick_callback(_run_catalog_index)
        grid_state["catalog_callback_id"] = callback_id

    def _initial_dashboard_load() -> None:
        _fast_latest_triplet_load()

    def _invalidate_uncertainty_trend_cache() -> None:
        grid_state["uncertainty_trend_points_key"] = ""
        grid_state["uncertainty_trend_points"] = None

    def _end_control_update_batch() -> None:
        grid_state["updating_controls"] = False

    def _set_select_value_if_needed(select: Select, value: str) -> None:
        target = str(value or "latest")
        if str(select.value or "") != target:
            select.value = target

    def _reset_to_latest_triplet_selectors() -> None:
        """Point all three file selectors at the live latest triplet."""
        grid_state["version_suffix"] = "latest"
        grid_state["surrogate_version_suffix"] = "latest"
        grid_state["next_x_version_suffix"] = "latest"
        grid_state["surrogate_manual"] = False
        grid_state["next_x_manual"] = False
        grid_state["updating_controls"] = True
        try:
            latest_values = {value for value, _label in version_select.options}
            if "latest" in latest_values:
                _set_select_value_if_needed(version_select, "latest")
            latest_values = {value for value, _label in surrogate_select.options}
            if "latest" in latest_values:
                _set_select_value_if_needed(surrogate_select, "latest")
            latest_values = {value for value, _label in next_x_select.options}
            if "latest" in latest_values:
                _set_select_value_if_needed(next_x_select, "latest")
        finally:
            # Bokeh may emit select callbacks after this function returns.
            doc.add_next_tick_callback(_end_control_update_batch)

    def _reload_current_view() -> None:
        grid_state["suppress_selector_reload"] = True
        try:
            _reset_to_latest_triplet_selectors()
            if grid_state.get("catalog_ready"):
                _sync_auxiliary_selectors_from_data(reset_manual=True)
            load_payload()
            _update_workflow_hint_from_bundle()
            rebuild_figures()
        finally:
            def _clear_selector_suppress() -> None:
                grid_state["suppress_selector_reload"] = False

            doc.add_next_tick_callback(_clear_selector_suppress)

    def _wrap_log_html(inner_html: str) -> str:
        return (
            f'<div style="font-family:monospace;font-size:12px;line-height:1.4;">'
            f'<div style="{_log_scroll_style}">{inner_html}</div></div>'
        )

    def _show_log_panel(inner_html: str) -> None:
        log_panel_div.text = _wrap_log_html(inner_html)
        log_panel_div.visible = True
        btn_toggle_status.label = "Hide status"

    def set_status(msg: str, ok: bool = True) -> None:
        color = "#0a0" if ok else "#a00"
        _show_log_panel(
            f'<div style="color:{color};white-space:pre-wrap;">{html.escape(msg)}</div>'
        )

    def _format_triplet_log_html(errors: List[str], warnings: List[str]) -> str:
        parts: List[str] = []
        if errors:
            parts.append(
                '<div style="color:#a00;font-weight:bold;margin:0 0 4px 0;">'
                "NSDF triplet errors</div>"
            )
            parts.extend(
                f'<div style="color:#a00;margin-left:8px;">• {html.escape(msg)}</div>'
                for msg in errors
            )
        if warnings:
            parts.append(
                '<div style="color:#b8860b;font-weight:bold;margin:8px 0 4px 0;">'
                "NSDF triplet warnings</div>"
            )
            parts.extend(
                f'<div style="color:#b8860b;margin-left:8px;">• {html.escape(msg)}</div>'
                for msg in warnings
            )
        return "".join(parts)

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

    def _set_loaded_status() -> None:
        last_status = grid_state["last_status"]
        if not last_status:
            return
        active_grid_size = plot_cfg.grid_size
        triplet_errors = list(last_status.get("triplet_errors") or [])
        triplet_warnings = list(last_status.get("triplet_warnings") or [])
        msg_parts = [
            f"Loaded NSDF measurement data: {last_status['measurement_count']} points.",
            f"Unique coordinates in data (sparse hint): {last_status['inferred_grid_size'][0]} x {last_status['inferred_grid_size'][1]}.",
            f"Plot grid size: {active_grid_size[0]} x {active_grid_size[1]}.",
            f"Grid source: {grid_state['active_source']} (bounds/surrogate define the scan canvas).",
            last_status.get("color_range_line")
            or _color_range_status_line(*grid_state.get("last_color_range", (0.0, 1.0))),
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
            msg_parts.insert(0, f"Files: {last_status['version_label']}.")
        if last_status.get("workflow_line"):
            msg_parts.insert(0, last_status["workflow_line"])
        if last_status["source_line"]:
            msg_parts.insert(0, last_status["source_line"])
        msg_parts.extend(last_status["messages"])
        msg_parts.extend(last_status["warnings"])

        status_color = "#a00" if triplet_errors else "#0a0"
        log_parts: List[str] = []
        triplet_html = _format_triplet_log_html(triplet_errors, triplet_warnings)
        if triplet_html:
            log_parts.append(triplet_html)
            log_parts.append('<div style="margin:8px 0 4px 0;border-top:1px solid #ddd;"></div>')
        log_parts.append(
            f'<div style="color:{status_color};white-space:pre-wrap;">'
            f"{html.escape(chr(10).join(msg_parts))}</div>"
        )
        log_panel_div.text = _wrap_log_html("".join(log_parts))
        if triplet_errors or triplet_warnings:
            log_panel_div.visible = True
            btn_toggle_status.label = "Hide status"
        else:
            log_panel_div.visible = False
            btn_toggle_status.label = "Show status"
        workflow_line = last_status.get("workflow_line") or ""
        if workflow_line:
            workflow_div.text = (
                f'<div style="font-family:monospace;font-size:14px;padding:6px 0;'
                f'min-height:1.35em;line-height:1.35em;"><b>{workflow_line}</b></div>'
            )
        else:
            workflow_div.text = _workflow_placeholder

    def _refresh_auxiliary_suffix_options(workflow_id: Optional[str] = None) -> None:
        base_paths = _resolve_base_paths()
        index = grid_state.get("triplet_index")
        wf = (workflow_id or grid_state.get("workflow_id") or "").strip()
        if grid_state.get("catalog_ready") and index is not None and wf and index.has_workflow(wf):
            grid_state["surrogate_suffix_options"] = index.surrogate_select_options(wf)
            grid_state["next_x_suffix_options"] = index.next_x_select_options(wf)
            return
        grid_state["surrogate_suffix_options"] = discover_nsdf_surrogate_version_options(
            base_paths,
            base_dir=_bd,
            save_dir=_sd,
            mongo_s3_auth=_dataset_s3_auth_override(),
            remote_linked=_remote_linked,
        )
        grid_state["next_x_suffix_options"] = discover_nsdf_next_x_version_options(
            base_paths,
            base_dir=_bd,
            save_dir=_sd,
            mongo_s3_auth=_dataset_s3_auth_override(),
            remote_linked=_remote_linked,
        )

    def _auxiliary_timestamp_values(options: list[tuple[str, str]]) -> list[str]:
        return [value for value, _label in options if value != "latest"]

    def _sync_auxiliary_selectors_from_data(*, reset_manual: bool = False) -> None:
        if reset_manual:
            grid_state["surrogate_manual"] = False
            grid_state["next_x_manual"] = False

        data_value = str(grid_state.get("version_suffix") or "latest")
        sur_options = list(grid_state.get("surrogate_suffix_options") or [])
        nx_options = list(grid_state.get("next_x_suffix_options") or [])
        if not sur_options:
            sur_options = [("latest", "Latest (surrogate.json)")]
        if not nx_options:
            nx_options = [("latest", "Latest (next_x.json)")]

        if not grid_state.get("surrogate_manual"):
            sur_value = resolve_auxiliary_suffix_for_data_snapshot(
                data_value,
                _auxiliary_timestamp_values(sur_options),
            )
            valid_sur = {value for value, _label in sur_options}
            if sur_value not in valid_sur:
                sur_value = "latest" if "latest" in valid_sur else sur_options[0][0]
            grid_state["surrogate_version_suffix"] = sur_value

        if not grid_state.get("next_x_manual"):
            nx_value = resolve_auxiliary_suffix_for_data_snapshot(
                data_value,
                _auxiliary_timestamp_values(nx_options),
            )
            valid_nx = {value for value, _label in nx_options}
            if nx_value not in valid_nx:
                nx_value = "latest" if "latest" in valid_nx else nx_options[0][0]
            grid_state["next_x_version_suffix"] = nx_value

        grid_state["updating_controls"] = True
        try:
            surrogate_select.options = sur_options
            _set_select_value_if_needed(
                surrogate_select,
                str(grid_state.get("surrogate_version_suffix") or "latest"),
            )
            next_x_select.options = nx_options
            _set_select_value_if_needed(
                next_x_select,
                str(grid_state.get("next_x_version_suffix") or "latest"),
            )
        finally:
            grid_state["updating_controls"] = False

    def _rebuild_triplet_catalog(
        *,
        preserve_workflow: bool = True,
        preserve_snapshot: bool = True,
    ) -> None:
        index = discover_nsdf_triplet_index(
            _resolve_base_paths(),
            base_dir=_bd,
            save_dir=_sd,
            mongo_s3_auth=_dataset_s3_auth_override(),
            remote_linked=_remote_linked,
        )
        grid_state["triplet_index"] = index
        grid_state["catalog_ready"] = True
        grid_state["uncertainty_trend_points_key"] = ""
        grid_state["uncertainty_trend_points"] = None

        workflow_options = index.workflow_select_options()
        if not workflow_options:
            workflow_options = [(NSDF_UNKNOWN_WORKFLOW_ID, "(unknown)")]

        current_workflow = str(grid_state.get("workflow_id") or "").strip()
        if (
            not preserve_workflow
            or not current_workflow
            or not index.has_workflow(current_workflow)
        ):
            current_workflow = resolve_default_workflow_selection(
                index,
                dataset_workflow_id=workflow_id_from_dataset_doc(_dataset_doc),
                query_workflow_id=_query_workflow_id0,
            )
        grid_state["workflow_id"] = current_workflow

        snapshot_options = index.snapshot_select_options(current_workflow)
        if not snapshot_options:
            snapshot_options = [("latest", "Latest (data.json)")]

        current_snapshot = str(grid_state.get("version_suffix") or "latest")
        valid_snapshots = {value for value, _label in snapshot_options}
        if not preserve_snapshot or current_snapshot not in valid_snapshots:
            current_snapshot = index.default_snapshot_value(current_workflow)
        grid_state["version_suffix"] = current_snapshot
        _refresh_auxiliary_suffix_options(current_workflow)

        grid_state["updating_controls"] = True
        try:
            workflow_select.options = workflow_options
            workflow_select.value = current_workflow
            version_select.options = snapshot_options
            version_select.value = current_snapshot
        finally:
            grid_state["updating_controls"] = False
        _sync_auxiliary_selectors_from_data(reset_manual=not preserve_snapshot)

    def _refresh_snapshot_options_for_workflow(workflow_id: str) -> None:
        index = grid_state.get("triplet_index")
        if index is None:
            snapshot_options = [("latest", "Latest (data.json)")]
            current_snapshot = "latest"
        else:
            snapshot_options = index.snapshot_select_options(workflow_id)
            if not snapshot_options:
                snapshot_options = [("latest", "Latest (data.json)")]
            current_snapshot = index.default_snapshot_value(workflow_id)
        grid_state["version_suffix"] = current_snapshot
        _refresh_auxiliary_suffix_options(workflow_id)
        grid_state["updating_controls"] = True
        try:
            version_select.options = snapshot_options
            version_select.value = current_snapshot
        finally:
            grid_state["updating_controls"] = False
        _sync_auxiliary_selectors_from_data(reset_manual=True)

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
        _sync_auxiliary_selectors_from_data(reset_manual=True)
        _defer_figures_work(_reload_figures_view, message="Loading snapshot...")

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

        _invalidate_uncertainty_trend_cache()
        p = _resolve_paths()
        try:
            bundle = load_nsdf_json_bundle(
                p,
                mongo_s3_auth=_dataset_s3_auth_override(),
                remote_linked=_remote_linked,
            )
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
        selected_workflow = str(grid_state.get("workflow_id") or "").strip()
        active_workflow_id = active_workflow_id_from_grid_state(selected_workflow)
        if active_workflow_id is None and selected_workflow != NSDF_UNKNOWN_WORKFLOW_ID:
            active_workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
        overlay_entry = _select_next_x_entry(next_x_info, active_workflow_id)
        if overlay_entry is not None and overlay_entry.coordinates.size:
            active_points = int(overlay_entry.coordinates.shape[0])
            next_x_summary = (
                f"next_x: {active_points} proposed point(s) for workflow "
                f"{overlay_entry.workflow_id}."
            )
        elif next_x_info.entries and active_workflow_id:
            active_points = 0
        elif next_x_info.entries:
            next_x_summary = "next_x: loaded but no active non-demo workflow entry."
        elif bundle.next_x is not None:
            next_x_summary = "next_x: loaded but no valid workflow entries."
        workflow_label = format_nsdf_workflow_select_label(
            selected_workflow or NSDF_UNKNOWN_WORKFLOW_ID
        )
        workflow_line = f"Workflow: {workflow_label}"
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
        def _file_label(selector_value: str, latest_label: str) -> str:
            value = str(selector_value or "latest").strip()
            return latest_label if value in ("", "latest") else value

        data_label = _file_label(
            grid_state.get("version_suffix"),
            "Latest (data.json)",
        )
        sur_label = _file_label(
            grid_state.get("surrogate_version_suffix"),
            "Latest (surrogate.json)",
        )
        nx_label = _file_label(
            grid_state.get("next_x_version_suffix"),
            "Latest (next_x.json)",
        )
        version_label = f"data={data_label}; surrogate={sur_label}; next_x={nx_label}"
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

    def _color_range_mode() -> str:
        mode = str(grid_state.get("color_range_mode") or "dynamic").strip()
        return mode if mode in ("dynamic", "manual") else "dynamic"

    def _sync_color_range_control_state() -> None:
        manual = _color_range_mode() == "manual"
        color_lo_input.disabled = not manual
        color_hi_input.disabled = not manual
        btn_reset_color_range.disabled = not manual

    def apply_plot_color_range() -> None:
        if _color_range_mode() == "manual":
            lo = _float_value(str(color_lo_input.value or ""))
            hi = _float_value(str(color_hi_input.value or ""))
            plot_cfg.estimate_color_low = lo
            plot_cfg.estimate_color_high = hi
        else:
            plot_cfg.estimate_color_low = None
            plot_cfg.estimate_color_high = None

    def _update_color_range_display(lo: float, hi: float) -> None:
        grid_state["last_color_range"] = (lo, hi)
        if _color_range_mode() != "dynamic":
            return
        grid_state["updating_controls"] = True
        try:
            color_lo_input.value = _format_color_limit(lo)
            color_hi_input.value = _format_color_limit(hi)
        finally:
            grid_state["updating_controls"] = False

    def _color_range_status_line(lo: float, hi: float) -> str:
        lo_text = _format_color_limit(lo)
        hi_text = _format_color_limit(hi)
        if _color_range_mode() == "manual":
            manual_lo = plot_cfg.estimate_color_low
            manual_hi = plot_cfg.estimate_color_high
            if (
                manual_lo is not None
                and manual_hi is not None
                and manual_lo < manual_hi
            ):
                return f"Plot 1–2 color range: fixed [{lo_text}, {hi_text}]."
            return (
                "Plot 1–2 color range: invalid manual min/max; "
                f"using auto [{lo_text}, {hi_text}]."
            )
        return f"Plot 1–2 color range: auto [{lo_text}, {hi_text}]."

    def _uncertainty_trend_points_cache_key(workflow_id: str, index: NSDFTripletIndex) -> str:
        resolved = _resolve_paths()
        return (
            f"{workflow_id}|{len(index.snapshots_for_workflow(workflow_id))}|"
            f"{resolved.s3_surrogate_key}|{resolved.surrogate_json_path}|"
            f"{resolved.surrogate_version_suffix}|{_remote_linked}|"
            f"{plot_cfg.grid_size[0]}x{plot_cfg.grid_size[1]}"
        )

    def _plot4_trend_disabled_series() -> UncertaintyTrendSeries:
        return UncertaintyTrendSeries(
            step_ids=[],
            y=np.array([], dtype=np.float64),
            labels=[],
            warnings=[
                'Plot 4 trend is off. Enable "Compute Plot 4 trend" to scan surrogate history.'
            ],
        )

    def _uncertainty_trend_for_dashboard() -> Optional[UncertaintyTrendSeries]:
        if not grid_state.get("compute_plot4_trend"):
            return _plot4_trend_disabled_series()

        current_snap = str(grid_state.get("version_suffix") or "latest")

        if not grid_state.get("catalog_ready"):
            quick = _uncertainty_trend_from_loaded_surrogate(current_snap)
            if quick is not None:
                return quick
            quick = build_uncertainty_trend_from_surrogate_paths(
                _resolve_paths(),
                current_snapshot=current_snap,
                grid_size=plot_cfg.grid_size,
                mongo_s3_auth=_dataset_s3_auth_override(),
                remote_linked=_remote_linked,
            )
            if quick is not None:
                return quick
            return UncertaintyTrendSeries(
                step_ids=[],
                y=np.array([], dtype=np.float64),
                labels=[],
                warnings=[
                    "Press Index workflows to build the full uncertainty trend across snapshots."
                ],
            )

        index = grid_state.get("triplet_index")
        if not isinstance(index, NSDFTripletIndex):
            return None
        workflow = str(grid_state.get("workflow_id") or "").strip()
        if not workflow or workflow == NSDF_UNKNOWN_WORKFLOW_ID:
            return None
        cache_key = _uncertainty_trend_points_cache_key(workflow, index)
        cached = grid_state.get("uncertainty_trend_points")
        if (
            grid_state.get("uncertainty_trend_points_key") == cache_key
            and isinstance(cached, UncertaintyTrendSeries)
        ):
            chrono = _snapshots_chronological(index.snapshots_for_workflow(workflow))
            current_index = _trend_current_index(
                cached.step_ids,
                current_snapshot=current_snap,
                chrono_snaps=chrono,
            )
            return replace(cached, current_index=current_index)

        series = build_uncertainty_trend_series(
            index,
            _resolve_base_paths(),
            workflow_id=workflow,
            current_snapshot=current_snap,
            grid_size=plot_cfg.grid_size,
            mongo_s3_auth=_dataset_s3_auth_override(),
            remote_linked=_remote_linked,
            surrogate_paths=_resolve_paths(),
            allow_per_snapshot_fallback=True,
        )
        grid_state["uncertainty_trend_points_key"] = cache_key
        grid_state["uncertainty_trend_points"] = replace(series, current_index=None)
        return series

    def rebuild_figures(*, show_loading_gap: bool = True) -> None:
        apply_grid_size()
        apply_plot_color_range()
        if show_loading_gap and not figures_column.children:
            figures_column.children = [_loading_placeholder_div("Building plots...")]
        if loaded_bundle is None:
            figures_column.children = [Div(text="<i>No NSDF data loaded.</i>")]
            return
        try:
            grids = build_strain_field_grids(loaded_bundle.data, plot_cfg, loaded_bundle.surrogate)
            measured_vals = grids.meta.get("measurement_values")
            if not isinstance(measured_vals, np.ndarray):
                measured_vals = np.array([], dtype=np.float64)
            est_lo, est_hi = resolve_estimate_color_limits(
                grids.estimate,
                measured_vals,
                manual_low=plot_cfg.estimate_color_low,
                manual_high=plot_cfg.estimate_color_high,
            )
            _update_color_range_display(est_lo, est_hi)
            if grid_state.get("last_status") is not None:
                grid_state["last_status"]["color_range_line"] = _color_range_status_line(
                    est_lo,
                    est_hi,
                )
            surrogate_info = validate_nsdf_surrogate_doc(loaded_bundle.surrogate)
            next_x_info = validate_nsdf_next_x_doc(loaded_bundle.next_x)
            selected_workflow = str(grid_state.get("workflow_id") or "").strip()
            active_workflow_id = active_workflow_id_from_grid_state(selected_workflow)
            if active_workflow_id is None and selected_workflow != NSDF_UNKNOWN_WORKFLOW_ID:
                active_workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
            uncertainty_trend: Optional[UncertaintyTrendSeries]
            try:
                uncertainty_trend = _uncertainty_trend_for_dashboard()
            except Exception as trend_exc:
                traceback.print_exc()
                uncertainty_trend = UncertaintyTrendSeries(
                    step_ids=[],
                    y=np.array([], dtype=np.float64),
                    labels=[],
                    warnings=[f"Plot 4 trend unavailable: {trend_exc}"],
                )
            triplet_row = make_strain_triplet_row(
                grids,
                plot_cfg,
                row_subtitle="dataset_y",
                next_x_info=next_x_info,
                active_workflow_id=active_workflow_id,
                uncertainty_trend=uncertainty_trend,
            )
            figures_column.children = [triplet_row]
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
        if grid_state["updating_controls"] or grid_state.get("suppress_selector_reload"):
            return
        if (grid_state.get("playback") or {}).get("active"):
            _stop_playback()
        grid_state["version_suffix"] = str(new or "latest")
        _sync_auxiliary_selectors_from_data(reset_manual=False)
        _defer_figures_work(_reload_figures_view, message="Loading data snapshot...")

    def on_surrogate_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"] or grid_state.get("suppress_selector_reload"):
            return
        grid_state["surrogate_version_suffix"] = str(new or "latest")
        grid_state["surrogate_manual"] = True
        _defer_figures_work(_reload_figures_view, message="Loading surrogate snapshot...")

    def on_next_x_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"] or grid_state.get("suppress_selector_reload"):
            return
        grid_state["next_x_version_suffix"] = str(new or "latest")
        grid_state["next_x_manual"] = True
        _defer_figures_work(_reload_figures_view, message="Loading next_x snapshot...")

    def on_workflow_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        if not grid_state.get("catalog_ready"):
            return
        if (grid_state.get("playback") or {}).get("active"):
            _stop_playback()
        grid_state["workflow_id"] = str(new or "").strip()
        _refresh_snapshot_options_for_workflow(grid_state["workflow_id"])
        _defer_figures_work(
            _reload_figures_view,
            message="Loading workflow and file selectors...",
        )

    def on_index_workflows() -> None:
        _defer_catalog_index()

    def on_reload() -> None:
        _defer_figures_work(_reload_current_view, message="Reloading latest triplet...")

    def on_external_refresh(_doc: Any = doc, _on_reload: Callable[[], None] = on_reload) -> None:
        _doc.add_next_tick_callback(_on_reload)

    def on_grid_control_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        manual_grid_size = _grid_size_from_controls()
        grid_state["manual_grid_size"] = manual_grid_size
        grid_state["active_source"] = "manual controls"
        plot_cfg.grid_size = manual_grid_size
        _defer_figures_work(rebuild_figures, message="Updating grid...")

    def on_reset_color_range() -> None:
        grid_state["color_range_mode"] = "dynamic"
        grid_state["updating_controls"] = True
        try:
            color_range_select.value = "dynamic"
        finally:
            grid_state["updating_controls"] = False
        _sync_color_range_control_state()
        _defer_figures_work(rebuild_figures, message="Resetting color range...")

    def on_color_range_mode_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        grid_state["color_range_mode"] = str(new or "dynamic")
        _sync_color_range_control_state()
        _defer_figures_work(rebuild_figures, message="Updating color range...")

    def on_color_limit_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        if _color_range_mode() != "manual":
            return
        _defer_figures_work(rebuild_figures, message="Updating color range...")

    def on_plot4_trend_change(attr: str, old: Any, new: Any) -> None:
        if grid_state["updating_controls"]:
            return
        enabled = bool(new)
        grid_state["compute_plot4_trend"] = enabled
        if enabled:
            _invalidate_uncertainty_trend_cache()
        message = "Computing Plot 4 trend..." if enabled else "Updating plots..."
        _defer_figures_work(rebuild_figures, message=message)

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
        _defer_figures_work(rebuild_figures, message="Resetting grid...")

    def on_toggle_status() -> None:
        log_panel_div.visible = not log_panel_div.visible
        btn_toggle_status.label = "Hide status" if log_panel_div.visible else "Show status"

    grid_w.on_change("value", on_grid_control_change)
    grid_h.on_change("value", on_grid_control_change)
    color_range_select.on_change("value", on_color_range_mode_change)
    color_lo_input.on_change("value", on_color_limit_change)
    color_hi_input.on_change("value", on_color_limit_change)
    plot4_trend_checkbox.on_change("active", on_plot4_trend_change)
    workflow_select.on_change("value", on_workflow_change)
    version_select.on_change("value", on_version_change)
    surrogate_select.on_change("value", on_surrogate_change)
    next_x_select.on_change("value", on_next_x_change)
    btn_play_forward.on_click(on_play_forward)
    btn_play_backward.on_click(on_play_backward)
    btn_play_stop.on_click(on_play_stop)
    btn_reset_grid.on_click(on_reset_grid)
    btn_reset_color_range.on_click(on_reset_color_range)
    btn_reload.on_click(on_reload)
    btn_index_workflows.on_click(on_index_workflows)
    btn_toggle_status.on_click(on_toggle_status)
    _busy_controls.extend(
        [
            workflow_select,
            version_select,
            surrogate_select,
            next_x_select,
            grid_w,
            grid_h,
            color_range_select,
            color_lo_input,
            color_hi_input,
            btn_reset_color_range,
            plot4_trend_checkbox,
            btn_index_workflows,
            btn_reload,
            btn_reset_grid,
            btn_play_forward,
            btn_play_backward,
            btn_play_stop,
            play_interval_input,
        ]
    )

    controls = column(
        row(
            workflow_select,
            version_select,
            surrogate_select,
            next_x_select,
            sizing_mode="scale_width",
        ),
        _control_row(
            grid_w,
            grid_h,
            color_range_select,
            color_lo_input,
            color_hi_input,
            btn_reset_color_range,
            plot4_trend_checkbox,
            btn_index_workflows,
            btn_reload,
            btn_reset_grid,
            btn_toggle_status,
        ),
        _control_row(
            play_interval_input,
            btn_play_backward,
            btn_play_forward,
            btn_play_stop,
        ),
        sizing_mode="stretch_width",
    )

    if is_scientistcloud_portal_data_mount_context(_bd, _sd):
        header = create_header_banner("ORNL CHESS Strain", "ScientistCloud")
    else:
        header = create_header_banner("ORNL CHESS NSDF measurements", "DIAL Dashboard")
    root = column(
        header,
        workflow_div,
        controls,
        figures_column,
        log_panel_div,
        sizing_mode="stretch_width",
    )
    figures_column.children = [_loading_placeholder_div("Loading dashboard...")]
    _sync_color_range_control_state()
    doc.add_root(root)

    if paths.local_data_dir or paths.has_s3_source() or paths.local_json_path or paths.json_url:
        _defer_figures_work(_initial_dashboard_load, message="Loading latest triplet...")
    else:
        _set_controls_busy(False)
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
            cancel_catalog: Callable[[], None] = _cancel_background_catalog,
        ) -> None:
            try:
                stop_playback()
            except Exception:
                pass
            try:
                cancel_catalog()
            except Exception:
                pass
            unregister(token)

        doc.on_session_destroyed(_cleanup_refresh_callback)
