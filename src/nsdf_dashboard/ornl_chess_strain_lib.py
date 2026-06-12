"""
ORNL / CHESS NSDF measurement helpers: JSON parsing, NSDF validation, grid construction,
and Bokeh heatmaps.

**Where JSON is loaded from (deployment order)**

1. **ScientistCloud data portal** (default when ``base_dir`` / ``save_dir`` live under
   ``/mnt/visus_datasets``, e.g. upload + converted trees):

   - ``upload`` — first ``*.json`` under dataset ``base_dir`` (prefers ``reduced_data.json``)
   - ``converted`` — same under ``save_dir``
   - ``query_path`` / ``query_url`` — Bokeh URL args (portal may append gateway HTTPS)
   - ``env_path`` / ``env_url`` — preferred ``ORNL_NSDF_DATA_JSON_PATH`` /
     ``ORNL_NSDF_DATA_JSON_URL``; legacy ``ORNL_STRAIN_JSON_PATH`` /
     ``ORNL_STRAIN_JSON_URL`` remain aliases for NSDF ``data.json``.

2. **Command line / local / CHESS checkout** (default when not on that mount): use
   **environment and URL args first**, then server dirs:

   - ``env_path``, ``env_url``, ``query_path``, ``query_url``, ``upload``, ``converted``

**Override without code changes**

- ``ORNL_STRAIN_RESOLVE_MODE`` — ``auto`` (default), ``portal`` (always server-first),
  ``cli`` (always env-first). Aliases: ``server`` / ``scientistcloud`` for portal;
  ``local`` / ``cmd`` for cli.

- ``ORNL_STRAIN_SOURCE_ORDER`` — explicit comma-separated tokens (overrides mode), e.g.
  ``env_path,env_url,upload,converted,query_url`` for a CHESS server that only sets env vars.

  Valid tokens: ``upload``, ``converted``, ``query_path``, ``query_url``, ``env_path``, ``env_url``.

**NSDF triplet files on S3 (CHESS live run)**

- **Live defaults (primary):** ``data.json``, ``surrogate.json``, ``next_x.json``. The
  dashboard loads these on startup, on Reload, and on event-triggered refresh; the pipeline
  writes new datapoints to these rolling files.
- **Backups (archive):** ``data_<timestamp>_<id>.json``, ``surrogate_<timestamp>_<id>.json``,
  ``next_x_<timestamp>_<id>.json``. Written when a snapshot is archived; used for workflow
  indexing, Plot 4 trend, and when the user explicitly picks a snapshot in the selectors.
  Load failures on the live trio mean the writer has not updated S3 yet—not that backups
  should replace them as the default source.

  **Triplet matching:** archived ``data`` / ``surrogate`` / ``next_x`` files are paired by
  shared ``workflow_id`` and the numeric **id** at the end of the filename (e.g. ``_48``).
  Timestamps in the middle of the suffix may differ across the three files.

- **Workflow catalog (optional):** ``catalog.json`` beside the triplet files lists archived
  snapshots (workflow id, suffix, trend scalar). The dashboard reads it on **Index workflows**
  when present; otherwise it scans JSON files and writes ``catalog.json`` locally or on S3.
"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (edit here or override via StrainDashboardPaths)
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: Tuple[int, int] = (26, 26)
NSDF_DATA_JSON_BASENAME = "data.json"
NSDF_CATALOG_JSON_BASENAME = "catalog.json"
NSDF_CATALOG_VERSION = 1
NSDF_VERSION_SUFFIX_RE = re.compile(r"^\d{8}T\d{6}Z$", re.IGNORECASE)
NSDF_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", re.IGNORECASE)
NSDF_COMPOSITE_SNAPSHOT_SUFFIX_RE = re.compile(
    r"^\d{8}T\d{6}Z_(?P<id>\d+)$",
    re.IGNORECASE,
)
NSDF_DATA_FILENAME_RE = re.compile(
    r"^data(?:_(?P<suffix>[A-Za-z0-9][A-Za-z0-9._-]*))?\.json$",
    re.IGNORECASE,
)


def is_valid_nsdf_snapshot_suffix(suffix: str) -> bool:
    """True for a non-empty ISO timestamp or opaque snapshot id (e.g. ``50``, ``20260608T120000Z_3``)."""
    key = (suffix or "").strip()
    if not key:
        return False
    if NSDF_VERSION_SUFFIX_RE.fullmatch(key):
        return True
    return bool(NSDF_SNAPSHOT_ID_RE.fullmatch(key))


def is_valid_nsdf_version_suffix(suffix: str) -> bool:
    """True for empty/latest alias or a valid snapshot suffix (ISO timestamp or id)."""
    key = (suffix or "").strip()
    return not key or is_valid_nsdf_snapshot_suffix(key)


def triplet_snapshot_id(suffix: str) -> str:
    """
    Shared snapshot id for matching trio members (``data`` / ``surrogate`` / ``next_x``).

    ``20260608T221354Z_8`` and ``20260608T214403Z_8`` both yield ``8``. ISO-only legacy
    suffixes without a trailing id return the full suffix string.
    """
    text = (suffix or "").strip()
    if not text:
        return ""
    label = triplet_suffix_trend_label(text)
    return "" if label == "Latest" else label


def triplet_suffix_trend_label(suffix: str) -> str:
    """
    Short Plot 4 x-axis label from a snapshot suffix or trend step id.

    ``20260608T222314Z_49`` -> ``49``; plain numeric ids pass through unchanged.
    """
    text = (suffix or "").strip()
    if not text or text.lower() == "latest":
        return "Latest"
    composite = NSDF_COMPOSITE_SNAPSHOT_SUFFIX_RE.match(text)
    if composite:
        return composite.group("id")
    if text.isdigit():
        return text
    if "_" in text:
        tail = text.rsplit("_", 1)[-1]
        if tail.isdigit():
            return tail
    return text


def _trend_labels_for_step_ids(step_ids: Sequence[str]) -> List[str]:
    return [triplet_suffix_trend_label(step_id) for step_id in step_ids]


def _trend_step_matches_snapshot(step_id: str, current_snapshot: str) -> bool:
    step_id = (step_id or "").strip()
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    if step_id == current_snapshot:
        return True
    if step_id.lower() == "latest" and current_snapshot.lower() == "latest":
        return True
    return triplet_suffix_trend_label(step_id) == triplet_suffix_trend_label(current_snapshot)


def _strip_env_quotes(value: str) -> str:
    v = (value or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def load_simple_env_file(path: str) -> Dict[str, str]:
    """Parse a small dotenv-style file without mutating ``os.environ``."""
    p = (path or "").strip()
    if not p:
        return {}
    out: Dict[str, str] = {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                out[key] = _strip_env_quotes(value)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read ORNL_S3_ENV_FILE={p!r}: {exc}") from exc
    return out


def _find_optional_env_file() -> str:
    """Return a repo/cwd ``.env`` path when ``ORNL_S3_ENV_FILE`` is not set."""
    candidates: List[str] = []
    cwd = os.getcwd()
    if cwd:
        candidates.append(os.path.join(cwd, ".env"))
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(pkg_root, ".env"))
    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm
    return ""


def local_files_first_for_testing() -> bool:
    """True when ``LOCAL_FILES_FIRST_FOR_TESTING`` enables ``LOCAL_DATA_DIR`` routing."""
    raw = (os.environ.get("LOCAL_FILES_FIRST_FOR_TESTING") or "").strip()
    if not raw:
        env_file = os.environ.get("ORNL_S3_ENV_FILE", "").strip() or _find_optional_env_file()
        if env_file:
            raw = (load_simple_env_file(env_file).get("LOCAL_FILES_FIRST_FOR_TESTING") or "").strip()
    return raw.lower() in ("1", "true", "yes")


@dataclass
class StrainDashboardPaths:
    """Where to load NSDF data and optional surrogate JSON."""

    local_json_path: str = ""
    json_url: str = ""
    surrogate_json_path: str = ""
    surrogate_json_url: str = ""
    next_x_json_path: str = ""
    next_x_json_url: str = ""
    local_data_dir: str = ""
    s3_env_file: str = ""
    s3_bucket: str = ""
    s3_data_key: str = ""
    s3_surrogate_key: str = ""
    s3_next_x_key: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"
    version_suffix: str = ""
    surrogate_version_suffix: str = ""
    next_x_version_suffix: str = ""
    strict_triplet_paths: bool = False

    @classmethod
    def from_environ(cls) -> "StrainDashboardPaths":
        env_file = os.environ.get("ORNL_S3_ENV_FILE", "").strip() or _find_optional_env_file()
        env_file_values = load_simple_env_file(env_file)

        def env_value(name: str, default: str = "") -> str:
            return (os.environ.get(name) or env_file_values.get(name) or default).strip()

        return cls(
            local_json_path=(
                os.environ.get("ORNL_NSDF_DATA_JSON_PATH")
                or os.environ.get("ORNL_STRAIN_JSON_PATH")
                or ""
            ).strip(),
            json_url=(
                os.environ.get("ORNL_NSDF_DATA_JSON_URL")
                or os.environ.get("ORNL_STRAIN_JSON_URL")
                or ""
            ).strip(),
            surrogate_json_path=os.environ.get("ORNL_SURROGATE_JSON_PATH", "").strip(),
            surrogate_json_url=os.environ.get("ORNL_SURROGATE_JSON_URL", "").strip(),
            next_x_json_path=os.environ.get("ORNL_NEXT_X_JSON_PATH", "").strip(),
            next_x_json_url=os.environ.get("ORNL_NEXT_X_JSON_URL", "").strip(),
            local_data_dir=env_value("LOCAL_DATA_DIR"),
            s3_env_file=env_file,
            s3_bucket=env_value("S3_BUCKET"),
            s3_data_key=env_value("S3_DATA_KEY"),
            s3_surrogate_key=env_value("S3_SURROGATE_KEY"),
            s3_next_x_key=env_value("S3_NEXT_X_KEY"),
            s3_endpoint_url=env_value("S3_ENDPOINT_URL"),
            s3_region=env_value("S3_REGION", "us-east-1") or "us-east-1",
        )

    def has_s3_source(self) -> bool:
        return bool((self.s3_bucket or "").strip() and (self.s3_data_key or "").strip())


def strain_paths_are_loadable(paths: StrainDashboardPaths) -> bool:
    """True when ``load_nsdf_json_bundle`` has a concrete data.json source to try."""
    if paths.has_s3_source():
        return True
    if (paths.json_url or "").strip():
        return True
    loc = (paths.local_json_path or "").strip()
    if loc:
        if _looks_like_http_url(loc):
            return True
        if _local_path_is_nsdf_data_json(loc):
            return True
    return _local_data_dir_active(paths)


def format_nsdf_path_resolution_hint(
    paths: StrainDashboardPaths,
    doc: Optional[Mapping[str, Any]],
    *,
    remote_linked: bool = False,
) -> str:
    """Short diagnostic text when path resolution did not yield a loadable source."""
    lines: List[str] = []
    if remote_linked:
        lines.append("Remote-linked dataset (server=true): JSON should come from URL args or Mongo.")
    if not isinstance(doc, Mapping):
        lines.append("Mongo dataset document not found for this uuid.")
    else:
        link = pick_strain_json_link_from_dataset_doc(doc)
        if link:
            lines.append(f"Mongo link field: {link[:120]}{'…' if len(link) > 120 else ''}")
        else:
            lines.append(
                "Mongo record has no viewer_url/download_url/google_drive_link/source_path "
                "ending in .json or starting with s3:// or https://."
            )
        if str(doc.get("s3_access_key_id") or "").strip():
            lines.append("Mongo stores s3_access_key_id (dashboard loads S3 after login without URL secrets).")
    if paths.has_s3_source():
        lines.append(f"Resolved S3: s3://{paths.s3_bucket}/{paths.s3_data_key}")
    elif (paths.json_url or "").strip():
        lines.append(f"Resolved URL: {paths.json_url[:120]}")
    elif (paths.local_json_path or "").strip():
        loc = paths.local_json_path
        if _local_path_is_nsdf_data_json(loc):
            lines.append(f"Resolved local data.json: {loc}")
        else:
            lines.append(
                f"Ignored non-data local JSON (e.g. catalog.json): {loc}"
            )
    elif (paths.local_data_dir or "").strip() and not _local_data_dir_active(paths):
        lines.append(
            f"LOCAL_DATA_DIR={paths.local_data_dir!r} is set but LOCAL_FILES_FIRST_FOR_TESTING is off."
        )
    return "\n".join(lines)


def _copy_s3_fields(src: StrainDashboardPaths, dst: StrainDashboardPaths) -> StrainDashboardPaths:
    """Merge env/session auxiliary fields from *src* into resolved *dst* (dst routing wins)."""
    if (src.local_data_dir or "").strip():
        dst.local_data_dir = src.local_data_dir
    if (src.s3_env_file or "").strip():
        dst.s3_env_file = src.s3_env_file
    for attr in (
        "s3_bucket",
        "s3_data_key",
        "s3_surrogate_key",
        "s3_next_x_key",
        "s3_endpoint_url",
        "s3_region",
    ):
        if not (getattr(dst, attr) or "").strip():
            setattr(dst, attr, getattr(src, attr))
    return dst


# ---------------------------------------------------------------------------
# JSON location resolution (portal vs CLI vs custom order)
# ---------------------------------------------------------------------------

_STRAIN_ORDER_PORTAL: Tuple[str, ...] = (
    "upload",
    "converted",
    "query_path",
    "query_url",
    "env_path",
    "env_url",
)
_STRAIN_ORDER_REMOTE_LINKED: Tuple[str, ...] = (
    "query_url",
    "query_path",
    "env_url",
    "env_path",
)
_STRAIN_ORDER_CLI: Tuple[str, ...] = (
    "env_path",
    "env_url",
    "query_path",
    "query_url",
    "upload",
    "converted",
)
_STRAIN_ORDER_TOKENS = frozenset(_STRAIN_ORDER_PORTAL)


def is_scientistcloud_portal_data_mount_context(base_dir: str, save_dir: str) -> bool:
    """True when dataset dirs look like ScientistCloud portal upload/converted mounts."""
    root = "/mnt/visus_datasets"
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()
    return bd.startswith(root) or sd.startswith(root)


def _local_path_is_nsdf_data_json(path: str) -> bool:
    """True when *path* points at an NSDF measurement ``data*.json`` file (not catalog/surrogate/next_x)."""
    p = (path or "").strip()
    if not p or _looks_like_http_url(p):
        return bool(p)
    if not os.path.isfile(p):
        return False
    base = os.path.basename(p)
    if base.lower() == NSDF_CATALOG_JSON_BASENAME.lower():
        return False
    if base.lower() in ("surrogate.json", "next_x.json"):
        return False
    if parse_nsdf_data_filename(base) is not None:
        return True
    return base.lower() in ("data.json", "reduced_data.json")


def find_strain_json_under_dataset_dir(directory: str) -> str:
    """
    Pick an NSDF **data** JSON file under ``directory`` (upload or converted tree for one UUID).

    Prefers ``data.json``; otherwise the first archived ``data_*.json`` (sorted by name).
    Ignores ``catalog.json``, surrogate, and next_x sidecars.
    """
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return ""
    for preferred_name in ("data.json", "reduced_data.json"):
        preferred = os.path.join(d, preferred_name)
        if os.path.isfile(preferred):
            return preferred
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return ""
    for name in names:
        if not name.lower().endswith(".json"):
            continue
        if name.lower() == NSDF_CATALOG_JSON_BASENAME.lower():
            continue
        if name.lower() in ("surrogate.json", "next_x.json"):
            continue
        if parse_nsdf_data_filename(name) is None:
            continue
        full = os.path.join(d, name)
        if os.path.isfile(full):
            return full
    return ""


def scientistcloud_dataset_is_remote_linked(
    doc: Optional[Mapping[str, Any]] = None,
    *,
    server_param: str = "",
) -> bool:
    """
    True for ScientistCloud datasets linked to remote S3/gateway storage.

    Mirrors ``sc_dataset_is_remote_linked()`` in dashboard_share_link.php.
    Uploaded-file datasets (local upload tree only) return False.
    """
    if str(server_param or "").strip().lower() == "true":
        return True
    if not isinstance(doc, Mapping):
        return False
    if str(doc.get("server") or "").strip().lower() == "true":
        return True
    link = pick_strain_json_link_from_dataset_doc(doc)
    if not link:
        return False
    low = link.lower()
    return low.startswith(("s3://", "http://", "https://"))


def _strain_effective_source_order(
    base_dir: str,
    save_dir: str,
    *,
    remote_linked: bool = False,
) -> Tuple[str, ...]:
    if remote_linked:
        return _STRAIN_ORDER_REMOTE_LINKED
    custom = (os.environ.get("ORNL_STRAIN_SOURCE_ORDER") or "").strip()
    if custom:
        parts = tuple(
            x.strip().lower()
            for x in custom.split(",")
            if x.strip() and x.strip().lower() in _STRAIN_ORDER_TOKENS
        )
        if parts:
            unknown = [
                x.strip().lower()
                for x in custom.split(",")
                if x.strip() and x.strip().lower() not in _STRAIN_ORDER_TOKENS
            ]
            if unknown:
                _LOG.warning("ORNL_STRAIN_SOURCE_ORDER: ignoring unknown tokens: %s", unknown)
            return parts
        _LOG.warning("ORNL_STRAIN_SOURCE_ORDER set but no valid tokens; falling back to mode/auto")

    mode = (os.environ.get("ORNL_STRAIN_RESOLVE_MODE") or "auto").strip().lower()
    if mode in ("portal", "server", "scientistcloud", "data_portal"):
        return _STRAIN_ORDER_PORTAL
    if mode in ("cli", "local", "command_line", "cmd", "dev"):
        return _STRAIN_ORDER_CLI
    if mode not in ("auto", ""):
        _LOG.warning("Unknown ORNL_STRAIN_RESOLVE_MODE=%r; using auto", mode)
    if is_scientistcloud_portal_data_mount_context(base_dir, save_dir):
        return _STRAIN_ORDER_PORTAL
    return _STRAIN_ORDER_CLI


def strain_resolve_order_summary(base_dir: str = "", save_dir: str = "") -> str:
    """One-line description of the active resolution order (for UI / logs)."""
    order = _strain_effective_source_order(base_dir, save_dir)
    return " → ".join(order)


def resolve_strain_paths_for_session(
    *,
    base_dir: str = "",
    save_dir: str = "",
    query_strain_json_path: str = "",
    query_strain_json_url: str = "",
    query_surrogate_json_path: str = "",
    query_surrogate_json_url: str = "",
    query_next_x_json_path: str = "",
    query_next_x_json_url: str = "",
    env: Optional[StrainDashboardPaths] = None,
    remote_linked: bool = False,
) -> StrainDashboardPaths:
    """
    Build ``StrainDashboardPaths`` using the configured source order.

    When a file is chosen from ``upload`` or ``converted``, ``json_url`` is still set to the
    first available of ``query_strain_json_url`` / ``env.json_url`` when present (for display /
    provenance); ``load_strain_json`` reads the file first.
    """
    env = env or StrainDashboardPaths.from_environ()
    order = _strain_effective_source_order(base_dir, save_dir, remote_linked=remote_linked)
    q_path = normalize_nsdf_remote_data_link((query_strain_json_path or "").strip())
    q_url = normalize_nsdf_remote_data_link((query_strain_json_url or "").strip())
    env_path = (env.local_json_path or "").strip()
    env_url = normalize_nsdf_remote_data_link((env.json_url or "").strip())
    surrogate_path = (query_surrogate_json_path or "").strip() or (env.surrogate_json_path or "").strip()
    surrogate_url = (query_surrogate_json_url or "").strip() or (env.surrogate_json_url or "").strip()
    next_x_path = (query_next_x_json_path or "").strip() or (env.next_x_json_path or "").strip()
    next_x_url = (query_next_x_json_url or "").strip() or (env.next_x_json_url or "").strip()
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()

    def with_auxiliary(p: StrainDashboardPaths) -> StrainDashboardPaths:
        p.surrogate_json_path = surrogate_path
        p.surrogate_json_url = surrogate_url
        p.next_x_json_path = next_x_path
        p.next_x_json_url = next_x_url
        p.local_data_dir = env.local_data_dir
        p.s3_env_file = env.s3_env_file
        p.s3_bucket = env.s3_bucket
        p.s3_data_key = env.s3_data_key
        p.s3_surrogate_key = env.s3_surrogate_key
        p.s3_next_x_key = env.s3_next_x_key
        p.s3_endpoint_url = env.s3_endpoint_url
        p.s3_region = env.s3_region
        return p

    def display_url() -> str:
        return q_url or env_url

    for token in order:
        if token == "upload":
            p = find_strain_json_under_dataset_dir(bd)
            if p:
                return with_auxiliary(StrainDashboardPaths(local_json_path=p, json_url=display_url()))
        elif token == "converted":
            p = find_strain_json_under_dataset_dir(sd)
            if p:
                return with_auxiliary(StrainDashboardPaths(local_json_path=p, json_url=display_url()))
        elif token == "query_path":
            if not q_path:
                continue
            if _looks_like_http_url(q_path):
                return with_auxiliary(StrainDashboardPaths(local_json_path="", json_url=q_path))
            if os.path.isfile(q_path):
                return with_auxiliary(StrainDashboardPaths(local_json_path=q_path, json_url=display_url()))
        elif token == "query_url":
            if q_url:
                return with_auxiliary(StrainDashboardPaths(local_json_path="", json_url=q_url))
        elif token == "env_path":
            if not env_path:
                continue
            if _looks_like_http_url(env_path):
                return with_auxiliary(StrainDashboardPaths(local_json_path="", json_url=env_path))
            if os.path.isfile(env_path):
                return with_auxiliary(StrainDashboardPaths(local_json_path=env_path, json_url=display_url()))
        elif token == "env_url":
            if env_url:
                return with_auxiliary(StrainDashboardPaths(local_json_path="", json_url=env_url))

    return with_auxiliary(StrainDashboardPaths(local_json_path="", json_url=""))


@dataclass
class StrainFieldPlotConfig:
    """Per-plot titles and axis labels (easy to change when scientists refine wording)."""

    grid_size: Tuple[int, int] = field(default_factory=lambda: DEFAULT_GRID_SIZE)
    x_axis_label: str = "labx"
    y_axis_label: str = "labz"
    title_measurements: str = "Measurement locations"
    title_estimate: str = "Prediction"
    title_variance: str = "Uncertainty"
    title_uncertainty_trend: str = "Uncertainty trend"
    trend_x_axis_label: str = "snapshot id"
    trend_y_axis_label: str = "avg uncertainty"
    flip_y_for_display: bool = False
    colormap_estimate: str = "Viridis256"
    colormap_variance: str = "Coolwarm256"
    colormap_mask: Tuple[str, str] = ("#ffffff", "#ffffff")
    estimate_color_low: Optional[float] = None
    estimate_color_high: Optional[float] = None


@dataclass
class StrainFieldGrids:
    """Three 2D numpy arrays aligned to the same pixel grid."""

    measurements: np.ndarray  # float 0/1 or 0..1 mask
    estimate: np.ndarray
    variance: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyTrendSeries:
    """Average uncertainty (plot 3) vs time-step id for one workflow."""

    step_ids: List[str]
    y: np.ndarray
    labels: List[str]
    current_index: Optional[int] = None
    source: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class NSDFMeasurementData:
    """Validated NSDF ``data.json`` arrays."""

    coordinates: np.ndarray
    observed_values: np.ndarray
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    bounds_source: str = "observed_minmax"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NSDFSurrogateData:
    """Validated optional surrogate model arrays and grid metadata."""

    surrogate: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    raw_uncertainty: Optional[np.ndarray] = None
    workflow_id: Optional[str] = None
    plot_dim: Optional[str] = None
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    points: Optional[int] = None
    points_to_predict: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def plottable_fields(self) -> List[str]:
        fields = ["dataset_y"]
        if self.surrogate is not None:
            fields.append("surrogate")
        if self.uncertainty is not None:
            fields.append("uncertainty")
        if self.raw_uncertainty is not None:
            fields.append("raw_uncertainty")
        return fields


@dataclass
class NSDFNextXEntry:
    """One proposed-scan workflow block from ``next_x.json``."""

    workflow_id: str
    coordinates: np.ndarray


@dataclass
class NSDFNextXData:
    """Validated optional ``next_x.json`` entries."""

    entries: List[NSDFNextXEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(int(entry.coordinates.shape[0]) for entry in self.entries)


@dataclass
class NSDFLoadedBundle:
    """Loaded NSDF data and optional surrogate / next_x metadata for the UI."""

    data: Dict[str, Any]
    surrogate: Optional[Dict[str, Any]] = None
    next_x: Optional[Any] = None
    messages: List[str] = field(default_factory=list)
    paths: StrainDashboardPaths = field(default_factory=StrainDashboardPaths)


NSDF_UNKNOWN_WORKFLOW_ID = "__unknown__"


@dataclass(frozen=True)
class NSDFSnapshotRef:
    """One timestamped (or latest) triplet attributed to a workflow."""

    suffix: str  # ``""`` = latest trio (`data.json`, etc.)
    workflow_id: str
    sort_key: str  # ISO suffix or ``latest`` for ordering
    uncertainty_trend_y: Optional[float] = None  # ``transformed_stddevs_avg`` peeked at index time
    has_surrogate_archive: bool = False
    has_next_x_archive: bool = False


@dataclass
class NSDFTripletIndex:
    """Workflow-scoped snapshot catalog built once per Reload."""

    snapshots: List[NSDFSnapshotRef] = field(default_factory=list)
    by_workflow: Dict[str, List[NSDFSnapshotRef]] = field(default_factory=dict)

    def has_workflow(self, workflow_id: str) -> bool:
        key = (workflow_id or "").strip()
        return bool(key) and key in self.by_workflow

    def workflow_ids_newest_first(self) -> List[str]:
        """Workflow ids ordered by each workflow's newest snapshot."""
        ranked: List[Tuple[str, str]] = []
        for workflow_id, snaps in self.by_workflow.items():
            if not snaps:
                continue
            ranked.append((snaps[0].sort_key, workflow_id))
        ranked.sort(reverse=True)
        return [workflow_id for _sort, workflow_id in ranked]

    def workflow_select_options(self) -> List[Tuple[str, str]]:
        options: List[Tuple[str, str]] = []
        for workflow_id in self.workflow_ids_newest_first():
            count = len(self.by_workflow.get(workflow_id) or [])
            label = format_nsdf_workflow_select_label(workflow_id)
            if count > 1:
                label = f"{label} ({count} snapshots)"
            options.append((workflow_id, label))
        return options

    def snapshots_for_workflow(self, workflow_id: str) -> List[NSDFSnapshotRef]:
        return list(self.by_workflow.get((workflow_id or "").strip()) or [])

    def snapshot_select_options(self, workflow_id: str) -> List[Tuple[str, str]]:
        return self._suffix_select_options(
            self.snapshots_for_workflow(workflow_id),
            latest_label="Latest (data.json)",
        )

    def surrogate_select_options(self, workflow_id: str) -> List[Tuple[str, str]]:
        snaps = [
            snap
            for snap in self.snapshots_for_workflow(workflow_id)
            if not snap.suffix or snap.has_surrogate_archive
        ]
        return self._suffix_select_options(
            snaps,
            latest_label="Latest (surrogate.json)",
        )

    def next_x_select_options(self, workflow_id: str) -> List[Tuple[str, str]]:
        snaps = [
            snap
            for snap in self.snapshots_for_workflow(workflow_id)
            if not snap.suffix or snap.has_next_x_archive
        ]
        return self._suffix_select_options(
            snaps,
            latest_label="Latest (next_x.json)",
        )

    def _suffix_select_options(
        self,
        snaps: Sequence[NSDFSnapshotRef],
        *,
        latest_label: str,
    ) -> List[Tuple[str, str]]:
        options: List[Tuple[str, str]] = []
        for snap in snaps:
            value = "latest" if not snap.suffix else snap.suffix
            label = latest_label if value == "latest" else snap.suffix
            options.append((value, label))
        return options

    def newest_workflow_id(self) -> str:
        ordered = self.workflow_ids_newest_first()
        return ordered[0] if ordered else NSDF_UNKNOWN_WORKFLOW_ID

    def default_snapshot_value(self, workflow_id: str) -> str:
        snaps = self.snapshots_for_workflow(workflow_id)
        if not snaps:
            return "latest"
        return "latest" if not snaps[0].suffix else snaps[0].suffix


@dataclass
class NSDFTripletIndexDiscoverResult:
    """Outcome of ``discover_nsdf_triplet_index`` (catalog read, scan, and optional write)."""

    index: NSDFTripletIndex
    source: str = "empty"  # ``catalog_json``, ``full_scan``, or ``empty``
    catalog_written: bool = False
    catalog_write_location: str = ""
    catalog_write_error: str = ""


def format_nsdf_workflow_select_label(workflow_id: str) -> str:
    if workflow_id == NSDF_UNKNOWN_WORKFLOW_ID:
        return "(unknown)"
    return workflow_id


def workflow_id_from_dataset_doc(doc: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Best-effort workflow id stored on a ScientistCloud dataset record."""
    if not isinstance(doc, Mapping):
        return None
    for key in ("workflow_id", "nsdf_workflow_id"):
        value = str(doc.get(key) or "").strip()
        if value:
            return value
    metadata = doc.get("metadata")
    if isinstance(metadata, Mapping):
        value = str(metadata.get("workflow_id") or "").strip()
        if value:
            return value
    return None


def resolve_default_workflow_selection(
    index: NSDFTripletIndex,
    *,
    dataset_workflow_id: Optional[str] = None,
    query_workflow_id: Optional[str] = None,
) -> str:
    """Prefer URL param, then dataset record, then globally newest snapshot."""
    for candidate in (query_workflow_id, dataset_workflow_id):
        if candidate and index.has_workflow(candidate):
            return candidate
    return index.newest_workflow_id()


def active_workflow_id_from_grid_state(workflow_id: str) -> Optional[str]:
    """Map selector value to plot/filter workflow id (``None`` for unknown)."""
    key = (workflow_id or "").strip()
    if not key or key == NSDF_UNKNOWN_WORKFLOW_ID:
        return None
    return key


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def _looks_like_http_url(value: str) -> bool:
    v = (value or "").strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _object_key_needs_nsdf_data_json(key: str) -> bool:
    """True when an S3 object key is a folder prefix, not an explicit JSON object."""
    normalized = (key or "").strip().strip("/")
    if not normalized:
        return True
    leaf = normalized.rsplit("/", 1)[-1].lower()
    if leaf == NSDF_DATA_JSON_BASENAME:
        return False
    if leaf.endswith(".json"):
        return False
    return True


def normalize_nsdf_data_object_key(key: str) -> str:
    """Append ``data.json`` when *key* names an NSDF folder prefix (direct S3 read, no mirror)."""
    raw = (key or "").strip().lstrip("/")
    if not _object_key_needs_nsdf_data_json(raw):
        return raw
    prefix = raw.rstrip("/")
    return f"{prefix}/{NSDF_DATA_JSON_BASENAME}" if prefix else NSDF_DATA_JSON_BASENAME


def normalize_nsdf_gateway_data_url(url: str) -> str:
    """
    When a gateway HTTPS URL points at an S3 prefix (e.g. ``…/test-chess/``), rewrite it to
    ``…/test-chess/data.json`` so SigV4 ``GetObject`` reads the NSDF file directly.
    """
    from urllib.parse import urlunparse, urlparse

    u = (url or "").strip()
    if not _looks_like_http_url(u):
        return u

    parts = urlparse(u)
    segments = [x for x in (parts.path or "").split("/") if x]
    if len(segments) < 2:
        return u

    bucket = segments[0]
    key = "/".join(segments[1:])
    new_key = normalize_nsdf_data_object_key(key)
    if new_key == key:
        return u

    new_path = "/" + "/".join([bucket, *new_key.split("/")])
    return urlunparse((parts.scheme, parts.netloc, new_path, parts.params, parts.query, parts.fragment))


def normalize_nsdf_remote_data_link(link: str) -> str:
    """Normalize portal/S3 remote links so strain JSON loads via direct object read."""
    u = (link or "").strip()
    if not u:
        return u
    low = u.lower()
    if low.startswith("s3://"):
        body = u[5:].strip()
        slash = body.find("/")
        if slash == -1:
            return u
        bucket = body[:slash]
        key = normalize_nsdf_data_object_key(body[slash + 1 :])
        return f"s3://{bucket}/{key}"
    if _looks_like_http_url(u):
        return normalize_nsdf_gateway_data_url(u)
    return u


def parse_s3_uri(link: str) -> Tuple[str, str]:
    """Return ``(bucket, object_key)`` from an ``s3://`` URI (key may be empty)."""
    u = normalize_nsdf_remote_data_link((link or "").strip())
    if not u.lower().startswith("s3://"):
        return "", ""
    body = u[5:].strip()
    slash = body.find("/")
    if slash == -1:
        return body, ""
    return body[:slash], body[slash + 1 :].lstrip("/")


def nsdf_triplet_basenames(version_suffix: str = "") -> Tuple[str, str, str]:
    """
    Return ``(data, surrogate, next_x)`` basenames for a version suffix.

    Empty suffix = live rolling trio (``data.json``, etc.). Non-empty suffix = archived
    backup files (``data_<timestamp>_<id>.json``, etc.).
    """
    suffix = (version_suffix or "").strip()
    if not suffix:
        return "data.json", "surrogate.json", "next_x.json"
    if not is_valid_nsdf_snapshot_suffix(suffix):
        raise ValueError(f"Invalid NSDF version suffix: {suffix!r}")
    return (
        f"data_{suffix}.json",
        f"surrogate_{suffix}.json",
        f"next_x_{suffix}.json",
    )


def parse_nsdf_data_filename(name: str) -> Optional[str]:
    """Parse ``data.json`` or ``data_<suffix>.json``; return suffix or ``None`` for latest."""
    m = NSDF_DATA_FILENAME_RE.match((name or "").strip())
    if not m:
        return None
    suffix = (m.group("suffix") or "").strip()
    if not suffix:
        return None
    if not is_valid_nsdf_snapshot_suffix(suffix):
        return None
    return suffix


def _parse_nsdf_triplet_filename_suffix(name: str, prefix: str) -> Optional[str]:
    """Parse ``<prefix>.json`` or ``<prefix>_<suffix>.json``; ``None`` means latest."""
    base = os.path.basename((name or "").strip())
    latest_name = f"{prefix}.json"
    if base.lower() == latest_name.lower():
        return None
    token = f"{prefix}_"
    if not (base.startswith(token) and base.endswith(".json")):
        return None
    middle = base[len(token) : -len(".json")]
    if not middle or not is_valid_nsdf_snapshot_suffix(middle):
        return None
    return middle


def _replace_path_basename(path: str, new_basename: str) -> str:
    p = (path or "").strip()
    if not p:
        return p
    return os.path.join(os.path.dirname(p), new_basename)


def _replace_s3_key_basename(key: str, new_basename: str) -> str:
    k = (key or "").strip()
    if not k:
        return k
    if "/" in k:
        return k.rsplit("/", 1)[0] + "/" + new_basename
    return new_basename


def _replace_url_basename(url: str, new_basename: str) -> str:
    from urllib.parse import urlparse, urlunparse

    u = (url or "").strip()
    if not _looks_like_http_url(u):
        return u
    parts = urlparse(u)
    segments = [x for x in (parts.path or "").split("/") if x]
    if not segments:
        return u
    if parse_nsdf_data_filename(segments[-1]) is None and segments[-1].lower() != "data.json":
        return u
    segments[-1] = new_basename
    new_path = "/" + "/".join(segments)
    return urlunparse((parts.scheme, parts.netloc, new_path, parts.params, parts.query, parts.fragment))


def _nsdf_selector_value_to_suffix(value: str) -> str:
    """Map UI selector value ``latest`` to an empty version suffix."""
    key = (value or "").strip()
    return "" if key in ("", "latest") else key


def _nsdf_suffix_to_selector_value(suffix: str) -> str:
    return "latest" if not (suffix or "").strip() else (suffix or "").strip()


def _parse_nsdf_iso_suffix_timestamp(suffix: str) -> Optional["datetime.datetime"]:
    from datetime import datetime

    key = (suffix or "").strip().upper()
    if not NSDF_VERSION_SUFFIX_RE.fullmatch(key):
        return None
    try:
        return datetime.strptime(key, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None


def _nearest_nsdf_version_suffix(
    target_suffix: str,
    available_suffixes: Sequence[str],
) -> Optional[str]:
    """Pick the closest ISO timestamp suffix to ``target_suffix``."""
    target_dt = _parse_nsdf_iso_suffix_timestamp(target_suffix)
    if target_dt is None:
        return None

    best: Optional[str] = None
    best_delta: Optional[float] = None
    for suffix in available_suffixes:
        candidate_dt = _parse_nsdf_iso_suffix_timestamp(suffix)
        if candidate_dt is None:
            continue
        delta = abs((candidate_dt - target_dt).total_seconds())
        if best is None or delta < best_delta or (
            delta == best_delta and suffix > (best or "")
        ):
            best = suffix
            best_delta = delta
    return best


def resolve_auxiliary_suffix_for_data_snapshot(
    data_selector_value: str,
    available_suffixes: Sequence[str],
) -> str:
    """
    Return a surrogate/next_x selector value aligned with a data snapshot.

    Matches exact suffix first, then any available suffix with the same snapshot id
    (trailing ``_<id>`` token). Falls back to ``latest`` when no id match exists.
    Timestamps in the suffix are not compared.
    """
    data_suffix = _nsdf_selector_value_to_suffix(data_selector_value)
    if not data_suffix:
        return "latest"

    available = sorted({(suffix or "").strip() for suffix in available_suffixes if (suffix or "").strip()})
    if not available:
        return "latest"
    if data_suffix in available:
        return data_suffix

    data_id = triplet_snapshot_id(data_suffix)
    if data_id:
        id_matches = [
            suffix for suffix in available if triplet_snapshot_id(suffix) == data_id
        ]
        if id_matches:
            return sorted(id_matches, key=_snapshot_sort_key, reverse=True)[0]
    return "latest"


def apply_nsdf_file_suffixes(
    paths: StrainDashboardPaths,
    *,
    data_suffix: str = "",
    surrogate_suffix: Optional[str] = None,
    next_x_suffix: Optional[str] = None,
    strict: bool = False,
) -> StrainDashboardPaths:
    """
    Point resolved paths at independent ``data`` / ``surrogate`` / ``next_x`` files.

    Empty / ``latest`` suffix selects the rolling live trio (``data.json``,
    ``surrogate.json``, ``next_x.json``). Non-empty suffix selects archived backups.

    When ``surrogate_suffix`` or ``next_x_suffix`` is omitted, they follow ``data_suffix``.
    """
    data_s = _nsdf_selector_value_to_suffix(data_suffix)
    sur_s = data_s if surrogate_suffix is None else _nsdf_selector_value_to_suffix(surrogate_suffix)
    nx_s = data_s if next_x_suffix is None else _nsdf_selector_value_to_suffix(next_x_suffix)
    data_fn, _, _ = nsdf_triplet_basenames(data_s)
    _, sur_fn, _ = nsdf_triplet_basenames(sur_s)
    _, _, nx_fn = nsdf_triplet_basenames(nx_s)
    out = StrainDashboardPaths(
        local_json_path=paths.local_json_path,
        json_url=paths.json_url,
        surrogate_json_path=paths.surrogate_json_path,
        surrogate_json_url=paths.surrogate_json_url,
        next_x_json_path=paths.next_x_json_path,
        next_x_json_url=paths.next_x_json_url,
        local_data_dir=paths.local_data_dir,
        s3_env_file=paths.s3_env_file,
        s3_bucket=paths.s3_bucket,
        s3_data_key=paths.s3_data_key,
        s3_surrogate_key=paths.s3_surrogate_key,
        s3_next_x_key=paths.s3_next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        version_suffix=data_s,
        surrogate_version_suffix=sur_s,
        next_x_version_suffix=nx_s,
        strict_triplet_paths=strict,
    )

    if out.local_data_dir and local_files_first_for_testing():
        base = out.local_data_dir.rstrip("/")
        out.local_json_path = os.path.join(base, data_fn)
        out.surrogate_json_path = os.path.join(base, sur_fn)
        out.next_x_json_path = os.path.join(base, nx_fn)
        return out

    if out.local_json_path and not _looks_like_http_url(out.local_json_path):
        base = os.path.dirname(out.local_json_path)
        out.local_json_path = os.path.join(base, data_fn)
        out.surrogate_json_path = os.path.join(base, sur_fn)
        out.next_x_json_path = os.path.join(base, nx_fn)
        return out

    if out.json_url:
        out.json_url = _replace_url_basename(out.json_url, data_fn)
        out.surrogate_json_url = _replace_url_basename(out.json_url, sur_fn)
        out.next_x_json_url = _replace_url_basename(out.json_url, nx_fn)

    if out.s3_data_key:
        out.s3_data_key = _replace_s3_key_basename(out.s3_data_key, data_fn)
        explicit_sur = (paths.s3_surrogate_key or "").strip()
        explicit_nx = (paths.s3_next_x_key or "").strip()
        _, expected_sur_fn, _ = nsdf_triplet_basenames(sur_s)
        _, _, expected_nx_fn = nsdf_triplet_basenames(nx_s)
        data_prefix = _nsdf_key_prefix(out.s3_data_key)
        if explicit_sur and _location_basename(explicit_sur) == expected_sur_fn:
            out.s3_surrogate_key = explicit_sur
        else:
            out.s3_surrogate_key = data_prefix + expected_sur_fn
        if explicit_nx and _location_basename(explicit_nx) == expected_nx_fn:
            out.s3_next_x_key = explicit_nx
        elif (
            explicit_sur
            and "/" in explicit_sur
            and not explicit_nx
            and sur_s == data_s
            and _location_basename(explicit_sur) == expected_sur_fn
        ):
            sur_prefix = explicit_sur.rsplit("/", 1)[0] + "/"
            out.s3_next_x_key = sur_prefix + expected_nx_fn
        else:
            out.s3_next_x_key = data_prefix + expected_nx_fn

    return out


def apply_nsdf_version_suffix(
    paths: StrainDashboardPaths,
    version_suffix: str = "",
) -> StrainDashboardPaths:
    """
    Point resolved paths at a timestamped triplet (``data_<ts>.json``, etc.).

    Empty suffix keeps the default latest files (``data.json``, ``surrogate.json``, ``next_x.json``).
    """
    return apply_nsdf_file_suffixes(paths, data_suffix=version_suffix, strict=False)


def nsdf_listing_directory(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    remote_linked: bool = False,
) -> str:
    """Best-effort local directory for discovering timestamped NSDF JSON backups."""
    if remote_linked:
        return ""
    candidates: List[str] = []
    if (paths.local_data_dir or "").strip() and local_files_first_for_testing():
        candidates.append((paths.local_data_dir or "").strip())
    loc = (paths.local_json_path or "").strip()
    if loc and not _looks_like_http_url(loc):
        candidates.append(os.path.dirname(loc))
    for dataset_dir in ((base_dir or "").strip(), (save_dir or "").strip()):
        if not dataset_dir:
            continue
        found = find_strain_json_under_dataset_dir(dataset_dir)
        if found:
            candidates.append(os.path.dirname(found))
        elif os.path.isdir(dataset_dir):
            candidates.append(dataset_dir)
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return ""


def _local_data_dir_active(paths: StrainDashboardPaths) -> bool:
    """True when ``LOCAL_DATA_DIR`` is the exclusive data source (testing mode)."""
    local_dir = (paths.local_data_dir or "").strip()
    return bool(local_dir and os.path.isdir(local_dir) and local_files_first_for_testing())


def _strip_local_data_dir_paths(paths: StrainDashboardPaths) -> StrainDashboardPaths:
    """Remove ``LOCAL_DATA_DIR`` routing unless testing mode is enabled."""
    if local_files_first_for_testing():
        return paths
    local_dir = os.path.abspath((paths.local_data_dir or "").strip())
    if not local_dir:
        return paths

    def under_local(p: str) -> bool:
        p = (p or "").strip()
        if not p or _looks_like_http_url(p):
            return False
        try:
            return os.path.commonpath([local_dir, os.path.abspath(p)]) == local_dir
        except ValueError:
            return False

    return StrainDashboardPaths(
        local_json_path="" if under_local(paths.local_json_path) else paths.local_json_path,
        json_url=paths.json_url,
        surrogate_json_path="" if under_local(paths.surrogate_json_path) else paths.surrogate_json_path,
        surrogate_json_url=paths.surrogate_json_url,
        next_x_json_path="" if under_local(paths.next_x_json_path) else paths.next_x_json_path,
        next_x_json_url=paths.next_x_json_url,
        local_data_dir="",
        s3_env_file=paths.s3_env_file,
        s3_bucket=paths.s3_bucket,
        s3_data_key=paths.s3_data_key,
        s3_surrogate_key=paths.s3_surrogate_key,
        s3_next_x_key=paths.s3_next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        version_suffix=paths.version_suffix,
        surrogate_version_suffix=paths.surrogate_version_suffix,
        next_x_version_suffix=paths.next_x_version_suffix,
        strict_triplet_paths=paths.strict_triplet_paths,
    )


def _remote_snapshot_listing_enabled(paths: StrainDashboardPaths) -> bool:
    """True when snapshots should be discovered from S3 / gateway URL."""
    if _local_data_dir_active(paths):
        return False
    if paths.has_s3_source():
        return True
    return bool((paths.json_url or "").strip())


def list_nsdf_version_suffixes_from_directory(directory: str) -> List[str]:
    """List timestamp suffixes from ``data_<suffix>.json`` files (newest first)."""
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return []
    suffixes: List[str] = []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for name in names:
        parsed = parse_nsdf_data_filename(name)
        if parsed:
            suffixes.append(parsed)
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def list_nsdf_surrogate_suffixes_from_directory(directory: str) -> List[str]:
    """List timestamp suffixes from ``surrogate_<suffix>.json`` files (newest first)."""
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return []
    suffixes: List[str] = []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for name in names:
        if not (name.startswith("surrogate_") and name.endswith(".json")):
            continue
        parsed = parse_nsdf_surrogate_filename(name)
        if parsed:
            suffixes.append(parsed)
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def list_nsdf_next_x_suffixes_from_directory(directory: str) -> List[str]:
    """List timestamp suffixes from ``next_x_<suffix>.json`` files (newest first)."""
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return []
    suffixes: List[str] = []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for name in names:
        parsed = parse_nsdf_next_x_filename(name)
        if parsed:
            suffixes.append(parsed)
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def list_nsdf_version_suffixes_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """List timestamp suffixes under the resolved S3 prefix (newest first)."""
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
            data_key = (cfg.get("key") or "").strip()
    if not bucket:
        return []

    prefix = ""
    if data_key:
        if "/" in data_key:
            prefix = data_key.rsplit("/", 1)[0] + "/"
        elif parse_nsdf_data_filename(data_key) is None:
            prefix = data_key.rstrip("/") + "/"

    list_paths = StrainDashboardPaths(
        s3_bucket=bucket,
        s3_data_key=data_key or "data.json",
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        s3_env_file=paths.s3_env_file,
        json_url=paths.json_url,
    )
    try:
        client = _make_nsdf_s3_client(list_paths, mongo_s3_auth=mongo_s3_auth)
    except Exception:
        return []

    suffixes: List[str] = []
    continuation: Optional[str] = None
    try:
        while True:
            params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation:
                params["ContinuationToken"] = continuation
            result = client.list_objects_v2(**params)
            for obj in result.get("Contents") or []:
                name = os.path.basename(str(obj.get("Key") or ""))
                parsed = parse_nsdf_data_filename(name)
                if parsed:
                    suffixes.append(parsed)
            if not result.get("IsTruncated"):
                break
            continuation = result.get("NextContinuationToken")
            if not continuation:
                break
    except Exception:
        return []
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def parse_nsdf_surrogate_filename(name: str) -> Optional[str]:
    """Parse ``surrogate_<suffix>.json``; return suffix or ``None`` for ``surrogate.json``."""
    return _parse_nsdf_triplet_filename_suffix(name, "surrogate")


def parse_nsdf_next_x_filename(name: str) -> Optional[str]:
    """Parse ``next_x_<suffix>.json``; return suffix or ``None`` for ``next_x.json``."""
    return _parse_nsdf_triplet_filename_suffix(name, "next_x")


def _nsdf_reference_data_suffix(
    data_name: str,
    *,
    data_suffixes: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Timestamp floor for auxiliary fallbacks.

    For ``data_<ts>.json`` returns ``ts``. For latest ``data.json`` uses the newest
    ``data_<ts>.json`` suffix in the directory when present.
    """
    parsed = parse_nsdf_data_filename(os.path.basename((data_name or "").strip()))
    if parsed:
        return parsed
    if data_suffixes:
        ordered = sorted(set(data_suffixes), key=_snapshot_sort_key, reverse=True)
        if ordered:
            return ordered[0]
    return None


def _nsdf_suffixes_after_reference(
    suffixes: Sequence[str],
    reference_suffix: Optional[str],
) -> List[str]:
    """Newest-first auxiliary suffixes strictly after ``reference_suffix``."""
    ordered = sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)
    if not reference_suffix:
        return ordered
    ref_key = _snapshot_sort_key(reference_suffix.strip())
    return [suffix for suffix in ordered if _snapshot_sort_key(suffix) > ref_key]


def list_nsdf_surrogate_suffixes_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """List timestamp suffixes from ``surrogate_<suffix>.json`` under the S3 prefix."""
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
            data_key = (cfg.get("key") or "").strip()
    if not bucket:
        return []

    prefix = ""
    if data_key:
        if "/" in data_key:
            prefix = data_key.rsplit("/", 1)[0] + "/"
        elif parse_nsdf_data_filename(data_key) is None:
            prefix = data_key.rstrip("/") + "/"

    list_paths = StrainDashboardPaths(
        s3_bucket=bucket,
        s3_data_key=data_key or "data.json",
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        s3_env_file=paths.s3_env_file,
        json_url=paths.json_url,
    )
    try:
        client = _make_nsdf_s3_client(list_paths, mongo_s3_auth=mongo_s3_auth)
    except Exception:
        return []

    suffixes: List[str] = []
    continuation: Optional[str] = None
    try:
        while True:
            params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation:
                params["ContinuationToken"] = continuation
            result = client.list_objects_v2(**params)
            for obj in result.get("Contents") or []:
                name = os.path.basename(str(obj.get("Key") or ""))
                parsed = parse_nsdf_surrogate_filename(name)
                if parsed:
                    suffixes.append(parsed)
            if not result.get("IsTruncated"):
                break
            continuation = result.get("NextContinuationToken")
            if not continuation:
                break
    except Exception:
        return []
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def list_nsdf_next_x_suffixes_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """List timestamp suffixes from ``next_x_<suffix>.json`` under the S3 prefix."""
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
            data_key = (cfg.get("key") or "").strip()
    if not bucket:
        return []

    prefix = ""
    if data_key:
        if "/" in data_key:
            prefix = data_key.rsplit("/", 1)[0] + "/"
        elif parse_nsdf_data_filename(data_key) is None:
            prefix = data_key.rstrip("/") + "/"

    list_paths = StrainDashboardPaths(
        s3_bucket=bucket,
        s3_data_key=data_key or "data.json",
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        s3_env_file=paths.s3_env_file,
        json_url=paths.json_url,
    )
    try:
        client = _make_nsdf_s3_client(list_paths, mongo_s3_auth=mongo_s3_auth)
    except Exception:
        return []

    suffixes: List[str] = []
    continuation: Optional[str] = None
    try:
        while True:
            params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation:
                params["ContinuationToken"] = continuation
            result = client.list_objects_v2(**params)
            for obj in result.get("Contents") or []:
                name = os.path.basename(str(obj.get("Key") or ""))
                parsed = parse_nsdf_next_x_filename(name)
                if parsed:
                    suffixes.append(parsed)
            if not result.get("IsTruncated"):
                break
            continuation = result.get("NextContinuationToken")
            if not continuation:
                break
    except Exception:
        return []
    return sorted(set(suffixes), key=_snapshot_sort_key, reverse=True)


def _merged_snapshot_suffixes_from_directory(directory: str) -> List[str]:
    return sorted(
        set(list_nsdf_version_suffixes_from_directory(directory))
        | set(list_nsdf_surrogate_suffixes_from_directory(directory)),
        reverse=True,
    )


def _merged_snapshot_suffixes_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    return sorted(
        set(list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth))
        | set(list_nsdf_surrogate_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)),
        reverse=True,
    )


def discover_nsdf_version_options(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> List[Tuple[str, str]]:
    """
    Return ``(value, label)`` pairs for a version selector.

    ``latest`` is always first; values are ISO-like suffixes such as ``20260606T223505Z``.

    With ``LOCAL_FILES_FIRST_FOR_TESTING=1``, ``LOCAL_DATA_DIR`` runs list only that
    folder. Otherwise S3 / gateway runs list only the resolved remote prefix. The two
    sources are never merged.
    """
    options: List[Tuple[str, str]] = [("latest", "Latest (data.json)")]
    seen: set[str] = set()

    if not remote_linked:
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        for suffix in list_nsdf_version_suffixes_from_directory(local_dir):
            if suffix not in seen:
                seen.add(suffix)
                options.append((suffix, suffix))

    if not _remote_snapshot_listing_enabled(paths):
        return options

    for suffix in list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth):
        if suffix not in seen:
            seen.add(suffix)
            options.append((suffix, suffix))

    return options


def discover_nsdf_surrogate_version_options(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> List[Tuple[str, str]]:
    """Return ``(value, label)`` pairs for a surrogate.json version selector."""
    options: List[Tuple[str, str]] = [("latest", "Latest (surrogate.json)")]
    seen: set[str] = set()

    if not remote_linked:
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        for suffix in list_nsdf_surrogate_suffixes_from_directory(local_dir):
            if suffix not in seen:
                seen.add(suffix)
                options.append((suffix, suffix))

    if _remote_snapshot_listing_enabled(paths):
        for suffix in list_nsdf_surrogate_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth):
            if suffix not in seen:
                seen.add(suffix)
                options.append((suffix, suffix))

    return options


def discover_nsdf_next_x_version_options(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> List[Tuple[str, str]]:
    """Return ``(value, label)`` pairs for a next_x.json version selector."""
    options: List[Tuple[str, str]] = [("latest", "Latest (next_x.json)")]
    seen: set[str] = set()

    if not remote_linked:
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        for suffix in list_nsdf_next_x_suffixes_from_directory(local_dir):
            if suffix not in seen:
                seen.add(suffix)
                options.append((suffix, suffix))

    if _remote_snapshot_listing_enabled(paths):
        for suffix in list_nsdf_next_x_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth):
            if suffix not in seen:
                seen.add(suffix)
                options.append((suffix, suffix))

    return options


def _suffix_from_data_basename(name: str) -> Optional[str]:
    base = os.path.basename((name or "").strip())
    if base == "data.json":
        return ""
    parsed = parse_nsdf_data_filename(base)
    return parsed if parsed else None


def _suffix_from_surrogate_basename(name: str) -> Optional[str]:
    base = os.path.basename((name or "").strip())
    if base == "surrogate.json":
        return ""
    return parse_nsdf_surrogate_filename(base)


def _suffix_from_next_x_basename(name: str) -> Optional[str]:
    base = os.path.basename((name or "").strip())
    if base == "next_x.json":
        return ""
    return parse_nsdf_next_x_filename(base)


def _snapshot_sort_key(suffix: str) -> str:
    """Sort newest-first: ISO timestamps, then numeric ids, then other ids; latest trio on top."""
    s = (suffix or "").strip()
    if not s:
        return "z_latest"
    if NSDF_VERSION_SUFFIX_RE.fullmatch(s):
        return f"a_ts_{s.upper()}"
    if s.isdigit():
        return f"b_num_{int(s):020d}"
    return f"c_id_{s.upper()}"


def _peek_workflow_id_from_json_doc(doc: Any) -> Optional[str]:
    if not isinstance(doc, Mapping):
        return None
    value = doc.get("workflow_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _peek_workflow_id_from_next_x_doc(doc: Any) -> Optional[str]:
    if isinstance(doc, Mapping):
        value = str(doc.get("workflow_id") or "").strip()
        return value or None
    if not isinstance(doc, list):
        return None
    found: set[str] = set()
    for item in doc:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get("workflow_id") or "").strip()
        if value:
            found.add(value)
    if len(found) == 1:
        return next(iter(found))
    return None


def _next_x_doc_is_recognized_format(doc: Any) -> bool:
    """True for the current object schema or the legacy array-of-blocks schema."""
    if isinstance(doc, Mapping):
        return True
    return isinstance(doc, list)


def _parse_next_x_coordinate_row(row_value: Any) -> Optional[Tuple[float, float]]:
    """
    Extract ``(labx, labz)`` from one ``next_x`` row.

    The pipeline stores a single scan location as ``[labx, labz]`` even when
    ``dataset_x_size`` describes the full GP input dimension (e.g. 16), not
    the row width.
    """
    if not isinstance(row_value, list) or len(row_value) < 2:
        return None
    if not (_is_number(row_value[0]) and _is_number(row_value[1])):
        return None
    labx, labz = float(row_value[0]), float(row_value[1])
    if not (math.isfinite(labx) and math.isfinite(labz)):
        return None
    return labx, labz


def _iter_next_x_workflow_blocks(doc: Any) -> List[Mapping[str, Any]]:
    """Normalize ``next_x.json`` to workflow blocks (object or legacy array)."""
    if isinstance(doc, Mapping):
        return [doc]
    if isinstance(doc, list):
        return [item for item in doc if isinstance(item, Mapping)]
    return []


def _parse_next_x_workflow_block(
    item: Mapping[str, Any],
    *,
    label: str,
    warnings: List[str],
) -> Optional[NSDFNextXEntry]:
    workflow_id = str(item.get("workflow_id") or "").strip()
    if not workflow_id:
        warnings.append(f"Skipping {label}: missing workflow_id.")
        return None
    data = item.get("data")
    if not isinstance(data, list) or not data:
        warnings.append(f"Skipping {label} ({workflow_id!r}): data must be a non-empty list.")
        return None
    coords: List[Tuple[float, float]] = []
    for j, row_value in enumerate(data):
        parsed = _parse_next_x_coordinate_row(row_value)
        if parsed is None:
            warnings.append(
                f"Skipping {label} ({workflow_id!r}): data[{j}] must contain at least "
                "two numeric labx/labz values."
            )
            return None
        coords.append(parsed)
    return NSDFNextXEntry(
        workflow_id=workflow_id,
        coordinates=np.asarray(coords, dtype=np.float64),
    )


def _collect_auxiliary_workflow_maps(
    groups: Dict[str, Dict[str, str]],
    read_doc: Any,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map timestamp suffix -> workflow_id from surrogate and next_x files."""
    surrogate_wf: Dict[str, str] = {}
    next_x_wf: Dict[str, str] = {}
    for suffix, files in groups.items():
        sur_path = (files.get("surrogate") or "").strip()
        if sur_path:
            wf = _peek_workflow_id_from_json_doc(read_doc(sur_path))
            if wf:
                surrogate_wf[suffix] = wf
        nx_path = (files.get("next_x") or "").strip()
        if nx_path:
            wf = _peek_workflow_id_from_next_x_doc(read_doc(nx_path))
            if wf:
                next_x_wf[suffix] = wf
    return surrogate_wf, next_x_wf


def _workflow_id_from_auxiliary_maps(
    data_suffix: str,
    surrogate_wf: Mapping[str, str],
    next_x_wf: Mapping[str, str],
) -> Optional[str]:
    """Resolve workflow_id from surrogate/next_x maps by exact suffix or shared snapshot id."""
    key = (data_suffix or "").strip()
    if not key:
        return surrogate_wf.get("") or next_x_wf.get("")

    if key in surrogate_wf:
        return surrogate_wf[key]
    if key in next_x_wf:
        return next_x_wf[key]

    data_id = triplet_snapshot_id(key)
    if not data_id:
        return None

    for wf_map in (surrogate_wf, next_x_wf):
        for suffix, workflow_id in wf_map.items():
            if triplet_snapshot_id(suffix) == data_id:
                return workflow_id
    return None


def _attribute_workflow_for_data_snapshot(
    *,
    data_suffix: str,
    data_doc: Any = None,
    surrogate_doc: Any = None,
    next_x_doc: Any = None,
    surrogate_wf_by_suffix: Optional[Mapping[str, str]] = None,
    next_x_wf_by_suffix: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Resolve workflow_id for a ``data.json`` snapshot.

    Order: explicit ``data.json`` id (new files), linked surrogate/next_x on the same
    snapshot id, then exact suffix in auxiliary maps.
    """
    workflow_id = _peek_workflow_id_from_json_doc(data_doc)
    if workflow_id:
        return workflow_id
    workflow_id = _peek_workflow_id_from_json_doc(surrogate_doc)
    if workflow_id:
        return workflow_id
    workflow_id = _peek_workflow_id_from_next_x_doc(next_x_doc)
    if workflow_id:
        return workflow_id

    surrogate_wf = surrogate_wf_by_suffix or {}
    next_x_wf = next_x_wf_by_suffix or {}
    matched = _workflow_id_from_auxiliary_maps(data_suffix, surrogate_wf, next_x_wf)
    if matched:
        return matched
    return NSDF_UNKNOWN_WORKFLOW_ID


def _attribute_workflow_to_triplet_files(
    *,
    surrogate_doc: Any = None,
    data_doc: Any = None,
    next_x_doc: Any = None,
) -> str:
    return _attribute_workflow_for_data_snapshot(
        data_suffix="",
        data_doc=data_doc,
        surrogate_doc=surrogate_doc,
        next_x_doc=next_x_doc,
    )


def _build_triplet_index_from_groups(
    groups: Dict[str, Dict[str, str]],
    *,
    read_doc: Any,
) -> NSDFTripletIndex:
    surrogate_wf_by_suffix, next_x_wf_by_suffix = _collect_auxiliary_workflow_maps(
        groups,
        read_doc,
    )
    snapshots: List[NSDFSnapshotRef] = []
    by_workflow: Dict[str, List[NSDFSnapshotRef]] = {}
    for suffix in sorted(groups.keys(), key=_snapshot_sort_key, reverse=True):
        if suffix and not is_valid_nsdf_snapshot_suffix(suffix):
            continue
        files = groups[suffix]
        data_loc = (files.get("data") or "").strip()
        if not data_loc:
            continue
        surrogate_doc = read_doc((files.get("surrogate") or "").strip())
        data_doc = read_doc(data_loc)
        next_x_doc = read_doc((files.get("next_x") or "").strip())
        workflow_id = _attribute_workflow_for_data_snapshot(
            data_suffix=suffix,
            data_doc=data_doc,
            surrogate_doc=surrogate_doc,
            next_x_doc=next_x_doc,
            surrogate_wf_by_suffix=surrogate_wf_by_suffix,
            next_x_wf_by_suffix=next_x_wf_by_suffix,
        )
        ref = NSDFSnapshotRef(
            suffix=suffix,
            workflow_id=workflow_id,
            sort_key=_snapshot_sort_key(suffix),
            uncertainty_trend_y=_peek_transformed_stddevs_avg_scalar(surrogate_doc),
            has_surrogate_archive=bool((files.get("surrogate") or "").strip()),
            has_next_x_archive=bool((files.get("next_x") or "").strip()),
        )
        snapshots.append(ref)
        by_workflow.setdefault(workflow_id, []).append(ref)
    for workflow_id in by_workflow:
        by_workflow[workflow_id].sort(key=lambda snap: snap.sort_key, reverse=True)
    return NSDFTripletIndex(snapshots=snapshots, by_workflow=by_workflow)


def _triplet_groups_from_object_keys(keys: Sequence[str]) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    id_to_data_suffix: Dict[str, str] = {}

    for key in keys:
        name = os.path.basename((key or "").strip())
        if not name:
            continue
        data_suffix = _suffix_from_data_basename(name)
        if data_suffix is None:
            continue
        groups.setdefault(data_suffix, {})["data"] = key
        snapshot_id = triplet_snapshot_id(data_suffix)
        if not snapshot_id:
            continue
        previous = id_to_data_suffix.get(snapshot_id)
        if not previous or _snapshot_sort_key(data_suffix) > _snapshot_sort_key(previous):
            id_to_data_suffix[snapshot_id] = data_suffix

    def _attach_auxiliary(role: str, suffix: Optional[str], key: str) -> None:
        if suffix is None:
            return
        snapshot_id = triplet_snapshot_id(suffix)
        target_suffix = id_to_data_suffix.get(snapshot_id) if snapshot_id else None
        group_key = target_suffix if target_suffix is not None else suffix
        groups.setdefault(group_key, {})[role] = key

    for key in keys:
        name = os.path.basename((key or "").strip())
        if not name:
            continue
        if _suffix_from_data_basename(name) is not None:
            continue
        surrogate_suffix = _suffix_from_surrogate_basename(name)
        if surrogate_suffix is not None:
            _attach_auxiliary("surrogate", surrogate_suffix, key)
            continue
        next_x_suffix = _suffix_from_next_x_basename(name)
        if next_x_suffix is not None:
            _attach_auxiliary("next_x", next_x_suffix, key)
    return groups


def _read_json_if_exists_local(path: str) -> Any:
    if not path or not os.path.isfile(path):
        return None
    try:
        return load_json_from_local_path(path)
    except Exception:
        return None


def _build_triplet_index_from_directory(directory: str) -> NSDFTripletIndex:
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return NSDFTripletIndex()
    try:
        names = os.listdir(d)
    except OSError:
        return NSDFTripletIndex()
    keys = [os.path.join(d, name) for name in names if name.endswith(".json")]
    groups = _triplet_groups_from_object_keys(keys)

    def read_doc(path: str) -> Any:
        return _read_json_if_exists_local(path)

    return _build_triplet_index_from_groups(groups, read_doc=read_doc)


def _list_nsdf_object_keys_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
            data_key = (cfg.get("key") or "").strip()
    if not bucket:
        return []

    prefix = ""
    if data_key:
        if "/" in data_key:
            prefix = data_key.rsplit("/", 1)[0] + "/"
        elif parse_nsdf_data_filename(data_key) is None and data_key.lower() != "data.json":
            prefix = data_key.rstrip("/") + "/"

    list_paths = StrainDashboardPaths(
        s3_bucket=bucket,
        s3_data_key=data_key or "data.json",
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        s3_env_file=paths.s3_env_file,
        json_url=paths.json_url,
    )
    try:
        client = _make_nsdf_s3_client(list_paths, mongo_s3_auth=mongo_s3_auth)
    except Exception:
        return []

    keys: List[str] = []
    continuation: Optional[str] = None
    try:
        while True:
            params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation:
                params["ContinuationToken"] = continuation
            result = client.list_objects_v2(**params)
            for obj in result.get("Contents") or []:
                key = str(obj.get("Key") or "").strip()
                if key.endswith(".json"):
                    keys.append(key)
            if not result.get("IsTruncated"):
                break
            continuation = result.get("NextContinuationToken")
            if not continuation:
                break
    except Exception:
        return []
    return keys


def _read_json_if_exists_s3(client: Any, bucket: str, key: str) -> Any:
    if not key:
        return None
    try:
        return _load_json_from_s3_key(client, bucket, key)
    except Exception as exc:
        if _s3_missing_error(exc):
            return None
        return None


def _build_triplet_index_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> NSDFTripletIndex:
    bucket = (paths.s3_bucket or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
    if not bucket:
        return NSDFTripletIndex()

    keys = _list_nsdf_object_keys_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    groups = _triplet_groups_from_object_keys(keys)
    try:
        client = _make_nsdf_s3_client(paths, mongo_s3_auth=mongo_s3_auth)
    except Exception:
        return NSDFTripletIndex()

    def read_doc(key: str) -> Any:
        return _read_json_if_exists_s3(client, bucket, key)

    return _build_triplet_index_from_groups(groups, read_doc=read_doc)


def _resolve_nsdf_s3_bucket_and_prefix(
    paths: StrainDashboardPaths,
) -> Tuple[str, str]:
    """Return ``(bucket, prefix)``; ``prefix`` has a trailing slash when non-empty."""
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket and (paths.json_url or "").strip():
        cfg = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if cfg:
            bucket = (cfg.get("bucket") or "").strip()
            data_key = (cfg.get("key") or "").strip()
    if not bucket:
        return "", ""
    prefix = ""
    if data_key:
        if "/" in data_key:
            prefix = data_key.rsplit("/", 1)[0] + "/"
        elif parse_nsdf_data_filename(data_key) is None and data_key.lower() != "data.json":
            prefix = data_key.rstrip("/") + "/"
    return bucket, prefix


def _nsdf_catalog_local_path(directory: str) -> str:
    return os.path.join((directory or "").strip(), NSDF_CATALOG_JSON_BASENAME)


def _nsdf_catalog_s3_key(paths: StrainDashboardPaths) -> str:
    _bucket, prefix = _resolve_nsdf_s3_bucket_and_prefix(paths)
    return f"{prefix}{NSDF_CATALOG_JSON_BASENAME}" if prefix else NSDF_CATALOG_JSON_BASENAME


def triplet_index_to_catalog_doc(index: NSDFTripletIndex) -> Dict[str, Any]:
    """Serialize a triplet index to ``catalog.json`` document format."""
    return {
        "version": NSDF_CATALOG_VERSION,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshots": [
            {
                "suffix": snap.suffix,
                "workflow_id": snap.workflow_id,
                "sort_key": snap.sort_key,
                "uncertainty_trend_y": snap.uncertainty_trend_y,
                "has_surrogate_archive": snap.has_surrogate_archive,
                "has_next_x_archive": snap.has_next_x_archive,
            }
            for snap in index.snapshots
        ],
    }


def triplet_index_from_catalog_doc(doc: Any) -> Optional[NSDFTripletIndex]:
    """Parse ``catalog.json`` into a triplet index; return ``None`` when invalid."""
    if not isinstance(doc, Mapping):
        return None
    if doc.get("version") != NSDF_CATALOG_VERSION:
        return None
    raw_snaps = doc.get("snapshots")
    if not isinstance(raw_snaps, list) or not raw_snaps:
        return None

    snapshots: List[NSDFSnapshotRef] = []
    for item in raw_snaps:
        if not isinstance(item, Mapping):
            continue
        suffix = str(item.get("suffix") or "")
        if suffix and not is_valid_nsdf_snapshot_suffix(suffix):
            continue
        workflow_id = str(item.get("workflow_id") or "").strip() or NSDF_UNKNOWN_WORKFLOW_ID
        sort_key = str(item.get("sort_key") or _snapshot_sort_key(suffix))
        trend_raw = item.get("uncertainty_trend_y")
        trend_y: Optional[float] = None
        if _is_number(trend_raw):
            trend_y = float(trend_raw)
        snapshots.append(
            NSDFSnapshotRef(
                suffix=suffix,
                workflow_id=workflow_id,
                sort_key=sort_key,
                uncertainty_trend_y=trend_y,
                has_surrogate_archive=bool(item.get("has_surrogate_archive")),
                has_next_x_archive=bool(item.get("has_next_x_archive")),
            )
        )
    if not snapshots:
        return None

    by_workflow: Dict[str, List[NSDFSnapshotRef]] = {}
    for snap in snapshots:
        by_workflow.setdefault(snap.workflow_id, []).append(snap)
    for workflow_id in by_workflow:
        by_workflow[workflow_id].sort(key=lambda snap: snap.sort_key, reverse=True)
    return NSDFTripletIndex(snapshots=snapshots, by_workflow=by_workflow)


def _load_catalog_index_from_local_directory(directory: str) -> Optional[NSDFTripletIndex]:
    path = _nsdf_catalog_local_path(directory)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:
        _LOG.warning("Could not read local catalog %s: %s", path, exc)
        return None
    index = triplet_index_from_catalog_doc(doc)
    if index is None:
        _LOG.warning("Local catalog %s is missing or has invalid format.", path)
    return index


def _load_catalog_index_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Optional[NSDFTripletIndex]:
    bucket, _prefix = _resolve_nsdf_s3_bucket_and_prefix(paths)
    if not bucket:
        return None
    key = _nsdf_catalog_s3_key(paths)
    try:
        client = _make_nsdf_s3_client(paths, mongo_s3_auth=mongo_s3_auth)
        doc = _load_json_from_s3_key(client, bucket, key)
    except FileNotFoundError:
        return None
    except Exception as exc:
        _LOG.warning("Could not read S3 catalog s3://%s/%s: %s", bucket, key, exc)
        return None
    index = triplet_index_from_catalog_doc(doc)
    if index is None:
        _LOG.warning("S3 catalog s3://%s/%s has invalid format.", bucket, key)
    return index


def _write_catalog_index_to_local_directory(
    directory: str,
    index: NSDFTripletIndex,
) -> Tuple[bool, str, str]:
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return False, "", "local directory not found"
    path = _nsdf_catalog_local_path(d)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(triplet_index_to_catalog_doc(index), fh, indent=2)
            fh.write("\n")
        return True, path, ""
    except OSError as exc:
        return False, "", str(exc)


def _write_catalog_index_to_s3(
    paths: StrainDashboardPaths,
    index: NSDFTripletIndex,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str]:
    bucket, _prefix = _resolve_nsdf_s3_bucket_and_prefix(paths)
    if not bucket:
        return False, "", "S3 bucket not configured"
    key = _nsdf_catalog_s3_key(paths)
    body = json.dumps(triplet_index_to_catalog_doc(index), indent=2).encode("utf-8")
    try:
        client = _make_nsdf_s3_client(paths, mongo_s3_auth=mongo_s3_auth)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return True, f"s3://{bucket}/{key}", ""
    except Exception as exc:
        return False, "", str(exc)


def _try_load_catalog_index(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> Optional[NSDFTripletIndex]:
    if _local_data_dir_active(paths):
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        if local_dir:
            return _load_catalog_index_from_local_directory(local_dir)
        return None

    if remote_linked:
        if not _remote_snapshot_listing_enabled(paths):
            local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
            if local_dir:
                return _load_catalog_index_from_local_directory(local_dir)
            return None
        remote_paths = (
            paths if paths.has_s3_source() else promote_gateway_json_url_to_s3_paths(paths)
        )
        index = _load_catalog_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)
        if index is not None and index.snapshots:
            return index
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        if local_dir:
            return _load_catalog_index_from_local_directory(local_dir)
        return None

    if _remote_snapshot_listing_enabled(paths):
        remote_paths = (
            paths if paths.has_s3_source() else promote_gateway_json_url_to_s3_paths(paths)
        )
        index = _load_catalog_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)
        if index is not None and index.snapshots:
            return index

    local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
    if local_dir:
        return _load_catalog_index_from_local_directory(local_dir)
    return None


def _scan_triplet_index(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> NSDFTripletIndex:
    """Full JSON scan (legacy index path) when ``catalog.json`` is absent or stale."""
    if _local_data_dir_active(paths):
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        if local_dir:
            return _build_triplet_index_from_directory(local_dir)
        return NSDFTripletIndex()

    if remote_linked:
        if not _remote_snapshot_listing_enabled(paths):
            return NSDFTripletIndex()
        remote_paths = (
            paths if paths.has_s3_source() else promote_gateway_json_url_to_s3_paths(paths)
        )
        return _build_triplet_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)

    if _remote_snapshot_listing_enabled(paths):
        remote_paths = (
            paths if paths.has_s3_source() else promote_gateway_json_url_to_s3_paths(paths)
        )
        index = _build_triplet_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)
        if index.snapshots:
            return index

    local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
    if local_dir:
        return _build_triplet_index_from_directory(local_dir)
    return NSDFTripletIndex()


def _try_write_catalog_index(
    paths: StrainDashboardPaths,
    index: NSDFTripletIndex,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> Tuple[bool, str, str]:
    """Write ``catalog.json`` beside triplet files (S3 when configured, else local dir)."""
    if not index.snapshots:
        return False, "", "empty index"

    if _local_data_dir_active(paths):
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        if local_dir:
            return _write_catalog_index_to_local_directory(local_dir, index)
        return False, "", "local directory not found"

    last_error = ""

    if remote_linked or _remote_snapshot_listing_enabled(paths):
        remote_paths = (
            paths if paths.has_s3_source() else promote_gateway_json_url_to_s3_paths(paths)
        )
        if remote_paths.has_s3_source() or _resolve_nsdf_s3_bucket_and_prefix(remote_paths)[0]:
            ok, loc, err = _write_catalog_index_to_s3(
                remote_paths,
                index,
                mongo_s3_auth=mongo_s3_auth,
            )
            if ok:
                return True, loc, ""
            last_error = err or last_error

    local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
    if local_dir:
        ok, loc, err = _write_catalog_index_to_local_directory(local_dir, index)
        if ok:
            return True, loc, ""
        last_error = err or last_error

    return False, "", last_error or "no writable catalog location"


def discover_nsdf_triplet_index(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
    force_rescan: bool = False,
) -> NSDFTripletIndexDiscoverResult:
    """
    Build a workflow-scoped snapshot catalog for workflow/snapshot UI selectors.

    Reads ``catalog.json`` when present (unless ``force_rescan``). Otherwise scans JSON
    files and writes ``catalog.json`` locally or on S3 when permitted.

    ScientistCloud S3-linked datasets must list only the remote prefix (never the
    sparse upload mirror). Uploaded-file datasets use the local upload tree.
    """
    if not force_rescan:
        catalog_index = _try_load_catalog_index(
            paths,
            base_dir=base_dir,
            save_dir=save_dir,
            mongo_s3_auth=mongo_s3_auth,
            remote_linked=remote_linked,
        )
        if catalog_index is not None and catalog_index.snapshots:
            return NSDFTripletIndexDiscoverResult(
                index=catalog_index,
                source="catalog_json",
            )

    index = _scan_triplet_index(
        paths,
        base_dir=base_dir,
        save_dir=save_dir,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
    )
    if not index.snapshots:
        return NSDFTripletIndexDiscoverResult(index=index, source="empty")

    wrote, location, write_error = _try_write_catalog_index(
        paths,
        index,
        base_dir=base_dir,
        save_dir=save_dir,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
    )
    return NSDFTripletIndexDiscoverResult(
        index=index,
        source="full_scan",
        catalog_written=wrote,
        catalog_write_location=location,
        catalog_write_error=write_error,
    )


def _credential_is_placeholder(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    if v == "...":
        return True
    return bool(re.fullmatch(r"\.+", v))


def pick_strain_json_link_from_dataset_doc(doc: Mapping[str, Any]) -> str:
    """First remote/local JSON link on a portal dataset document (matches SC_Web PHP)."""
    for field in ("viewer_url", "download_url", "google_drive_link", "source_path"):
        u = str(doc.get(field) or "").strip()
        if not u:
            continue
        low = u.lower()
        if (
            low.endswith(".json")
            or ".json?" in low
            or low.startswith("s3://")
            or low.startswith("http://")
            or low.startswith("https://")
        ):
            return u
    return ""


def apply_gateway_credentials_to_url(url: str, access_key: str, secret_key: str) -> str:
    """Inject or replace gateway query credentials on an https URL."""
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    url = (url or "").strip()
    access_key = (access_key or "").strip()
    secret_key = (secret_key or "").strip()
    if not url or not access_key or not secret_key:
        return url
    if not _looks_like_http_url(url):
        return url

    parts = urlparse(url)
    query = dict(parse_qsl(parts.query or "", keep_blank_values=True))
    for key in list(query.keys()):
        if key in ("access_key", "access_key_id", "secret_key", "secret_access_key"):
            if _credential_is_placeholder(str(query[key])):
                del query[key]
    query["access_key"] = access_key
    query["secret_key"] = secret_key
    new_query = urlencode(query)
    return urlunparse(
        (parts.scheme, parts.netloc, parts.path, parts.params, new_query, parts.fragment)
    )


def resolve_strain_json_remote_link_from_dataset(doc: Mapping[str, Any]) -> str:
    """Remote strain JSON URL with real gateway credentials when stored on the dataset."""
    link = normalize_nsdf_remote_data_link(pick_strain_json_link_from_dataset_doc(doc))
    if not link:
        return ""
    ak = str(doc.get("s3_access_key_id") or doc.get("accesskey") or "").strip()
    sk = str(doc.get("s3_secret_access_key") or doc.get("secretkey") or "").strip()
    if _credential_is_placeholder(ak):
        ak = ""
    if _credential_is_placeholder(sk):
        sk = ""
    if ak and sk and _looks_like_http_url(link):
        return apply_gateway_credentials_to_url(link, ak, sk)
    return link


def promote_gateway_json_url_to_s3_paths(paths: StrainDashboardPaths) -> StrainDashboardPaths:
    """
    When ``json_url`` is a ScientistCloud gateway link, populate ``s3_bucket`` / ``s3_data_key``.

    Portal datasets usually pass gateway HTTPS URLs without explicit S3 fields. Promoting
    enables direct S3 reads and timestamped surrogate fallback for ``data.json`` / latest.
    """
    if paths.has_s3_source():
        return paths
    url = normalize_nsdf_gateway_data_url((paths.json_url or "").strip())
    if not url:
        return paths
    cfg = _parse_gateway_url_with_query_keys(url)
    if not cfg:
        return paths
    bucket = (cfg.get("bucket") or "").strip()
    key = normalize_nsdf_data_object_key((cfg.get("key") or "").strip())
    if not bucket or not key:
        return paths
    return StrainDashboardPaths(
        local_json_path=paths.local_json_path,
        json_url=url,
        surrogate_json_path=paths.surrogate_json_path,
        surrogate_json_url=paths.surrogate_json_url,
        next_x_json_path=paths.next_x_json_path,
        next_x_json_url=paths.next_x_json_url,
        local_data_dir=paths.local_data_dir,
        s3_env_file=paths.s3_env_file,
        s3_bucket=bucket,
        s3_data_key=key,
        s3_surrogate_key=paths.s3_surrogate_key,
        s3_next_x_key=paths.s3_next_x_key,
        s3_endpoint_url=(cfg.get("endpoint_url") or paths.s3_endpoint_url or "").strip(),
        s3_region=(cfg.get("region_name") or paths.s3_region or "us-east-1").strip() or "us-east-1",
        version_suffix=paths.version_suffix,
        surrogate_version_suffix=paths.surrogate_version_suffix,
        next_x_version_suffix=paths.next_x_version_suffix,
        strict_triplet_paths=paths.strict_triplet_paths,
    )


def _finalize_strain_paths(paths: StrainDashboardPaths) -> StrainDashboardPaths:
    return promote_gateway_json_url_to_s3_paths(paths)


def _paths_without_local_json_files(paths: StrainDashboardPaths) -> StrainDashboardPaths:
    """Drop portal upload-mirror file paths; keep remote URL / S3 routing."""
    return StrainDashboardPaths(
        local_json_path="",
        json_url=paths.json_url,
        surrogate_json_path="",
        surrogate_json_url=paths.surrogate_json_url,
        next_x_json_path="",
        next_x_json_url=paths.next_x_json_url,
        local_data_dir=paths.local_data_dir,
        s3_env_file=paths.s3_env_file,
        s3_bucket=paths.s3_bucket,
        s3_data_key=paths.s3_data_key,
        s3_surrogate_key=paths.s3_surrogate_key,
        s3_next_x_key=paths.s3_next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        version_suffix=paths.version_suffix,
        surrogate_version_suffix=paths.surrogate_version_suffix,
        next_x_version_suffix=paths.next_x_version_suffix,
        strict_triplet_paths=paths.strict_triplet_paths,
    )


def _prepare_nsdf_load_paths(
    paths: StrainDashboardPaths,
    *,
    remote_linked: bool = False,
) -> StrainDashboardPaths:
    """
    Normalize paths for bundle/surrogate loads without widening strict triplet scope.

    Gateway promotion can add ``s3_bucket`` / ``s3_data_key`` after suffix resolution;
    re-apply suffixes so surrogate/next_x keys exist and ``strict_triplet_paths`` stays on.
    """
    load_paths = promote_gateway_json_url_to_s3_paths(_strip_local_data_dir_paths(paths))
    if remote_linked:
        load_paths = _paths_without_local_json_files(load_paths)
    if remote_linked or load_paths.has_s3_source():
        load_paths = StrainDashboardPaths(
            local_json_path="",
            json_url=load_paths.json_url,
            surrogate_json_path="",
            surrogate_json_url=load_paths.surrogate_json_url,
            next_x_json_path="",
            next_x_json_url=load_paths.next_x_json_url,
            local_data_dir=load_paths.local_data_dir,
            s3_env_file=load_paths.s3_env_file,
            s3_bucket=load_paths.s3_bucket,
            s3_data_key=load_paths.s3_data_key,
            s3_surrogate_key=load_paths.s3_surrogate_key,
            s3_next_x_key=load_paths.s3_next_x_key,
            s3_endpoint_url=load_paths.s3_endpoint_url,
            s3_region=load_paths.s3_region,
            version_suffix=load_paths.version_suffix,
            surrogate_version_suffix=load_paths.surrogate_version_suffix,
            next_x_version_suffix=load_paths.next_x_version_suffix,
            strict_triplet_paths=load_paths.strict_triplet_paths,
        )
    if paths.strict_triplet_paths and load_paths.has_s3_source():
        load_paths = apply_nsdf_file_suffixes(
            load_paths,
            data_suffix=paths.version_suffix or "",
            surrogate_suffix=paths.surrogate_version_suffix,
            next_x_suffix=paths.next_x_version_suffix,
            strict=True,
        )
    return load_paths


def apply_scientistcloud_storage_policy(
    paths: StrainDashboardPaths,
    doc: Optional[Mapping[str, Any]] = None,
    *,
    server_param: str = "",
    base_dir: str = "",
    save_dir: str = "",
) -> StrainDashboardPaths:
    """
    Enforce ScientistCloud storage mode on resolved session paths.

    Remote-linked datasets use S3 / gateway only. Uploaded-file datasets keep
    the portal upload/converted tree.
    """
    if not scientistcloud_dataset_is_remote_linked(doc, server_param=server_param):
        return _finalize_strain_paths(paths)

    cleared = _paths_without_local_json_files(paths)
    finalized = _finalize_strain_paths(cleared)
    if finalized.has_s3_source() or (finalized.json_url or "").strip():
        return finalized

    if isinstance(doc, Mapping):
        return enrich_strain_paths_from_dataset_doc(
            _copy_s3_fields(
                paths,
                StrainDashboardPaths(
                    local_json_path="",
                    json_url="",
                    surrogate_json_path=paths.surrogate_json_path,
                    surrogate_json_url=paths.surrogate_json_url,
                    next_x_json_path=paths.next_x_json_path,
                    next_x_json_url=paths.next_x_json_url,
                ),
            ),
            doc,
            base_dir=base_dir,
            save_dir=save_dir,
            remote_linked=True,
        )
    return finalized


def enrich_strain_paths_from_dataset_doc(
    paths: StrainDashboardPaths,
    doc: Optional[Mapping[str, Any]],
    *,
    base_dir: str = "",
    save_dir: str = "",
    remote_linked: bool = False,
) -> StrainDashboardPaths:
    """
    When URL args and env did not resolve JSON, use the Mongo dataset record
    (same fields as portal dashboard share links).
    """
    loc = (paths.local_json_path or "").strip()
    if loc and not _local_path_is_nsdf_data_json(loc):
        loc = ""
    if loc or (paths.json_url or "").strip():
        return _finalize_strain_paths(
            StrainDashboardPaths(
                local_json_path=loc,
                json_url=paths.json_url,
                surrogate_json_path=paths.surrogate_json_path,
                surrogate_json_url=paths.surrogate_json_url,
                next_x_json_path=paths.next_x_json_path,
                next_x_json_url=paths.next_x_json_url,
                local_data_dir=paths.local_data_dir,
                s3_env_file=paths.s3_env_file,
                s3_bucket=paths.s3_bucket,
                s3_data_key=paths.s3_data_key,
                s3_surrogate_key=paths.s3_surrogate_key,
                s3_next_x_key=paths.s3_next_x_key,
                s3_endpoint_url=paths.s3_endpoint_url,
                s3_region=paths.s3_region,
            )
        )
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()
    mirror = ""
    if not remote_linked:
        mirror = find_strain_json_under_dataset_dir(bd) or find_strain_json_under_dataset_dir(sd)
    if mirror:
        return _finalize_strain_paths(
            _copy_s3_fields(
                paths,
                StrainDashboardPaths(
                    local_json_path=mirror,
                    json_url="",
                    surrogate_json_path=paths.surrogate_json_path,
                    surrogate_json_url=paths.surrogate_json_url,
                    next_x_json_path=paths.next_x_json_path,
                    next_x_json_url=paths.next_x_json_url,
                ),
            )
        )
    if not doc:
        return _finalize_strain_paths(paths)
    link = resolve_strain_json_remote_link_from_dataset(doc)
    if not link:
        return _finalize_strain_paths(paths)
    if link.lower().startswith("s3://"):
        bucket, data_key = parse_s3_uri(link)
        if bucket and data_key:
            ep = str(doc.get("s3_endpoint_url") or "").strip()
            reg = str(doc.get("s3_region_name") or "us-east-1").strip() or "us-east-1"
            return _finalize_strain_paths(
                _copy_s3_fields(
                    paths,
                    StrainDashboardPaths(
                        local_json_path="",
                        json_url="",
                        surrogate_json_path=paths.surrogate_json_path,
                        surrogate_json_url=paths.surrogate_json_url,
                        next_x_json_path=paths.next_x_json_path,
                        next_x_json_url=paths.next_x_json_url,
                        s3_bucket=bucket,
                        s3_data_key=data_key,
                        s3_endpoint_url=ep,
                        s3_region=reg,
                    ),
                )
            )
    if _looks_like_http_url(link):
        return _finalize_strain_paths(
            _copy_s3_fields(
                paths,
                StrainDashboardPaths(
                    local_json_path="",
                    json_url=link,
                    surrogate_json_path=paths.surrogate_json_path,
                    surrogate_json_url=paths.surrogate_json_url,
                    next_x_json_path=paths.next_x_json_path,
                    next_x_json_url=paths.next_x_json_url,
                ),
            )
        )
    if os.path.isfile(link):
        return _finalize_strain_paths(
            _copy_s3_fields(
                paths,
                StrainDashboardPaths(
                    local_json_path=link,
                    json_url="",
                    surrogate_json_path=paths.surrogate_json_path,
                    surrogate_json_url=paths.surrogate_json_url,
                    next_x_json_path=paths.next_x_json_path,
                    next_x_json_url=paths.next_x_json_url,
                ),
            )
        )
    # s3:// and other schemes: pass as URL for downstream loaders
    return _finalize_strain_paths(
        _copy_s3_fields(
            paths,
            StrainDashboardPaths(
                local_json_path="",
                json_url=link,
                surrogate_json_path=paths.surrogate_json_path,
                surrogate_json_url=paths.surrogate_json_url,
                next_x_json_path=paths.next_x_json_path,
                next_x_json_url=paths.next_x_json_url,
            ),
        )
    )


def load_json_from_local_path(path: str) -> Dict[str, Any]:
    if _looks_like_http_url(path):
        return load_json_from_url(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_qsl_preserve_plus(query: str) -> Dict[str, str]:
    """
    Parse ``application/x-www-form-urlencoded`` query string for gateway URLs.

    ``urllib.parse.parse_qs`` / ``parse_qsl`` treat ``+`` as a space in values. S3/RGW
    credentials often contain literal ``+``; that corruption yields InvalidAccessKeyId.
    We split on ``&`` / ``=`` and use ``unquote`` on values (``unquote`` does *not* map ``+`` → space).
    """
    from urllib.parse import unquote

    out: Dict[str, str] = {}
    if not query:
        return out
    for part in query.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        k = unquote(k.replace("+", " "), errors="replace").strip()
        v = unquote(v, errors="replace")
        if k:
            out[k] = v
    return out


def _parse_gateway_url_with_query_keys(url: str) -> Optional[Dict[str, str]]:
    """
    Detect ScientistCloud-style gateway URLs:
    https://<host>/<bucket>/<object_key>?access_key=...&secret_key=...
    (Ceph/RGW and similar often reject unsigned GET; use SigV4 via boto3.)
    """
    from urllib.parse import unquote, urlparse

    u = (url or "").strip()
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    qs = _parse_qsl_preserve_plus(p.query or "")
    ak = (
        qs.get("access_key")
        or qs.get("AWSAccessKeyId")
        or qs.get("AccessKeyId")
        or ""
    ).strip()
    sk = (
        qs.get("secret_key")
        or qs.get("AWSSecretKey")
        or qs.get("SecretAccessKey")
        or qs.get("SecretKey")
        or ""
    ).strip()
    if not ak or not sk:
        return None
    segments = [unquote(x) for x in (p.path or "").split("/") if x]
    if len(segments) < 2:
        return None
    bucket, key = segments[0], "/".join(segments[1:])
    if not key:
        return None
    region = (qs.get("region") or "us-east-1").strip() or "us-east-1"
    session_tok = (
        qs.get("session_token")
        or qs.get("SessionToken")
        or qs.get("security_token")
        or qs.get("X-Amz-Security-Token")
        or ""
    ).strip()
    out: Dict[str, str] = {
        "endpoint_url": f"{p.scheme}://{p.netloc}",
        "bucket": bucket,
        "key": key,
        "access_key_id": ak,
        "secret_access_key": sk,
        "region_name": region,
    }
    if session_tok:
        out["aws_session_token"] = session_tok
    return out


def _load_json_via_s3_query_client(cfg: Dict[str, str]) -> Dict[str, Any]:
    """
    Signed download for gateway URLs (?access_key=…&secret_key=…).

    Use ``get_object`` instead of ``download_fileobj``: the latter issues
    ``HeadObject`` first, which many Ceph/RGW bucket policies omit while still
    allowing ``GetObject`` — that mismatch surfaces as 403 AccessDenied.

    Uses **path-style** ``https://endpoint/bucket/key`` requests, which match URLs like
    ``https://us-east-1.gw.example.com/scientistcloud/.../file.json`` and the gateway TLS cert.

    **Virtual-hosted** fallback (``bucket.endpoint``) is only attempted for ``*.amazonaws.com``
    endpoints: on custom RGW hosts, virtual style changes the hostname (e.g.
    ``scientistcloud.us-east-1.gw...``) so the certificate no longer matches
    ``us-east-1.gw...`` and TLS fails.
    """
    from urllib.parse import urlparse

    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError

    token = (cfg.get("aws_session_token") or "").strip() or None
    session = boto3.session.Session(
        aws_access_key_id=cfg["access_key_id"].strip(),
        aws_secret_access_key=cfg["secret_access_key"].strip(),
        aws_session_token=token,
        region_name=(cfg.get("region_name") or "us-east-1").strip() or "us-east-1",
    )

    def _get(style: str) -> Dict[str, Any]:
        client = session.client(
            "s3",
            endpoint_url=cfg["endpoint_url"],
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": style},
            ),
        )
        resp = client.get_object(Bucket=cfg["bucket"], Key=cfg["key"])
        raw = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))

    host = (urlparse(cfg.get("endpoint_url") or "").netloc or "").lower()
    try_aws_virtual = "amazonaws.com" in host

    try:
        return _get("path")
    except ClientError as e:
        if not try_aws_virtual:
            raise
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code not in (
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
            "InvalidRequest",
            "AuthorizationHeaderMalformed",
            "AccessDenied",
        ):
            raise
        _LOG.debug("S3 get_object path-style failed (%s); retrying virtual-hosted (AWS endpoint)", code)
        return _get("virtual")


def _apply_s3_auth_override(cfg: Dict[str, str], ov: Optional[Dict[str, str]]) -> None:
    """
    Replace query-string credentials with values from Mongo (or another trusted source).

    Portal iframe URLs can truncate ``secret_key=``; dataset docs store full
    ``s3_access_key_id`` / ``s3_secret_access_key`` from upload.
    """
    if not ov:
        return
    ak = (ov.get("access_key_id") or "").strip()
    sk = (ov.get("secret_access_key") or "").strip()
    if ak and sk:
        cfg["access_key_id"] = ak
        cfg["secret_access_key"] = sk
        _LOG.debug("S3 gateway load: using credential override (Mongo / server), not URL query string")
    ep = (ov.get("endpoint_url") or "").strip()
    if ep:
        cfg["endpoint_url"] = ep.rstrip("/")
    reg = (ov.get("region_name") or "").strip()
    if reg:
        cfg["region_name"] = reg
    tok = (ov.get("aws_session_token") or "").strip()
    if tok:
        cfg["aws_session_token"] = tok


def load_json_from_url(
    url: str,
    *,
    timeout_s: float = 120.0,
    s3_auth_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Load JSON from http(s): plain GET, or boto3 SigV4 when URL carries access_key/secret_key (gateway style).

    ``s3_auth_override``: optional ``access_key_id``, ``secret_access_key``, and optionally
    ``endpoint_url``, ``region_name``, ``aws_session_token`` — used when the URL query
    credentials are missing or invalid (e.g. truncated in an iframe).
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    u = normalize_nsdf_gateway_data_url((url or "").strip())
    if not u.lower().startswith(("http://", "https://")):
        raise ValueError("JSON URL must start with http:// or https://")

    s3_cfg = _parse_gateway_url_with_query_keys(u)
    if s3_cfg:
        _apply_s3_auth_override(s3_cfg, s3_auth_override)
        try:
            return _load_json_via_s3_query_client(s3_cfg)
        except ImportError as ie:
            raise FileNotFoundError(
                "This JSON URL needs boto3 (S3-compatible signed download). "
                "Install boto3 in the dashboard image or use a presigned GET URL."
            ) from ie
        except Exception as ex:
            try:
                from botocore.exceptions import ClientError

                if isinstance(ex, ClientError):
                    err = (ex.response or {}).get("Error") or {}
                    code = err.get("Code", "ClientError")
                    msg = err.get("Message", str(ex))
                    raise FileNotFoundError(
                        f"S3 gateway download failed ({code}): {msg}. "
                        f"Object: s3://{s3_cfg.get('bucket', '')}/{s3_cfg.get('key', '')} "
                        f"via {s3_cfg.get('endpoint_url', '')}. "
                        "Unsigned curl/browser GET often returns 403 on this gateway; the dashboard "
                        "uses SigV4. If this persists, verify the dataset S3 access/secret keys in the "
                        "portal and that the object exists."
                    ) from ex
            except ImportError:
                pass
            _LOG.debug("S3 client load failed, trying HTTP GET: %s", ex, exc_info=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 ScientistCloud-ORNL-strain"
        ),
        "Accept": "application/json, text/plain, */*",
    }
    req = Request(u, headers=headers)
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except HTTPError as e:
        err_body = ""
        try:
            err_body = (e.read() or b"").decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        if s3_cfg and e.code in (401, 403):
            try:
                _apply_s3_auth_override(s3_cfg, s3_auth_override)
                return _load_json_via_s3_query_client(s3_cfg)
            except Exception as e2:
                raise FileNotFoundError(
                    f"HTTP {e.code} for JSON URL; signed S3 retry failed ({e2}). "
                    f"Response: {err_body or '(empty)'}"
                ) from e2
        raise FileNotFoundError(
            f"HTTP {e.code} loading JSON URL: {u}. Response: {err_body or '(empty)'}"
        ) from e
    except URLError as e:
        raise FileNotFoundError(f"Network error loading JSON URL: {e.reason}") from e
    return json.loads(raw.decode("utf-8"))


def load_strain_json(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Load document: prefer ``local_json_path`` (file path or http(s) URL), else ``json_url``.

    ``mongo_s3_auth``: when loading a gateway HTTPS URL, optional credentials from the
    dataset Mongo document (``s3_access_key_id`` / ``s3_secret_access_key``) override
    fragile query-string keys in the URL.
    """
    loc = (paths.local_json_path or "").strip()
    if loc and _looks_like_http_url(loc):
        return load_json_from_url(loc, s3_auth_override=mongo_s3_auth)
    if loc:
        return load_json_from_local_path(loc)
    jurl = (paths.json_url or "").strip()
    if jurl:
        return load_json_from_url(jurl, s3_auth_override=mongo_s3_auth)
    raise FileNotFoundError(
        "Set ORNL_NSDF_DATA_JSON_PATH for a local data.json path, or ORNL_NSDF_DATA_JSON_URL / "
        "nsdf_data_json_url for a full https://... link. Legacy ORNL_STRAIN_JSON_* aliases "
        "are still accepted for NSDF data.json."
    )


def _sibling_surrogate_path(data_path: str) -> str:
    p = (data_path or "").strip()
    if not p or _looks_like_http_url(p):
        return ""
    suffix = parse_nsdf_data_filename(os.path.basename(p))
    if suffix is None and os.path.basename(p).lower() != "data.json":
        return ""
    if suffix:
        return os.path.join(os.path.dirname(p), f"surrogate_{suffix}.json")
    return os.path.join(os.path.dirname(p), "surrogate.json")


def _sibling_surrogate_url(data_url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    u = (data_url or "").strip()
    if not _looks_like_http_url(u):
        return ""
    parts = urlparse(u)
    segments = [x for x in (parts.path or "").split("/") if x]
    if not segments:
        return ""
    suffix = parse_nsdf_data_filename(segments[-1])
    if suffix is None and segments[-1].lower() != "data.json":
        return ""
    segments[-1] = f"surrogate_{suffix}.json" if suffix else "surrogate.json"
    new_path = "/" + "/".join(segments)
    return urlunparse((parts.scheme, parts.netloc, new_path, parts.params, parts.query, parts.fragment))


def load_optional_surrogate_json(
    paths: StrainDashboardPaths,
    *,
    data_doc: Optional[Mapping[str, Any]] = None,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str], StrainDashboardPaths]:
    """
    Load optional ``surrogate.json`` from explicit path/URL or inferred sibling.

    Missing inferred siblings are non-fatal. Malformed explicit surrogate locations are
    reported as messages and skipped so the dashboard can still render measurements.
    """
    messages: List[str] = []
    explicit_path = (paths.surrogate_json_path or "").strip()
    explicit_url = (paths.surrogate_json_url or "").strip()
    effective = StrainDashboardPaths(
        local_json_path=paths.local_json_path,
        json_url=paths.json_url,
        surrogate_json_path=explicit_path,
        surrogate_json_url=explicit_url,
        local_data_dir=paths.local_data_dir,
    )

    candidates: List[Tuple[str, str, bool]] = []
    if explicit_path:
        candidates.append(("path", explicit_path, True))
    if explicit_url:
        candidates.append(("url", explicit_url, True))
    if not paths.strict_triplet_paths:
        if not candidates:
            sib_path = _sibling_surrogate_path(paths.local_json_path)
            if sib_path:
                candidates.append(("path", sib_path, False))
            else:
                sib_url = _sibling_surrogate_url(paths.json_url)
                if sib_url:
                    candidates.append(("url", sib_url, False))

        tried = {value for _, value, _ in candidates}
        for path in _iter_surrogate_path_candidates(paths):
            if path not in tried:
                candidates.append(("path", path, False))
                tried.add(path)
        if paths.json_url:
            from urllib.parse import urlparse

            data_name = (
                os.path.basename(urlparse(paths.json_url).path or "")
                if _looks_like_http_url(paths.json_url)
                else ""
            )
            data_suffixes = list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
            reference = _nsdf_reference_data_suffix(data_name, data_suffixes=data_suffixes)
            sur_suffixes = list_nsdf_surrogate_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
            for suffix in _nsdf_suffixes_after_reference(sur_suffixes, reference):
                if not is_valid_nsdf_version_suffix(suffix):
                    continue
                try:
                    _, sur_fn, _ = nsdf_triplet_basenames(suffix)
                except ValueError:
                    continue
                sur_url = _replace_url_basename(paths.json_url, sur_fn)
                if sur_url not in tried:
                    candidates.append(("url", sur_url, False))
                    tried.add(sur_url)

    display_size: Optional[Tuple[int, int]] = None
    if isinstance(data_doc, Mapping):
        display_size, _ = resolve_nsdf_grid_size(data_doc)

    for kind, value, explicit in candidates:
        try:
            if kind == "path":
                doc = load_json_from_local_path(value)
                effective.surrogate_json_path = value
            else:
                doc = load_json_from_url(value, s3_auth_override=mongo_s3_auth)
                effective.surrogate_json_url = value
            if display_size is not None and not _surrogate_doc_is_usable_for_display(
                doc,
                display_size=display_size,
            ):
                length = len((doc or {}).get("surrogate") or [])
                level = "Configured" if explicit else "Inferred"
                messages.append(
                    f"{level} surrogate JSON skipped ({value}): model grid too small "
                    f"for display ({length} values for {display_size[0]}x{display_size[1]} grid)."
                )
                continue
            messages.append(f"Loaded surrogate JSON from {kind}: {value}")
            return doc, messages, effective
        except FileNotFoundError as exc:
            level = "Configured" if explicit else "Inferred"
            messages.append(f"{level} surrogate JSON not loaded: {exc}")
        except Exception as exc:
            level = "Configured" if explicit else "Inferred"
            messages.append(f"{level} surrogate JSON skipped: {exc}")

    return None, messages, effective


def _sibling_next_x_path(data_path: str) -> str:
    p = (data_path or "").strip()
    if not p or _looks_like_http_url(p):
        return ""
    suffix = parse_nsdf_data_filename(os.path.basename(p))
    if suffix is None and os.path.basename(p).lower() != "data.json":
        return ""
    if suffix:
        return os.path.join(os.path.dirname(p), f"next_x_{suffix}.json")
    return os.path.join(os.path.dirname(p), "next_x.json")


def _sibling_next_x_url(data_url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    u = (data_url or "").strip()
    if not _looks_like_http_url(u):
        return ""
    parts = urlparse(u)
    segments = [x for x in (parts.path or "").split("/") if x]
    if not segments:
        return ""
    suffix = parse_nsdf_data_filename(segments[-1])
    if suffix is None and segments[-1].lower() != "data.json":
        return ""
    segments[-1] = f"next_x_{suffix}.json" if suffix else "next_x.json"
    new_path = "/" + "/".join(segments)
    return urlunparse((parts.scheme, parts.netloc, new_path, parts.params, parts.query, parts.fragment))


def _sibling_next_x_s3_key(data_key: str) -> str:
    key = (data_key or "").strip()
    if not key:
        return ""
    basename = key.rsplit("/", 1)[-1]
    suffix = parse_nsdf_data_filename(basename)
    if suffix is None and basename.lower() != "data.json":
        return ""
    prefix = key[: -len(basename)]
    if suffix:
        return prefix + f"next_x_{suffix}.json"
    return prefix + "next_x.json"


def validate_nsdf_next_x_doc(value: Any) -> NSDFNextXData:
    """
    Validate ``next_x.json``.

    Current schema (single proposed point)::

        {"workflow_id": "...", "dataset_x_size": 16, "data": [[labx, labz]]}

    ``dataset_x_size`` is GP input metadata; plotting uses the first two values
    in each ``data`` row as ``(labx, labz)``. Legacy array-of-blocks schema
    is still accepted.
    """
    warnings: List[str] = []
    if value is None:
        return NSDFNextXData(warnings=warnings)
    if isinstance(value, list):
        blocks = _iter_next_x_workflow_blocks(value)
        if not blocks and value:
            return NSDFNextXData(
                warnings=["Skipping next_x JSON: legacy array entries must be JSON objects."],
            )
    elif isinstance(value, Mapping):
        blocks = [value]
    else:
        return NSDFNextXData(
            warnings=["Skipping next_x JSON: expected a JSON object or array."],
        )

    entries: List[NSDFNextXEntry] = []
    for i, item in enumerate(blocks):
        label = "next_x" if isinstance(value, Mapping) else f"next_x[{i}]"
        parsed = _parse_next_x_workflow_block(item, label=label, warnings=warnings)
        if parsed is not None:
            entries.append(parsed)
    return NSDFNextXData(entries=entries, warnings=warnings)


def load_optional_next_x_json(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Any], List[str], StrainDashboardPaths]:
    """
    Load optional ``next_x.json`` from explicit path/URL or inferred sibling.

    Missing inferred siblings are non-fatal.
    """
    messages: List[str] = []
    explicit_path = (paths.next_x_json_path or "").strip()
    explicit_url = (paths.next_x_json_url or "").strip()
    effective = StrainDashboardPaths(
        local_json_path=paths.local_json_path,
        json_url=paths.json_url,
        surrogate_json_path=paths.surrogate_json_path,
        surrogate_json_url=paths.surrogate_json_url,
        next_x_json_path=explicit_path,
        next_x_json_url=explicit_url,
        local_data_dir=paths.local_data_dir,
    )

    candidates: List[Tuple[str, str, bool]] = []
    if explicit_path:
        candidates.append(("path", explicit_path, True))
    if explicit_url:
        candidates.append(("url", explicit_url, True))
    if not paths.strict_triplet_paths:
        if not candidates:
            sib_path = _sibling_next_x_path(paths.local_json_path)
            if sib_path:
                candidates.append(("path", sib_path, False))
            else:
                sib_url = _sibling_next_x_url(paths.json_url)
                if sib_url:
                    candidates.append(("url", sib_url, False))

        tried = {value for _, value, _ in candidates}
        for path in _iter_next_x_path_candidates(paths):
            if path not in tried:
                candidates.append(("path", path, False))
                tried.add(path)
        if paths.json_url:
            from urllib.parse import urlparse

            data_name = (
                os.path.basename(urlparse(paths.json_url).path or "")
                if _looks_like_http_url(paths.json_url)
                else ""
            )
            data_suffixes = list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
            reference = _nsdf_reference_data_suffix(data_name, data_suffixes=data_suffixes)
            nx_suffixes = list_nsdf_next_x_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
            for suffix in _nsdf_suffixes_after_reference(nx_suffixes, reference):
                if not is_valid_nsdf_version_suffix(suffix):
                    continue
                try:
                    _, _, nx_fn = nsdf_triplet_basenames(suffix)
                except ValueError:
                    continue
                nx_url = _replace_url_basename(paths.json_url, nx_fn)
                if nx_url not in tried:
                    candidates.append(("url", nx_url, False))
                    tried.add(nx_url)

    for kind, value, explicit in candidates:
        try:
            if kind == "path":
                doc = load_json_from_local_path(value)
                effective.next_x_json_path = value
            else:
                doc = load_json_from_url(value, s3_auth_override=mongo_s3_auth)
                effective.next_x_json_url = value
            if not _next_x_doc_is_recognized_format(doc):
                raise ValueError("next_x.json must be a JSON object or array.")
            messages.append(f"Loaded next_x JSON from {kind}: {value}")
            return doc, messages, effective
        except FileNotFoundError as exc:
            level = "Configured" if explicit else "Inferred"
            messages.append(f"{level} next_x JSON not loaded: {exc}")
        except Exception as exc:
            level = "Configured" if explicit else "Inferred"
            messages.append(f"{level} next_x JSON skipped: {exc}")

    return None, messages, effective


def load_nsdf_json_bundle_from_local_data_dir(paths: StrainDashboardPaths) -> Optional[NSDFLoadedBundle]:
    """Load fixed-name data/surrogate JSON files from LOCAL_DATA_DIR when present."""
    local_dir = (paths.local_data_dir or "").strip()
    if not local_dir:
        return None
    data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
    _, sur_fn, _ = nsdf_triplet_basenames(
        paths.surrogate_version_suffix if paths.strict_triplet_paths else paths.version_suffix or ""
    )
    _, _, nx_fn = nsdf_triplet_basenames(
        paths.next_x_version_suffix if paths.strict_triplet_paths else paths.version_suffix or ""
    )
    data_path = os.path.join(local_dir, data_fn)
    if not os.path.isfile(data_path):
        return None

    data = load_json_from_local_path(data_path)
    display_size, _ = resolve_nsdf_grid_size(data)
    surrogate = None
    messages = [f"Loaded NSDF data JSON from local path: {data_path}"]
    effective = StrainDashboardPaths(
        local_json_path=data_path,
        surrogate_json_path="",
        next_x_json_path="",
        local_data_dir=local_dir,
        version_suffix=paths.version_suffix,
        surrogate_version_suffix=paths.surrogate_version_suffix,
        next_x_version_suffix=paths.next_x_version_suffix,
        strict_triplet_paths=paths.strict_triplet_paths,
    )
    surrogate_paths = (
        [os.path.join(local_dir, sur_fn)]
        if paths.strict_triplet_paths
        else _iter_surrogate_path_candidates(paths, data_path=data_path, local_dir=local_dir)
    )
    for surrogate_path in surrogate_paths:
        if not os.path.isfile(surrogate_path):
            continue
        try:
            candidate_doc = load_json_from_local_path(surrogate_path)
            if not _surrogate_doc_is_usable_for_display(
                candidate_doc,
                display_size=display_size,
            ):
                length = len((candidate_doc or {}).get("surrogate") or [])
                messages.append(
                    f"Local surrogate JSON skipped ({surrogate_path}): "
                    f"model grid too small for display ({length} values for "
                    f"{display_size[0]}x{display_size[1]} grid)."
                )
                continue
            surrogate = candidate_doc
            effective.surrogate_json_path = surrogate_path
            messages.append(f"Loaded surrogate JSON from local path: {surrogate_path}")
            break
        except Exception as exc:
            messages.append(f"Local surrogate JSON skipped ({surrogate_path}): {exc}")
    next_x = None
    next_x_paths = (
        [os.path.join(local_dir, nx_fn)]
        if paths.strict_triplet_paths
        else _iter_next_x_path_candidates(paths, data_path=data_path, local_dir=local_dir)
    )
    for next_x_path in next_x_paths:
        if not os.path.isfile(next_x_path):
            continue
        try:
            next_x_doc = load_json_from_local_path(next_x_path)
            if _next_x_doc_is_recognized_format(next_x_doc):
                next_x = next_x_doc
                effective.next_x_json_path = next_x_path
                messages.append(f"Loaded next_x JSON from local path: {next_x_path}")
                break
            messages.append(
                f"Local next_x JSON skipped ({next_x_path}): expected a JSON object or array."
            )
        except Exception as exc:
            messages.append(f"Local next_x JSON skipped ({next_x_path}): {exc}")
    return NSDFLoadedBundle(data=data, surrogate=surrogate, next_x=next_x, messages=messages, paths=effective)


def load_nsdf_json_bundle(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> NSDFLoadedBundle:
    if _local_data_dir_active(paths):
        local_bundle = load_nsdf_json_bundle_from_local_data_dir(paths)
        if local_bundle is not None:
            return local_bundle
        data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
        local_dir = (paths.local_data_dir or "").strip()
        raise FileNotFoundError(
            f"Local NSDF file not found: {os.path.join(local_dir, data_fn)} "
            f"(LOCAL_DATA_DIR={local_dir!r})"
        )

    load_paths = _prepare_nsdf_load_paths(paths, remote_linked=remote_linked)
    if load_paths.has_s3_source():
        return load_nsdf_json_bundle_from_s3(load_paths, mongo_s3_auth=mongo_s3_auth)
    data = load_strain_json(load_paths, mongo_s3_auth=mongo_s3_auth)
    surrogate, messages, effective = load_optional_surrogate_json(
        load_paths,
        data_doc=data,
        mongo_s3_auth=mongo_s3_auth,
    )
    next_x, next_x_messages, effective = load_optional_next_x_json(
        effective,
        mongo_s3_auth=mongo_s3_auth,
    )
    messages.extend(next_x_messages)
    return NSDFLoadedBundle(
        data=data,
        surrogate=surrogate,
        next_x=next_x,
        messages=messages,
        paths=effective,
    )


def _nsdf_key_prefix(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return ""
    basename = key.rsplit("/", 1)[-1]
    return key[: -len(basename)] if basename else ""


def _iter_surrogate_path_candidates(
    paths: StrainDashboardPaths,
    *,
    data_path: str = "",
    local_dir: str = "",
) -> List[str]:
    """Ordered surrogate.json paths to try, including timestamped fallbacks."""
    candidates: List[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        p = (path or "").strip()
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    if paths.strict_triplet_paths:
        add((paths.surrogate_json_path or "").strip())
        return candidates

    data_path = (data_path or paths.local_json_path or "").strip()
    local_dir = (local_dir or paths.local_data_dir or "").strip()
    directory = local_dir or (os.path.dirname(data_path) if data_path else "")

    add((paths.surrogate_json_path or "").strip())
    add(_sibling_surrogate_path(data_path))

    data_basename = os.path.basename(data_path) if data_path else ""
    data_suffixes = list_nsdf_version_suffixes_from_directory(directory)
    reference = _nsdf_reference_data_suffix(data_basename, data_suffixes=data_suffixes)
    sur_suffixes = list_nsdf_surrogate_suffixes_from_directory(directory)
    for listed_suffix in _nsdf_suffixes_after_reference(sur_suffixes, reference):
        if not is_valid_nsdf_version_suffix(listed_suffix):
            continue
        try:
            _, sur_fn, _ = nsdf_triplet_basenames(listed_suffix)
        except ValueError:
            continue
        if directory:
            add(os.path.join(directory, sur_fn))
    return candidates


def _iter_next_x_path_candidates(
    paths: StrainDashboardPaths,
    *,
    data_path: str = "",
    local_dir: str = "",
) -> List[str]:
    """Ordered next_x.json paths to try, including timestamped fallbacks."""
    candidates: List[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        p = (path or "").strip()
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    if paths.strict_triplet_paths:
        add((paths.next_x_json_path or "").strip())
        return candidates

    data_path = (data_path or paths.local_json_path or "").strip()
    local_dir = (local_dir or paths.local_data_dir or "").strip()
    directory = local_dir or (os.path.dirname(data_path) if data_path else "")

    add((paths.next_x_json_path or "").strip())
    add(_sibling_next_x_path(data_path))

    data_basename = os.path.basename(data_path) if data_path else ""
    data_suffixes = list_nsdf_version_suffixes_from_directory(directory)
    reference = _nsdf_reference_data_suffix(data_basename, data_suffixes=data_suffixes)
    nx_suffixes = list_nsdf_next_x_suffixes_from_directory(directory)
    for listed_suffix in _nsdf_suffixes_after_reference(nx_suffixes, reference):
        if not is_valid_nsdf_version_suffix(listed_suffix):
            continue
        try:
            _, _, nx_fn = nsdf_triplet_basenames(listed_suffix)
        except ValueError:
            continue
        if directory:
            add(os.path.join(directory, nx_fn))
    return candidates


def _data_s3_key_candidates(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Ordered data.json keys to try when env prefixes disagree (e.g. chess-data vs test-chess)."""
    return _iter_data_s3_key_candidates(paths, mongo_s3_auth=mongo_s3_auth)


def _iter_data_s3_key_candidates(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Ordered live ``data.json`` keys to try (no timestamped backup fallback for latest)."""
    data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
    candidates: List[str] = []
    seen: set[str] = set()

    def add(key: str) -> None:
        k = (key or "").strip()
        if k and k not in seen:
            seen.add(k)
            candidates.append(k)

    add((paths.s3_data_key or "").strip())
    for aux_key in (
        (paths.s3_surrogate_key or "").strip(),
        (paths.s3_next_x_key or "").strip(),
    ):
        if aux_key and "/" in aux_key:
            add(_replace_s3_key_basename(aux_key, data_fn))

    return candidates


def _add_s3_keys_for_matching_snapshot_id(
    add: Any,
    *,
    prefix: str,
    snapshot_id: str,
    listed_suffixes: Sequence[str],
    role: str,
) -> None:
    if not snapshot_id:
        return
    for listed_suffix in listed_suffixes:
        if triplet_snapshot_id(listed_suffix) != snapshot_id:
            continue
        try:
            data_fn, sur_fn, nx_fn = nsdf_triplet_basenames(listed_suffix)
        except ValueError:
            continue
        basename = {"data": data_fn, "surrogate": sur_fn, "next_x": nx_fn}.get(role, "")
        if basename:
            add(prefix + basename)


def _iter_surrogate_s3_key_candidates(
    paths: StrainDashboardPaths,
    data_key: str,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Ordered surrogate S3 keys to try, including timestamped fallbacks."""
    candidates: List[str] = []
    seen: set[str] = set()

    def add(key: str) -> None:
        k = (key or "").strip()
        if k and k not in seen:
            seen.add(k)
            candidates.append(k)

    if paths.strict_triplet_paths:
        data_basename = data_key.rsplit("/", 1)[-1] if data_key else ""
        versioned_data = bool(parse_nsdf_data_filename(data_basename))
        sur_suffix = (paths.surrogate_version_suffix or "").strip()
        prefix = _nsdf_key_prefix(data_key)
        _, live_sur_fn, _ = nsdf_triplet_basenames("")
        if versioned_data:
            if not sur_suffix:
                add(prefix + live_sur_fn)
            else:
                snapshot_id = triplet_snapshot_id(parse_nsdf_data_filename(data_basename) or "")
                if snapshot_id:
                    _add_s3_keys_for_matching_snapshot_id(
                        add,
                        prefix=prefix,
                        snapshot_id=snapshot_id,
                        listed_suffixes=list_nsdf_surrogate_suffixes_from_s3(
                            paths,
                            mongo_s3_auth=mongo_s3_auth,
                        ),
                        role="surrogate",
                    )
                add(_sibling_surrogate_s3_key(data_key))
                add((paths.s3_surrogate_key or "").strip())
                if not snapshot_id:
                    _add_s3_keys_for_matching_snapshot_id(
                        add,
                        prefix=prefix,
                        snapshot_id=snapshot_id,
                        listed_suffixes=list_nsdf_surrogate_suffixes_from_s3(
                            paths,
                            mongo_s3_auth=mongo_s3_auth,
                        ),
                        role="surrogate",
                    )
                add(prefix + live_sur_fn)
        else:
            add((paths.s3_surrogate_key or "").strip())
            add(_sibling_surrogate_s3_key(data_key))
        return candidates

    data_key = (data_key or "").strip()
    prefix = _nsdf_key_prefix(data_key)

    add((paths.s3_surrogate_key or "").strip())
    add(_sibling_surrogate_s3_key(data_key))

    data_basename = data_key.rsplit("/", 1)[-1] if data_key else ""
    data_suffixes = list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    reference = _nsdf_reference_data_suffix(data_basename, data_suffixes=data_suffixes)
    sur_suffixes = list_nsdf_surrogate_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    for listed_suffix in _nsdf_suffixes_after_reference(sur_suffixes, reference):
        if not is_valid_nsdf_version_suffix(listed_suffix):
            continue
        try:
            _, sur_fn, _ = nsdf_triplet_basenames(listed_suffix)
        except ValueError:
            continue
        add(prefix + sur_fn)
    return candidates


def _iter_next_x_s3_key_candidates(
    paths: StrainDashboardPaths,
    data_key: str,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Ordered next_x S3 keys to try, including timestamped fallbacks."""
    candidates: List[str] = []
    seen: set[str] = set()

    def add(key: str) -> None:
        k = (key or "").strip()
        if k and k not in seen:
            seen.add(k)
            candidates.append(k)

    if paths.strict_triplet_paths:
        data_basename = data_key.rsplit("/", 1)[-1] if data_key else ""
        versioned_data = bool(parse_nsdf_data_filename(data_basename))
        nx_suffix = (paths.next_x_version_suffix or "").strip()
        prefix = _nsdf_key_prefix(data_key)
        _, _, live_nx_fn = nsdf_triplet_basenames("")
        if versioned_data:
            if not nx_suffix:
                add(prefix + live_nx_fn)
            else:
                snapshot_id = triplet_snapshot_id(parse_nsdf_data_filename(data_basename) or "")
                if snapshot_id:
                    _add_s3_keys_for_matching_snapshot_id(
                        add,
                        prefix=prefix,
                        snapshot_id=snapshot_id,
                        listed_suffixes=list_nsdf_next_x_suffixes_from_s3(
                            paths,
                            mongo_s3_auth=mongo_s3_auth,
                        ),
                        role="next_x",
                    )
                add(_sibling_next_x_s3_key(data_key))
                add((paths.s3_next_x_key or "").strip())
                if not snapshot_id:
                    _add_s3_keys_for_matching_snapshot_id(
                        add,
                        prefix=prefix,
                        snapshot_id=snapshot_id,
                        listed_suffixes=list_nsdf_next_x_suffixes_from_s3(
                            paths,
                            mongo_s3_auth=mongo_s3_auth,
                        ),
                        role="next_x",
                    )
                add(prefix + live_nx_fn)
        else:
            add((paths.s3_next_x_key or "").strip())
            add(_sibling_next_x_s3_key(data_key))
        return candidates

    data_key = (data_key or "").strip()
    prefix = _nsdf_key_prefix(data_key)

    add((paths.s3_next_x_key or "").strip())
    add(_sibling_next_x_s3_key(data_key))

    data_basename = data_key.rsplit("/", 1)[-1] if data_key else ""
    data_suffixes = list_nsdf_version_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    reference = _nsdf_reference_data_suffix(data_basename, data_suffixes=data_suffixes)
    nx_suffixes = list_nsdf_next_x_suffixes_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    for listed_suffix in _nsdf_suffixes_after_reference(nx_suffixes, reference):
        if not is_valid_nsdf_version_suffix(listed_suffix):
            continue
        try:
            _, _, nx_fn = nsdf_triplet_basenames(listed_suffix)
        except ValueError:
            continue
        add(prefix + nx_fn)
    return candidates


def _sibling_surrogate_s3_key(data_key: str) -> str:
    key = (data_key or "").strip()
    if not key:
        return ""
    basename = key.rsplit("/", 1)[-1]
    suffix = parse_nsdf_data_filename(basename)
    if suffix is None and basename.lower() != "data.json":
        return ""
    prefix = key[: -len(basename)]
    if suffix:
        return prefix + f"surrogate_{suffix}.json"
    return prefix + "surrogate.json"


def _s3_env_values(paths: StrainDashboardPaths) -> Dict[str, str]:
    file_values = load_simple_env_file(paths.s3_env_file)

    def value(name: str, default: str = "") -> str:
        return (os.environ.get(name) or file_values.get(name) or default).strip()

    return {
        "aws_access_key_id": value("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": value("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": value("AWS_SESSION_TOKEN"),
        "endpoint_url": paths.s3_endpoint_url or value("S3_ENDPOINT_URL"),
        "region_name": paths.s3_region or value("S3_REGION", "us-east-1") or "us-east-1",
    }


def _make_nsdf_s3_client(
    paths: StrainDashboardPaths,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
):
    import boto3
    from botocore.config import Config as BotoConfig

    cfg = _s3_env_values(paths)
    if mongo_s3_auth:
        ak = (mongo_s3_auth.get("access_key_id") or "").strip()
        sk = (mongo_s3_auth.get("secret_access_key") or "").strip()
        if ak and sk:
            cfg["aws_access_key_id"] = ak
            cfg["aws_secret_access_key"] = sk
        ep = (mongo_s3_auth.get("endpoint_url") or "").strip()
        if ep:
            cfg["endpoint_url"] = ep.rstrip("/")
        reg = (mongo_s3_auth.get("region_name") or "").strip()
        if reg:
            cfg["region_name"] = reg
        tok = (mongo_s3_auth.get("aws_session_token") or "").strip()
        if tok:
            cfg["aws_session_token"] = tok
    if not (cfg["aws_access_key_id"] and cfg["aws_secret_access_key"]):
        gateway = _parse_gateway_url_with_query_keys((paths.json_url or "").strip())
        if gateway:
            cfg["aws_access_key_id"] = (gateway.get("access_key_id") or "").strip()
            cfg["aws_secret_access_key"] = (gateway.get("secret_access_key") or "").strip()
            if not cfg["endpoint_url"]:
                cfg["endpoint_url"] = (gateway.get("endpoint_url") or "").strip()
            if gateway.get("region_name"):
                cfg["region_name"] = (gateway.get("region_name") or cfg["region_name"]).strip()
            tok = (gateway.get("aws_session_token") or "").strip()
            if tok:
                cfg["aws_session_token"] = tok

    kwargs: Dict[str, Any] = {
        "region_name": cfg["region_name"],
        "config": BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    }
    if cfg["endpoint_url"]:
        kwargs["endpoint_url"] = cfg["endpoint_url"]
    if cfg["aws_access_key_id"] and cfg["aws_secret_access_key"]:
        kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
    if cfg["aws_session_token"]:
        kwargs["aws_session_token"] = cfg["aws_session_token"]
    return boto3.client("s3", **kwargs)


def _format_s3_get_object_error(exc: Exception, *, bucket: str, key: str) -> str:
    key_label = (key or "").strip() or "(empty key)"
    bucket_label = (bucket or "").strip() or "(empty bucket)"
    try:
        from botocore.exceptions import ClientError

        if isinstance(exc, ClientError):
            err = (exc.response or {}).get("Error") or {}
            code = str(err.get("Code") or "ClientError")
            message = str(err.get("Message") or "").strip()
            detail = message or str(exc)
            return (
                f"S3 GetObject failed ({code}) for s3://{bucket_label}/{key_label}: {detail}"
            )
    except Exception:
        pass
    return f"S3 GetObject failed for s3://{bucket_label}/{key_label}: {exc}"


def _load_json_from_s3_key(client: Any, bucket: str, key: str) -> Any:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise FileNotFoundError(_format_s3_get_object_error(exc, bucket=bucket, key=key)) from exc
    raw = resp["Body"].read()
    return json.loads(raw.decode("utf-8"))


def _s3_missing_error(exc: Exception) -> bool:
    try:
        from botocore.exceptions import ClientError
    except Exception:
        return False
    if not isinstance(exc, ClientError):
        return False
    code = str(((exc.response or {}).get("Error") or {}).get("Code") or "")
    return code in ("NoSuchKey", "NoSuchBucket", "404", "NotFound")


def _parse_nsdf_points_to_predict(
    surrogate_doc: Mapping[str, Any],
    warnings: List[str],
) -> Optional[np.ndarray]:
    """Parse ``points_to_predict`` coordinate pairs from ``surrogate.json``."""
    raw = surrogate_doc.get("points_to_predict")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        warnings.append("Skipping points_to_predict: expected a non-empty coordinate list.")
        return None
    coords: List[Tuple[float, float]] = []
    for i, row_value in enumerate(raw):
        if not isinstance(row_value, list) or len(row_value) < 2:
            warnings.append(f"points_to_predict[{i}] must contain at least two numeric values.")
            continue
        x, z = row_value[0], row_value[1]
        if not (_is_number(x) and _is_number(z)):
            warnings.append(f"points_to_predict[{i}] must contain numeric labx/labz values.")
            continue
        fx, fz = float(x), float(z)
        if not (math.isfinite(fx) and math.isfinite(fz)):
            warnings.append(f"points_to_predict[{i}] labx/labz values must be finite.")
            continue
        coords.append((fx, fz))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def _surrogate_values_are_point_aligned(
    surrogate_doc: Optional[Mapping[str, Any]],
) -> bool:
    """Return True when flattened surrogate arrays align with ``points_to_predict``."""
    if not isinstance(surrogate_doc, Mapping):
        return False
    pts = surrogate_doc.get("points_to_predict")
    sur = surrogate_doc.get("surrogate")
    if not isinstance(pts, list) or not isinstance(sur, list):
        return False
    return len(pts) > 0 and len(pts) == len(sur)


def _surrogate_list_is_usable_for_display(
    surrogate_values: Any,
    *,
    display_size: Tuple[int, int],
    surrogate_doc: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Reject stub surrogate arrays (e.g. two zeros) that flatten estimate/variance heatmaps."""
    if _surrogate_values_are_point_aligned(surrogate_doc):
        return True
    if not isinstance(surrogate_values, list) or len(surrogate_values) < 4:
        return False
    for value in surrogate_values:
        if not _is_number(value):
            return False
        if not math.isfinite(float(value)):
            return False
    if len(surrogate_values) <= 4 and all(float(value) == 0.0 for value in surrogate_values):
        return False
    length = len(surrogate_values)
    display_nx, display_ny = display_size
    if length == display_nx * display_ny:
        return True
    side = int(round(math.sqrt(length)))
    if side >= 2 and side * side == length:
        return True
    for ny in range(2, int(math.sqrt(length)) + 1):
        if length % ny != 0:
            continue
        nx = length // ny
        if nx >= ny:
            return True
    return False


def _surrogate_doc_is_usable_for_display(
    doc: Optional[Mapping[str, Any]],
    *,
    display_size: Tuple[int, int],
) -> bool:
    if not isinstance(doc, Mapping):
        return False
    return _surrogate_list_is_usable_for_display(
        doc.get("surrogate"),
        display_size=display_size,
        surrogate_doc=doc,
    )


def load_nsdf_json_bundle_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> NSDFLoadedBundle:
    bucket = (paths.s3_bucket or "").strip()
    if not bucket:
        raise FileNotFoundError(
            "Set S3_BUCKET and S3_DATA_KEY to load NSDF data from S3."
        )

    client = _make_nsdf_s3_client(paths, mongo_s3_auth=mongo_s3_auth)
    data = None
    data_key = ""
    data_errors: List[str] = []
    data_candidates = _iter_data_s3_key_candidates(paths, mongo_s3_auth=mongo_s3_auth)
    for candidate_key in data_candidates:
        try:
            data = _load_json_from_s3_key(client, bucket, candidate_key)
            data_key = candidate_key
            break
        except FileNotFoundError as exc:
            data_errors.append(str(exc))
        except Exception as exc:
            if _s3_missing_error(exc):
                data_errors.append(_format_s3_get_object_error(exc, bucket=bucket, key=candidate_key))
                continue
            raise
    if data is None or not data_key:
        tried = ", ".join(data_candidates) or "(none)"
        detail = data_errors[-1] if data_errors else "object not found"
        if not (paths.version_suffix or "").strip():
            live_key = (paths.s3_data_key or "").strip() or "data.json"
            detail = (
                f"Live {live_key} not found on s3://{bucket}/. "
                f"The dashboard watches the rolling live triplet (data.json, surrogate.json, "
                f"next_x.json); timestamped backups are loaded only when selected explicitly. "
                f"{detail}"
            )
        raise FileNotFoundError(
            f"Could not load NSDF data.json from s3://{bucket}/. Tried: {tried}. {detail}"
        )
    loaded_suffix = parse_nsdf_data_filename(data_key.rsplit("/", 1)[-1]) or ""
    messages = [f"Loaded NSDF data JSON from s3://{bucket}/{data_key}"]
    aux_paths = paths
    if loaded_suffix:
        aux_paths = apply_nsdf_file_suffixes(
            paths,
            data_suffix=loaded_suffix,
            surrogate_suffix=(paths.surrogate_version_suffix or "").strip(),
            next_x_suffix=(paths.next_x_version_suffix or "").strip(),
            strict=paths.strict_triplet_paths,
        )
    display_size, _ = resolve_nsdf_grid_size(data)

    surrogate = None
    surrogate_key = ""
    surrogate_candidates = _iter_surrogate_s3_key_candidates(
        aux_paths,
        data_key,
        mongo_s3_auth=mongo_s3_auth,
    )
    surrogate_probe_skips: List[str] = []
    for candidate_key in surrogate_candidates:
        try:
            candidate_doc = _load_json_from_s3_key(client, bucket, candidate_key)
            if not _surrogate_doc_is_usable_for_display(
                candidate_doc,
                display_size=display_size,
            ):
                length = len((candidate_doc or {}).get("surrogate") or [])
                surrogate_probe_skips.append(
                    f"S3 surrogate JSON skipped ({candidate_key}): "
                    f"model grid too small for display ({length} values for "
                    f"{display_size[0]}x{display_size[1]} grid)."
                )
                continue
            surrogate = candidate_doc
            surrogate_key = candidate_key
            messages.append(f"Loaded surrogate JSON from s3://{bucket}/{candidate_key}")
            break
        except FileNotFoundError as exc:
            surrogate_probe_skips.append(f"S3 surrogate JSON skipped ({candidate_key}): {exc}")
            continue
        except Exception as exc:
            if _s3_missing_error(exc):
                surrogate_probe_skips.append(
                    f"S3 surrogate JSON skipped ({candidate_key}): "
                    f"{_format_s3_get_object_error(exc, bucket=bucket, key=candidate_key)}"
                )
                continue
            messages.append(f"S3 surrogate JSON skipped ({candidate_key}): {exc}")
            break
    if surrogate is None:
        messages.extend(surrogate_probe_skips)

    next_x = None
    next_x_key = ""
    next_x_probe_skips: List[str] = []
    for candidate_key in _iter_next_x_s3_key_candidates(
        aux_paths,
        data_key,
        mongo_s3_auth=mongo_s3_auth,
    ):
        try:
            next_x_doc = _load_json_from_s3_key(client, bucket, candidate_key)
            if _next_x_doc_is_recognized_format(next_x_doc):
                next_x = next_x_doc
                next_x_key = candidate_key
                if loaded_suffix and _location_basename(candidate_key) == nsdf_triplet_basenames("")[2]:
                    messages.append(
                        f"Loaded live next_x JSON from s3://{bucket}/{candidate_key} "
                        f"(overlay for data snapshot {loaded_suffix})"
                    )
                else:
                    messages.append(f"Loaded next_x JSON from s3://{bucket}/{candidate_key}")
                break
            next_x_probe_skips.append(
                f"S3 next_x JSON skipped ({candidate_key}): expected a JSON object or array."
            )
        except FileNotFoundError as exc:
            next_x_probe_skips.append(f"S3 next_x JSON skipped ({candidate_key}): {exc}")
            continue
        except Exception as exc:
            if _s3_missing_error(exc):
                next_x_probe_skips.append(
                    f"S3 next_x JSON skipped ({candidate_key}): "
                    f"{_format_s3_get_object_error(exc, bucket=bucket, key=candidate_key)}"
                )
                continue
            messages.append(f"S3 next_x JSON skipped ({candidate_key}): {exc}")
            break
    if next_x is None:
        messages.extend(next_x_probe_skips)

    effective_version = (paths.version_suffix or "").strip() or loaded_suffix
    surrogate_version = (aux_paths.surrogate_version_suffix or "").strip()
    next_x_version = (aux_paths.next_x_version_suffix or "").strip()
    live_sur_basename = nsdf_triplet_basenames("")[1]
    live_nx_basename = nsdf_triplet_basenames("")[2]
    if surrogate_key:
        parsed_sur_suffix = parse_nsdf_surrogate_filename(_location_basename(surrogate_key))
        if parsed_sur_suffix is not None:
            surrogate_version = parsed_sur_suffix
        elif _location_basename(surrogate_key) == live_sur_basename and loaded_suffix:
            surrogate_version = ""
    if next_x_key:
        parsed_nx_suffix = parse_nsdf_next_x_filename(_location_basename(next_x_key))
        if parsed_nx_suffix is not None:
            next_x_version = parsed_nx_suffix
        elif _location_basename(next_x_key) == live_nx_basename and loaded_suffix:
            next_x_version = ""

    effective = StrainDashboardPaths(
        s3_env_file=paths.s3_env_file,
        local_data_dir=paths.local_data_dir,
        s3_bucket=bucket,
        s3_data_key=data_key,
        s3_surrogate_key=surrogate_key,
        s3_next_x_key=next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        version_suffix=effective_version,
        surrogate_version_suffix=surrogate_version,
        next_x_version_suffix=next_x_version,
        strict_triplet_paths=paths.strict_triplet_paths,
    )
    if not surrogate and surrogate_candidates:
        if (effective_version or "").strip():
            messages.append(
                "S3 surrogate archive not found for this data snapshot "
                "(data-only backup); plots 2–3 use measurement interpolation."
            )
        else:
            messages.append(
                "S3 surrogate JSON not found for the live triplet (surrogate.json)."
            )

    return NSDFLoadedBundle(data=data, surrogate=surrogate, next_x=next_x, messages=messages, paths=effective)


def _location_basename(location: str) -> str:
    loc = (location or "").strip()
    if not loc:
        return ""
    if _looks_like_http_url(loc):
        from urllib.parse import urlparse

        path = (urlparse(loc).path or "").strip()
        return os.path.basename(path) if path else loc
    return os.path.basename(loc)


def primary_nsdf_triplet_locations(paths: StrainDashboardPaths) -> Dict[str, str]:
    """Expected primary ``data`` / ``surrogate`` / ``next_x`` paths for the active snapshot."""
    data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
    sur_suffix = (
        paths.surrogate_version_suffix
        if paths.strict_triplet_paths
        else paths.version_suffix or ""
    )
    nx_suffix = (
        paths.next_x_version_suffix
        if paths.strict_triplet_paths
        else paths.version_suffix or ""
    )
    _, sur_fn, _ = nsdf_triplet_basenames(sur_suffix)
    _, _, nx_fn = nsdf_triplet_basenames(nx_suffix)
    loc: Dict[str, str] = {"data": "", "surrogate": "", "next_x": ""}

    if paths.has_s3_source():
        prefix = _nsdf_key_prefix(paths.s3_data_key)
        loc["data"] = (paths.s3_data_key or "").strip()
        loc["surrogate"] = prefix + sur_fn
        loc["next_x"] = prefix + nx_fn
        return loc

    if (paths.local_data_dir or "").strip():
        base = paths.local_data_dir.rstrip("/")
        loc["data"] = os.path.join(base, data_fn)
        loc["surrogate"] = os.path.join(base, sur_fn)
        loc["next_x"] = os.path.join(base, nx_fn)
        return loc

    if (paths.local_json_path or "").strip() and not _looks_like_http_url(paths.local_json_path):
        base = os.path.dirname(paths.local_json_path)
        loc["data"] = os.path.join(base, data_fn)
        loc["surrogate"] = os.path.join(base, sur_fn)
        loc["next_x"] = os.path.join(base, nx_fn)
        return loc

    if (paths.json_url or "").strip():
        loc["data"] = (paths.json_url or "").strip()
        loc["surrogate"] = (paths.surrogate_json_url or "").strip() or _sibling_surrogate_url(
            paths.json_url
        )
        loc["next_x"] = (paths.next_x_json_url or "").strip() or _sibling_next_x_url(paths.json_url)
    return loc


def _parsed_triplet_role_suffix(basename: str, role: str) -> Optional[str]:
    """Suffix token from a triplet basename, or ``None`` for the live rolling file."""
    parsers = {
        "data": parse_nsdf_data_filename,
        "surrogate": parse_nsdf_surrogate_filename,
        "next_x": parse_nsdf_next_x_filename,
    }
    parser = parsers.get(role)
    if not parser:
        return None
    return parser((basename or "").strip())


def _triplet_role_files_match_by_snapshot_id(
    primary_basename: str,
    loaded_basename: str,
    role: str,
) -> bool:
    """
    True when two triplet files are the same object or share a snapshot id.

    DIAL may write ``data_<tsA>_8.json`` and ``surrogate_<tsB>_8.json`` with different
    timestamps; matching by trailing id is intentional, not a load failure.
    """
    primary_base = (primary_basename or "").strip()
    loaded_base = (loaded_basename or "").strip()
    if not primary_base or not loaded_base:
        return False
    if primary_base == loaded_base:
        return True
    primary_suffix = _parsed_triplet_role_suffix(primary_base, role)
    loaded_suffix = _parsed_triplet_role_suffix(loaded_base, role)
    if primary_suffix is None or loaded_suffix is None:
        return False
    primary_id = triplet_snapshot_id(primary_suffix)
    if not primary_id:
        return False
    return primary_id == triplet_snapshot_id(loaded_suffix)


def _message_targets_location(message: str, location: str) -> bool:
    msg = (message or "").strip()
    loc = (location or "").strip()
    if not msg or not loc:
        return False
    if loc in msg:
        return True
    base = _location_basename(loc)
    return bool(base and base in msg)


def _viewing_live_triplet(paths: StrainDashboardPaths) -> bool:
    """True when selectors point at the rolling live trio (``data.json``, etc.)."""
    return not (paths.version_suffix or "").strip()


def collect_nsdf_triplet_load_issues(
    paths: StrainDashboardPaths,
    bundle: NSDFLoadedBundle,
    *,
    grid_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Summarize problems with the primary ``data.json`` / ``surrogate.json`` / ``next_x.json`` triplet.

    Returns ``(errors, warnings)``. Errors indicate the live triplet or selected archive
    surrogate is missing when required for the primary view. Warnings cover optional
    ``next_x.json`` gaps and data-only archived snapshots (measurements without a
    matching surrogate backup on S3).
    """
    errors: List[str] = []
    warnings: List[str] = []
    live_triplet = _viewing_live_triplet(paths)
    primary = primary_nsdf_triplet_locations(paths)
    data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
    sur_suffix = (
        paths.surrogate_version_suffix
        if paths.strict_triplet_paths
        else paths.version_suffix or ""
    )
    nx_suffix = (
        paths.next_x_version_suffix
        if paths.strict_triplet_paths
        else paths.version_suffix or ""
    )
    _, sur_fn, _ = nsdf_triplet_basenames(sur_suffix)
    _, _, nx_fn = nsdf_triplet_basenames(nx_suffix)
    loaded_paths = bundle.paths

    loaded_sur = (
        (loaded_paths.s3_surrogate_key or "").strip()
        or (loaded_paths.surrogate_json_path or "").strip()
        or (loaded_paths.surrogate_json_url or "").strip()
    )
    loaded_nx = (
        (loaded_paths.s3_next_x_key or "").strip()
        or (loaded_paths.next_x_json_path or "").strip()
        or (loaded_paths.next_x_json_url or "").strip()
    )
    primary_sur = (primary.get("surrogate") or "").strip()
    primary_nx = (primary.get("next_x") or "").strip()

    sur_id_match = bool(
        bundle.surrogate is not None
        and primary_sur
        and loaded_sur
        and _triplet_role_files_match_by_snapshot_id(
            _location_basename(primary_sur),
            _location_basename(loaded_sur),
            "surrogate",
        )
    )
    nx_id_match = bool(
        bundle.next_x is not None
        and primary_nx
        and loaded_nx
        and _triplet_role_files_match_by_snapshot_id(
            _location_basename(primary_nx),
            _location_basename(loaded_nx),
            "next_x",
        )
    )

    for msg in bundle.messages:
        if _message_targets_location(msg, primary_sur) and any(
            token in msg.lower() for token in ("skipped", "not found", "not loaded", "missing")
        ):
            if sur_id_match:
                continue
            detail = msg.split(":", 1)[-1].strip() if ":" in msg else msg
            if live_triplet:
                errors.append(f"{sur_fn}: {detail}")
            else:
                warnings.append(
                    f"{sur_fn}: surrogate archive not on S3 for this data snapshot "
                    f"({detail}); plots 2–3 use measurement interpolation."
                )
        elif _message_targets_location(msg, primary_nx) and any(
            token in msg.lower() for token in ("skipped", "not found", "not loaded", "missing")
        ):
            if nx_id_match:
                continue
            detail = msg.split(":", 1)[-1].strip() if ":" in msg else msg
            if "expected a json object or array" in msg.lower():
                errors.append(f"{nx_fn}: {detail}")
            else:
                warnings.append(f"{nx_fn}: {detail}")

    if bundle.surrogate is None:
        if not any(sur_fn in item for item in errors + warnings):
            if live_triplet:
                errors.append(f"{sur_fn}: not loaded (missing or every fallback failed).")
            else:
                warnings.append(
                    f"{sur_fn}: data-only archive (surrogate backup not on S3); "
                    "plots 2–3 use measurement interpolation."
                )
    elif primary_sur:
        primary_base = _location_basename(primary_sur)
        loaded_base = _location_basename(loaded_sur)
        if (
            primary_base
            and loaded_base
            and primary_base != loaded_base
            and not _triplet_role_files_match_by_snapshot_id(
                primary_base,
                loaded_base,
                "surrogate",
            )
        ):
            if live_triplet:
                errors.append(
                    f"{sur_fn}: using fallback {loaded_base} "
                    f"(primary {primary_base} missing or invalid)."
                )

    if bundle.next_x is None and primary_nx:
        if not any(nx_fn in item for item in errors + warnings):
            warnings.append(f"{nx_fn}: not loaded (optional file missing).")

    if isinstance(grid_meta, Mapping):
        estimate_source = str(grid_meta.get("estimate_source") or "")
        if estimate_source == "dataset_y_idw":
            for note in grid_meta.get("warnings") or []:
                if "Surrogate model grid was empty" in str(note):
                    if not any(sur_fn in item for item in errors):
                        errors.append(
                            f"{sur_fn}: empty or stub model grid; "
                            "estimate interpolated from measurements."
                        )
                    break

    if not bundle.data:
        errors.append(f"{data_fn}: not loaded.")

    return errors, warnings


# ---------------------------------------------------------------------------
# NSDF field discovery
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_1d_surrogate_array_or_none(
    doc: Optional[Mapping[str, Any]],
    key: str,
    warnings: List[str],
) -> Optional[np.ndarray]:
    """Parse a surrogate model 1D field (length independent of measurement count)."""
    if not doc or key not in doc:
        return None
    value = doc.get(key)
    if not isinstance(value, list) or not value:
        warnings.append(f"Skipping surrogate field {key!r}: expected a non-empty numeric 1D list.")
        return None
    if any((not _is_number(x)) for x in value):
        warnings.append(f"Skipping surrogate field {key!r}: all values must be numeric.")
        return None
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        warnings.append(f"Skipping surrogate field {key!r}: values must be finite.")
        return None
    return arr


def _best_factor_grid_shape(
    length: int,
    *,
    target: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """
    Pick a non-degenerate (nx, ny) factorization for a flattened surrogate grid.

    When ``target`` is set (from bounds or display size), choose the shape whose
    dimensions are closest to the target. This avoids mis-reading a 2D model as
    1×N (vertical stripes on the dashboard).
    """
    candidates: List[Tuple[int, int]] = []
    for ny in range(1, length + 1):
        if length % ny != 0:
            continue
        candidates.append((length // ny, ny))
    if not candidates:
        return None
    multi_d = [shape for shape in candidates if shape[0] > 1 and shape[1] > 1]
    pool = multi_d if multi_d else candidates
    if target and target[0] > 0 and target[1] > 0:
        tx, ty = target
        return min(pool, key=lambda shape: abs(shape[0] - tx) + abs(shape[1] - ty))
    return min(pool, key=lambda shape: max(shape[0], shape[1]) / max(1, min(shape[0], shape[1])))


def _surrogate_source_grid_size(
    length: int,
    surrogate_info: NSDFSurrogateData,
    surrogate_doc: Optional[Mapping[str, Any]],
    display_size: Tuple[int, int],
    warnings: List[str],
) -> Optional[Tuple[int, int]]:
    """Infer the model grid shape for a flattened surrogate field."""
    if surrogate_info.bounds:
        bounds_size = infer_nsdf_bounds_grid_size({"bounds": list(surrogate_info.bounds)})
        if bounds_size and bounds_size[0] * bounds_size[1] == length:
            return bounds_size
    if isinstance(surrogate_doc, Mapping):
        bounds_size = infer_nsdf_bounds_grid_size(surrogate_doc)
        if bounds_size and bounds_size[0] * bounds_size[1] == length:
            return bounds_size
    display_nx, display_ny = display_size
    if length == display_nx * display_ny:
        return display_nx, display_ny
    side = int(round(math.sqrt(length)))
    if side > 0 and side * side == length:
        return side, side

    target: Optional[Tuple[int, int]] = None
    if surrogate_info.bounds:
        target = infer_nsdf_bounds_grid_size({"bounds": list(surrogate_info.bounds)})
    if target is None and isinstance(surrogate_doc, Mapping):
        target = infer_nsdf_bounds_grid_size(surrogate_doc)
    if target is None:
        target = display_size
    inferred = _best_factor_grid_shape(length, target=target)
    if inferred is not None:
        if target and inferred[0] * inferred[1] != target[0] * target[1]:
            warnings.append(
                f"Surrogate grid length {length} does not match bounds "
                f"{target[0]}x{target[1]}={target[0] * target[1]}; "
                f"using inferred shape {inferred[0]}x{inferred[1]}."
            )
        return inferred

    warnings.append(
        f"Could not infer surrogate grid shape for field length {length}; "
        "provide surrogate bounds."
    )
    return None


def _embed_grid_in_display(
    grid: np.ndarray,
    target_nx: int,
    target_ny: int,
    *,
    warnings: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Place a native model grid on the bounds canvas without resampling.

    Values are copied cell-for-cell from the origin; cells outside the model
    grid (or beyond the bounds canvas) are left as NaN so the UI shows blank.
    """
    src_ny, src_nx = grid.shape
    if src_nx == target_nx and src_ny == target_ny:
        return grid
    canvas = np.full((target_ny, target_nx), np.nan, dtype=np.float64)
    copy_nx = min(src_nx, target_nx)
    copy_ny = min(src_ny, target_ny)
    canvas[:copy_ny, :copy_nx] = grid[:copy_ny, :copy_nx]
    if warnings is None:
        return canvas
    if copy_nx < src_nx or copy_ny < src_ny:
        warnings.append(
            f"Surrogate grid {src_nx}x{src_ny} clipped to fit "
            f"{target_nx}x{target_ny} bounds canvas."
        )
    elif copy_nx < target_nx or copy_ny < target_ny:
        warnings.append(
            f"Surrogate grid {src_nx}x{src_ny} placed on "
            f"{target_nx}x{target_ny} bounds canvas (uncovered cells left blank)."
        )
    return canvas


def _scatter_field_to_display_grid(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    warnings: List[str],
) -> np.ndarray:
    """Place one scalar per ``points_to_predict`` coordinate on the bounds canvas."""
    canvas = np.full((ny, nx), np.nan, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        warnings.append("points_to_predict must be a 2D coordinate array.")
        return canvas
    if values.shape[0] != coordinates.shape[0]:
        warnings.append(
            f"Surrogate field length ({values.shape[0]}) does not match "
            f"points_to_predict ({coordinates.shape[0]})."
        )
        return canvas
    gx, gy = _norm_coordinates_to_grid(coordinates, nx, ny, bounds)
    placed = 0
    for x, y, val in zip(gx, gy, values):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if not math.isfinite(float(val)):
            continue
        ix = int(np.clip(round(float(x)), 0, nx - 1))
        iy = int(np.clip(round(float(y)), 0, ny - 1))
        canvas[iy, ix] = float(val)
        placed += 1
    if placed == 0:
        warnings.append("No surrogate values could be placed from points_to_predict.")
    return canvas


def _display_grid_physical_coords(
    nx: int,
    ny: int,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-center ``labx`` / ``labz`` meshes shaped ``(ny, nx)``."""
    (xlo, xhi), (zlo, zhi) = bounds[0], bounds[1]
    if nx <= 1:
        xs = np.array([(xlo + xhi) / 2.0], dtype=np.float64)
    else:
        xs = xlo + np.arange(nx, dtype=np.float64) * (xhi - xlo) / (nx - 1)
    if ny <= 1:
        zs = np.array([(zlo + zhi) / 2.0], dtype=np.float64)
    else:
        zs = zlo + np.arange(ny, dtype=np.float64) * (zhi - zlo) / (ny - 1)
    return np.meshgrid(xs, zs)


def _snap_coordinates_to_lattice_axes(
    coordinates: np.ndarray,
    values: np.ndarray,
    *,
    atol: float = 0.08,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect a labx×labz lattice in ``points_to_predict``.

    Returns ``(unique_labx, unique_labz, grid[labz, labx])`` when at least 85% of
    points snap to a rectangular lattice.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        return None
    labx = coordinates[:, 0]
    labz = coordinates[:, 1]
    vals = np.asarray(values, dtype=np.float64)
    ux = np.unique(np.round(labx, 4))
    uz = np.unique(np.round(labz, 4))
    if len(ux) < 2 or len(uz) < 2:
        return None
    expected = len(ux) * len(uz)
    if len(vals) < expected * 0.85 or len(vals) > expected * 1.05:
        return None

    grid = np.full((len(uz), len(ux)), np.nan, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int64)
    for x, z, val in zip(labx, labz, vals):
        if not (math.isfinite(x) and math.isfinite(z) and math.isfinite(float(val))):
            continue
        xi = int(np.argmin(np.abs(ux - x)))
        zi = int(np.argmin(np.abs(uz - z)))
        if abs(float(ux[xi]) - float(x)) > atol or abs(float(uz[zi]) - float(z)) > atol:
            return None
        if counts[zi, xi]:
            grid[zi, xi] = (grid[zi, xi] * counts[zi, xi] + float(val)) / (counts[zi, xi] + 1)
        else:
            grid[zi, xi] = float(val)
        counts[zi, xi] += 1
    if int(np.sum(counts > 0)) < len(vals) * 0.85:
        return None
    return ux, uz, grid


def _interpolate_lattice_to_display(
    ux: np.ndarray,
    uz: np.ndarray,
    grid: np.ndarray,
    *,
    xx: np.ndarray,
    zz: np.ndarray,
    warnings: List[str],
) -> Optional[np.ndarray]:
    """Bilinear interpolation from a lab lattice onto the display canvas."""
    try:
        from scipy.interpolate import RegularGridInterpolator
    except Exception as exc:
        warnings.append(f"Structured lattice interpolation unavailable ({exc}).")
        return None

    interp = RegularGridInterpolator(
        (uz, ux),
        grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    query = np.column_stack([zz.ravel(), xx.ravel()])
    filled = interp(query).reshape(zz.shape)
    if not np.any(np.isfinite(filled)):
        return None
    if np.any(~np.isfinite(filled)):
        try:
            from scipy.interpolate import NearestNDInterpolator

            known = np.isfinite(grid)
            if np.any(known):
                zi, xi = np.where(known)
                pts = np.column_stack([uz[zi], ux[xi]])
                nearest = NearestNDInterpolator(pts, grid[known])
                nan_mask = ~np.isfinite(filled)
                filled[nan_mask] = nearest(
                    np.column_stack([zz[nan_mask], xx[nan_mask]])
                )
        except Exception:
            pass
    return filled


def _physical_idw_fill_grid(
    labx: np.ndarray,
    labz: np.ndarray,
    values: np.ndarray,
    *,
    xx: np.ndarray,
    zz: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Inverse-distance fill using physical ``labx`` / ``labz`` distances."""
    pts_x = np.asarray(labx, dtype=np.float64).reshape(1, 1, -1)
    pts_z = np.asarray(labz, dtype=np.float64).reshape(1, 1, -1)
    pts_v = np.asarray(values, dtype=np.float64).reshape(1, 1, -1)
    grid_x = xx.reshape(1, xx.shape[1], 1)
    grid_z = zz.reshape(zz.shape[0], 1, 1)
    mask = np.isfinite(pts_x) & np.isfinite(pts_z) & np.isfinite(pts_v)
    pts_x = np.where(mask, pts_x, np.nan)
    pts_z = np.where(mask, pts_z, np.nan)
    pts_v = np.where(mask, pts_v, np.nan)
    d2 = (pts_x - grid_x) ** 2 + (pts_z - grid_z) ** 2 + eps
    w = d2 ** (-power / 2.0)
    w = np.where(np.isfinite(w), w, 0.0)
    pts_v = np.where(np.isfinite(pts_v), pts_v, 0.0)
    denom = np.sum(w, axis=2)
    numer = np.sum(w * pts_v, axis=2)
    out = np.full((zz.shape[0], xx.shape[1]), np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def _interpolate_scattered_points_to_display(
    labx: np.ndarray,
    labz: np.ndarray,
    values: np.ndarray,
    *,
    xx: np.ndarray,
    zz: np.ndarray,
    warnings: List[str],
) -> np.ndarray:
    """Unstructured fallback: linear ND in lab space, then physical IDW."""
    points = np.column_stack([labx, labz])
    vals = np.asarray(values, dtype=np.float64)
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        linear = LinearNDInterpolator(points, vals)
        filled = linear(xx, zz)
        if np.any(~np.isfinite(filled)):
            nearest = NearestNDInterpolator(points, vals)
            nan_mask = ~np.isfinite(filled)
            filled[nan_mask] = nearest(xx[nan_mask], zz[nan_mask])
        if np.any(np.isfinite(filled)):
            return filled
    except Exception as exc:
        warnings.append(f"Linear ND interpolation unavailable ({exc}); using physical IDW.")
    return _physical_idw_fill_grid(labx, labz, vals, xx=xx, zz=zz)


def _scatter_and_interpolate_field_to_display_grid(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    warnings: List[str],
) -> np.ndarray:
    """
    Interpolate ``points_to_predict`` onto the bounds canvas in physical lab space.

    CHESS surrogate points usually form a labx×labz lattice; those are bilinearly
    interpolated first. Unstructured clouds fall back to linear ND / physical IDW.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        warnings.append("points_to_predict must be a 2D coordinate array.")
        return np.full((ny, nx), np.nan, dtype=np.float64)
    if values.shape[0] != coordinates.shape[0]:
        warnings.append(
            f"Surrogate field length ({values.shape[0]}) does not match "
            f"points_to_predict ({coordinates.shape[0]})."
        )
        return np.full((ny, nx), np.nan, dtype=np.float64)
    if bounds is None:
        warnings.append("Cannot interpolate surrogate points without bounds.")
        return _scatter_field_to_display_grid(
            values, coordinates, nx=nx, ny=ny, bounds=bounds, warnings=warnings
        )

    labx = coordinates[:, 0]
    labz = coordinates[:, 1]
    xx, zz = _display_grid_physical_coords(nx, ny, bounds)

    lattice = _snap_coordinates_to_lattice_axes(coordinates, values)
    if lattice is not None:
        ux, uz, grid = lattice
        filled = _interpolate_lattice_to_display(ux, uz, grid, xx=xx, zz=zz, warnings=warnings)
        if filled is not None and np.any(np.isfinite(filled)):
            return filled

    return _interpolate_scattered_points_to_display(
        labx,
        labz,
        values,
        xx=xx,
        zz=zz,
        warnings=warnings,
    )


def _surrogate_field_to_display_grid(
    values: Optional[np.ndarray],
    *,
    key: str,
    surrogate_info: NSDFSurrogateData,
    surrogate_doc: Optional[Mapping[str, Any]],
    display_size: Tuple[int, int],
    warnings: List[str],
) -> Optional[np.ndarray]:
    """Map a surrogate model field onto the bounds canvas."""
    if values is None:
        return None
    display_nx, display_ny = display_size
    pts = surrogate_info.points_to_predict
    if pts is not None and pts.shape[0] == values.shape[0]:
        return _scatter_and_interpolate_field_to_display_grid(
            values,
            pts,
            nx=display_nx,
            ny=display_ny,
            bounds=surrogate_info.bounds,
            warnings=warnings,
        )

    length = int(values.shape[0])
    source_size = _surrogate_source_grid_size(
        length,
        surrogate_info,
        surrogate_doc,
        display_size,
        warnings,
    )
    if source_size is None:
        warnings.append(f"Skipping surrogate field {key!r}: unable to map length {length} to a grid.")
        return None
    src_nx, src_ny = source_size
    grid = values.reshape(src_ny, src_nx)
    display_nx, display_ny = display_size
    if (src_nx, src_ny) != (display_nx, display_ny):
        grid = _embed_grid_in_display(
            grid,
            display_nx,
            display_ny,
            warnings=warnings,
        )
    return grid


def _validate_bounds(value: Any) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    out: List[Tuple[float, float]] = []
    for idx in range(2):
        axis = value[idx]
        if not isinstance(axis, list) or len(axis) < 2:
            return None
        lo, hi = axis[0], axis[1]
        if not (_is_number(lo) and _is_number(hi)):
            return None
        flo, fhi = float(lo), float(hi)
        if not (math.isfinite(flo) and math.isfinite(fhi)) or flo == fhi:
            return None
        out.append((flo, fhi))
    return out[0], out[1]


def _coerce_bounds_pair(
    value: Any,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Normalize bounds stored as nested lists or tuples."""
    if value is None:
        return None
    if isinstance(value, list):
        return _validate_bounds(value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(axis, (list, tuple)) and len(axis) >= 2 for axis in value)
    ):
        as_list = [list(axis[:2]) for axis in value]
        return _validate_bounds(as_list)
    return None


def _resolve_strain_plot_bounds(
    meta: Mapping[str, Any],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Bounds for mapping lab coordinates onto the dashboard grid."""
    for key in ("measurement_bounds", "surrogate_bounds"):
        bounds = _coerce_bounds_pair(meta.get(key))
        if bounds is not None:
            return bounds
    return None


def validate_nsdf_measurement_doc(doc: Mapping[str, Any]) -> NSDFMeasurementData:
    """Validate native NSDF ``data.json`` and return normalized arrays."""
    if not isinstance(doc, Mapping):
        raise ValueError("NSDF data.json must be a JSON object.")
    if "dataset_x" not in doc:
        raise ValueError("NSDF data.json is missing required key 'dataset_x'.")
    if "dataset_y" not in doc:
        raise ValueError("NSDF data.json is missing required key 'dataset_y'.")

    dataset_x = doc.get("dataset_x")
    dataset_y = doc.get("dataset_y")
    if not isinstance(dataset_x, list):
        raise ValueError("'dataset_x' must be a list of coordinate pairs.")
    if not isinstance(dataset_y, list):
        raise ValueError("'dataset_y' must be a numeric 1D list.")
    if len(dataset_x) == 0 and len(dataset_y) == 0:
        bounds = _validate_bounds(doc.get("bounds"))
        metadata = {k: v for k, v in doc.items() if k not in ("dataset_x", "dataset_y")}
        return NSDFMeasurementData(
            coordinates=np.zeros((0, 2), dtype=np.float64),
            observed_values=np.zeros(0, dtype=np.float64),
            bounds=bounds,
            bounds_source="bounds" if bounds else "none",
            metadata=metadata,
        )
    if not dataset_x:
        raise ValueError("'dataset_x' must be a non-empty list of coordinate pairs.")
    if not dataset_y:
        raise ValueError("'dataset_y' must be a non-empty numeric 1D list.")
    if len(dataset_x) != len(dataset_y):
        raise ValueError(
            f"'dataset_x' length ({len(dataset_x)}) must match 'dataset_y' length ({len(dataset_y)})."
        )

    coords: List[Tuple[float, float]] = []
    for i, row_value in enumerate(dataset_x):
        if not isinstance(row_value, list) or len(row_value) < 2:
            raise ValueError(f"'dataset_x[{i}]' must contain at least two numeric values.")
        x, z = row_value[0], row_value[1]
        if not (_is_number(x) and _is_number(z)):
            raise ValueError(f"'dataset_x[{i}]' must contain numeric labx/labz values.")
        fx, fz = float(x), float(z)
        if not (math.isfinite(fx) and math.isfinite(fz)):
            raise ValueError(f"'dataset_x[{i}]' labx/labz values must be finite.")
        coords.append((fx, fz))

    if any((not _is_number(v)) for v in dataset_y):
        raise ValueError("'dataset_y' must contain only numeric values.")
    observed = np.asarray(dataset_y, dtype=np.float64)
    if observed.ndim != 1 or not np.all(np.isfinite(observed)):
        raise ValueError("'dataset_y' must be a finite numeric 1D list.")

    bounds = _validate_bounds(doc.get("bounds"))
    metadata = {k: v for k, v in doc.items() if k not in ("dataset_x", "dataset_y")}
    return NSDFMeasurementData(
        coordinates=np.asarray(coords, dtype=np.float64),
        observed_values=observed,
        bounds=bounds,
        bounds_source="bounds" if bounds else "observed_minmax",
        metadata=metadata,
    )


def _parse_nsdf_points(value: Any) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return value
    if _is_number(value):
        parsed = int(value)
        if parsed > 0:
            return parsed
    return None


def _valid_grid_size(value: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if value is None or len(value) < 2:
        return None
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def parse_nsdf_plot_dim(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``dim`` as a plot type string (``1D``, ``2D``, ``3D``)."""
    if value is None:
        return None, None
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"1D", "2D", "3D"}:
            return normalized, None
        return None, f"Skipping surrogate 'dim': expected '1D', '2D', or '3D', got {value!r}."
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return None, (
            "surrogate 'dim' as [width, height] is deprecated; use dim: \"2D\" "
            "and bounds for grid size."
        )
    if isinstance(value, Mapping):
        return None, (
            "surrogate 'dim' as {width, height} is deprecated; use dim: \"2D\" "
            "and bounds for grid size."
        )
    return None, f"Skipping surrogate 'dim': expected a plot type string, got {type(value).__name__}."


def _legacy_grid_size_from_dim_array(value: Any) -> Optional[Tuple[int, int]]:
    """Deprecated fallback: grid width/height encoded as a two-element ``dim`` array."""
    if isinstance(value, Mapping):
        for width_key, height_key in (
            ("width", "height"),
            ("grid_width", "grid_height"),
            ("nx", "ny"),
        ):
            if width_key in value and height_key in value:
                return _valid_grid_size((value[width_key], value[height_key]))
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _valid_grid_size((value[0], value[1]))
    return None


def infer_grid_size_from_dim(value: Any) -> Optional[Tuple[int, int]]:
    """Deprecated alias for legacy ``dim: [width, height]`` grid metadata."""
    return _legacy_grid_size_from_dim_array(value)


def _is_demo_workflow_id(workflow_id: str) -> bool:
    normalized = str(workflow_id or "").strip().lower()
    if not normalized:
        return True
    if normalized in {"dashboard-demo", "demo", "test", "test-workflow"}:
        return True
    return normalized.startswith("dashboard-demo")


def resolve_nsdf_workflow_id(
    surrogate_info: NSDFSurrogateData,
    next_x_info: NSDFNextXData,
) -> Optional[str]:
    """Return the active workflow id (surrogate first, then first real next_x entry)."""
    if surrogate_info.workflow_id and not _is_demo_workflow_id(surrogate_info.workflow_id):
        return surrogate_info.workflow_id
    for entry in next_x_info.entries:
        if entry.workflow_id and not _is_demo_workflow_id(entry.workflow_id):
            return entry.workflow_id
    return None


def _select_next_x_entry(
    next_x_info: NSDFNextXData,
    workflow_id: Optional[str],
) -> Optional[NSDFNextXEntry]:
    """Pick the next_x workflow block to plot (preferred id, else first real entry)."""
    preferred = (workflow_id or "").strip()
    if preferred:
        for entry in next_x_info.entries:
            if entry.workflow_id == preferred:
                return entry
    for entry in next_x_info.entries:
        if entry.workflow_id and not _is_demo_workflow_id(entry.workflow_id):
            return entry
    return None


def next_x_grid_coords_for_workflow(
    next_x_info: NSDFNextXData,
    workflow_id: Optional[str],
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map proposed next-scan coordinates onto the dashboard grid for one workflow."""
    entry = _select_next_x_entry(next_x_info, workflow_id)
    if entry is None or entry.coordinates.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    arr = np.asarray(entry.coordinates, dtype=np.float64)
    return _norm_positions_to_grid(arr[:, 0], arr[:, 1], nx, ny, bounds)


def _grid_display_coords(
    gx: np.ndarray,
    gy: np.ndarray,
    nx: int,
    ny: int,
    flip_y: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized grid indices to Bokeh figure coordinates (cell centers)."""
    if gx.size == 0:
        return gx, gy
    ix = np.clip(np.rint(gx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(gy).astype(int), 0, ny - 1)
    px = ix.astype(np.float64) + 0.5
    py = (ny - 1 - iy + 0.5).astype(np.float64) if flip_y else iy.astype(np.float64) + 0.5
    return px, py


def _attach_point_legend_below(
    figure: Any,
    legend_items: List[Tuple[str, Any]],
) -> None:
    """Place marker legend below the plot frame; always reserve footer space."""
    from bokeh.models import Legend, LegendItem

    if legend_items:
        legend = Legend(
            items=[LegendItem(label=label, renderers=[renderer]) for label, renderer in legend_items],
            location="center",
            orientation="horizontal",
            click_policy="hide",
            background_fill_alpha=0.92,
            border_line_alpha=0.0,
            label_text_font_size="9pt",
            spacing=14,
            margin=0,
            padding=6,
        )
        figure.add_layout(legend, "below")


def _add_grid_point_overlay(
    figure: Any,
    gx: np.ndarray,
    gy: np.ndarray,
    nx: int,
    ny: int,
    flip_y: bool,
    *,
    color: str,
    marker: str,
    size: int,
    line_color: Optional[str] = None,
    line_width: float = 1.5,
    fill_alpha: float = 0.95,
) -> Optional[Any]:
    if gx.size == 0:
        return None
    px, py = _grid_display_coords(gx, gy, nx, ny, flip_y)
    scatter_kwargs: Dict[str, Any] = {
        "size": size,
        "marker": marker,
        "color": color,
        "line_color": line_color or color,
        "line_width": line_width,
        "fill_alpha": fill_alpha,
        "line_alpha": 1.0,
    }
    return figure.scatter(px, py, **scatter_kwargs)


def _add_proposed_next_scan_overlay(
    figure: Any,
    gx: np.ndarray,
    gy: np.ndarray,
    nx: int,
    ny: int,
    flip_y: bool,
    *,
    size: int = 15,
    line_width: float = 2.0,
) -> Optional[Any]:
    """Hollow black ring over proposed next-scan cells; sampled viridis squares show inside."""
    if gx.size == 0:
        return None
    px, py = _grid_display_coords(gx, gy, nx, ny, flip_y)
    return figure.scatter(
        px,
        py,
        marker="circle",
        size=size,
        color=None,
        fill_color=None,
        fill_alpha=0.0,
        line_color="#000000",
        line_width=line_width,
        line_alpha=1.0,
    )


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def _lerp_rgb(
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return tuple(int(round(start[i] + (end[i] - start[i]) * t)) for i in range(3))  # type: ignore[return-value]


def _sample_three_stop_diverging_palette(
    low_hex: str,
    mid_hex: str,
    high_hex: str,
    *,
    steps: int = 256,
) -> List[str]:
    """Smooth diverging palette (low -> mid -> high), matplotlib coolwarm-style."""
    low = _hex_to_rgb(low_hex)
    mid = _hex_to_rgb(mid_hex)
    high = _hex_to_rgb(high_hex)
    if steps <= 1:
        return [_rgb_to_hex(*mid)]
    colors: List[str] = []
    for index in range(steps):
        t = index / (steps - 1)
        if t <= 0.5:
            rgb = _lerp_rgb(low, mid, t / 0.5)
        else:
            rgb = _lerp_rgb(mid, high, (t - 0.5) / 0.5)
        colors.append(_rgb_to_hex(*rgb))
    return colors


def _continuous_palette_generators() -> Dict[str, Any]:
    """Named smooth palettes not bundled as flat lists in Bokeh."""
    return {
        "coolwarm256": lambda: _sample_three_stop_diverging_palette("#3b4cc0", "#f7f7f7", "#b40426"),
        "coolwarm": lambda: _sample_three_stop_diverging_palette("#3b4cc0", "#f7f7f7", "#b40426"),
    }


def _resolve_bokeh_palette(palette_name: str) -> List[Any]:
    """Resolve a Bokeh palette name (e.g. Coolwarm256, Viridis256) to a color list."""
    import re

    import bokeh.palettes as palettes_module
    from bokeh.palettes import Viridis256

    custom = _continuous_palette_generators().get(palette_name.lower())
    if custom is not None:
        return custom()

    direct = getattr(palettes_module, palette_name, None)
    if isinstance(direct, (list, tuple)):
        return list(direct)

    try:
        from bokeh.palettes import all_palettes  # type: ignore

        palette = all_palettes.get(palette_name)
        if isinstance(palette, (list, tuple)):
            return list(palette)
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", palette_name)
        if match:
            family, size_str = match.group(1), match.group(2)
            family_palette = all_palettes.get(family)
            if isinstance(family_palette, dict):
                sized = family_palette.get(int(size_str))
                if isinstance(sized, (list, tuple)):
                    return list(sized)
    except Exception:
        pass
    return list(Viridis256)


def _symmetric_rdbu_limits(
    *arrays: np.ndarray,
    fallback: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[float, float]:
    """Symmetric limits centered at zero for diverging RdBu / coolwarm maps."""
    chunks: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        finite = np.asarray(arr, dtype=np.float64)[np.isfinite(arr)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return fallback
    combined = np.concatenate(chunks)
    abs_max = max(float(np.max(np.abs(combined))), 1e-12)
    return -abs_max, abs_max


def _shared_field_limits(
    *arrays: np.ndarray,
    fallback: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    """Shared min/max color scale for measurement + prediction panels."""
    chunks: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        finite = np.asarray(arr, dtype=np.float64)[np.isfinite(arr)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return fallback
    combined = np.concatenate(chunks)
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if lo == hi:
        hi = lo + 1e-12
    return lo, hi


def resolve_estimate_color_limits(
    *arrays: np.ndarray,
    manual_low: Optional[float] = None,
    manual_high: Optional[float] = None,
    fallback: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    """
    Color scale for measurement + prediction panels.

    Uses ``manual_low`` / ``manual_high`` when both are finite and ``low < high``;
    otherwise falls back to data-driven min/max via ``_shared_field_limits``.
    """
    if manual_low is not None and manual_high is not None:
        lo = float(manual_low)
        hi = float(manual_high)
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            return lo, hi
    return _shared_field_limits(*arrays, fallback=fallback)


_STRAIN_COLORBAR_MARGIN = 75
_STRAIN_FRAME_BASE = 360
_STRAIN_MIN_BORDER_LEFT = 58
_STRAIN_MIN_BORDER_TOP = 42
_STRAIN_LEGEND_FOOTER_HEIGHT = 32
_STRAIN_MIN_BORDER_BOTTOM = 36  # axis label only; legend footer is outside the figure


def _strain_legend_footer_div(
    legend_items: List[Tuple[str, Any]],
    *,
    width: int = 0,
) -> Any:
    """Fixed-height footer aligned under one triplet panel."""
    from bokeh.models import Div

    chips: List[str] = []
    for label, _renderer in legend_items:
        if label == "Sampled":
            chips.append('<span style="color:#333;">&#9632; Sampled</span>')
        elif label == "Proposed next scan":
            chips.append(
                '<span style="color:#000;">&#9675; Proposed next scan</span>'
            )
        else:
            chips.append(f"<span>{label}</span>")
    html = "&nbsp;&nbsp;&nbsp;".join(chips)
    return Div(
        text=html,
        width=width or None,
        height=_STRAIN_LEGEND_FOOTER_HEIGHT,
        sizing_mode="fixed",
        styles={
            "padding": "4px 0 0 0",
            "margin": "0",
            "border": "0",
            "text-align": "center",
            "font-size": "9pt",
            "line-height": "1.4",
            "overflow": "hidden",
            "box-sizing": "border-box",
        },
    )


@dataclass(frozen=True)
class StrainFigureLayout:
    frame_width: int
    frame_height: int
    outer_width: int
    outer_height: int


def _strain_figure_layout(nx: int, ny: int) -> StrainFigureLayout:
    """Pixel layout shared by all triplet panels (frame + chrome + colorbar gutter)."""
    frame_w, frame_h = _proportional_frame_size(nx, ny, base=_STRAIN_FRAME_BASE)
    outer_w = frame_w + _STRAIN_MIN_BORDER_LEFT + _STRAIN_COLORBAR_MARGIN
    outer_h = frame_h + _STRAIN_MIN_BORDER_TOP + _STRAIN_MIN_BORDER_BOTTOM
    return StrainFigureLayout(
        frame_width=frame_w,
        frame_height=frame_h,
        outer_width=outer_w,
        outer_height=outer_h,
    )


def _lock_strain_figure_layout(figure: Any, layout: StrainFigureLayout) -> None:
    """Force identical outer/frame dimensions after glyphs, color bars, and legends."""
    figure.frame_width = layout.frame_width
    figure.frame_height = layout.frame_height
    figure.width = layout.outer_width
    figure.height = layout.outer_height
    figure.min_border_left = _STRAIN_MIN_BORDER_LEFT
    figure.min_border_right = _STRAIN_COLORBAR_MARGIN
    figure.min_border_top = _STRAIN_MIN_BORDER_TOP
    figure.min_border_bottom = _STRAIN_MIN_BORDER_BOTTOM
    figure.sizing_mode = "fixed"
    figure.toolbar_location = "above"


def _strain_plot_figure(
    title: str,
    nx: int,
    ny: int,
    cfg: StrainFieldPlotConfig,
    *,
    layout: Optional[StrainFigureLayout] = None,
) -> Any:
    """Shared Bokeh figure: equal lab-unit aspect, fixed frame, colorbar gutter."""
    from bokeh.models import FixedTicker
    from bokeh.plotting import figure

    panel_layout = layout or _strain_figure_layout(nx, ny)
    p = figure(
        title=title,
        x_range=(0, nx),
        y_range=(0, ny),
        frame_width=panel_layout.frame_width,
        frame_height=panel_layout.frame_height,
        width=panel_layout.outer_width,
        height=panel_layout.outer_height,
        min_border_left=_STRAIN_MIN_BORDER_LEFT,
        min_border_right=_STRAIN_COLORBAR_MARGIN,
        min_border_top=_STRAIN_MIN_BORDER_TOP,
        min_border_bottom=_STRAIN_MIN_BORDER_BOTTOM,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        match_aspect=True,
        aspect_scale=1,
        sizing_mode="fixed",
        background_fill_color="#ffffff",
        border_fill_color="#ffffff",
    )
    p.xaxis.axis_label = cfg.x_axis_label
    p.yaxis.axis_label = cfg.y_axis_label
    p.xaxis.ticker = FixedTicker(ticks=list(range(0, nx + 1, max(1, nx // 5))))
    p.yaxis.ticker = FixedTicker(ticks=list(range(0, ny + 1, max(1, ny // 5))))
    return p


def _add_strain_white_grid_image(
    figure: Any,
    nx: int,
    ny: int,
    cfg: StrainFieldPlotConfig,
) -> None:
    """Full-grid image layer so scatter panels share heatmap sizing/aspect behavior."""
    from bokeh.models import LinearColorMapper

    zd = _maybe_flip_y(np.ones((ny, nx), dtype=np.float64), cfg.flip_y_for_display)
    mapper = LinearColorMapper(palette=["#ffffff", "#ffffff"], low=0.0, high=1.0)
    figure.image(
        image=[zd],
        x=0,
        y=0,
        dw=nx,
        dh=ny,
        color_mapper=mapper,
        level="image",
    )


def _attach_heatmap_colorbar(
    figure: Any,
    mapper: Any,
    lo: float,
    hi: float,
) -> None:
    from bokeh.models import BasicTicker, ColorBar, NumeralTickFormatter

    span = abs(hi - lo)
    if span >= 100 or max(abs(lo), abs(hi)) >= 1000:
        tick_format = "0.00e0"
    elif span >= 1:
        tick_format = "0.00"
    elif span >= 0.01:
        tick_format = "0.0000"
    else:
        tick_format = "0.00e0"
    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(),
        formatter=NumeralTickFormatter(format=tick_format),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        width=12,
    )
    figure.add_layout(color_bar, "right")


def _add_colored_square_overlay(
    figure: Any,
    gx: np.ndarray,
    gy: np.ndarray,
    values: np.ndarray,
    nx: int,
    ny: int,
    flip_y: bool,
    mapper: Any,
    *,
    size: int = 11,
) -> Optional[Any]:
    if gx.size == 0:
        return None
    from bokeh.models import ColumnDataSource

    px, py = _grid_display_coords(gx, gy, nx, ny, flip_y)
    source = ColumnDataSource(data={"x": px, "y": py, "value": np.asarray(values, dtype=np.float64)})
    return figure.scatter(
        "x",
        "y",
        source=source,
        marker="square",
        size=size,
        line_color="#333333",
        line_width=0.8,
        fill_alpha=0.95,
        color={"field": "value", "transform": mapper},
    )


def format_nsdf_workflow_display(
    surrogate_info: NSDFSurrogateData,
    next_x_info: "NSDFNextXData",
) -> str:
    """Show the single active workflow id for the dashboard banner."""
    workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
    if not workflow_id:
        return ""
    return f"Workflow ID: {workflow_id}"


def _coerce_trend_step_id(value: Any) -> Optional[str]:
    """Normalize a time-step id for the uncertainty trend x-axis."""
    if value is None:
        return None
    if isinstance(value, str):
        step_id = value.strip()
        return step_id or None
    if _is_number(value):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if numeric == int(numeric):
            return str(int(numeric))
        return str(numeric)
    text = str(value).strip()
    return text or None


def _parse_trend_y_value(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _peek_transformed_stddevs_avg_scalar(
    surrogate_doc: Any,
) -> Optional[float]:
    """
    Return one average-uncertainty value from a surrogate snapshot.

    Accepts the current pipeline format ``transformed_stddevs_avg: <float>`` as
    well as a single ``[[id, avg]]`` pair. Returns ``None`` when the field holds
    a multi-step history (handled via the selected ``surrogate.json`` instead).
    """
    if not isinstance(surrogate_doc, Mapping):
        return None
    raw = surrogate_doc.get("transformed_stddevs_avg")
    if raw is None:
        return None
    scalar = _parse_trend_y_value(raw)
    if scalar is not None:
        return scalar
    points, _ = parse_transformed_stddevs_avg_points(raw)
    if len(points) == 1:
        return float(points[0][1])
    return None


def _append_transformed_stddevs_avg_point(
    points: List[Tuple[str, float]],
    *,
    step_id_raw: Any,
    y_raw: Any,
    index: int,
    warnings: List[str],
) -> None:
    step_id = _coerce_trend_step_id(step_id_raw)
    y_val = _parse_trend_y_value(y_raw)
    if step_id is None:
        warnings.append(
            f"Skipping transformed_stddevs_avg[{index}]: time-step id must be a non-empty string."
        )
        return
    if y_val is None:
        warnings.append(
            f"Skipping transformed_stddevs_avg[{index}]: average uncertainty must be numeric."
        )
        return
    points.append((step_id, y_val))


def parse_transformed_stddevs_avg_points(
    value: Any,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Parse ``transformed_stddevs_avg`` from ``surrogate.json``.

    Primary format: ``[[id, avg], ...]`` where ``id`` is a time-step string and
    ``avg`` is a float. Also accepts legacy numeric ``[[x, y], ...]``,
    ``{"id": [...], "transformed_stddevs_avg": [...]}``, and ``{"x": [...], "y": [...]}``.
    """
    warnings: List[str] = []
    if value is None:
        return [], warnings
    scalar = _parse_trend_y_value(value)
    if scalar is not None:
        return [], warnings
    if isinstance(value, Mapping):
        ids = value.get("id")
        if ids is None:
            ids = value.get("x")
        ys = value.get("transformed_stddevs_avg")
        if ys is None:
            ys = value.get("y")
        if isinstance(ids, list) and isinstance(ys, list):
            if len(ids) != len(ys):
                warnings.append(
                    "Skipping transformed_stddevs_avg: id and value arrays have different lengths."
                )
                return [], warnings
            points: List[Tuple[str, float]] = []
            for i, (id_raw, y_raw) in enumerate(zip(ids, ys)):
                _append_transformed_stddevs_avg_point(
                    points,
                    step_id_raw=id_raw,
                    y_raw=y_raw,
                    index=i,
                    warnings=warnings,
                )
            return points, warnings
        if ids is not None or ys is not None:
            _append_transformed_stddevs_avg_point(
                points := [],
                step_id_raw=ids,
                y_raw=ys,
                index=0,
                warnings=warnings,
            )
            return points, warnings
        warnings.append(
            "Skipping transformed_stddevs_avg: expected [id, avg] pairs or parallel id/value arrays."
        )
        return [], warnings
    if isinstance(value, list):
        if not value:
            return [], warnings
        if len(value) == 2 and not isinstance(value[0], (list, Mapping)):
            points = []
            _append_transformed_stddevs_avg_point(
                points,
                step_id_raw=value[0],
                y_raw=value[1],
                index=0,
                warnings=warnings,
            )
            return points, warnings
        points = []
        for i, item in enumerate(value):
            if isinstance(item, list) and len(item) >= 2:
                id_raw, y_raw = item[0], item[1]
            elif isinstance(item, Mapping):
                id_raw = item.get("id")
                if id_raw is None:
                    id_raw = item.get("x")
                y_raw = item.get("transformed_stddevs_avg")
                if y_raw is None:
                    y_raw = item.get("y")
            else:
                warnings.append(
                    f"Skipping transformed_stddevs_avg[{i}]: expected [id, avg] or {{id, transformed_stddevs_avg}}."
                )
                continue
            _append_transformed_stddevs_avg_point(
                points,
                step_id_raw=id_raw,
                y_raw=y_raw,
                index=i,
                warnings=warnings,
            )
        return points, warnings
    warnings.append(
        "Skipping transformed_stddevs_avg: expected a list or object with id/value pairs."
    )
    return [], warnings


def _mean_uncertainty_from_surrogate_doc(
    surrogate_doc: Optional[Mapping[str, Any]],
    *,
    grid_size: Tuple[int, int],
) -> Optional[float]:
    """Fallback average of the surrogate uncertainty grid (same units as plot 3 stddev)."""
    if not isinstance(surrogate_doc, Mapping):
        return None
    warnings: List[str] = []
    surrogate = validate_nsdf_surrogate_doc(surrogate_doc)
    warnings.extend(surrogate.warnings)
    if surrogate.uncertainty is None:
        return None
    grid = _surrogate_field_to_display_grid(
        surrogate.uncertainty,
        key="uncertainty",
        surrogate_info=surrogate,
        surrogate_doc=surrogate_doc,
        display_size=grid_size,
        warnings=warnings,
    )
    if grid is None:
        return None
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _uncertainty_point_from_surrogate_doc(
    surrogate_doc: Optional[Mapping[str, Any]],
    *,
    grid_size: Tuple[int, int],
    step_index: int,
    step_id: str = "",
) -> Tuple[Optional[str], Optional[float], str]:
    """Return (step_id, y, source) for one snapshot's contribution to the trend line."""
    if not isinstance(surrogate_doc, Mapping):
        return None, None, "none"
    scalar = _peek_transformed_stddevs_avg_scalar(surrogate_doc)
    if scalar is not None:
        label = (step_id or "").strip() or str(step_index)
        return label, scalar, "transformed_stddevs_avg"
    points, _ = parse_transformed_stddevs_avg_points(
        surrogate_doc.get("transformed_stddevs_avg")
    )
    if len(points) == 1:
        return points[0][0], points[0][1], "transformed_stddevs_avg"
    if len(points) > 1:
        step, y_val = points[-1]
        return step, y_val, "transformed_stddevs_avg"
    mean_val = _mean_uncertainty_from_surrogate_doc(surrogate_doc, grid_size=grid_size)
    if mean_val is not None:
        fallback_id = (step_id or "").strip() or str(step_index)
        return fallback_id, mean_val, "uncertainty_grid_mean"
    return None, None, "none"


def load_surrogate_doc_for_paths(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load only ``surrogate.json`` for the resolved/versioned paths."""
    load_paths = _prepare_nsdf_load_paths(paths, remote_linked=remote_linked)
    if load_paths.has_s3_source():
        bucket = (load_paths.s3_bucket or "").strip()
        data_key = (load_paths.s3_data_key or "").strip()
        if not bucket or not data_key:
            return None
        client = _make_nsdf_s3_client(load_paths, mongo_s3_auth=mongo_s3_auth)
        for candidate_key in _iter_surrogate_s3_key_candidates(
            load_paths,
            data_key,
            mongo_s3_auth=mongo_s3_auth,
        ):
            doc = _read_json_if_exists_s3(client, bucket, candidate_key)
            if isinstance(doc, Mapping):
                return dict(doc)
        return None
    surrogate, _, _ = load_optional_surrogate_json(
        load_paths,
        mongo_s3_auth=mongo_s3_auth,
    )
    return surrogate if isinstance(surrogate, Mapping) else None


def _snapshots_chronological(snaps: Sequence[NSDFSnapshotRef]) -> List[NSDFSnapshotRef]:
    return sorted(snaps, key=lambda snap: snap.sort_key)


def _trend_current_index(
    step_ids: Sequence[str],
    *,
    current_snapshot: str,
    chrono_snaps: Sequence[NSDFSnapshotRef],
) -> Optional[int]:
    """Highlight the point matching the active data snapshot when possible."""
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    if not step_ids:
        return None
    if current_snapshot != "latest":
        for idx, step_id in enumerate(step_ids):
            if _trend_step_matches_snapshot(step_id, current_snapshot):
                return idx
    fallback = _snapshot_index_in_series(chrono_snaps, current_snapshot, len(step_ids))
    return fallback


def uncertainty_trend_from_surrogate_doc(
    surrogate_doc: Optional[Mapping[str, Any]],
    *,
    current_snapshot: str = "latest",
    grid_size: Tuple[int, int],
) -> Optional[UncertaintyTrendSeries]:
    """
    Build plot-4 trend data from one loaded ``surrogate.json``.

    Prefers ``transformed_stddevs_avg``; when absent, falls back to the mean of the
    uncertainty grid so early acquisitions still show one trend point.
    """
    if not isinstance(surrogate_doc, Mapping):
        return None
    scalar = _peek_transformed_stddevs_avg_scalar(surrogate_doc)
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    if scalar is not None:
        step_id = current_snapshot if current_snapshot != "latest" else "latest"
        return UncertaintyTrendSeries(
            step_ids=[step_id],
            y=np.asarray([scalar], dtype=np.float64),
            labels=[triplet_suffix_trend_label(step_id)],
            current_index=0,
            source="transformed_stddevs_avg",
            warnings=[],
        )
    full_points, parse_warnings = parse_transformed_stddevs_avg_points(
        surrogate_doc.get("transformed_stddevs_avg")
    )
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    if len(full_points) >= 1:
        step_ids = [point[0] for point in full_points]
        ys = np.asarray([point[1] for point in full_points], dtype=np.float64)
        current_index: Optional[int] = None
        if current_snapshot != "latest":
            for idx, step_id in enumerate(step_ids):
                if _trend_step_matches_snapshot(step_id, current_snapshot):
                    current_index = idx
                    break
        elif step_ids:
            current_index = len(step_ids) - 1
        return UncertaintyTrendSeries(
            step_ids=step_ids,
            y=ys,
            labels=_trend_labels_for_step_ids(step_ids),
            current_index=current_index,
            source="transformed_stddevs_avg",
            warnings=parse_warnings,
        )

    mean_val = _mean_uncertainty_from_surrogate_doc(surrogate_doc, grid_size=grid_size)
    if mean_val is None:
        return None
    step_id = current_snapshot
    warnings = list(parse_warnings)
    warnings.append(
        "transformed_stddevs_avg missing; using mean uncertainty from surrogate grid."
    )
    return UncertaintyTrendSeries(
        step_ids=[step_id],
        y=np.asarray([mean_val], dtype=np.float64),
        labels=[triplet_suffix_trend_label(step_id)],
        current_index=0,
        source="uncertainty_grid_mean",
        warnings=warnings,
    )


def _build_per_snapshot_uncertainty_trend(
    chrono: Sequence[NSDFSnapshotRef],
    base_paths: StrainDashboardPaths,
    *,
    grid_size: Tuple[int, int],
    current_snapshot: str,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
    allow_file_load: bool = True,
) -> UncertaintyTrendSeries:
    """One trend point per ``surrogate_<suffix>.json`` (x = file id, y = scalar avg)."""
    warnings: List[str] = []
    step_ids_out: List[str] = []
    ys_out: List[float] = []
    labels_out: List[str] = []
    current_index: Optional[int] = None
    current_snapshot = (current_snapshot or "latest").strip() or "latest"

    for step_idx, snap in enumerate(chrono, start=1):
        suffix = snap.suffix or "latest"
        step_key = suffix
        display_label = triplet_suffix_trend_label(step_key)
        if suffix != "latest" and not is_valid_nsdf_version_suffix(suffix):
            warnings.append(f"Snapshot {display_label}: invalid version suffix; skipped.")
            continue
        y_val = snap.uncertainty_trend_y
        if y_val is None and allow_file_load:
            try:
                snap_paths = apply_nsdf_version_suffix(
                    base_paths,
                    "" if suffix == "latest" else suffix,
                )
            except ValueError as exc:
                warnings.append(f"Snapshot {display_label}: {exc}")
                continue
            surrogate_doc = load_surrogate_doc_for_paths(
                snap_paths,
                mongo_s3_auth=mongo_s3_auth,
                remote_linked=remote_linked,
            )
            _step_id, loaded_y, _point_source = _uncertainty_point_from_surrogate_doc(
                surrogate_doc,
                grid_size=grid_size,
                step_index=step_idx,
                step_id=step_key,
            )
            if loaded_y is not None:
                y_val = loaded_y
        if y_val is None:
            if allow_file_load:
                warnings.append(
                    f"Snapshot {display_label}: no transformed_stddevs_avg or uncertainty grid."
                )
            continue
        step_ids_out.append(step_key)
        ys_out.append(float(y_val))
        labels_out.append(display_label)
        if suffix == current_snapshot:
            current_index = len(step_ids_out) - 1

    return UncertaintyTrendSeries(
        step_ids=step_ids_out,
        y=np.asarray(ys_out, dtype=np.float64),
        labels=labels_out,
        current_index=current_index,
        source="per_snapshot_transformed_stddevs_avg" if step_ids_out else "per_snapshot",
        warnings=warnings,
    )


def build_uncertainty_trend_from_surrogate_paths(
    surrogate_paths: StrainDashboardPaths,
    *,
    current_snapshot: str = "latest",
    grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> Optional[UncertaintyTrendSeries]:
    """
    Fast uncertainty trend from one ``surrogate.json``.

    Returns ``None`` when the selected surrogate has no usable trend data.
    """
    trend_doc = load_surrogate_doc_for_paths(
        surrogate_paths,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
    )
    return uncertainty_trend_from_surrogate_doc(
        trend_doc,
        current_snapshot=current_snapshot,
        grid_size=grid_size,
    )


def build_uncertainty_trend_series(
    triplet_index: NSDFTripletIndex,
    base_paths: StrainDashboardPaths,
    *,
    workflow_id: str,
    current_snapshot: str = "latest",
    grid_size: Tuple[int, int],
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
    surrogate_paths: Optional[StrainDashboardPaths] = None,
    allow_per_snapshot_fallback: bool = True,
) -> UncertaintyTrendSeries:
    """
    Build avg-uncertainty vs time-step series for the active workflow.

    Prefers one ``transformed_stddevs_avg`` scalar per ``surrogate_<suffix>.json``
    (x = snapshot id). Falls back to a multi-step history array in the selected
    ``surrogate.json``, then per-snapshot uncertainty-grid means.
    """
    warnings: List[str] = []
    snaps = triplet_index.snapshots_for_workflow(workflow_id)
    if not snaps:
        return UncertaintyTrendSeries(
            step_ids=[],
            y=np.zeros(0, dtype=np.float64),
            labels=[],
            warnings=["No snapshots found for the active workflow."],
        )

    chrono = _snapshots_chronological(snaps)
    per_snapshot = _build_per_snapshot_uncertainty_trend(
        chrono,
        base_paths,
        grid_size=grid_size,
        current_snapshot=current_snapshot,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
        allow_file_load=allow_per_snapshot_fallback,
    )
    warnings.extend(per_snapshot.warnings)
    if per_snapshot.step_ids:
        if per_snapshot.current_index is None and current_snapshot == "latest" and chrono:
            per_snapshot = replace(per_snapshot, current_index=len(per_snapshot.step_ids) - 1)
        return replace(
            per_snapshot,
            warnings=warnings,
        )

    trend_surrogate_paths = surrogate_paths or apply_nsdf_version_suffix(base_paths, "")
    trend_doc = load_surrogate_doc_for_paths(
        trend_surrogate_paths,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
    )
    full_points, parse_warnings = parse_transformed_stddevs_avg_points(
        (trend_doc or {}).get("transformed_stddevs_avg")
    )
    warnings.extend(parse_warnings)
    if len(full_points) >= 1:
        step_ids = [point[0] for point in full_points]
        ys = np.asarray([point[1] for point in full_points], dtype=np.float64)
        current_index = _trend_current_index(
            step_ids,
            current_snapshot=current_snapshot,
            chrono_snaps=chrono,
        )
        if current_index is None and current_snapshot == "latest" and step_ids:
            current_index = len(step_ids) - 1
        return UncertaintyTrendSeries(
            step_ids=step_ids,
            y=ys,
            labels=_trend_labels_for_step_ids(step_ids),
            current_index=current_index,
            source="transformed_stddevs_avg",
            warnings=warnings,
        )

    quick = uncertainty_trend_from_surrogate_doc(
        trend_doc,
        current_snapshot=current_snapshot,
        grid_size=grid_size,
    )
    if quick is not None:
        return replace(quick, warnings=[*warnings, *quick.warnings])

    if not allow_per_snapshot_fallback:
        return UncertaintyTrendSeries(
            step_ids=[],
            y=np.zeros(0, dtype=np.float64),
            labels=[],
            warnings=warnings
            or ["Uncertainty trend will update after the snapshot catalog finishes loading."],
        )

    # Last resort: mean uncertainty grid per snapshot (loads full surrogate each time).
    step_ids_out: List[str] = []
    ys_out: List[float] = []
    labels_out: List[str] = []
    current_index: Optional[int] = None
    current_snapshot = (current_snapshot or "latest").strip() or "latest"

    for step_idx, snap in enumerate(chrono, start=1):
        suffix = snap.suffix or "latest"
        step_key = suffix
        display_label = triplet_suffix_trend_label(step_key)
        if suffix != "latest" and not is_valid_nsdf_version_suffix(suffix):
            warnings.append(f"Snapshot {display_label}: invalid version suffix; skipped.")
            continue
        try:
            snap_paths = apply_nsdf_version_suffix(
                base_paths,
                "" if suffix == "latest" else suffix,
            )
        except ValueError as exc:
            warnings.append(f"Snapshot {display_label}: {exc}")
            continue
        surrogate_doc = load_surrogate_doc_for_paths(
            snap_paths,
            mongo_s3_auth=mongo_s3_auth,
            remote_linked=remote_linked,
        )
        mean_val = _mean_uncertainty_from_surrogate_doc(surrogate_doc, grid_size=grid_size)
        if mean_val is None:
            warnings.append(
                f"Snapshot {display_label}: no transformed_stddevs_avg or uncertainty grid."
            )
            continue
        step_ids_out.append(step_key)
        ys_out.append(float(mean_val))
        labels_out.append(display_label)
        if suffix == current_snapshot:
            current_index = len(step_ids_out) - 1

    if not step_ids_out:
        return UncertaintyTrendSeries(
            step_ids=[],
            y=np.zeros(0, dtype=np.float64),
            labels=[],
            warnings=warnings or ["No uncertainty trend points could be built."],
        )

    return UncertaintyTrendSeries(
        step_ids=step_ids_out,
        y=np.asarray(ys_out, dtype=np.float64),
        labels=labels_out,
        current_index=current_index,
        source="per_snapshot",
        warnings=warnings,
    )


def _snapshot_index_in_series(
    chrono_snaps: Sequence[NSDFSnapshotRef],
    current_snapshot: str,
    series_length: int,
) -> Optional[int]:
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    if series_length <= 0:
        return None
    for idx, snap in enumerate(chrono_snaps):
        suffix = snap.suffix or "latest"
        if suffix == current_snapshot:
            return min(idx, series_length - 1)
    if current_snapshot == "latest":
        return series_length - 1
    return None


def validate_nsdf_surrogate_doc(
    surrogate_doc: Optional[Mapping[str, Any]],
) -> NSDFSurrogateData:
    warnings: List[str] = []
    if surrogate_doc is None:
        return NSDFSurrogateData(warnings=warnings)
    if not isinstance(surrogate_doc, Mapping):
        return NSDFSurrogateData(warnings=["Skipping surrogate JSON: expected a JSON object."])
    workflow_id: Optional[str] = None
    raw_workflow_id = surrogate_doc.get("workflow_id")
    if isinstance(raw_workflow_id, str) and raw_workflow_id.strip():
        workflow_id = raw_workflow_id.strip()
    plot_dim, plot_dim_warning = parse_nsdf_plot_dim(surrogate_doc.get("dim"))
    if plot_dim_warning:
        warnings.append(plot_dim_warning)
    bounds = _validate_bounds(surrogate_doc.get("bounds"))
    points = _parse_nsdf_points(surrogate_doc.get("points"))
    points_to_predict = _parse_nsdf_points_to_predict(surrogate_doc, warnings)
    surrogate_arr = _numeric_1d_surrogate_array_or_none(surrogate_doc, "surrogate", warnings)
    if (
        points_to_predict is not None
        and surrogate_arr is not None
        and points_to_predict.shape[0] != surrogate_arr.shape[0]
    ):
        warnings.append(
            "points_to_predict length does not match surrogate array; "
            "ignoring coordinate-aligned placement."
        )
        points_to_predict = None
    return NSDFSurrogateData(
        surrogate=surrogate_arr,
        uncertainty=_numeric_1d_surrogate_array_or_none(surrogate_doc, "uncertainty", warnings),
        raw_uncertainty=_numeric_1d_surrogate_array_or_none(
            surrogate_doc,
            "raw_uncertainty",
            warnings,
        ),
        workflow_id=workflow_id,
        plot_dim=plot_dim,
        bounds=bounds,
        points=points,
        points_to_predict=points_to_predict,
        warnings=warnings,
    )


def list_nsdf_field_headers(
    data_doc: Mapping[str, Any],
    surrogate_doc: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    measurement = validate_nsdf_measurement_doc(data_doc)
    surrogate = validate_nsdf_surrogate_doc(surrogate_doc)
    return surrogate.plottable_fields


def infer_nsdf_grid_size(data_doc: Mapping[str, Any]) -> Tuple[int, int]:
    """Infer grid width/height from unique labx/labz coordinates in ``dataset_x``."""
    measurement = validate_nsdf_measurement_doc(data_doc)
    if measurement.coordinates.shape[0] == 0:
        bounds_size = infer_nsdf_bounds_grid_size(data_doc)
        if bounds_size:
            return bounds_size
        return DEFAULT_GRID_SIZE
    unique_x = np.unique(measurement.coordinates[:, 0])
    unique_z = np.unique(measurement.coordinates[:, 1])
    nx = int(unique_x.shape[0])
    ny = int(unique_z.shape[0])
    if nx <= 0 or ny <= 0:
        return DEFAULT_GRID_SIZE
    return nx, ny


def infer_nsdf_bounds_grid_size(data_doc: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Read grid width/height from ``bounds=[[0, width], [0, height], ...]``."""
    if not isinstance(data_doc, Mapping):
        return None
    bounds = data_doc.get("bounds")
    if not isinstance(bounds, list) or len(bounds) < 1:
        return None

    out: List[int] = []
    for axis in bounds:
        if not isinstance(axis, list) or len(axis) < 2:
            return None
        lo, hi = axis[0], axis[1]
        if not (_is_number(lo) and _is_number(hi)):
            return None
        flo, fhi = float(lo), float(hi)
        if not (math.isfinite(flo) and math.isfinite(fhi)):
            return None
        if abs(flo) > 1e-9 or fhi <= 0:
            return None
        rounded = int(round(fhi))
        if abs(fhi - rounded) > 1e-9 or rounded <= 0:
            return None
        out.append(rounded)
    if len(out) == 1:
        return out[0], 1
    return out[0], out[1]


def surrogate_doc_defines_grid_size(surrogate_doc: Optional[Mapping[str, Any]]) -> bool:
    """Return True when surrogate JSON carries pre-capture grid metadata."""
    if not isinstance(surrogate_doc, Mapping):
        return False
    if infer_nsdf_bounds_grid_size(surrogate_doc) is not None:
        return True
    return _legacy_grid_size_from_dim_array(surrogate_doc.get("dim")) is not None


def resolve_nsdf_grid_size(
    data_doc: Mapping[str, Any],
    *,
    surrogate_doc: Optional[Mapping[str, Any]] = None,
    env_grid_size: Optional[Tuple[int, int]] = None,
    manual_grid_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int], str]:
    """Resolve grid size from manual controls, surrogate metadata, data bounds, env, dataset_x."""
    manual_size = _valid_grid_size(manual_grid_size)
    if manual_size:
        return manual_size, "manual controls"
    if isinstance(surrogate_doc, Mapping):
        surrogate_bounds_size = infer_nsdf_bounds_grid_size(surrogate_doc)
        if surrogate_bounds_size:
            return surrogate_bounds_size, "surrogate bounds"
        legacy_dim_size = _legacy_grid_size_from_dim_array(surrogate_doc.get("dim"))
        if legacy_dim_size:
            return legacy_dim_size, "surrogate dim (deprecated)"
    bounds_size = infer_nsdf_bounds_grid_size(data_doc)
    if bounds_size:
        return bounds_size, "bounds"
    env_size = _valid_grid_size(env_grid_size)
    if env_size:
        return env_size, "environment"
    return infer_nsdf_grid_size(data_doc), "dataset_x"


# ---------------------------------------------------------------------------
# Grid construction (1D sparse → 2D, or direct 2D)
# ---------------------------------------------------------------------------


def _norm_positions_to_grid(
    labx: Sequence[Number],
    labz: Sequence[Number],
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    lx = np.asarray(labx, dtype=np.float64)
    lz = np.asarray(labz, dtype=np.float64)
    if lx.shape != lz.shape:
        raise ValueError("labx and labz must have the same length")
    if lx.size == 0:
        return np.zeros(0), np.zeros(0)

    def scale_axis(a: np.ndarray, n: int, axis_bounds: Optional[Tuple[float, float]]) -> np.ndarray:
        if axis_bounds is None:
            amin, amax = np.nanmin(a), np.nanmax(a)
        else:
            amin, amax = axis_bounds
        if not math.isfinite(amin) or not math.isfinite(amax) or amax == amin:
            return np.full_like(a, (n - 1) / 2.0)
        return (a - amin) / (amax - amin) * (n - 1)

    x_bounds = bounds[0] if bounds else None
    z_bounds = bounds[1] if bounds else None
    gx = scale_axis(lx, nx, x_bounds)
    gy = scale_axis(lz, ny, z_bounds)
    return gx, gy


def _norm_coordinates_to_grid(
    coordinates: np.ndarray,
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must be a 2D array with at least two columns")
    return _norm_positions_to_grid(coordinates[:, 0], coordinates[:, 1], nx, ny, bounds)


def _idw_fill_grid(
    px: np.ndarray,
    py: np.ndarray,
    values: np.ndarray,
    nx: int,
    ny: int,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Inverse-distance weights onto a dense grid (NumPy only)."""
    out = np.zeros((ny, nx), dtype=np.float64)
    xs = np.arange(nx, dtype=np.float64)
    ys = np.arange(ny, dtype=np.float64)
    for i, y in enumerate(ys):
        dy = py - y
        for j, x in enumerate(xs):
            dx = px - x
            d2 = dx * dx + dy * dy + eps
            w = 1.0 / np.power(d2, power / 2.0)
            mask = np.isfinite(values) & np.isfinite(px) & np.isfinite(py)
            if not np.any(mask):
                out[i, j] = np.nan
                continue
            ws = w[mask]
            vs = values[mask]
            out[i, j] = float(np.sum(ws * vs) / np.sum(ws))
    return out


def _distance_weighted_variance_placeholder(
    mask: np.ndarray,
    base_scale: float,
) -> np.ndarray:
    """
    When no GP variance is supplied: high uncertainty away from measurements
    (distance transform), scaled by ``base_scale`` (e.g. mean stdev).
    """
    try:
        from scipy import ndimage  # type: ignore
    except Exception:
        ndimage = None  # type: ignore

    if ndimage is None:
        occupied = mask > 0.5
        dist = np.full(mask.shape, np.inf, dtype=np.float64)
        dist[occupied] = 0.0
        for _ in range(max(mask.shape) * 2):
            nxt = np.minimum(dist, np.roll(dist, 1, axis=0))
            nxt = np.minimum(nxt, np.roll(dist, -1, axis=0))
            nxt = np.minimum(nxt, np.roll(dist, 1, axis=1))
            nxt = np.minimum(nxt, np.roll(dist, -1, axis=1))
            dist = np.minimum(dist, nxt + 1.0)
        dist[~np.isfinite(dist)] = (
            float(np.nanmax(dist[np.isfinite(dist)])) if np.any(np.isfinite(dist)) else 1.0
        )
        return (dist / (dist.max() + 1e-9)) * base_scale

    inv = 1.0 - (mask > 0.5).astype(np.float64)
    dt = ndimage.distance_transform_edt(inv)
    return (dt / (dt.max() + 1e-9)) * base_scale


def build_strain_field_grids(
    doc: Mapping[str, Any],
    cfg: StrainFieldPlotConfig,
    surrogate_doc: Optional[Mapping[str, Any]] = None,
) -> StrainFieldGrids:
    """Build measurement mask, estimate, and variance directly from native NSDF schema."""
    nx, ny = cfg.grid_size[0], cfg.grid_size[1]
    measurement = validate_nsdf_measurement_doc(doc)
    surrogate = validate_nsdf_surrogate_doc(surrogate_doc)
    values = measurement.observed_values
    display_size = (nx, ny)
    gx, gy = _norm_coordinates_to_grid(measurement.coordinates, nx, ny, measurement.bounds)
    mask = np.zeros((ny, nx), dtype=np.float64)
    for x, y in zip(gx, gy):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ix = int(np.clip(round(float(x)), 0, nx - 1))
        iy = int(np.clip(round(float(y)), 0, ny - 1))
        mask[iy, ix] = 1.0

    meta: Dict[str, Any] = {
        "field": "dataset_y",
        "mode": "nsdf_sparse",
        "n_points": int(values.shape[0]),
        "bounds_source": measurement.bounds_source,
        "measurement_bounds": measurement.bounds,
        "surrogate_bounds": surrogate.bounds,
        "measurement_gx": gx,
        "measurement_gy": gy,
        "measurement_values": values.copy(),
        "plottable_fields": surrogate.plottable_fields,
        "warnings": list(surrogate.warnings),
    }

    est_grid = _surrogate_field_to_display_grid(
        surrogate.surrogate,
        key="surrogate",
        surrogate_info=surrogate,
        surrogate_doc=surrogate_doc,
        display_size=display_size,
        warnings=meta["warnings"],
    )
    if est_grid is not None:
        est = est_grid
        meta["estimate_source"] = (
            "surrogate_points_interp"
            if surrogate.points_to_predict is not None
            else "surrogate_grid"
        )
        if values.shape[0] > 0:
            finite_est = est[np.isfinite(est)]
            if (
                finite_est.size
                and np.allclose(finite_est, 0.0)
                and not np.allclose(values, 0.0)
            ):
                est = _idw_fill_grid(gx, gy, values, nx, ny)
                meta["estimate_source"] = "dataset_y_idw"
                meta["warnings"].append(
                    "Surrogate model grid was empty; estimate uses inverse-distance "
                    "interpolation from measurements."
                )
    elif values.shape[0] > 0:
        est = _idw_fill_grid(gx, gy, values, nx, ny)
        meta["estimate_source"] = "dataset_y_idw"
    else:
        est = np.full((ny, nx), np.nan, dtype=np.float64)
        meta["estimate_source"] = "none"
        meta["warnings"].append("No measurements yet; estimate panel shows surrogate only when available.")

    var_grid = _surrogate_field_to_display_grid(
        surrogate.uncertainty,
        key="uncertainty",
        surrogate_info=surrogate,
        surrogate_doc=surrogate_doc,
        display_size=display_size,
        warnings=meta["warnings"],
    )
    if var_grid is not None and meta["estimate_source"] != "dataset_y_idw":
        var = np.square(np.maximum(var_grid, 0.0))
        meta["variance_source"] = (
            "uncertainty_squared_points_interp"
            if surrogate.points_to_predict is not None
            else "uncertainty_squared_grid"
        )
    else:
        scale = float(np.nanmean(np.abs(values)) or 1e-6) * 0.25
        var = _distance_weighted_variance_placeholder(mask, scale)
        meta["variance_source"] = "distance_placeholder"

    return StrainFieldGrids(mask, est, var, meta)


# ---------------------------------------------------------------------------
# Bokeh figures
# ---------------------------------------------------------------------------


def _maybe_flip_y(arr: np.ndarray, flip: bool) -> np.ndarray:
    return np.flipud(arr) if flip else arr


def _proportional_frame_size(
    nx: int,
    ny: int,
    *,
    base: int = 320,
) -> Tuple[int, int]:
    """Data-frame width/height preserving nx:ny aspect (longest axis = base pixels)."""
    if nx <= 0 or ny <= 0:
        return base, base
    if nx >= ny:
        frame_width = base
        frame_height = max(120, int(round(base * ny / nx)))
    else:
        frame_height = base
        frame_width = max(120, int(round(base * nx / ny)))
    return frame_width, frame_height


def make_strain_heatmap_figure(
    title: str,
    z: np.ndarray,
    cfg: StrainFieldPlotConfig,
    *,
    palette_name: str = "Viridis256",
    low_high: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = False,
    layout: Optional[StrainFigureLayout] = None,
):
    from bokeh.models import LinearColorMapper

    nx, ny = cfg.grid_size
    zd = _maybe_flip_y(np.asarray(z, dtype=np.float64), cfg.flip_y_for_display)

    palette = _resolve_bokeh_palette(palette_name)
    if low_high is not None:
        lo, hi = low_high
    else:
        finite = zd[np.isfinite(zd)]
        lo = float(np.nanmin(finite)) if finite.size else 0.0
        hi = float(np.nanmax(finite)) if finite.size else 1.0
        if lo == hi:
            hi = lo + 1e-12
    mapper = LinearColorMapper(palette=palette, low=lo, high=hi, nan_color="#ffffff")

    p = _strain_plot_figure(title, nx, ny, cfg, layout=layout)
    _add_strain_white_grid_image(p, nx, ny, cfg)
    p.image(image=[zd], x=0, y=0, dw=nx, dh=ny, color_mapper=mapper)
    if show_colorbar:
        _attach_heatmap_colorbar(p, mapper, lo, hi)
    if layout is not None:
        _lock_strain_figure_layout(p, layout)
    return p


def make_strain_measurement_figure(
    title: str,
    cfg: StrainFieldPlotConfig,
    *,
    gx: np.ndarray,
    gy: np.ndarray,
    values: np.ndarray,
    mapper: Any,
    lo: float,
    hi: float,
    layout: Optional[StrainFigureLayout] = None,
) -> Tuple[Any, Optional[Any]]:
    """White canvas with square markers colored by dataset_y and matching colorbar."""
    nx, ny = cfg.grid_size
    p = _strain_plot_figure(title, nx, ny, cfg, layout=layout)
    _add_strain_white_grid_image(p, nx, ny, cfg)
    renderer = _add_colored_square_overlay(
        p,
        gx,
        gy,
        values,
        nx,
        ny,
        cfg.flip_y_for_display,
        mapper,
    )
    _attach_heatmap_colorbar(p, mapper, lo, hi)
    if layout is not None:
        _lock_strain_figure_layout(p, layout)
    return p, renderer


def _build_strain_triplet_figures(
    grids: StrainFieldGrids,
    cfg: StrainFieldPlotConfig,
    *,
    row_subtitle: str = "",
    next_x_info: Optional[NSDFNextXData] = None,
    active_workflow_id: Optional[str] = None,
) -> Tuple[Any, Any, Any, List[Tuple[str, Any]]]:
    from bokeh.models import LinearColorMapper

    sub = f" — {row_subtitle}" if row_subtitle else ""
    est = grids.estimate
    nx, ny = cfg.grid_size
    bounds = _resolve_strain_plot_bounds(grids.meta)

    measured_gx = grids.meta.get("measurement_gx")
    measured_gy = grids.meta.get("measurement_gy")
    measured_vals = grids.meta.get("measurement_values")
    if not isinstance(measured_gx, np.ndarray):
        measured_gx = np.array([], dtype=np.int64)
    if not isinstance(measured_gy, np.ndarray):
        measured_gy = np.array([], dtype=np.int64)
    if not isinstance(measured_vals, np.ndarray):
        measured_vals = np.array([], dtype=np.float64)

    est_palette = _resolve_bokeh_palette(cfg.colormap_estimate)
    est_lo, est_hi = resolve_estimate_color_limits(
        est,
        measured_vals,
        manual_low=cfg.estimate_color_low,
        manual_high=cfg.estimate_color_high,
    )
    est_mapper = LinearColorMapper(palette=est_palette, low=est_lo, high=est_hi)
    panel_layout = _strain_figure_layout(nx, ny)

    if measured_gx.size:
        p0, sampled_renderer = make_strain_measurement_figure(
            f"{cfg.title_measurements}{sub}",
            cfg,
            gx=measured_gx,
            gy=measured_gy,
            values=measured_vals,
            mapper=est_mapper,
            lo=est_lo,
            hi=est_hi,
            layout=panel_layout,
        )
    else:
        p0 = _strain_plot_figure(f"{cfg.title_measurements}{sub}", nx, ny, cfg, layout=panel_layout)
        _add_strain_white_grid_image(p0, nx, ny, cfg)
        _attach_heatmap_colorbar(p0, est_mapper, est_lo, est_hi)
        _lock_strain_figure_layout(p0, panel_layout)
        sampled_renderer = None

    point_legend_items: List[Tuple[str, Any]] = []
    if sampled_renderer is not None:
        point_legend_items.append(("Sampled", sampled_renderer))
    if next_x_info is not None and active_workflow_id:
        nx_gx, nx_gy = next_x_grid_coords_for_workflow(
            next_x_info,
            active_workflow_id,
            nx,
            ny,
            bounds,
        )
        if nx_gx.size:
            proposed_renderer = _add_proposed_next_scan_overlay(
                p0,
                nx_gx,
                nx_gy,
                nx,
                ny,
                cfg.flip_y_for_display,
            )
            if proposed_renderer is not None:
                point_legend_items.append(("Proposed next scan", proposed_renderer))
    p1 = make_strain_heatmap_figure(
        f"{cfg.title_estimate}{sub}",
        grids.estimate,
        cfg,
        palette_name=cfg.colormap_estimate,
        low_high=(est_lo, est_hi),
        show_colorbar=True,
        layout=panel_layout,
    )
    var = grids.variance
    vf = var[np.isfinite(var)]
    vlo = float(np.nanmin(vf)) if vf.size else 0.0
    vhi = float(np.nanmax(vf)) if vf.size else 1.0
    if vlo == vhi:
        vhi = vlo + 1e-12
    p2 = make_strain_heatmap_figure(
        f"{cfg.title_variance}{sub}",
        grids.variance,
        cfg,
        palette_name=cfg.colormap_variance,
        low_high=(vlo, vhi),
        show_colorbar=True,
        layout=panel_layout,
    )
    _lock_strain_figure_layout(p0, panel_layout)
    _lock_strain_figure_layout(p1, panel_layout)
    _lock_strain_figure_layout(p2, panel_layout)
    return p0, p1, p2, point_legend_items


def make_strain_triplet_figures(
    grids: StrainFieldGrids,
    cfg: StrainFieldPlotConfig,
    *,
    row_subtitle: str = "",
    next_x_info: Optional[NSDFNextXData] = None,
    active_workflow_id: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    p0, p1, p2, _legend_items = _build_strain_triplet_figures(
        grids,
        cfg,
        row_subtitle=row_subtitle,
        next_x_info=next_x_info,
        active_workflow_id=active_workflow_id,
    )
    return p0, p1, p2


def _trend_axis_label_budget(point_count: int, *, panel_width: int) -> int:
    """
    Choose how many x-axis labels fit for a trend series.

    Scales down as acquisitions grow and respects the panel width so ISO-like
    step ids stay legible without overlapping.
    """
    n = max(0, int(point_count))
    if n <= 0:
        return 0
    if n <= 8:
        return n
    width = max(120, int(panel_width or 0))
    # Rotated timestamp ids need roughly this much horizontal space each.
    width_cap = max(4, min(n, width // 50))
    if n <= 16:
        return min(n, max(width_cap, 6))
    if n <= 32:
        return min(n, max(width_cap, 5))
    return min(n, width_cap)


def _sparse_trend_axis_ticks(
    step_ids: Sequence[str],
    *,
    panel_width: int = 360,
    max_labels: Optional[int] = None,
    highlight_id: Optional[str] = None,
) -> List[str]:
    """Pick a readable subset of categorical x-axis labels for dense trend series."""
    ids = list(step_ids)
    n = len(ids)
    label_budget = max_labels if max_labels is not None else _trend_axis_label_budget(
        n,
        panel_width=panel_width,
    )
    if n <= label_budget:
        ticks = list(ids)
    elif label_budget <= 1:
        ticks = [ids[-1]]
    else:
        positions = sorted(
            {int(round(i * (n - 1) / (label_budget - 1))) for i in range(label_budget)}
        )
        ticks = [ids[pos] for pos in positions]
    if highlight_id and highlight_id in ids and highlight_id not in ticks:
        ticks.append(highlight_id)
        order = {step_id: idx for idx, step_id in enumerate(ids)}
        ticks.sort(key=lambda step_id: order[step_id])
    return ticks


def make_uncertainty_trend_figure(
    series: UncertaintyTrendSeries,
    cfg: StrainFieldPlotConfig,
    *,
    layout: StrainFigureLayout,
) -> Any:
    """Line plot of average uncertainty vs scan step (4th dashboard panel)."""
    from bokeh.models import ColumnDataSource, Div
    from bokeh.plotting import figure

    if not series.step_ids:
        if series.warnings:
            hint = "<br>".join(
                f"<span style='color:#666;'>{html.escape(msg)}</span>"
                for msg in series.warnings[:2]
            )
        else:
            hint = "<i>No uncertainty trend data for this workflow yet.</i>"
        return Div(
            text=(
                f"<b>{html.escape(cfg.title_uncertainty_trend)}</b><br>"
                f"{hint}"
            ),
            width=layout.outer_width,
            height=layout.outer_height,
            sizing_mode="fixed",
            styles={
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "text-align": "center",
                "font-size": "9pt",
                "padding": "8px",
                "box-sizing": "border-box",
            },
        )

    from bokeh.models import FixedTicker

    point_count = len(series.step_ids)
    tick_text = {
        step_id: (
            series.labels[idx]
            if idx < len(series.labels) and series.labels[idx]
            else triplet_suffix_trend_label(step_id)
        )
        for idx, step_id in enumerate(series.step_ids)
    }
    highlight_id = None
    if series.current_index is not None and 0 <= series.current_index < point_count:
        highlight_id = series.step_ids[series.current_index]
    sparse_ticks = _sparse_trend_axis_ticks(
        series.step_ids,
        panel_width=layout.frame_width,
        highlight_id=highlight_id,
    )
    sparse_positions = [series.step_ids.index(step_id) for step_id in sparse_ticks]

    p = figure(
        title=cfg.title_uncertainty_trend,
        width=layout.outer_width,
        height=layout.outer_height,
        frame_width=layout.frame_width,
        frame_height=layout.frame_height,
        min_border_left=_STRAIN_MIN_BORDER_LEFT,
        min_border_right=_STRAIN_COLORBAR_MARGIN,
        min_border_top=_STRAIN_MIN_BORDER_TOP,
        min_border_bottom=_STRAIN_MIN_BORDER_BOTTOM,
        sizing_mode="fixed",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label=cfg.trend_x_axis_label,
        y_axis_label=cfg.trend_y_axis_label,
        x_range=(-0.5, max(point_count - 1, 0) + 0.5),
        toolbar_location="above",
    )
    _lock_strain_figure_layout(p, layout)
    p.xaxis.ticker = FixedTicker(ticks=sparse_positions)
    p.xaxis.major_label_overrides = {
        pos: tick_text[step_id] for pos, step_id in zip(sparse_positions, sparse_ticks)
    }
    shown_labels = [tick_text[step_id] for step_id in sparse_ticks]
    p.xaxis.major_label_orientation = (
        math.pi / 4 if any(len(label) > 4 for label in shown_labels) else 0.0
    )
    source = ColumnDataSource(
        data={
            "x": list(range(point_count)),
            "step_id": list(series.step_ids),
            "y": series.y.tolist(),
            "label": series.labels or _trend_labels_for_step_ids(series.step_ids),
        }
    )
    p.line("x", "y", source=source, line_width=2, color="#2c7bb6")
    p.scatter("x", "y", source=source, size=6, color="#2c7bb6", alpha=0.7, line_color="#1f4f73")
    if series.current_index is not None and 0 <= series.current_index < point_count:
        p.scatter(
            [series.current_index],
            [float(series.y[series.current_index])],
            size=11,
            color="#ffffff",
            line_color="#d7191c",
            line_width=2,
        )
    return p


def make_strain_triplet_row(
    grids: StrainFieldGrids,
    cfg: StrainFieldPlotConfig,
    *,
    row_subtitle: str = "",
    next_x_info: Optional[NSDFNextXData] = None,
    active_workflow_id: Optional[str] = None,
    uncertainty_trend: Optional[UncertaintyTrendSeries] = None,
) -> Any:
    """Spatial triplet plus uncertainty trend as a fourth panel in the same row."""
    from bokeh.layouts import column, row

    p0, p1, p2, legend_items = _build_strain_triplet_figures(
        grids,
        cfg,
        row_subtitle=row_subtitle,
        next_x_info=next_x_info,
        active_workflow_id=active_workflow_id,
    )
    panel_w = int(p0.width or 0)
    panel_h = int(p0.height or 0)
    nx, ny = cfg.grid_size
    panel_layout = _strain_figure_layout(nx, ny)
    trend_series = uncertainty_trend
    if trend_series is None:
        trend_series = UncertaintyTrendSeries(
            step_ids=[],
            y=np.zeros(0, dtype=np.float64),
            labels=[],
        )
    trend_fig = make_uncertainty_trend_figure(trend_series, cfg, layout=panel_layout)

    plot_row = row(
        p0,
        p1,
        p2,
        trend_fig,
        sizing_mode="fixed",
        width=panel_w * 4 if panel_w else None,
        height=panel_h or None,
    )
    footer_row = row(
        _strain_legend_footer_div(legend_items, width=panel_w),
        _strain_legend_footer_div([], width=panel_w),
        _strain_legend_footer_div([], width=panel_w),
        _strain_legend_footer_div([], width=panel_w),
        sizing_mode="fixed",
        width=panel_w * 4 if panel_w else None,
        height=_STRAIN_LEGEND_FOOTER_HEIGHT,
    )
    total_h = (panel_h or 0) + _STRAIN_LEGEND_FOOTER_HEIGHT
    return column(
        plot_row,
        footer_row,
        sizing_mode="fixed",
        width=panel_w * 4 if panel_w else None,
        height=total_h or None,
    )
