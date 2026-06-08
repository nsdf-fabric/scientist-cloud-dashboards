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
"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (edit here or override via StrainDashboardPaths)
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE: Tuple[int, int] = (26, 26)
NSDF_DATA_JSON_BASENAME = "data.json"
NSDF_VERSION_SUFFIX_RE = re.compile(r"^\d{8}T\d{6}Z$", re.IGNORECASE)
NSDF_DATA_FILENAME_RE = re.compile(
    r"^data(?:_(?P<suffix>\d{8}T\d{6}Z))?\.json$",
    re.IGNORECASE,
)


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


def _copy_s3_fields(src: StrainDashboardPaths, dst: StrainDashboardPaths) -> StrainDashboardPaths:
    dst.local_data_dir = src.local_data_dir
    dst.s3_env_file = src.s3_env_file
    dst.s3_bucket = src.s3_bucket
    dst.s3_data_key = src.s3_data_key
    dst.s3_surrogate_key = src.s3_surrogate_key
    dst.s3_next_x_key = src.s3_next_x_key
    dst.s3_endpoint_url = src.s3_endpoint_url
    dst.s3_region = src.s3_region
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


def find_strain_json_under_dataset_dir(directory: str) -> str:
    """
    Pick an NSDF JSON file under ``directory`` (upload or converted tree for one UUID).

    Prefers ``data.json``; otherwise the first ``*.json`` (sorted by name).
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
    trend_x_axis_label: str = "time step"
    trend_y_axis_label: str = "avg uncertainty"
    flip_y_for_display: bool = True
    colormap_estimate: str = "Viridis256"
    colormap_variance: str = "Coolwarm256"
    colormap_mask: Tuple[str, str] = ("#ffffff", "#ffffff")


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
        options: List[Tuple[str, str]] = []
        for snap in self.snapshots_for_workflow(workflow_id):
            value = "latest" if not snap.suffix else snap.suffix
            if value == "latest":
                label = "Latest (data.json)"
            else:
                label = snap.suffix
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
    """Return ``(data, surrogate, next_x)`` basenames for a version suffix (empty = latest)."""
    suffix = (version_suffix or "").strip()
    if not suffix:
        return "data.json", "surrogate.json", "next_x.json"
    if not NSDF_VERSION_SUFFIX_RE.fullmatch(suffix):
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
    return (m.group("suffix") or "").strip() or None


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

    Uses an exact timestamp match when present, otherwise the nearest available
    timestamp, then ``latest`` when no candidates exist.
    """
    data_suffix = _nsdf_selector_value_to_suffix(data_selector_value)
    if not data_suffix:
        return "latest"

    available = sorted({(suffix or "").strip() for suffix in available_suffixes if (suffix or "").strip()})
    if not available:
        return "latest"
    if data_suffix in available:
        return data_suffix

    nearest = _nearest_nsdf_version_suffix(data_suffix, available)
    if nearest:
        return nearest
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
        out.s3_surrogate_key = _replace_s3_key_basename(out.s3_data_key, sur_fn)
        out.s3_next_x_key = _replace_s3_key_basename(out.s3_data_key, nx_fn)

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
    return sorted(set(suffixes), reverse=True)


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
    return sorted(set(suffixes), reverse=True)


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
    return sorted(set(suffixes), reverse=True)


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
    return sorted(set(suffixes), reverse=True)


def parse_nsdf_surrogate_filename(name: str) -> Optional[str]:
    """Parse ``surrogate_<suffix>.json``; return suffix or ``None`` for ``surrogate.json``."""
    base = os.path.basename((name or "").strip())
    if base == "surrogate.json":
        return None
    if not (base.startswith("surrogate_") and base.endswith(".json")):
        return None
    middle = base[len("surrogate_") : -len(".json")]
    return middle or None


def parse_nsdf_next_x_filename(name: str) -> Optional[str]:
    """Parse ``next_x_<suffix>.json``; return suffix or ``None`` for ``next_x.json``."""
    base = os.path.basename((name or "").strip())
    if base == "next_x.json":
        return None
    if not (base.startswith("next_x_") and base.endswith(".json")):
        return None
    middle = base[len("next_x_") : -len(".json")]
    return middle or None


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
        ordered = sorted(set(data_suffixes), reverse=True)
        if ordered:
            return ordered[0]
    return None


def _nsdf_suffixes_after_reference(
    suffixes: Sequence[str],
    reference_suffix: Optional[str],
) -> List[str]:
    """Newest-first auxiliary suffixes strictly after ``reference_suffix``."""
    ordered = sorted(set(suffixes), reverse=True)
    if not reference_suffix:
        return ordered
    ref = reference_suffix.strip().upper()
    return [suffix for suffix in ordered if suffix.strip().upper() > ref]


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
    return sorted(set(suffixes), reverse=True)


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
    return sorted(set(suffixes), reverse=True)


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
    """Sort newest-first: ISO timestamps lexicographic; latest trio sorts above all."""
    return suffix if suffix else "z_latest"


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


def _next_x_coord_size(item: Mapping[str, Any]) -> int:
    raw = item.get("dataset_x_size")
    if isinstance(raw, int) and raw > 0:
        return raw
    if _is_number(raw):
        parsed = int(raw)
        if parsed > 0:
            return parsed
    return 2


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
    coord_size = _next_x_coord_size(item)
    coords: List[Tuple[float, float]] = []
    for j, row_value in enumerate(data):
        if not isinstance(row_value, list) or len(row_value) < coord_size:
            warnings.append(
                f"Skipping {label} ({workflow_id!r}): data[{j}] must contain "
                f"at least {coord_size} numeric coordinate value(s)."
            )
            return None
        values = row_value[:coord_size]
        if not all(_is_number(value) for value in values):
            warnings.append(
                f"Skipping {label} ({workflow_id!r}): data[{j}] must contain numeric values."
            )
            return None
        floats = [float(value) for value in values]
        if not all(math.isfinite(value) for value in floats):
            warnings.append(
                f"Skipping {label} ({workflow_id!r}): data[{j}] values must be finite."
            )
            return None
        if coord_size < 2:
            warnings.append(
                f"Skipping {label} ({workflow_id!r}): dataset_x_size must be at least 2 for plotting."
            )
            return None
        coords.append((floats[0], floats[1]))
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


def _nearest_auxiliary_workflow_id(
    data_suffix: str,
    surrogate_wf: Mapping[str, str],
    next_x_wf: Mapping[str, str],
) -> Optional[str]:
    """
    Infer workflow_id for a data snapshot from nearby surrogate/next_x timestamps.

    Prefers surrogate workflow_id when surrogate and next_x are equally close.
    """
    key = (data_suffix or "").strip()
    if not key:
        return surrogate_wf.get("") or next_x_wf.get("")

    target_dt = _parse_nsdf_iso_suffix_timestamp(key)
    if target_dt is None:
        return None

    best_wf: Optional[str] = None
    best_delta: Optional[float] = None
    best_is_surrogate = False

    for wf_map, is_surrogate in ((surrogate_wf, True), (next_x_wf, False)):
        for suffix, workflow_id in wf_map.items():
            if not (suffix or "").strip():
                continue
            candidate_dt = _parse_nsdf_iso_suffix_timestamp(suffix)
            if candidate_dt is None:
                continue
            delta = abs((candidate_dt - target_dt).total_seconds())
            if best_wf is None or delta < best_delta or (
                delta == best_delta and is_surrogate and not best_is_surrogate
            ):
                best_wf = workflow_id
                best_delta = delta
                best_is_surrogate = is_surrogate
    return best_wf


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

    Order: explicit ``data.json`` id (new files), same-suffix surrogate/next_x,
    then nearest-timestamp surrogate/next_x for backward compatibility.
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
    nearest = _nearest_auxiliary_workflow_id(data_suffix, surrogate_wf, next_x_wf)
    if nearest:
        return nearest
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
        )
        snapshots.append(ref)
        by_workflow.setdefault(workflow_id, []).append(ref)
    for workflow_id in by_workflow:
        by_workflow[workflow_id].sort(key=lambda snap: snap.sort_key, reverse=True)
    return NSDFTripletIndex(snapshots=snapshots, by_workflow=by_workflow)


def _triplet_groups_from_object_keys(keys: Sequence[str]) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for key in keys:
        name = os.path.basename((key or "").strip())
        if not name:
            continue
        data_suffix = _suffix_from_data_basename(name)
        if data_suffix is not None:
            groups.setdefault(data_suffix, {})["data"] = key
            continue
        surrogate_suffix = _suffix_from_surrogate_basename(name)
        if surrogate_suffix is not None:
            groups.setdefault(surrogate_suffix, {})["surrogate"] = key
            continue
        next_x_suffix = _suffix_from_next_x_basename(name)
        if next_x_suffix is not None:
            groups.setdefault(next_x_suffix, {})["next_x"] = key
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


def discover_nsdf_triplet_index(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> NSDFTripletIndex:
    """
    Build a workflow-scoped snapshot catalog (list + lightweight JSON peeks).

    Called once per Reload; snapshot/workflow UI filters use the cached index.

    ScientistCloud S3-linked datasets must list only the remote prefix (never the
    sparse upload mirror). Uploaded-file datasets use the local upload tree.
    """
    if _local_data_dir_active(paths):
        local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
        if local_dir:
            return _build_triplet_index_from_directory(local_dir)
        return NSDFTripletIndex()

    if remote_linked:
        if not _remote_snapshot_listing_enabled(paths):
            return NSDFTripletIndex()
        remote_paths = (
            paths
            if paths.has_s3_source()
            else promote_gateway_json_url_to_s3_paths(paths)
        )
        return _build_triplet_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)

    if _remote_snapshot_listing_enabled(paths):
        remote_paths = (
            paths
            if paths.has_s3_source()
            else promote_gateway_json_url_to_s3_paths(paths)
        )
        index = _build_triplet_index_from_s3(remote_paths, mongo_s3_auth=mongo_s3_auth)
        if index.snapshots:
            return index

    local_dir = nsdf_listing_directory(paths, base_dir=base_dir, save_dir=save_dir)
    if local_dir:
        return _build_triplet_index_from_directory(local_dir)
    return NSDFTripletIndex()


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
        )
    return finalized


def enrich_strain_paths_from_dataset_doc(
    paths: StrainDashboardPaths,
    doc: Optional[Mapping[str, Any]],
    *,
    base_dir: str = "",
    save_dir: str = "",
) -> StrainDashboardPaths:
    """
    When URL args and env did not resolve JSON, use the Mongo dataset record
    (same fields as portal dashboard share links).
    """
    if (paths.local_json_path or "").strip() or (paths.json_url or "").strip():
        return _finalize_strain_paths(paths)
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()
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
                _, sur_fn, _ = nsdf_triplet_basenames(suffix)
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

        {"workflow_id": "...", "dataset_x_size": 2, "data": [[labx, labz]]}

    Legacy schema (array of workflow blocks) is still accepted.
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
                _, _, nx_fn = nsdf_triplet_basenames(suffix)
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
        _, sur_fn, _ = nsdf_triplet_basenames(listed_suffix)
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
        _, _, nx_fn = nsdf_triplet_basenames(listed_suffix)
        if directory:
            add(os.path.join(directory, nx_fn))
    return candidates


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
        add((paths.s3_surrogate_key or "").strip())
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
        _, sur_fn, _ = nsdf_triplet_basenames(listed_suffix)
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
        add((paths.s3_next_x_key or "").strip())
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
        _, _, nx_fn = nsdf_triplet_basenames(listed_suffix)
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


def _load_json_from_s3_key(client: Any, bucket: str, key: str) -> Any:
    resp = client.get_object(Bucket=bucket, Key=key)
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


def _surrogate_list_is_usable_for_display(
    surrogate_values: Any,
    *,
    display_size: Tuple[int, int],
) -> bool:
    """Reject stub surrogate arrays (e.g. two zeros) that flatten estimate/variance heatmaps."""
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
    )


def load_nsdf_json_bundle_from_s3(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> NSDFLoadedBundle:
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket or not data_key:
        raise FileNotFoundError(
            "Set S3_BUCKET and S3_DATA_KEY to load NSDF data from S3."
        )

    client = _make_nsdf_s3_client(paths, mongo_s3_auth=mongo_s3_auth)
    data = _load_json_from_s3_key(client, bucket, data_key)
    messages = [f"Loaded NSDF data JSON from s3://{bucket}/{data_key}"]
    display_size, _ = resolve_nsdf_grid_size(data)

    surrogate = None
    surrogate_key = ""
    surrogate_candidates = _iter_surrogate_s3_key_candidates(
        paths,
        data_key,
        mongo_s3_auth=mongo_s3_auth,
    )
    for candidate_key in surrogate_candidates:
        try:
            candidate_doc = _load_json_from_s3_key(client, bucket, candidate_key)
            if not _surrogate_doc_is_usable_for_display(
                candidate_doc,
                display_size=display_size,
            ):
                length = len((candidate_doc or {}).get("surrogate") or [])
                messages.append(
                    f"S3 surrogate JSON skipped ({candidate_key}): "
                    f"model grid too small for display ({length} values for "
                    f"{display_size[0]}x{display_size[1]} grid)."
                )
                continue
            surrogate = candidate_doc
            surrogate_key = candidate_key
            messages.append(f"Loaded surrogate JSON from s3://{bucket}/{candidate_key}")
            break
        except Exception as exc:
            if _s3_missing_error(exc):
                continue
            messages.append(f"S3 surrogate JSON skipped ({candidate_key}): {exc}")
            break

    next_x = None
    next_x_key = ""
    for candidate_key in _iter_next_x_s3_key_candidates(
        paths,
        data_key,
        mongo_s3_auth=mongo_s3_auth,
    ):
        try:
            next_x_doc = _load_json_from_s3_key(client, bucket, candidate_key)
            if _next_x_doc_is_recognized_format(next_x_doc):
                next_x = next_x_doc
                next_x_key = candidate_key
                messages.append(f"Loaded next_x JSON from s3://{bucket}/{candidate_key}")
                break
            messages.append(
                f"S3 next_x JSON skipped ({candidate_key}): expected a JSON object or array."
            )
        except Exception as exc:
            if _s3_missing_error(exc):
                continue
            messages.append(f"S3 next_x JSON skipped ({candidate_key}): {exc}")
            break

    effective = StrainDashboardPaths(
        s3_env_file=paths.s3_env_file,
        local_data_dir=paths.local_data_dir,
        s3_bucket=bucket,
        s3_data_key=data_key,
        s3_surrogate_key=surrogate_key,
        s3_next_x_key=next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
        version_suffix=paths.version_suffix,
    )
    if not surrogate and surrogate_candidates:
        messages.append(
            "S3 surrogate JSON not found for latest triplet or timestamped fallbacks."
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


def _message_targets_location(message: str, location: str) -> bool:
    msg = (message or "").strip()
    loc = (location or "").strip()
    if not msg or not loc:
        return False
    if loc in msg:
        return True
    base = _location_basename(loc)
    return bool(base and base in msg)


def collect_nsdf_triplet_load_issues(
    paths: StrainDashboardPaths,
    bundle: NSDFLoadedBundle,
    *,
    grid_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Summarize problems with the primary ``data.json`` / ``surrogate.json`` / ``next_x.json`` triplet.

    Returns ``(errors, warnings)``. Errors indicate the primary snapshot file is missing or unusable
    (including when a fallback surrogate was used). Warnings cover optional ``next_x.json`` gaps.
    """
    errors: List[str] = []
    warnings: List[str] = []
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

    for msg in bundle.messages:
        if _message_targets_location(msg, primary_sur) and any(
            token in msg.lower() for token in ("skipped", "not found", "not loaded", "missing")
        ):
            detail = msg.split(":", 1)[-1].strip() if ":" in msg else msg
            errors.append(f"{sur_fn}: {detail}")
        elif _message_targets_location(msg, primary_nx) and any(
            token in msg.lower() for token in ("skipped", "not found", "not loaded", "missing")
        ):
            detail = msg.split(":", 1)[-1].strip() if ":" in msg else msg
            if "expected a json object or array" in msg.lower():
                errors.append(f"{nx_fn}: {detail}")
            else:
                warnings.append(f"{nx_fn}: {detail}")

    if bundle.surrogate is None:
        if not any(sur_fn in item for item in errors):
            errors.append(f"{sur_fn}: not loaded (missing or every fallback failed).")
    elif primary_sur:
        primary_base = _location_basename(primary_sur)
        loaded_base = _location_basename(loaded_sur)
        if primary_base and loaded_base and primary_base != loaded_base:
            errors.append(
                f"{sur_fn}: using fallback {loaded_base} (primary {primary_base} missing or invalid)."
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
    for ny in range(1, int(math.sqrt(length)) + 1):
        if length % ny != 0:
            continue
        nx = length // ny
        if nx >= ny:
            return nx, ny
    warnings.append(
        f"Could not infer surrogate grid shape for field length {length}; "
        "provide surrogate bounds."
    )
    return None


def _align_grid_to_display(
    grid: np.ndarray,
    target_nx: int,
    target_ny: int,
) -> np.ndarray:
    """Resize a model grid onto the dashboard display grid."""
    src_ny, src_nx = grid.shape
    if src_nx == target_nx and src_ny == target_ny:
        return grid
    try:
        from scipy import ndimage  # type: ignore

        zoom_y = target_ny / float(src_ny)
        zoom_x = target_nx / float(src_nx)
        return ndimage.zoom(grid, (zoom_y, zoom_x), order=1)
    except Exception:
        y_idx = np.clip(
            np.rint(np.linspace(0, src_ny - 1, target_ny)).astype(int),
            0,
            src_ny - 1,
        )
        x_idx = np.clip(
            np.rint(np.linspace(0, src_nx - 1, target_nx)).astype(int),
            0,
            src_nx - 1,
        )
        return grid[np.ix_(y_idx, x_idx)]


def _surrogate_field_to_display_grid(
    values: Optional[np.ndarray],
    *,
    key: str,
    surrogate_info: NSDFSurrogateData,
    surrogate_doc: Optional[Mapping[str, Any]],
    display_size: Tuple[int, int],
    warnings: List[str],
) -> Optional[np.ndarray]:
    """Reshape a flattened surrogate model field and align it to the display grid."""
    if values is None:
        return None
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
        grid = _align_grid_to_display(grid, display_nx, display_ny)
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


def next_x_grid_coords_for_workflow(
    next_x_info: NSDFNextXData,
    workflow_id: Optional[str],
    nx: int,
    ny: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map proposed next-scan coordinates onto the dashboard grid for one workflow."""
    if not workflow_id:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    coords: List[Tuple[float, float]] = []
    for entry in next_x_info.entries:
        if entry.workflow_id != workflow_id:
            continue
        for row in entry.coordinates:
            coords.append((float(row[0]), float(row[1])))
    if not coords:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    arr = np.asarray(coords, dtype=np.float64)
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
            if step_id == current_snapshot:
                return idx
    fallback = _snapshot_index_in_series(chrono_snaps, current_snapshot, len(step_ids))
    return fallback


def build_uncertainty_trend_from_surrogate_paths(
    surrogate_paths: StrainDashboardPaths,
    *,
    current_snapshot: str = "latest",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
    remote_linked: bool = False,
) -> Optional[UncertaintyTrendSeries]:
    """
    Fast uncertainty trend from one ``surrogate.json`` (``transformed_stddevs_avg`` only).

    Returns ``None`` when the selected surrogate does not contain a multi-point series.
    """
    trend_doc = load_surrogate_doc_for_paths(
        surrogate_paths,
        mongo_s3_auth=mongo_s3_auth,
        remote_linked=remote_linked,
    )
    full_points, parse_warnings = parse_transformed_stddevs_avg_points(
        (trend_doc or {}).get("transformed_stddevs_avg")
    )
    if len(full_points) < 2:
        return None
    step_ids = [point[0] for point in full_points]
    ys = np.asarray([point[1] for point in full_points], dtype=np.float64)
    current_snapshot = (current_snapshot or "latest").strip() or "latest"
    current_index: Optional[int] = None
    if current_snapshot != "latest":
        for idx, step_id in enumerate(step_ids):
            if step_id == current_snapshot:
                current_index = idx
                break
    return UncertaintyTrendSeries(
        step_ids=step_ids,
        y=ys,
        labels=list(step_ids),
        current_index=current_index,
        source="transformed_stddevs_avg",
        warnings=parse_warnings,
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

    Prefers ``transformed_stddevs_avg`` ``[[id, avg], ...]`` from the selected
    ``surrogate.json``; falls back to per-snapshot means from uncertainty grids.
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
    if len(full_points) >= 2:
        step_ids = [point[0] for point in full_points]
        ys = np.asarray([point[1] for point in full_points], dtype=np.float64)
        current_index = _trend_current_index(
            step_ids,
            current_snapshot=current_snapshot,
            chrono_snaps=chrono,
        )
        return UncertaintyTrendSeries(
            step_ids=step_ids,
            y=ys,
            labels=list(step_ids),
            current_index=current_index,
            source="transformed_stddevs_avg",
            warnings=warnings,
        )

    if not allow_per_snapshot_fallback:
        return UncertaintyTrendSeries(
            step_ids=[],
            y=np.zeros(0, dtype=np.float64),
            labels=[],
            warnings=warnings
            or ["Uncertainty trend will update after the snapshot catalog finishes loading."],
        )

    step_ids_out: List[str] = []
    ys_out: List[float] = []
    labels_out: List[str] = []
    current_index: Optional[int] = None
    current_snapshot = (current_snapshot or "latest").strip() or "latest"

    for step_idx, snap in enumerate(chrono, start=1):
        suffix = snap.suffix or "latest"
        label = "Latest" if not snap.suffix else snap.suffix
        snap_paths = apply_nsdf_version_suffix(
            base_paths,
            "" if suffix == "latest" else suffix,
        )
        surrogate_doc = load_surrogate_doc_for_paths(
            snap_paths,
            mongo_s3_auth=mongo_s3_auth,
            remote_linked=remote_linked,
        )
        step_id, y_val, _source = _uncertainty_point_from_surrogate_doc(
            surrogate_doc,
            grid_size=grid_size,
            step_index=step_idx,
            step_id=label,
        )
        if y_val is None:
            warnings.append(f"Snapshot {label}: no transformed_stddevs_avg or uncertainty grid.")
            continue
        if step_id is None:
            step_id = label
        step_ids_out.append(step_id)
        ys_out.append(float(y_val))
        labels_out.append(label)
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
    return NSDFSurrogateData(
        surrogate=_numeric_1d_surrogate_array_or_none(surrogate_doc, "surrogate", warnings),
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
        meta["estimate_source"] = "surrogate_grid"
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
        meta["variance_source"] = "uncertainty_squared_grid"
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
    mapper = LinearColorMapper(palette=palette, low=lo, high=hi)

    p = _strain_plot_figure(title, nx, ny, cfg, layout=layout)
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
    bounds = grids.meta.get("measurement_bounds")
    if not isinstance(bounds, tuple):
        bounds = None

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
    est_lo, est_hi = _shared_field_limits(est, measured_vals)
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
        return Div(
            text=(
                f"<b>{html.escape(cfg.title_uncertainty_trend)}</b><br>"
                "<i>No uncertainty trend data for this workflow yet.</i>"
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

    from bokeh.models import FactorRange

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
        x_range=FactorRange(factors=list(series.step_ids)),
        toolbar_location="above",
    )
    _lock_strain_figure_layout(p, layout)
    highlight_id = None
    if series.current_index is not None and 0 <= series.current_index < len(series.step_ids):
        highlight_id = series.step_ids[series.current_index]
    sparse_ticks = _sparse_trend_axis_ticks(
        series.step_ids,
        panel_width=layout.frame_width,
        highlight_id=highlight_id,
    )
    sparse_set = set(sparse_ticks)
    # FactorRange is categorical: blank overrides hide labels without changing point positions.
    p.xaxis.major_label_overrides = {
        step_id: step_id if step_id in sparse_set else ""
        for step_id in series.step_ids
    }
    p.xaxis.major_label_orientation = math.pi / 4 if len(sparse_ticks) > 4 else 0.0
    source = ColumnDataSource(
        data={
            "step_id": list(series.step_ids),
            "y": series.y.tolist(),
            "label": series.labels,
        }
    )
    p.line("step_id", "y", source=source, line_width=2, color="#2c7bb6")
    p.circle("step_id", "y", source=source, size=6, color="#2c7bb6", alpha=0.7, line_color="#1f4f73")
    if series.current_index is not None and 0 <= series.current_index < len(series.step_ids):
        p.circle(
            [series.step_ids[series.current_index]],
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
