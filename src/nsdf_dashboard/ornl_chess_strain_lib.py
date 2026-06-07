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


def _strain_effective_source_order(base_dir: str, save_dir: str) -> Tuple[str, ...]:
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
) -> StrainDashboardPaths:
    """
    Build ``StrainDashboardPaths`` using the configured source order.

    When a file is chosen from ``upload`` or ``converted``, ``json_url`` is still set to the
    first available of ``query_strain_json_url`` / ``env.json_url`` when present (for display /
    provenance); ``load_strain_json`` reads the file first.
    """
    env = env or StrainDashboardPaths.from_environ()
    order = _strain_effective_source_order(base_dir, save_dir)
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
    title_estimate: str = "Estimate"
    title_variance: str = "Variance"
    flip_y_for_display: bool = True
    colormap_estimate: str = "Viridis256"
    colormap_variance: str = "Viridis256"
    colormap_mask: Tuple[str, str] = ("#2d1b4e", "#fde724")


@dataclass
class StrainFieldGrids:
    """Three 2D numpy arrays aligned to the same pixel grid."""

    measurements: np.ndarray  # float 0/1 or 0..1 mask
    estimate: np.ndarray
    variance: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


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
    """One workflow block from ``next_x.json``."""

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
    next_x: Optional[List[Dict[str, Any]]] = None
    messages: List[str] = field(default_factory=list)
    paths: StrainDashboardPaths = field(default_factory=StrainDashboardPaths)


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


def apply_nsdf_version_suffix(
    paths: StrainDashboardPaths,
    version_suffix: str = "",
) -> StrainDashboardPaths:
    """
    Point resolved paths at a timestamped triplet (``data_<ts>.json``, etc.).

    Empty suffix keeps the default latest files (``data.json``, ``surrogate.json``, ``next_x.json``).
    """
    suffix = (version_suffix or "").strip()
    data_fn, sur_fn, nx_fn = nsdf_triplet_basenames(suffix)
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
        version_suffix=suffix,
    )

    if out.local_data_dir:
        base = out.local_data_dir.rstrip("/")
        out.local_json_path = os.path.join(base, data_fn)
        if suffix:
            out.surrogate_json_path = os.path.join(base, sur_fn)
            out.next_x_json_path = os.path.join(base, nx_fn)
        return out

    if out.local_json_path and not _looks_like_http_url(out.local_json_path):
        base = os.path.dirname(out.local_json_path)
        out.local_json_path = os.path.join(base, data_fn)
        if suffix:
            out.surrogate_json_path = os.path.join(base, sur_fn)
            out.next_x_json_path = os.path.join(base, nx_fn)
        return out

    if out.json_url:
        out.json_url = _replace_url_basename(out.json_url, data_fn)
        if suffix:
            out.surrogate_json_url = _replace_url_basename(out.json_url, sur_fn)
            out.next_x_json_url = _replace_url_basename(out.json_url, nx_fn)

    if out.s3_data_key:
        out.s3_data_key = _replace_s3_key_basename(out.s3_data_key, data_fn)
        if suffix:
            out.s3_surrogate_key = _replace_s3_key_basename(out.s3_data_key, sur_fn)
            out.s3_next_x_key = _replace_s3_key_basename(out.s3_data_key, nx_fn)
        else:
            out.s3_surrogate_key = (paths.s3_surrogate_key or "").strip() or _sibling_surrogate_s3_key(
                out.s3_data_key
            )
            out.s3_next_x_key = (paths.s3_next_x_key or "").strip() or _sibling_next_x_s3_key(out.s3_data_key)

    return out


def nsdf_listing_directory(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
) -> str:
    """Best-effort local directory for discovering timestamped NSDF JSON backups."""
    candidates: List[str] = []
    if (paths.local_data_dir or "").strip():
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


def _local_data_dir_configured(paths: StrainDashboardPaths) -> bool:
    """True when ``LOCAL_DATA_DIR`` is set — local runs must not touch S3."""
    local_dir = (paths.local_data_dir or "").strip()
    return bool(local_dir and os.path.isdir(local_dir))


def _remote_snapshot_listing_enabled(paths: StrainDashboardPaths) -> bool:
    """True when snapshots should be discovered from S3 / gateway URL."""
    if _local_data_dir_configured(paths):
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


def discover_nsdf_version_options(
    paths: StrainDashboardPaths,
    *,
    base_dir: str = "",
    save_dir: str = "",
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    Return ``(value, label)`` pairs for a version selector.

    ``latest`` is always first; values are ISO-like suffixes such as ``20260606T223505Z``.

    Local ``LOCAL_DATA_DIR`` runs list only that folder. S3 / gateway runs list only
    the resolved remote prefix. The two sources are never merged.
    """
    options: List[Tuple[str, str]] = [("latest", "Latest (data.json)")]
    seen: set[str] = set()

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
        return paths
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()
    mirror = find_strain_json_under_dataset_dir(bd) or find_strain_json_under_dataset_dir(sd)
    if mirror:
        return _copy_s3_fields(
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
    if not doc:
        return paths
    link = resolve_strain_json_remote_link_from_dataset(doc)
    if not link:
        return paths
    if link.lower().startswith("s3://"):
        bucket, data_key = parse_s3_uri(link)
        if bucket and data_key:
            ep = str(doc.get("s3_endpoint_url") or "").strip()
            reg = str(doc.get("s3_region_name") or "us-east-1").strip() or "us-east-1"
            return _copy_s3_fields(
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
    if _looks_like_http_url(link):
        return _copy_s3_fields(
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
    if os.path.isfile(link):
        return _copy_s3_fields(
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
    # s3:// and other schemes: pass as URL for downstream loaders
    return _copy_s3_fields(
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
    if not candidates:
        sib_path = _sibling_surrogate_path(paths.local_json_path)
        if sib_path:
            candidates.append(("path", sib_path, False))
        else:
            sib_url = _sibling_surrogate_url(paths.json_url)
            if sib_url:
                candidates.append(("url", sib_url, False))

    for kind, value, explicit in candidates:
        try:
            if kind == "path":
                doc = load_json_from_local_path(value)
                effective.surrogate_json_path = value
            else:
                doc = load_json_from_url(value, s3_auth_override=mongo_s3_auth)
                effective.surrogate_json_url = value
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
    """Validate ``next_x.json``: a list of workflow blocks with coordinate pairs."""
    warnings: List[str] = []
    if value is None:
        return NSDFNextXData(warnings=warnings)
    if not isinstance(value, list):
        return NSDFNextXData(warnings=["Skipping next_x JSON: expected a JSON array."])

    entries: List[NSDFNextXEntry] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(f"Skipping next_x[{i}]: expected a JSON object.")
            continue
        workflow_id = str(item.get("workflow_id") or "").strip()
        if not workflow_id:
            warnings.append(f"Skipping next_x[{i}]: missing workflow_id.")
            continue
        data = item.get("data")
        if not isinstance(data, list) or not data:
            warnings.append(f"Skipping next_x[{i}] ({workflow_id!r}): data must be a non-empty list.")
            continue
        coords: List[Tuple[float, float]] = []
        bad_row = False
        for j, row_value in enumerate(data):
            if not isinstance(row_value, list) or len(row_value) < 2:
                warnings.append(
                    f"Skipping next_x[{i}] ({workflow_id!r}): data[{j}] must contain labx/labz values."
                )
                bad_row = True
                break
            x, z = row_value[0], row_value[1]
            if not (_is_number(x) and _is_number(z)):
                warnings.append(
                    f"Skipping next_x[{i}] ({workflow_id!r}): data[{j}] must contain numeric values."
                )
                bad_row = True
                break
            fx, fz = float(x), float(z)
            if not (math.isfinite(fx) and math.isfinite(fz)):
                warnings.append(
                    f"Skipping next_x[{i}] ({workflow_id!r}): data[{j}] values must be finite."
                )
                bad_row = True
                break
            coords.append((fx, fz))
        if bad_row:
            continue
        entries.append(
            NSDFNextXEntry(
                workflow_id=workflow_id,
                coordinates=np.asarray(coords, dtype=np.float64),
            )
        )
    return NSDFNextXData(entries=entries, warnings=warnings)


def load_optional_next_x_json(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str], StrainDashboardPaths]:
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
    if not candidates:
        sib_path = _sibling_next_x_path(paths.local_json_path)
        if sib_path:
            candidates.append(("path", sib_path, False))
        else:
            sib_url = _sibling_next_x_url(paths.json_url)
            if sib_url:
                candidates.append(("url", sib_url, False))

    for kind, value, explicit in candidates:
        try:
            if kind == "path":
                doc = load_json_from_local_path(value)
                effective.next_x_json_path = value
            else:
                doc = load_json_from_url(value, s3_auth_override=mongo_s3_auth)
                effective.next_x_json_url = value
            if not isinstance(doc, list):
                raise ValueError("next_x.json must be a JSON array.")
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
    data_fn, sur_fn, nx_fn = nsdf_triplet_basenames(paths.version_suffix or "")
    data_path = os.path.join(local_dir, data_fn)
    if not os.path.isfile(data_path):
        return None

    data = load_json_from_local_path(data_path)
    surrogate = None
    messages = [f"Loaded NSDF data JSON from local path: {data_path}"]
    surrogate_path = os.path.join(local_dir, sur_fn)
    effective = StrainDashboardPaths(
        local_json_path=data_path,
        surrogate_json_path="",
        local_data_dir=local_dir,
        version_suffix=paths.version_suffix,
    )
    if os.path.isfile(surrogate_path):
        try:
            surrogate = load_json_from_local_path(surrogate_path)
            effective.surrogate_json_path = surrogate_path
            messages.append(f"Loaded surrogate JSON from local path: {surrogate_path}")
        except Exception as exc:
            messages.append(f"Local surrogate JSON skipped: {exc}")
    next_x = None
    next_x_path = os.path.join(local_dir, nx_fn)
    if os.path.isfile(next_x_path):
        try:
            next_x_doc = load_json_from_local_path(next_x_path)
            if isinstance(next_x_doc, list):
                next_x = next_x_doc
                effective.next_x_json_path = next_x_path
                messages.append(f"Loaded next_x JSON from local path: {next_x_path}")
            else:
                messages.append("Local next_x JSON skipped: expected a JSON array.")
        except Exception as exc:
            messages.append(f"Local next_x JSON skipped: {exc}")
    return NSDFLoadedBundle(data=data, surrogate=surrogate, next_x=next_x, messages=messages, paths=effective)


def load_nsdf_json_bundle(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> NSDFLoadedBundle:
    if _local_data_dir_configured(paths):
        local_bundle = load_nsdf_json_bundle_from_local_data_dir(paths)
        if local_bundle is not None:
            return local_bundle
        data_fn, _, _ = nsdf_triplet_basenames(paths.version_suffix or "")
        local_dir = (paths.local_data_dir or "").strip()
        raise FileNotFoundError(
            f"Local NSDF file not found: {os.path.join(local_dir, data_fn)} "
            f"(LOCAL_DATA_DIR={local_dir!r})"
        )

    local_bundle = load_nsdf_json_bundle_from_local_data_dir(paths)
    if local_bundle is not None:
        return local_bundle
    if paths.has_s3_source():
        return load_nsdf_json_bundle_from_s3(paths, mongo_s3_auth=mongo_s3_auth)
    data = load_strain_json(paths, mongo_s3_auth=mongo_s3_auth)
    surrogate, messages, effective = load_optional_surrogate_json(
        paths,
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

    surrogate = None
    surrogate_key = (paths.s3_surrogate_key or "").strip() or _sibling_surrogate_s3_key(data_key)
    next_x_key = (paths.s3_next_x_key or "").strip() or _sibling_next_x_s3_key(data_key)
    effective = StrainDashboardPaths(
        s3_env_file=paths.s3_env_file,
        local_data_dir=paths.local_data_dir,
        s3_bucket=bucket,
        s3_data_key=data_key,
        s3_surrogate_key=surrogate_key,
        s3_next_x_key=next_x_key,
        s3_endpoint_url=paths.s3_endpoint_url,
        s3_region=paths.s3_region,
    )
    if surrogate_key:
        try:
            surrogate = _load_json_from_s3_key(client, bucket, surrogate_key)
            messages.append(f"Loaded surrogate JSON from s3://{bucket}/{surrogate_key}")
        except Exception as exc:
            if _s3_missing_error(exc):
                messages.append(f"S3 surrogate JSON not found: s3://{bucket}/{surrogate_key}")
            else:
                messages.append(f"S3 surrogate JSON skipped: {exc}")

    next_x = None
    if next_x_key:
        try:
            next_x_doc = _load_json_from_s3_key(client, bucket, next_x_key)
            if isinstance(next_x_doc, list):
                next_x = next_x_doc
                messages.append(f"Loaded next_x JSON from s3://{bucket}/{next_x_key}")
            else:
                messages.append(f"S3 next_x JSON skipped: expected a JSON array at s3://{bucket}/{next_x_key}")
        except Exception as exc:
            if _s3_missing_error(exc):
                messages.append(f"S3 next_x JSON not found: s3://{bucket}/{next_x_key}")
            else:
                messages.append(f"S3 next_x JSON skipped: {exc}")

    return NSDFLoadedBundle(data=data, surrogate=surrogate, next_x=next_x, messages=messages, paths=effective)


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
    """Place marker legend in the layout below the plot frame (not over the heatmap)."""
    if not legend_items:
        return
    from bokeh.models import Legend, LegendItem

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
    figure.min_border_bottom = 10


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


def format_nsdf_workflow_display(
    surrogate_info: NSDFSurrogateData,
    next_x_info: "NSDFNextXData",
) -> str:
    """Show the single active workflow id for the dashboard banner."""
    workflow_id = resolve_nsdf_workflow_id(surrogate_info, next_x_info)
    if not workflow_id:
        return ""
    return f"Workflow ID: {workflow_id}"


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
    if var_grid is not None:
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


def make_strain_heatmap_figure(
    title: str,
    z: np.ndarray,
    cfg: StrainFieldPlotConfig,
    *,
    palette_name: str = "Viridis256",
    discrete_mask: bool = False,
    low_high: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = False,
):
    from bokeh.models import BasicTicker, ColorBar, FixedTicker, LinearColorMapper, NumeralTickFormatter
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    nx, ny = cfg.grid_size
    zd = _maybe_flip_y(np.asarray(z, dtype=np.float64), cfg.flip_y_for_display)

    lo = 0.0
    hi = 1.0
    if discrete_mask:
        lo_c, hi_c = cfg.colormap_mask
        palette = [lo_c, hi_c]
        mapper = LinearColorMapper(palette=palette, low=0.0, high=1.0)
    else:
        try:
            from bokeh.palettes import all_palettes  # type: ignore

            palette = all_palettes.get(palette_name) or Viridis256
        except Exception:
            palette = Viridis256
        if low_high is not None:
            lo, hi = low_high
        else:
            finite = zd[np.isfinite(zd)]
            lo = float(np.nanmin(finite)) if finite.size else 0.0
            hi = float(np.nanmax(finite)) if finite.size else 1.0
            if lo == hi:
                hi = lo + 1e-12
        mapper = LinearColorMapper(palette=palette, low=lo, high=hi)

    p = figure(
        title=title,
        x_range=(0, nx),
        y_range=(0, ny),
        width=380 if show_colorbar else 320,
        height=320,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        match_aspect=True,
        aspect_scale=1,
    )
    p.xaxis.axis_label = cfg.x_axis_label
    p.yaxis.axis_label = cfg.y_axis_label
    p.xaxis.ticker = FixedTicker(ticks=list(range(0, nx + 1, max(1, nx // 5))))
    p.yaxis.ticker = FixedTicker(ticks=list(range(0, ny + 1, max(1, ny // 5))))

    p.image(image=[zd], x=0, y=0, dw=nx, dh=ny, color_mapper=mapper)
    if show_colorbar and not discrete_mask:
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
        )
        p.add_layout(color_bar, "right")
    return p


def make_strain_triplet_figures(
    grids: StrainFieldGrids,
    cfg: StrainFieldPlotConfig,
    *,
    row_subtitle: str = "",
    next_x_info: Optional[NSDFNextXData] = None,
    active_workflow_id: Optional[str] = None,
):
    sub = f" — {row_subtitle}" if row_subtitle else ""
    est = grids.estimate
    finite = est[np.isfinite(est)]
    lo = float(np.nanmin(finite)) if finite.size else 0.0
    hi = float(np.nanmax(finite)) if finite.size else 1.0
    if lo == hi:
        hi = lo + 1e-12

    nx, ny = cfg.grid_size
    bounds = grids.meta.get("measurement_bounds")
    if not isinstance(bounds, tuple):
        bounds = None

    p0 = make_strain_heatmap_figure(
        f"{cfg.title_measurements}{sub}",
        grids.measurements,
        cfg,
        palette_name=cfg.colormap_estimate,
        discrete_mask=True,
        low_high=(0.0, 1.0),
    )
    point_legend_items: List[Tuple[str, Any]] = []
    measured_gx = grids.meta.get("measurement_gx")
    measured_gy = grids.meta.get("measurement_gy")
    if isinstance(measured_gx, np.ndarray) and isinstance(measured_gy, np.ndarray) and measured_gx.size:
        sampled_renderer = _add_grid_point_overlay(
            p0,
            measured_gx,
            measured_gy,
            nx,
            ny,
            cfg.flip_y_for_display,
            color=cfg.colormap_mask[1],
            marker="circle",
            size=9,
            line_color="#ffffff",
            line_width=1.25,
            fill_alpha=0.95,
        )
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
            proposed_renderer = _add_grid_point_overlay(
                p0,
                nx_gx,
                nx_gy,
                nx,
                ny,
                cfg.flip_y_for_display,
                color="#ff6600",
                marker="cross",
                size=16,
                line_color="#ff6600",
                line_width=2.5,
                fill_alpha=0.0,
            )
            if proposed_renderer is not None:
                point_legend_items.append(("Proposed next scan", proposed_renderer))
    _attach_point_legend_below(p0, point_legend_items)
    p1 = make_strain_heatmap_figure(
        f"{cfg.title_estimate}{sub}",
        grids.estimate,
        cfg,
        palette_name=cfg.colormap_estimate,
        low_high=(lo, hi),
        show_colorbar=True,
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
    )
    return p0, p1, p2
