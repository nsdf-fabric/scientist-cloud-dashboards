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


@dataclass
class StrainDashboardPaths:
    """Where to load NSDF data and optional surrogate JSON."""

    local_json_path: str = ""
    json_url: str = ""
    surrogate_json_path: str = ""
    surrogate_json_url: str = ""
    local_data_dir: str = ""
    s3_env_file: str = ""
    s3_bucket: str = ""
    s3_data_key: str = ""
    s3_surrogate_key: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"

    @classmethod
    def from_environ(cls) -> "StrainDashboardPaths":
        env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())

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
            local_data_dir=env_value("ORNL_NSDF_LOCAL_DATA_DIR"),
            s3_env_file=os.environ.get("ORNL_S3_ENV_FILE", "").strip(),
            s3_bucket=env_value("ORNL_NSDF_S3_BUCKET"),
            s3_data_key=env_value("ORNL_NSDF_S3_DATA_KEY"),
            s3_surrogate_key=env_value("ORNL_NSDF_S3_SURROGATE_KEY"),
            s3_endpoint_url=env_value("ORNL_NSDF_S3_ENDPOINT_URL"),
            s3_region=env_value("ORNL_NSDF_S3_REGION", "us-east-1") or "us-east-1",
        )

    def has_s3_source(self) -> bool:
        return bool((self.s3_bucket or "").strip() and (self.s3_data_key or "").strip())


def _copy_s3_fields(src: StrainDashboardPaths, dst: StrainDashboardPaths) -> StrainDashboardPaths:
    dst.local_data_dir = src.local_data_dir
    dst.s3_env_file = src.s3_env_file
    dst.s3_bucket = src.s3_bucket
    dst.s3_data_key = src.s3_data_key
    dst.s3_surrogate_key = src.s3_surrogate_key
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
    q_path = (query_strain_json_path or "").strip()
    q_url = (query_strain_json_url or "").strip()
    env_path = (env.local_json_path or "").strip()
    env_url = (env.json_url or "").strip()
    surrogate_path = (query_surrogate_json_path or "").strip() or (env.surrogate_json_path or "").strip()
    surrogate_url = (query_surrogate_json_url or "").strip() or (env.surrogate_json_url or "").strip()
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()

    def with_surrogate(p: StrainDashboardPaths) -> StrainDashboardPaths:
        p.surrogate_json_path = surrogate_path
        p.surrogate_json_url = surrogate_url
        p.local_data_dir = env.local_data_dir
        p.s3_env_file = env.s3_env_file
        p.s3_bucket = env.s3_bucket
        p.s3_data_key = env.s3_data_key
        p.s3_surrogate_key = env.s3_surrogate_key
        p.s3_endpoint_url = env.s3_endpoint_url
        p.s3_region = env.s3_region
        return p

    def display_url() -> str:
        return q_url or env_url

    for token in order:
        if token == "upload":
            p = find_strain_json_under_dataset_dir(bd)
            if p:
                return with_surrogate(StrainDashboardPaths(local_json_path=p, json_url=display_url()))
        elif token == "converted":
            p = find_strain_json_under_dataset_dir(sd)
            if p:
                return with_surrogate(StrainDashboardPaths(local_json_path=p, json_url=display_url()))
        elif token == "query_path":
            if not q_path:
                continue
            if _looks_like_http_url(q_path):
                return with_surrogate(StrainDashboardPaths(local_json_path="", json_url=q_path))
            if os.path.isfile(q_path):
                return with_surrogate(StrainDashboardPaths(local_json_path=q_path, json_url=display_url()))
        elif token == "query_url":
            if q_url:
                return with_surrogate(StrainDashboardPaths(local_json_path="", json_url=q_url))
        elif token == "env_path":
            if not env_path:
                continue
            if _looks_like_http_url(env_path):
                return with_surrogate(StrainDashboardPaths(local_json_path="", json_url=env_path))
            if os.path.isfile(env_path):
                return with_surrogate(StrainDashboardPaths(local_json_path=env_path, json_url=display_url()))
        elif token == "env_url":
            if env_url:
                return with_surrogate(StrainDashboardPaths(local_json_path="", json_url=env_url))

    return with_surrogate(StrainDashboardPaths(local_json_path="", json_url=""))


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
    """Validated length-compatible optional surrogate arrays."""

    surrogate: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    raw_uncertainty: Optional[np.ndarray] = None
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
class NSDFLoadedBundle:
    """Loaded NSDF data and optional surrogate metadata for the UI."""

    data: Dict[str, Any]
    surrogate: Optional[Dict[str, Any]] = None
    messages: List[str] = field(default_factory=list)
    paths: StrainDashboardPaths = field(default_factory=StrainDashboardPaths)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def _looks_like_http_url(value: str) -> bool:
    v = (value or "").strip().lower()
    return v.startswith("http://") or v.startswith("https://")


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
    link = pick_strain_json_link_from_dataset_doc(doc)
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
            ),
        )
    if not doc:
        return paths
    link = resolve_strain_json_remote_link_from_dataset(doc)
    if not link:
        return paths
    if _looks_like_http_url(link):
        return _copy_s3_fields(
            paths,
            StrainDashboardPaths(
                local_json_path="",
                json_url=link,
                surrogate_json_path=paths.surrogate_json_path,
                surrogate_json_url=paths.surrogate_json_url,
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

    u = (url or "").strip()
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
    if os.path.basename(p).lower() != "data.json":
        return ""
    return os.path.join(os.path.dirname(p), "surrogate.json")


def _sibling_surrogate_url(data_url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    u = (data_url or "").strip()
    if not _looks_like_http_url(u):
        return ""
    parts = urlparse(u)
    if not parts.path.lower().endswith("/data.json"):
        return ""
    new_path = parts.path[: -len("data.json")] + "surrogate.json"
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


def load_nsdf_json_bundle_from_local_data_dir(paths: StrainDashboardPaths) -> Optional[NSDFLoadedBundle]:
    """Load fixed-name data/surrogate JSON files from ORNL_NSDF_LOCAL_DATA_DIR when present."""
    local_dir = (paths.local_data_dir or "").strip()
    if not local_dir:
        return None
    data_path = os.path.join(local_dir, "data.json")
    if not os.path.isfile(data_path):
        return None

    data = load_json_from_local_path(data_path)
    surrogate = None
    messages = [f"Loaded NSDF data JSON from local path: {data_path}"]
    surrogate_path = os.path.join(local_dir, "surrogate.json")
    effective = StrainDashboardPaths(
        local_json_path=data_path,
        surrogate_json_path="",
        local_data_dir=local_dir,
    )
    if os.path.isfile(surrogate_path):
        try:
            surrogate = load_json_from_local_path(surrogate_path)
            effective.surrogate_json_path = surrogate_path
            messages.append(f"Loaded surrogate JSON from local path: {surrogate_path}")
        except Exception as exc:
            messages.append(f"Local surrogate JSON skipped: {exc}")
    return NSDFLoadedBundle(data=data, surrogate=surrogate, messages=messages, paths=effective)


def load_nsdf_json_bundle(
    paths: StrainDashboardPaths,
    *,
    mongo_s3_auth: Optional[Dict[str, str]] = None,
) -> NSDFLoadedBundle:
    local_bundle = load_nsdf_json_bundle_from_local_data_dir(paths)
    if local_bundle is not None:
        return local_bundle
    if paths.has_s3_source():
        return load_nsdf_json_bundle_from_s3(paths)
    data = load_strain_json(paths, mongo_s3_auth=mongo_s3_auth)
    surrogate, messages, effective = load_optional_surrogate_json(
        paths,
        mongo_s3_auth=mongo_s3_auth,
    )
    return NSDFLoadedBundle(data=data, surrogate=surrogate, messages=messages, paths=effective)


def _sibling_surrogate_s3_key(data_key: str) -> str:
    key = (data_key or "").strip()
    if not key.lower().endswith("data.json"):
        return ""
    return key[: -len("data.json")] + "surrogate.json"


def _s3_env_values(paths: StrainDashboardPaths) -> Dict[str, str]:
    file_values = load_simple_env_file(paths.s3_env_file)

    def value(name: str, default: str = "") -> str:
        return (os.environ.get(name) or file_values.get(name) or default).strip()

    return {
        "aws_access_key_id": value("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": value("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": value("AWS_SESSION_TOKEN"),
        "endpoint_url": paths.s3_endpoint_url or value("ORNL_NSDF_S3_ENDPOINT_URL"),
        "region_name": paths.s3_region or value("ORNL_NSDF_S3_REGION", "us-east-1") or "us-east-1",
    }


def _make_nsdf_s3_client(paths: StrainDashboardPaths):
    import boto3

    cfg = _s3_env_values(paths)
    kwargs: Dict[str, Any] = {
        "region_name": cfg["region_name"],
    }
    if cfg["endpoint_url"]:
        kwargs["endpoint_url"] = cfg["endpoint_url"]
    if cfg["aws_access_key_id"] and cfg["aws_secret_access_key"]:
        kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
    if cfg["aws_session_token"]:
        kwargs["aws_session_token"] = cfg["aws_session_token"]
    return boto3.client("s3", **kwargs)


def _load_json_from_s3_key(client: Any, bucket: str, key: str) -> Dict[str, Any]:
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


def load_nsdf_json_bundle_from_s3(paths: StrainDashboardPaths) -> NSDFLoadedBundle:
    bucket = (paths.s3_bucket or "").strip()
    data_key = (paths.s3_data_key or "").strip()
    if not bucket or not data_key:
        raise FileNotFoundError(
            "Set ORNL_NSDF_S3_BUCKET and ORNL_NSDF_S3_DATA_KEY to load NSDF data from S3."
        )

    client = _make_nsdf_s3_client(paths)
    data = _load_json_from_s3_key(client, bucket, data_key)
    messages = [f"Loaded NSDF data JSON from s3://{bucket}/{data_key}"]

    surrogate = None
    surrogate_key = (paths.s3_surrogate_key or "").strip() or _sibling_surrogate_s3_key(data_key)
    effective = StrainDashboardPaths(
        s3_env_file=paths.s3_env_file,
        local_data_dir=paths.local_data_dir,
        s3_bucket=bucket,
        s3_data_key=data_key,
        s3_surrogate_key=surrogate_key,
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

    return NSDFLoadedBundle(data=data, surrogate=surrogate, messages=messages, paths=effective)


# ---------------------------------------------------------------------------
# NSDF field discovery
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_1d_array_or_none(
    doc: Optional[Mapping[str, Any]],
    key: str,
    expected_len: int,
    warnings: List[str],
) -> Optional[np.ndarray]:
    if not doc or key not in doc:
        return None
    value = doc.get(key)
    if not isinstance(value, list) or not value:
        warnings.append(f"Skipping surrogate field {key!r}: expected a non-empty numeric 1D list.")
        return None
    if len(value) != expected_len:
        warnings.append(
            f"Skipping surrogate field {key!r}: length {len(value)} does not match "
            f"dataset_y length {expected_len}."
        )
        return None
    if any((not _is_number(x)) for x in value):
        warnings.append(f"Skipping surrogate field {key!r}: all values must be numeric.")
        return None
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        warnings.append(f"Skipping surrogate field {key!r}: values must be finite.")
        return None
    return arr


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
    if not isinstance(dataset_x, list) or not dataset_x:
        raise ValueError("'dataset_x' must be a non-empty list of coordinate pairs.")
    if not isinstance(dataset_y, list) or not dataset_y:
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


def validate_nsdf_surrogate_doc(
    surrogate_doc: Optional[Mapping[str, Any]],
    expected_len: int,
) -> NSDFSurrogateData:
    warnings: List[str] = []
    if surrogate_doc is None:
        return NSDFSurrogateData(warnings=warnings)
    if not isinstance(surrogate_doc, Mapping):
        return NSDFSurrogateData(warnings=["Skipping surrogate JSON: expected a JSON object."])
    return NSDFSurrogateData(
        surrogate=_numeric_1d_array_or_none(surrogate_doc, "surrogate", expected_len, warnings),
        uncertainty=_numeric_1d_array_or_none(surrogate_doc, "uncertainty", expected_len, warnings),
        raw_uncertainty=_numeric_1d_array_or_none(
            surrogate_doc,
            "raw_uncertainty",
            expected_len,
            warnings,
        ),
        warnings=warnings,
    )


def list_nsdf_field_headers(
    data_doc: Mapping[str, Any],
    surrogate_doc: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    measurement = validate_nsdf_measurement_doc(data_doc)
    surrogate = validate_nsdf_surrogate_doc(surrogate_doc, measurement.observed_values.shape[0])
    return surrogate.plottable_fields


def infer_nsdf_grid_size(data_doc: Mapping[str, Any]) -> Tuple[int, int]:
    """Infer grid width/height from unique labx/labz coordinates in ``dataset_x``."""
    measurement = validate_nsdf_measurement_doc(data_doc)
    unique_x = np.unique(measurement.coordinates[:, 0])
    unique_z = np.unique(measurement.coordinates[:, 1])
    nx = int(unique_x.shape[0])
    ny = int(unique_z.shape[0])
    if nx <= 0 or ny <= 0:
        return DEFAULT_GRID_SIZE
    return nx, ny


def infer_nsdf_bounds_grid_size(data_doc: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Read explicit grid width/height from ``bounds=[[0, width], [0, height]]``."""
    if not isinstance(data_doc, Mapping):
        return None
    bounds = data_doc.get("bounds")
    if not isinstance(bounds, list) or len(bounds) < 2:
        return None

    out: List[int] = []
    for axis in bounds[:2]:
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
    return out[0], out[1]


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


def resolve_nsdf_grid_size(
    data_doc: Mapping[str, Any],
    *,
    env_grid_size: Optional[Tuple[int, int]] = None,
    manual_grid_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int], str]:
    """Resolve grid size and source using bounds, manual controls, env, then dataset_x."""
    bounds_size = infer_nsdf_bounds_grid_size(data_doc)
    if bounds_size:
        return bounds_size, "bounds"
    manual_size = _valid_grid_size(manual_grid_size)
    if manual_size:
        return manual_size, "manual controls"
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
    surrogate = validate_nsdf_surrogate_doc(surrogate_doc, measurement.observed_values.shape[0])
    values = measurement.observed_values
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
        "plottable_fields": surrogate.plottable_fields,
        "warnings": list(surrogate.warnings),
    }

    if surrogate.surrogate is not None:
        est = _idw_fill_grid(gx, gy, surrogate.surrogate, nx, ny)
        meta["estimate_source"] = "surrogate"
    else:
        est = _idw_fill_grid(gx, gy, values, nx, ny)
        meta["estimate_source"] = "dataset_y_idw"

    if surrogate.uncertainty is not None:
        var_samples = np.square(np.maximum(surrogate.uncertainty, 0.0))
        var = _idw_fill_grid(gx, gy, var_samples, nx, ny)
        meta["variance_source"] = "uncertainty_squared"
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
):
    from bokeh.models import FixedTicker, LinearColorMapper
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    nx, ny = cfg.grid_size
    zd = _maybe_flip_y(np.asarray(z, dtype=np.float64), cfg.flip_y_for_display)

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
        width=320,
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
    return p


def make_strain_triplet_figures(
    grids: StrainFieldGrids,
    cfg: StrainFieldPlotConfig,
    *,
    row_subtitle: str = "",
):
    sub = f" — {row_subtitle}" if row_subtitle else ""
    est = grids.estimate
    finite = est[np.isfinite(est)]
    lo = float(np.nanmin(finite)) if finite.size else 0.0
    hi = float(np.nanmax(finite)) if finite.size else 1.0
    if lo == hi:
        hi = lo + 1e-12

    p0 = make_strain_heatmap_figure(
        f"{cfg.title_measurements}{sub}",
        grids.measurements,
        cfg,
        palette_name=cfg.colormap_estimate,
        discrete_mask=True,
        low_high=(0.0, 1.0),
    )
    p1 = make_strain_heatmap_figure(
        f"{cfg.title_estimate}{sub}",
        grids.estimate,
        cfg,
        palette_name=cfg.colormap_estimate,
        low_high=(lo, hi),
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
    )
    return p0, p1, p2
