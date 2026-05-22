"""
ORNL / CHESS strain-field helpers: JSON parsing, header discovery, grid construction,
and Bokeh heatmaps. Lives under ``scientistCloudLib/SCLib_Dashboards`` (listed in
``ORNL_CHESS_strain.json`` ``shared_utilities``) so Docker and local runs match other dashboard utils.

**Where JSON is loaded from (deployment order)**

1. **ScientistCloud data portal** (default when ``base_dir`` / ``save_dir`` live under
   ``/mnt/visus_datasets``, e.g. upload + converted trees):

   - ``upload`` — first ``*.json`` under dataset ``base_dir`` (prefers ``reduced_data.json``)
   - ``converted`` — same under ``save_dir``
   - ``query_path`` / ``query_url`` — Bokeh URL args (portal may append gateway HTTPS)
   - ``env_path`` / ``env_url`` — ``ORNL_STRAIN_JSON_PATH`` / ``ORNL_STRAIN_JSON_URL``

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

DEFAULT_HEADER_REGEX = re.compile(
    r"^\d+/data/(uniform_strain|unconstrained_strain)$",
    re.IGNORECASE,
)

DEFAULT_GRID_SIZE: Tuple[int, int] = (26, 26)

DEFAULT_ROW_HEADERS: Tuple[str, ...] = (
    "0/data/uniform_strain",
    "0/data/unconstrained_strain",
)


@dataclass
class StrainDashboardPaths:
    """Where to load reduced JSON from (local path vs full http(s) URL you distribute)."""

    local_json_path: str = ""
    json_url: str = ""

    @classmethod
    def from_environ(cls) -> "StrainDashboardPaths":
        return cls(
            local_json_path=os.environ.get("ORNL_STRAIN_JSON_PATH", "").strip(),
            json_url=os.environ.get("ORNL_STRAIN_JSON_URL", "").strip(),
        )


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
    Pick a strain JSON file under ``directory`` (upload or converted tree for one UUID).

    Prefers ``reduced_data.json``; otherwise the first ``*.json`` (sorted by name).
    """
    d = (directory or "").strip()
    if not d or not os.path.isdir(d):
        return ""
    preferred = os.path.join(d, "reduced_data.json")
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
    bd = (base_dir or "").strip()
    sd = (save_dir or "").strip()

    def display_url() -> str:
        return q_url or env_url

    for token in order:
        if token == "upload":
            p = find_strain_json_under_dataset_dir(bd)
            if p:
                return StrainDashboardPaths(local_json_path=p, json_url=display_url())
        elif token == "converted":
            p = find_strain_json_under_dataset_dir(sd)
            if p:
                return StrainDashboardPaths(local_json_path=p, json_url=display_url())
        elif token == "query_path":
            if not q_path:
                continue
            if _looks_like_http_url(q_path):
                return StrainDashboardPaths(local_json_path="", json_url=q_path)
            if os.path.isfile(q_path):
                return StrainDashboardPaths(local_json_path=q_path, json_url=display_url())
        elif token == "query_url":
            if q_url:
                return StrainDashboardPaths(local_json_path="", json_url=q_url)
        elif token == "env_path":
            if not env_path:
                continue
            if _looks_like_http_url(env_path):
                return StrainDashboardPaths(local_json_path="", json_url=env_path)
            if os.path.isfile(env_path):
                return StrainDashboardPaths(local_json_path=env_path, json_url=display_url())
        elif token == "env_url":
            if env_url:
                return StrainDashboardPaths(local_json_path="", json_url=env_url)

    return StrainDashboardPaths(local_json_path="", json_url="")


@dataclass
class StrainFieldPlotConfig:
    """Per-plot titles and axis labels (easy to change when scientists refine wording)."""

    grid_size: Tuple[int, int] = field(default_factory=lambda: DEFAULT_GRID_SIZE)
    x_axis_label: str = "x index"
    y_axis_label: str = "y index"
    title_measurements: str = "Measurement locations"
    title_estimate: str = "GP estimate"
    title_variance: str = "GP variance"
    header_regex: re.Pattern = field(default_factory=lambda: DEFAULT_HEADER_REGEX)
    labx_key: str = "labx"
    labz_key: str = "labz"
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
        return StrainDashboardPaths(local_json_path=mirror, json_url="")
    if not doc:
        return paths
    link = resolve_strain_json_remote_link_from_dataset(doc)
    if not link:
        return paths
    if _looks_like_http_url(link):
        return StrainDashboardPaths(local_json_path="", json_url=link)
    if os.path.isfile(link):
        return StrainDashboardPaths(local_json_path=link, json_url="")
    # s3:// and other schemes: pass as URL for downstream loaders
    return StrainDashboardPaths(local_json_path="", json_url=link)


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
        "Set ORNL_STRAIN_JSON_PATH for a local file path, or ORNL_STRAIN_JSON_URL / "
        "strain_json_url for a full https://… link."
    )


# ---------------------------------------------------------------------------
# Header discovery
# ---------------------------------------------------------------------------


def _is_numeric_1d(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if isinstance(first, (list, dict)):
        return False
    return isinstance(first, (int, float)) or first is None


def _is_numeric_2d_nested(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    row0 = value[0]
    if not isinstance(row0, list):
        return False
    return all(isinstance(x, (int, float)) or x is None for x in row0)


def _to_float_array_2d(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D numeric array")
    return arr


def _to_float_array_1d(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D numeric array")
    return arr


def list_strain_field_headers(
    doc: Mapping[str, Any],
    *,
    header_regex: Optional[re.Pattern] = None,
    include_non_matching: bool = False,
) -> List[str]:
    """
    Return sorted JSON keys that look like strain *series* headers:
    - Default: ``<scan>/data/uniform_strain`` or ``unconstrained_strain`` (regex).
    - If ``include_non_matching``, also include any top-level 1D numeric array keys
      (useful while exploring new exports).
    """
    rx = header_regex or DEFAULT_HEADER_REGEX
    keys = []
    extra: List[str] = []
    for k, v in doc.items():
        if not isinstance(k, str):
            continue
        if _is_numeric_2d_nested(v):
            if rx.search(k):
                keys.append(k)
            elif include_non_matching:
                extra.append(k)
            continue
        if _is_numeric_1d(v):
            if rx.search(k):
                keys.append(k)
            elif include_non_matching:
                extra.append(k)
    keys.sort()
    extra.sort()
    return keys + (extra if include_non_matching else [])


def guess_variance_key(header: str, doc: Mapping[str, Any]) -> Optional[str]:
    """
    Map ``…/unconstrained_strain`` → ``…/unconstrained_strain_stdev`` when present.
    Uniform strain has no standard sidecar in current schema; returns None.
    """
    if header.endswith("/unconstrained_strain"):
        candidate = header + "_stdev"
        if candidate in doc:
            return candidate
    return None


def guess_gp_estimate_key(header: str, doc: Mapping[str, Any]) -> Optional[str]:
    """Optional explicit GP grid in JSON, e.g. ``0/data/uniform_strain_gp``."""
    for suffix in ("_gp_estimate", "_gp", "/gp_estimate"):
        k = f"{header}{suffix}" if suffix.startswith("_") else header + suffix
        if k in doc:
            return k
    return None


def guess_gp_variance_key(header: str, doc: Mapping[str, Any]) -> Optional[str]:
    for suffix in ("_gp_variance", "_gp_var", "/gp_variance"):
        k = header + suffix
        if k in doc:
            return k
    return None


# ---------------------------------------------------------------------------
# Grid construction (1D sparse → 2D, or direct 2D)
# ---------------------------------------------------------------------------


def _norm_positions_to_grid(
    labx: Sequence[Number],
    labz: Sequence[Number],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lx = np.asarray(labx, dtype=np.float64)
    lz = np.asarray(labz, dtype=np.float64)
    if lx.shape != lz.shape:
        raise ValueError("labx and labz must have the same length")
    if lx.size == 0:
        return np.zeros(0), np.zeros(0)

    def scale_axis(a: np.ndarray, n: int) -> np.ndarray:
        amin, amax = np.nanmin(a), np.nanmax(a)
        if not math.isfinite(amin) or not math.isfinite(amax) or amax == amin:
            return np.full_like(a, (n - 1) / 2.0)
        return (a - amin) / (amax - amin) * (n - 1)

    gx = scale_axis(lx, nx)
    gy = scale_axis(lz, ny)
    return gx, gy


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
    header: str,
    cfg: StrainFieldPlotConfig,
) -> StrainFieldGrids:
    """
    Build (measurements mask, GP estimate, GP variance) for ``header``.

    Resolution order:
    1. If optional keys ``*_gp_estimate`` / ``*_gp_variance`` exist as 2D arrays, use them.
    2. If ``header`` value is already 2D, use as estimate; variance from optional key or placeholder.
    3. If 1D: use ``labx`` / ``labz`` to place sparse samples, IDW interpolation for estimate,
       variance from ``unconstrained_strain_stdev`` when available else distance placeholder.
    """
    nx, ny = cfg.grid_size[0], cfg.grid_size[1]
    raw = doc.get(header)
    if raw is None:
        raise KeyError(f"Missing JSON key: {header}")

    meta: Dict[str, Any] = {"header": header, "mode": "unknown"}

    gp_e_key = guess_gp_estimate_key(header, doc)
    gp_v_key = guess_gp_variance_key(header, doc)
    if gp_e_key and gp_v_key:
        est = _to_float_array_2d(doc[gp_e_key])
        var = _to_float_array_2d(doc[gp_v_key])
        meas = np.isfinite(est).astype(np.float64)
        meta["mode"] = "explicit_gp_keys"
        meta["gp_estimate_key"] = gp_e_key
        meta["gp_variance_key"] = gp_v_key
        return StrainFieldGrids(meas, est, var, meta)

    if _is_numeric_2d_nested(raw):
        est = _to_float_array_2d(raw)
        meas = np.isfinite(est).astype(np.float64)
        if gp_v_key:
            var = _to_float_array_2d(doc[gp_v_key])
        else:
            var = _distance_weighted_variance_placeholder(meas, float(np.nanstd(est) or 1.0))
        meta["mode"] = "dense_header"
        return StrainFieldGrids(meas, est, var, meta)

    if not _is_numeric_1d(raw):
        raise TypeError(f"Unsupported value type for {header!r}")

    values = _to_float_array_1d(raw)
    labx = doc.get(cfg.labx_key, [])
    labz = doc.get(cfg.labz_key, [])
    if not isinstance(labx, list) or not isinstance(labz, list):
        raise ValueError(f"Need numeric lists {cfg.labx_key!r} and {cfg.labz_key!r} for sparse mode")

    gx, gy = _norm_positions_to_grid(labx, labz, nx, ny)
    m = min(int(gx.shape[0]), int(gy.shape[0]), int(values.shape[0]))
    gx, gy, values = gx[:m], gy[:m], values[:m]
    mask = np.zeros((ny, nx), dtype=np.float64)
    for x, y in zip(gx, gy):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ix = int(np.clip(round(float(x)), 0, nx - 1))
        iy = int(np.clip(round(float(y)), 0, ny - 1))
        mask[iy, ix] = 1.0

    est = _idw_fill_grid(gx, gy, values, nx, ny)
    v_key = guess_variance_key(header, doc)
    if v_key and v_key in doc and _is_numeric_1d(doc[v_key]):
        vvals = _to_float_array_1d(doc[v_key])[:m]
        if vvals.shape[0] == values.shape[0]:
            var = _idw_fill_grid(gx, gy, vvals, nx, ny)
            var = np.square(np.maximum(var, 0.0))
            meta["variance_key"] = v_key
        else:
            var = _distance_weighted_variance_placeholder(mask, float(np.nanmean(np.abs(values)) or 1e-6))
    else:
        scale = float(np.nanmean(np.abs(values)) or 1e-6) * 0.25
        var = _distance_weighted_variance_placeholder(mask, scale)

    meta["mode"] = "sparse_idw"
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


def default_row_headers(n_rows: int) -> List[str]:
    base = list(DEFAULT_ROW_HEADERS)
    if n_rows <= len(base):
        return base[:n_rows]
    out = base[:]
    last = base[-1] if base else "0/data/uniform_strain"
    while len(out) < n_rows:
        out.append(last)
    return out
