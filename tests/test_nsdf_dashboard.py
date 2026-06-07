#!/usr/bin/env python3
"""Lightweight smoke checks for native NSDF dashboard helpers."""
from __future__ import annotations

import math
import json
import os
import tempfile

import numpy as np
from fastapi import HTTPException

import nsdf_dashboard.ornl_chess_strain_lib as lib
from nsdf_dashboard import refresh_api, refresh_bus
from nsdf_dashboard.ornl_chess_strain_lib import (
    StrainDashboardPaths,
    StrainFieldPlotConfig,
    apply_nsdf_version_suffix,
    build_strain_field_grids,
    discover_nsdf_version_options,
    enrich_strain_paths_from_dataset_doc,
    infer_nsdf_bounds_grid_size,
    infer_nsdf_grid_size,
    infer_grid_size_from_dim,
    parse_nsdf_plot_dim,
    format_nsdf_workflow_display,
    next_x_grid_coords_for_workflow,
    _grid_display_coords,
    resolve_nsdf_workflow_id,
    list_nsdf_field_headers,
    list_nsdf_version_suffixes_from_directory,
    load_simple_env_file,
    load_json_from_url,
    load_nsdf_json_bundle,
    make_strain_triplet_figures,
    normalize_nsdf_gateway_data_url,
    normalize_nsdf_remote_data_link,
    nsdf_triplet_basenames,
    parse_nsdf_data_filename,
    resolve_nsdf_grid_size,
    surrogate_doc_defines_grid_size,
    validate_nsdf_measurement_doc,
    validate_nsdf_next_x_doc,
    validate_nsdf_surrogate_doc,
)


def _base_data() -> dict:
    return {
        "dataset_x": [[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]],
        "dataset_y": [1.0, 2.0, 3.0],
        "bounds": [[0.0, 10.0], [0.0, 10.0]],
        "backend": "sklearn",
        "kernel": "rbf",
    }


def _assert_mask_at(mask: np.ndarray, x: int, y: int) -> None:
    assert mask[y, x] == 1.0, f"expected measurement mask at ({x}, {y})"


def test_empty_measurement_doc_allows_pre_capture_view() -> None:
    data = {
        "dataset_x": [],
        "dataset_y": [],
        "bounds": [[0, 24], [0, 24]],
    }
    surrogate = {
        "workflow_id": "wf-pre",
        "dim": "2D",
        "bounds": [[0, 5], [0, 5]],
        "surrogate": [float(i) for i in range(25)],
        "uncertainty": [0.1 for _ in range(25)],
    }
    next_x = [
        {"workflow_id": "wf-pre", "data": [[12.0, 12.0], [18.0, 18.0]]},
    ]
    measurement = validate_nsdf_measurement_doc(data)
    assert measurement.observed_values.shape == (0,)
    assert resolve_nsdf_grid_size(data, surrogate_doc=surrogate) == ((5, 5), "surrogate bounds")
    cfg = StrainFieldPlotConfig()
    cfg.grid_size = (5, 5)
    grids = build_strain_field_grids(data, cfg, surrogate)
    assert grids.meta["estimate_source"] == "surrogate_grid"
    assert grids.meta["n_points"] == 0
    info = validate_nsdf_next_x_doc(next_x)
    p0, _, _ = make_strain_triplet_figures(
        grids,
        cfg,
        next_x_info=info,
        active_workflow_id="wf-pre",
    )
    assert p0 is not None


def test_valid_without_surrogate() -> None:
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    grids = build_strain_field_grids(_base_data(), cfg)
    assert grids.meta["estimate_source"] == "dataset_y_idw"
    assert grids.meta["variance_source"] == "distance_placeholder"
    assert grids.measurements.shape == (11, 11)
    _assert_mask_at(grids.measurements, 0, 0)
    _assert_mask_at(grids.measurements, 10, 10)
    _assert_mask_at(grids.measurements, 5, 5)


def test_valid_with_matching_surrogate() -> None:
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    surrogate = {
        "dim": "2D",
        "bounds": [[0, 11], [0, 11]],
        "surrogate": [float(i) for i in range(121)],
        "uncertainty": [0.1 for _ in range(121)],
        "raw_uncertainty": [0.01 for _ in range(121)],
    }
    grids = build_strain_field_grids(_base_data(), cfg, surrogate)
    assert grids.meta["estimate_source"] == "surrogate_grid"
    assert grids.meta["variance_source"] == "uncertainty_squared_grid"
    assert list_nsdf_field_headers(_base_data(), surrogate) == [
        "dataset_y",
        "surrogate",
        "uncertainty",
        "raw_uncertainty",
    ]
    assert math.isclose(float(grids.variance[0, 0]), 0.01, rel_tol=1e-6, abs_tol=1e-6)


def test_length_mismatch_raises_clear_error() -> None:
    bad = {"dataset_x": [[0.0, 0.0], [1.0, 1.0]], "dataset_y": [1.0]}
    try:
        validate_nsdf_measurement_doc(bad)
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("expected dataset_x/dataset_y length mismatch to raise")


def test_bounds_are_used_for_normalization() -> None:
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    data = {
        "dataset_x": [[5.0, 5.0], [10.0, 10.0]],
        "dataset_y": [1.0, 2.0],
        "bounds": [[0.0, 10.0], [0.0, 10.0]],
    }
    grids = build_strain_field_grids(data, cfg)
    assert grids.meta["bounds_source"] == "bounds"
    _assert_mask_at(grids.measurements, 5, 5)
    _assert_mask_at(grids.measurements, 10, 10)


def test_bounds_fallback_to_observed_minmax() -> None:
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    data = {
        "dataset_x": [[5.0, 5.0], [10.0, 10.0]],
        "dataset_y": [1.0, 2.0],
        "bounds": [["bad", 10.0], [0.0, 10.0]],
    }
    grids = build_strain_field_grids(data, cfg)
    assert grids.meta["bounds_source"] == "observed_minmax"
    _assert_mask_at(grids.measurements, 0, 0)
    _assert_mask_at(grids.measurements, 10, 10)


def test_grid_size_is_inferred_from_unique_coordinates() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(4) for x in range(7)],
        "dataset_y": [float(i) for i in range(28)],
    }
    assert infer_nsdf_grid_size(data) == (7, 4)


def test_bounds_grid_size_is_explicit_when_bounds_start_at_zero() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(2) for x in range(3)],
        "dataset_y": [float(i) for i in range(6)],
        "bounds": [[0, 21], [0, 13]],
    }
    assert infer_nsdf_bounds_grid_size(data) == (21, 13)
    assert resolve_nsdf_grid_size(data, env_grid_size=(26, 26), manual_grid_size=(9, 9)) == (
        (9, 9),
        "manual controls",
    )


def test_bounds_grid_size_ignores_physical_ranges_not_starting_at_zero() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(13) for x in range(21)],
        "dataset_y": [float(i) for i in range(273)],
        "bounds": [[-25.0, 25.0], [2.5, 32.5]],
    }
    assert infer_nsdf_bounds_grid_size(data) is None
    assert resolve_nsdf_grid_size(data, env_grid_size=(26, 26)) == ((26, 26), "environment")


def test_manual_grid_size_overrides_env_without_bounds() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(4) for x in range(7)],
        "dataset_y": [float(i) for i in range(28)],
    }
    assert resolve_nsdf_grid_size(data, env_grid_size=(26, 26), manual_grid_size=(9, 8)) == (
        (9, 8),
        "manual controls",
    )


def test_reset_grid_size_returns_to_auto_source() -> None:
    data_with_bounds = {
        "dataset_x": [[float(x), float(z)] for z in range(2) for x in range(3)],
        "dataset_y": [float(i) for i in range(6)],
        "bounds": [[0, 21], [0, 13]],
    }
    data_without_bounds = {
        "dataset_x": [[float(x), float(z)] for z in range(4) for x in range(7)],
        "dataset_y": [float(i) for i in range(28)],
    }
    assert resolve_nsdf_grid_size(data_with_bounds, env_grid_size=(26, 26), manual_grid_size=None) == (
        (21, 13),
        "bounds",
    )
    assert resolve_nsdf_grid_size(data_without_bounds, env_grid_size=(26, 26), manual_grid_size=None) == (
        (26, 26),
        "environment",
    )


def test_refresh_bounds_replaces_manual_grid_size() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(2) for x in range(3)],
        "dataset_y": [float(i) for i in range(6)],
        "bounds": [[0, 21], [0, 13]],
    }
    assert resolve_nsdf_grid_size(data, env_grid_size=(26, 26), manual_grid_size=(5, 5)) == (
        (5, 5),
        "manual controls",
    )


def test_surrogate_bounds_overrides_sparse_data_inference() -> None:
    data = {
        "dataset_x": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        "dataset_y": [1.0, 2.0, 3.0],
    }
    surrogate = {
        "workflow_id": "abc123",
        "dim": "2D",
        "bounds": [[0, 24], [0, 24]],
    }
    assert resolve_nsdf_grid_size(data, surrogate_doc=surrogate) == ((24, 24), "surrogate bounds")
    assert surrogate_doc_defines_grid_size(surrogate) is True


def test_legacy_surrogate_dim_array_still_resolves_grid_size() -> None:
    data = {
        "dataset_x": [[0.0, 0.0], [1.0, 1.0]],
        "dataset_y": [1.0, 2.0],
    }
    surrogate = {"workflow_id": "abc123", "dim": [24, 24]}
    assert resolve_nsdf_grid_size(data, surrogate_doc=surrogate) == (
        (24, 24),
        "surrogate dim (deprecated)",
    )
    info = validate_nsdf_surrogate_doc(surrogate)
    assert info.plot_dim is None
    assert any("deprecated" in warning for warning in info.warnings)


def test_surrogate_bounds_grid_size() -> None:
    data = {
        "dataset_x": [[0.0, 0.0], [1.0, 1.0]],
        "dataset_y": [1.0, 2.0],
    }
    surrogate = {"bounds": [[0, 21], [0, 13]], "points": 273}
    assert resolve_nsdf_grid_size(data, surrogate_doc=surrogate) == ((21, 13), "surrogate bounds")
    info = validate_nsdf_surrogate_doc(surrogate)
    assert info.points == 273


def test_surrogate_workflow_id_display() -> None:
    surrogate = validate_nsdf_surrogate_doc(
        {"workflow_id": "6a24964fbc355556959d697f", "surrogate": [1.0, 2.0]},
    )
    next_x = validate_nsdf_next_x_doc(
        [
            {"workflow_id": "dashboard-demo", "data": [[0.0, 0.0], [1.0, 1.0]]},
            {"workflow_id": "other-id", "data": [[2.0, 2.0]]},
        ],
    )
    assert resolve_nsdf_workflow_id(surrogate, next_x) == "6a24964fbc355556959d697f"
    display = format_nsdf_workflow_display(surrogate, next_x)
    assert display == "Workflow ID: 6a24964fbc355556959d697f"
    assert "other-id" not in display
    assert "dashboard-demo" not in display


def test_workflow_id_falls_back_to_next_x_when_surrogate_missing() -> None:
    surrogate = validate_nsdf_surrogate_doc({"surrogate": [1.0, 2.0]})
    next_x = validate_nsdf_next_x_doc(
        [
            {"workflow_id": "dashboard-demo", "data": [[0.0, 0.0]]},
            {"workflow_id": "6a249da9bc355556959d6980", "data": [[1.0, 1.0], [2.0, 2.0]]},
        ],
    )
    assert resolve_nsdf_workflow_id(surrogate, next_x) == "6a249da9bc355556959d6980"
    assert format_nsdf_workflow_display(surrogate, next_x) == (
        "Workflow ID: 6a249da9bc355556959d6980"
    )


def test_next_x_grid_coords_for_active_workflow() -> None:
    data = {
        "dataset_x": [[0.0, 0.0], [10.0, 10.0]],
        "dataset_y": [1.0, 2.0],
        "bounds": [[0.0, 10.0], [0.0, 10.0]],
    }
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    grids = build_strain_field_grids(data, cfg)
    next_x = validate_nsdf_next_x_doc(
        [
            {"workflow_id": "dashboard-demo", "data": [[5.0, 5.0]]},
            {
                "workflow_id": "wf-active",
                "data": [[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]],
            },
        ],
    )
    gx, gy = next_x_grid_coords_for_workflow(next_x, "wf-active", 11, 11, grids.meta["measurement_bounds"])
    assert gx.shape == (3,)
    assert gy.shape == (3,)
    _assert_mask_at(grids.measurements, 0, 0)
    px, py = _grid_display_coords(gx, gy, 11, 11, flip_y=True)
    assert float(px[0]) == 0.5
    assert float(py[0]) == 10.5


def test_parse_nsdf_plot_dim() -> None:
    assert parse_nsdf_plot_dim("2D") == ("2D", None)
    assert parse_nsdf_plot_dim("3d") == ("3D", None)
    assert parse_nsdf_plot_dim("1D") == ("1D", None)
    plot_dim, warning = parse_nsdf_plot_dim([21, 13])
    assert plot_dim is None
    assert warning is not None and "deprecated" in warning
    plot_dim, warning = parse_nsdf_plot_dim("invalid")
    assert plot_dim is None
    assert warning is not None


def test_infer_grid_size_from_dim_legacy_formats() -> None:
    assert infer_grid_size_from_dim([21, 13]) == (21, 13)
    assert infer_grid_size_from_dim({"width": 21, "height": 13}) == (21, 13)
    assert infer_grid_size_from_dim({"nx": 21, "ny": 13}) == (21, 13)
    assert infer_grid_size_from_dim("2D") is None


def test_surrogate_model_grid_independent_of_measurement_count() -> None:
    data = {
        "dataset_x": [[float(x), float(z)] for z in range(3) for x in range(4)],
        "dataset_y": [float(i) for i in range(12)],
        "bounds": [[0, 4], [0, 3]],
    }
    surrogate = {
        "workflow_id": "abc123",
        "dim": "2D",
        "bounds": [[0, 5], [0, 5]],
        "surrogate": [float(i) for i in range(25)],
        "uncertainty": [0.1 for _ in range(25)],
    }
    info = validate_nsdf_surrogate_doc(surrogate)
    assert info.surrogate is not None
    assert info.surrogate.shape[0] == 25
    cfg = StrainFieldPlotConfig()
    cfg.grid_size = resolve_nsdf_grid_size(data, surrogate_doc=surrogate)[0]
    grids = build_strain_field_grids(data, cfg, surrogate)
    assert cfg.grid_size == (5, 5)
    assert grids.estimate.shape == (5, 5)
    assert grids.meta["estimate_source"] == "surrogate_grid"
    assert grids.meta["variance_source"] == "uncertainty_squared_grid"


def test_local_surrogate_sibling_inference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "data.json")
        surrogate_path = os.path.join(tmp, "surrogate.json")
        with open(data_path, "w", encoding="utf-8") as fh:
            json.dump(_base_data(), fh)
        with open(surrogate_path, "w", encoding="utf-8") as fh:
            json.dump({"surrogate": [10.0, 20.0, 30.0]}, fh)

        bundle = load_nsdf_json_bundle(StrainDashboardPaths(local_json_path=data_path))
        assert bundle.surrogate is not None
        assert bundle.paths.surrogate_json_path == surrogate_path


def test_local_next_x_sibling_inference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "data.json")
        next_x_path = os.path.join(tmp, "next_x.json")
        with open(data_path, "w", encoding="utf-8") as fh:
            json.dump(_base_data(), fh)
        with open(next_x_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"workflow_id": "test-workflow", "data": [[4.0, 5.0], [6.0, 7.0]]},
                    {"workflow_id": "0001", "data": [[6.0, 7.0]]},
                ],
                fh,
            )

        bundle = load_nsdf_json_bundle(StrainDashboardPaths(local_json_path=data_path))
        assert bundle.next_x is not None
        assert bundle.paths.next_x_json_path == next_x_path
        info = validate_nsdf_next_x_doc(bundle.next_x)
        assert len(info.entries) == 2
        assert info.total_points == 3
        assert info.entries[0].workflow_id == "test-workflow"


def test_next_x_validation_skips_bad_rows() -> None:
    info = validate_nsdf_next_x_doc(
        [
            {"workflow_id": "ok", "data": [[1.0, 2.0]]},
            {"workflow_id": "", "data": [[3.0, 4.0]]},
            {"workflow_id": "bad", "data": [[5.0]]},
        ]
    )
    assert len(info.entries) == 1
    assert info.entries[0].workflow_id == "ok"
    assert info.warnings


def test_normalize_gateway_prefix_to_data_json() -> None:
    base = (
        "https://us-east-1.gw.example.com/scientistcloud/test-chess"
        "?access_key=AK&secret_key=SK"
    )
    normalized = normalize_nsdf_gateway_data_url(base)
    assert normalized.endswith("/test-chess/data.json?access_key=AK&secret_key=SK")
    assert normalize_nsdf_remote_data_link("s3://scientistcloud/test-chess/") == (
        "s3://scientistcloud/test-chess/data.json"
    )
    explicit = normalize_nsdf_gateway_data_url(
        "https://gw.example.com/bucket/prefix/data.json?access_key=AK&secret_key=SK"
    )
    assert "/prefix/data.json?" in explicit


def test_nsdf_version_suffix_triplet_and_apply() -> None:
    assert nsdf_triplet_basenames("") == ("data.json", "surrogate.json", "next_x.json")
    assert nsdf_triplet_basenames("20260606T223505Z") == (
        "data_20260606T223505Z.json",
        "surrogate_20260606T223505Z.json",
        "next_x_20260606T223505Z.json",
    )
    assert parse_nsdf_data_filename("data.json") is None
    assert parse_nsdf_data_filename("data_20260606T223505Z.json") == "20260606T223505Z"

    base = StrainDashboardPaths(
        local_data_dir="/tmp/chess-data",
    )
    versioned = apply_nsdf_version_suffix(base, "20260606T223505Z")
    assert versioned.local_json_path.endswith("data_20260606T223505Z.json")
    assert versioned.surrogate_json_path.endswith("surrogate_20260606T223505Z.json")
    assert versioned.next_x_json_path.endswith("next_x_20260606T223505Z.json")

    remote = StrainDashboardPaths(
        json_url="https://gw.example.com/bucket/chess-data/data.json?access_key=AK&secret_key=SK",
        s3_bucket="bucket",
        s3_data_key="chess-data/data.json",
    )
    versioned_remote = apply_nsdf_version_suffix(remote, "20260606T223505Z")
    assert "data_20260606T223505Z.json" in versioned_remote.json_url
    assert versioned_remote.s3_data_key.endswith("data_20260606T223505Z.json")
    assert versioned_remote.s3_surrogate_key.endswith("surrogate_20260606T223505Z.json")


def test_list_nsdf_version_suffixes_from_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "data.json"), "w", encoding="utf-8").close()
        open(os.path.join(tmp, "data_20260606T215241Z.json"), "w", encoding="utf-8").close()
        open(os.path.join(tmp, "data_20260607T023932Z.json"), "w", encoding="utf-8").close()
        suffixes = list_nsdf_version_suffixes_from_directory(tmp)
        assert suffixes == ["20260607T023932Z", "20260606T215241Z"]
        options = discover_nsdf_version_options(StrainDashboardPaths(local_data_dir=tmp))
        assert options[0] == ("latest", "Latest (data.json)")
        assert ("20260607T023932Z", "20260607T023932Z") in options


def test_enrich_directory_link_resolves_s3_prefix_for_direct_read() -> None:
    doc = {
        "google_drive_link": (
            "https://us-east-1.gw.example.com/scientistcloud/test-chess"
            "?access_key=AK&secret_key=SK"
        ),
        "s3_access_key_id": "AK",
        "s3_secret_access_key": "SK",
        "s3_endpoint_url": "https://us-east-1.gw.example.com",
    }
    paths = enrich_strain_paths_from_dataset_doc(StrainDashboardPaths(), doc)
    assert "/test-chess/data.json" in paths.json_url
    assert "access_key=AK" in paths.json_url


def test_load_json_from_url_reads_prefix_via_data_json_key() -> None:
    payload = _base_data()
    requested: list[str] = []

    class FakeBody:
        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    class FakeClient:
        def get_object(self, *, Bucket: str, Key: str) -> dict:
            requested.append(Key)
            assert Bucket == "scientistcloud"
            assert Key == "test-chess/data.json"
            return {"Body": FakeBody()}

    old = lib._load_json_via_s3_query_client
    try:
        lib._load_json_via_s3_query_client = lambda cfg: (
            FakeClient().get_object(Bucket=cfg["bucket"], Key=cfg["key"]) and payload
        )

        def _fake_load(cfg: dict) -> dict:
            requested.append(cfg["key"])
            assert cfg["bucket"] == "scientistcloud"
            assert cfg["key"] == "test-chess/data.json"
            return payload

        lib._load_json_via_s3_query_client = _fake_load
        url = (
            "https://us-east-1.gw.example.com/scientistcloud/test-chess"
            "?access_key=AK&secret_key=SK"
        )
        doc = load_json_from_url(url)
        assert doc["dataset_y"] == payload["dataset_y"]
        assert requested == ["test-chess/data.json"]
    finally:
        lib._load_json_via_s3_query_client = old


def test_env_file_parser_and_s3_config_detection() -> None:
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, "s3.env")
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "\n".join(
                        [
                            "# ignored",
                            "AWS_ACCESS_KEY_ID=file-ak",
                            "AWS_SECRET_ACCESS_KEY='file-sk'",
                            "S3_BUCKET=my-bucket",
                            'S3_DATA_KEY="prefix/data.json"',
                            "S3_REGION=us-west-2",
                            "LOCAL_DATA_DIR=/tmp/nsdf-local",
                        ]
                    )
                )
            parsed = load_simple_env_file(env_path)
            assert parsed["AWS_SECRET_ACCESS_KEY"] == "file-sk"
            assert parsed["S3_DATA_KEY"] == "prefix/data.json"

            os.environ.clear()
            os.environ["ORNL_S3_ENV_FILE"] = env_path
            paths = StrainDashboardPaths.from_environ()
            assert paths.has_s3_source()
            assert paths.s3_bucket == "my-bucket"
            assert paths.s3_data_key == "prefix/data.json"
            assert paths.s3_region == "us-west-2"
            assert paths.local_data_dir == "/tmp/nsdf-local"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


class _FakeBody:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeS3Client:
    def __init__(self):
        self.keys_requested: list[str] = []
        self.listed_suffixes: list[str] = []

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        self.keys_requested.append(Key)
        if Bucket != "my-bucket":
            raise AssertionError(f"unexpected bucket {Bucket}")
        if Key == "prefix/data.json":
            return {"Body": _FakeBody(_base_data())}
        if Key == "prefix/surrogate.json":
            return {"Body": _FakeBody({"surrogate": [10.0, 20.0, 30.0]})}
        if Key == "prefix/next_x.json":
            return {"Body": _FakeBody([])}
        raise AssertionError(f"unexpected key {Key}")

    def list_objects_v2(self, **kwargs) -> dict:  # noqa: N803
        contents = [
            {"Key": f"prefix/data_{suffix}.json"}
            for suffix in self.listed_suffixes
        ]
        return {"Contents": contents, "IsTruncated": False}


def test_s3_bundle_loads_and_infers_surrogate_key() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths, mongo_s3_auth=None: fake_client
        paths = StrainDashboardPaths(
            s3_bucket="my-bucket",
            s3_data_key="prefix/data.json",
        )
        bundle = load_nsdf_json_bundle(paths)
        assert bundle.data["dataset_y"] == [1.0, 2.0, 3.0]
        assert bundle.surrogate == {"surrogate": [10.0, 20.0, 30.0]}
        assert bundle.paths.s3_surrogate_key == "prefix/surrogate.json"
        assert fake_client.keys_requested == [
            "prefix/data.json",
            "prefix/surrogate.json",
            "prefix/next_x.json",
        ]
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_local_data_dir_takes_priority_over_s3() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths, mongo_s3_auth=None: fake_client
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "data.json"), "w", encoding="utf-8") as fh:
                json.dump(_base_data(), fh)
            paths = StrainDashboardPaths(
                local_data_dir=tmp,
                s3_bucket="my-bucket",
                s3_data_key="prefix/data.json",
            )
            bundle = load_nsdf_json_bundle(paths)
            assert bundle.data["dataset_y"] == [1.0, 2.0, 3.0]
            assert bundle.surrogate is None
            assert bundle.paths.local_json_path == os.path.join(tmp, "data.json")
            assert bundle.paths.surrogate_json_path == ""
            assert fake_client.keys_requested == []
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_local_data_dir_loads_local_surrogate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "data.json")
        surrogate_path = os.path.join(tmp, "surrogate.json")
        with open(data_path, "w", encoding="utf-8") as fh:
            json.dump(_base_data(), fh)
        with open(surrogate_path, "w", encoding="utf-8") as fh:
            json.dump({"surrogate": [10.0, 20.0, 30.0]}, fh)
        bundle = load_nsdf_json_bundle(StrainDashboardPaths(local_data_dir=tmp))
        assert bundle.data["dataset_y"] == [1.0, 2.0, 3.0]
        assert bundle.surrogate == {"surrogate": [10.0, 20.0, 30.0]}
        assert bundle.paths.local_json_path == data_path
        assert bundle.paths.surrogate_json_path == surrogate_path


def test_missing_local_data_dir_does_not_fall_back_to_s3() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths, mongo_s3_auth=None: fake_client
        with tempfile.TemporaryDirectory() as tmp:
            paths = StrainDashboardPaths(
                local_data_dir=tmp,
                s3_bucket="my-bucket",
                s3_data_key="prefix/data.json",
            )
            try:
                load_nsdf_json_bundle(paths)
            except FileNotFoundError as exc:
                assert "Local NSDF file not found" in str(exc)
            else:
                raise AssertionError("expected missing local data.json to raise FileNotFoundError")
            assert fake_client.keys_requested == []
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_local_snapshot_discovery_excludes_s3() -> None:
    fake_client = _FakeS3Client()
    fake_client.listed_suffixes = ["20990101T000000Z"]
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths, mongo_s3_auth=None: fake_client
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "data.json"), "w", encoding="utf-8").close()
            open(os.path.join(tmp, "data_20260606T215241Z.json"), "w", encoding="utf-8").close()
            paths = StrainDashboardPaths(
                local_data_dir=tmp,
                s3_bucket="my-bucket",
                s3_data_key="prefix/data.json",
            )
            options = discover_nsdf_version_options(paths)
            values = [value for value, _label in options]
            assert "20260606T215241Z" in values
            assert "20990101T000000Z" not in values
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_s3_snapshot_discovery_uses_remote_prefix_only() -> None:
    fake_client = _FakeS3Client()
    fake_client.listed_suffixes = ["20260606T215241Z", "20260607T023932Z"]
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths, mongo_s3_auth=None: fake_client
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "data_20990101T000000Z.json"), "w", encoding="utf-8").close()
            paths = StrainDashboardPaths(
                s3_bucket="my-bucket",
                s3_data_key="prefix/data.json",
            )
            options = discover_nsdf_version_options(paths)
            values = [value for value, _label in options]
            assert "20260606T215241Z" in values
            assert "20260607T023932Z" in values
            assert "20990101T000000Z" not in values
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_refresh_api_key_loads_from_env_file() -> None:
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, "s3.env")
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.write("REFRESH_API_KEY=from-file\n")

            os.environ.clear()
            os.environ["ORNL_S3_ENV_FILE"] = env_path
            assert refresh_api.expected_api_key() == "from-file"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_refresh_bus_register_trigger_unregister() -> None:
    refresh_bus.clear_refresh_callbacks()
    calls: list[str] = []
    token = refresh_bus.register_refresh_callback(lambda: calls.append("called"))
    assert refresh_bus.registered_count() == 1
    assert refresh_bus.trigger_refresh() == 1
    assert calls == ["called"]
    refresh_bus.unregister_refresh_callback(token)
    assert refresh_bus.registered_count() == 0
    assert refresh_bus.trigger_refresh() == 0


def test_refresh_api_rejects_missing_or_bad_api_key() -> None:
    old_env = dict(os.environ)
    refresh_bus.clear_refresh_callbacks()
    try:
        os.environ.pop("ORNL_S3_ENV_FILE", None)
        os.environ["REFRESH_API_KEY"] = "secret"
        endpoint = next(
            route.endpoint
            for route in refresh_api.create_app().routes
            if getattr(route, "path", "") == "/refresh"
        )
        for api_key in ("", "wrong"):
            try:
                endpoint(x_api_key=api_key)
            except HTTPException as exc:
                assert exc.status_code == 401
            else:
                raise AssertionError("expected unauthorized refresh request to raise")
    finally:
        refresh_bus.clear_refresh_callbacks()
        os.environ.clear()
        os.environ.update(old_env)


def test_refresh_api_authorized_trigger() -> None:
    old_env = dict(os.environ)
    refresh_bus.clear_refresh_callbacks()
    calls: list[str] = []
    try:
        os.environ.pop("ORNL_S3_ENV_FILE", None)
        os.environ["REFRESH_API_KEY"] = "secret"
        refresh_bus.register_refresh_callback(lambda: calls.append("refresh"))
        endpoint = next(
            route.endpoint
            for route in refresh_api.create_app().routes
            if getattr(route, "path", "") == "/refresh"
        )
        response = endpoint(x_api_key="secret")
        assert response == {"ok": True, "triggered": 1}
        assert calls == ["refresh"]
    finally:
        refresh_bus.clear_refresh_callbacks()
        os.environ.clear()
        os.environ.update(old_env)


def main() -> None:
    tests = [
        test_valid_without_surrogate,
        test_valid_with_matching_surrogate,
        test_length_mismatch_raises_clear_error,
        test_bounds_are_used_for_normalization,
        test_bounds_fallback_to_observed_minmax,
        test_grid_size_is_inferred_from_unique_coordinates,
        test_bounds_grid_size_is_explicit_when_bounds_start_at_zero,
        test_bounds_grid_size_ignores_physical_ranges_not_starting_at_zero,
        test_manual_grid_size_overrides_env_without_bounds,
        test_reset_grid_size_returns_to_auto_source,
        test_refresh_bounds_replaces_manual_grid_size,
        test_local_surrogate_sibling_inference,
        test_local_next_x_sibling_inference,
        test_next_x_validation_skips_bad_rows,
        test_normalize_gateway_prefix_to_data_json,
        test_nsdf_version_suffix_triplet_and_apply,
        test_list_nsdf_version_suffixes_from_directory,
        test_enrich_directory_link_resolves_s3_prefix_for_direct_read,
        test_load_json_from_url_reads_prefix_via_data_json_key,
        test_env_file_parser_and_s3_config_detection,
        test_s3_bundle_loads_and_infers_surrogate_key,
        test_local_data_dir_takes_priority_over_s3,
        test_local_data_dir_loads_local_surrogate,
        test_missing_local_data_dir_does_not_fall_back_to_s3,
        test_local_snapshot_discovery_excludes_s3,
        test_s3_snapshot_discovery_uses_remote_prefix_only,
        test_refresh_api_key_loads_from_env_file,
        test_refresh_bus_register_trigger_unregister,
        test_refresh_api_rejects_missing_or_bad_api_key,
        test_refresh_api_authorized_trigger,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
