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
    build_strain_field_grids,
    infer_nsdf_bounds_grid_size,
    infer_nsdf_grid_size,
    list_nsdf_field_headers,
    load_simple_env_file,
    load_nsdf_json_bundle,
    resolve_nsdf_grid_size,
    validate_nsdf_measurement_doc,
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
        "surrogate": [10.0, 20.0, 30.0],
        "uncertainty": [0.1, 0.2, 0.3],
        "raw_uncertainty": [0.01, 0.02, 0.03],
    }
    grids = build_strain_field_grids(_base_data(), cfg, surrogate)
    assert grids.meta["estimate_source"] == "surrogate"
    assert grids.meta["variance_source"] == "uncertainty_squared"
    assert list_nsdf_field_headers(_base_data(), surrogate) == [
        "dataset_y",
        "surrogate",
        "uncertainty",
        "raw_uncertainty",
    ]
    assert math.isclose(float(grids.variance[0, 0]), 0.01, rel_tol=1e-6, abs_tol=1e-6)


def test_mismatched_surrogate_is_skipped() -> None:
    cfg = StrainFieldPlotConfig(grid_size=(11, 11))
    surrogate = {"surrogate": [10.0], "uncertainty": [0.1]}
    grids = build_strain_field_grids(_base_data(), cfg, surrogate)
    assert grids.meta["estimate_source"] == "dataset_y_idw"
    assert grids.meta["variance_source"] == "distance_placeholder"
    assert grids.meta["warnings"]
    assert list_nsdf_field_headers(_base_data(), surrogate) == ["dataset_y"]


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
        (21, 13),
        "bounds",
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
        (21, 13),
        "bounds",
    )


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
                            "ORNL_NSDF_S3_BUCKET=my-bucket",
                            'ORNL_NSDF_S3_DATA_KEY="prefix/data.json"',
                            "ORNL_NSDF_S3_REGION=us-west-2",
                            "ORNL_NSDF_LOCAL_DATA_DIR=/tmp/nsdf-local",
                        ]
                    )
                )
            parsed = load_simple_env_file(env_path)
            assert parsed["AWS_SECRET_ACCESS_KEY"] == "file-sk"
            assert parsed["ORNL_NSDF_S3_DATA_KEY"] == "prefix/data.json"

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

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        self.keys_requested.append(Key)
        if Bucket != "my-bucket":
            raise AssertionError(f"unexpected bucket {Bucket}")
        if Key == "prefix/data.json":
            return {"Body": _FakeBody(_base_data())}
        if Key == "prefix/surrogate.json":
            return {"Body": _FakeBody({"surrogate": [10.0, 20.0, 30.0]})}
        raise AssertionError(f"unexpected key {Key}")


def test_s3_bundle_loads_and_infers_surrogate_key() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths: fake_client
        paths = StrainDashboardPaths(
            s3_bucket="my-bucket",
            s3_data_key="prefix/data.json",
        )
        bundle = load_nsdf_json_bundle(paths)
        assert bundle.data["dataset_y"] == [1.0, 2.0, 3.0]
        assert bundle.surrogate == {"surrogate": [10.0, 20.0, 30.0]}
        assert bundle.paths.s3_surrogate_key == "prefix/surrogate.json"
        assert fake_client.keys_requested == ["prefix/data.json", "prefix/surrogate.json"]
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_local_data_dir_takes_priority_over_s3() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths: fake_client
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


def test_missing_local_data_dir_falls_back_to_s3() -> None:
    fake_client = _FakeS3Client()
    old_make_client = lib._make_nsdf_s3_client
    try:
        lib._make_nsdf_s3_client = lambda paths: fake_client
        with tempfile.TemporaryDirectory() as tmp:
            paths = StrainDashboardPaths(
                local_data_dir=tmp,
                s3_bucket="my-bucket",
                s3_data_key="prefix/data.json",
            )
            bundle = load_nsdf_json_bundle(paths)
            assert bundle.data["dataset_y"] == [1.0, 2.0, 3.0]
            assert bundle.surrogate == {"surrogate": [10.0, 20.0, 30.0]}
            assert fake_client.keys_requested == ["prefix/data.json", "prefix/surrogate.json"]
    finally:
        lib._make_nsdf_s3_client = old_make_client


def test_refresh_api_key_loads_from_env_file() -> None:
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = os.path.join(tmp, "s3.env")
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.write("ORNL_REFRESH_API_KEY=from-file\n")

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
        os.environ["ORNL_REFRESH_API_KEY"] = "secret"
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
        os.environ["ORNL_REFRESH_API_KEY"] = "secret"
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
        test_mismatched_surrogate_is_skipped,
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
        test_env_file_parser_and_s3_config_detection,
        test_s3_bundle_loads_and_infers_surrogate_key,
        test_local_data_dir_takes_priority_over_s3,
        test_local_data_dir_loads_local_surrogate,
        test_missing_local_data_dir_falls_back_to_s3,
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
