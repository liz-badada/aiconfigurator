# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

EXPERIMENT = Path(__file__).resolve().parents[3] / "tools" / "afd_multimodel_mtp_experiment.py"
RENDERER = Path(__file__).resolve().parents[3] / "tools" / "render_afd_multimodel_mtp_report.py"


@pytest.fixture(scope="module")
def experiment_module():
    spec = importlib.util.spec_from_file_location("afd_multimodel_mtp_experiment", EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def renderer_module():
    spec = importlib.util.spec_from_file_location("render_afd_multimodel_mtp_report", RENDERER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_afd_service_units_cover_every_node_aligned_size(experiment_module):
    assert experiment_module.afd_service_unit_grid((16, 24, 36, 48, 72)) == tuple(range(8, 73, 4))


def test_afd_service_units_stop_at_largest_selected_pool(experiment_module):
    assert experiment_module.afd_service_unit_grid((16, 24)) == (8, 12, 16, 20, 24)
    assert experiment_module.afd_service_unit_grid(()) == ()


def test_measured_only_rejects_a_profile_for_another_system(experiment_module):
    profile = SimpleNamespace(entries=(SimpleNamespace(key=SimpleNamespace(system="b200_sxm")),))

    with pytest.raises(ValueError, match=r"b200_sxm.*gb200"):
        experiment_module.require_profile_system(profile, "gb200")


def test_every_model_has_explicit_matched_backend_profiles(experiment_module):
    for model in experiment_module.MODELS:
        by_family = {
            family: experiment_module.selected_profiles(model, "all", {family})
            for family in ("megamoe", "deepep_deepgemm", "trtllm")
        }

        assert all(len(profiles) == 1 for profiles in by_family.values())
        assert by_family["megamoe"][0].primary
        assert by_family["megamoe"][0].measured_moe_backend == "megamoe"
        assert by_family["deepep_deepgemm"][0].measured_moe_backend == "deepep_deepgemm"
        assert by_family["trtllm"][0].measured_moe_backend is None


def test_primary_scope_does_not_silently_select_a_control_backend(experiment_module):
    model = experiment_module.MODEL_BY_KEY["qwen3_235b"]

    assert experiment_module.selected_profiles(model, "primary", {"trtllm"}) == ()
    selected = experiment_module.selected_profiles(model, "primary", {"megamoe"})
    assert tuple(profile.backend_family for profile in selected) == ("megamoe",)


@pytest.mark.parametrize("backend_family", ["megamoe", "deepep_deepgemm"])
def test_missing_measurement_is_labeled_as_generic_control(experiment_module, backend_family):
    spec = experiment_module.MODEL_BY_KEY["qwen3_235b"]
    requested = experiment_module.selected_profiles(spec, "all", {backend_family})[0]

    contract = experiment_module.moe_backend_contract(spec, requested, None)

    assert contract == {
        "moe_backend": "generic-trtllm",
        "moe_time_source": "aic-database",
        "moe_kernel": "sglang_mxfp4_flashinfer_trtllm_moe",
    }


def test_run_sweeps_fixed_agg_pools_and_all_fitting_afd_units(experiment_module, monkeypatch, tmp_path):
    agg_sizes = []
    afd_sizes = []

    def fake_agg(*args, **kwargs):
        del kwargs
        agg_sizes.append(args[4])
        return [], []

    def fake_afd(*args, **kwargs):
        del kwargs
        afd_sizes.append(args[4])
        return [], []

    monkeypatch.setattr(experiment_module, "agg_cluster_rows", fake_agg)
    monkeypatch.setattr(experiment_module, "afd_cluster_rows", fake_afd)
    monkeypatch.setattr(experiment_module, "git_value", lambda *args: "test")
    args = SimpleNamespace(
        models=["qwen3_235b"],
        workloads=["8k"],
        total_gpus=[16, 24],
        afd_moe_profile=None,
        profile_scope="primary",
        require_measured_moe=False,
        output=tmp_path / "sweep.json",
    )

    payload = experiment_module.run(args)

    assert agg_sizes == [16, 24, 16, 24]
    assert afd_sizes == [8, 12, 16, 20, 24] * 2
    assert payload["contract"]["afd_service_unit_gpu_grid"] == [8, 12, 16, 20, 24]


def test_renderer_labels_exact_megamoe_backend(renderer_module):
    contract = {
        "framework": "SGLang 0.5.14",
        "moe_backend": "measured-megamoe",
        "moe_kernel": "measured-profile",
        "moe_precision": "W4A8_MXFP4_MXFP8_TRTLLM",
    }

    assert renderer_module.is_megamoe_backend(contract["moe_backend"])
    assert "MegaMoE (exact measured profile)" in renderer_module.compact_backend_contract(contract)


def test_renderer_requires_exact_measured_moe_on_every_arm(renderer_module):
    measured = {
        "moe_backend": "measured-megamoe",
        "moe_time_source": "exact-measured-profile",
    }
    contracts = {arm: dict(measured) for arm in ("AGG", "AGG + AFD", "AGG + MTP", "AGG + AFD + MTP")}

    assert renderer_module.all_arms_use_exact_measured_moe(contracts)

    contracts["AGG + MTP"]["moe_time_source"] = "aic-database"
    assert not renderer_module.all_arms_use_exact_measured_moe(contracts)


def test_renderer_summarizes_external_moe_reference_without_calibrating(renderer_module, tmp_path):
    entries = []
    for backend, stage, topology, microbatches, latency in (
        ("megamoe", "agg", "ep8", 1, 5.0),
        ("megamoe", "afd", "4A4F", 2, 8.0),
        ("deepep_deepgemm", "agg", "ep8", 1, 9.0),
        ("deepep_deepgemm", "afd", "4A4F", 2, 12.0),
    ):
        entries.append(
            {
                "model_path": "Model/Test",
                "system": "b200_sxm",
                "stage": stage,
                "moe_backend": backend,
                "topology": topology,
                "logical_batch_per_source_rank": 96,
                "mtp_nextn": 3,
                "microbatches": microbatches,
                "moe_precision": "w4a8_mxfp4_mxfp8",
                "latency_ms": latency,
                "validation": {"matched_speedup": 2.0, "matched_speedup_lower_bound": 1.8},
            }
        )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": "aic.afd-moe-stage-profile.v2", "entries": entries}))

    reference = renderer_module.load_moe_reference(path, "https://example.test/reference")

    assert reference["systems"] == ["b200_sxm"]
    assert reference["entries"] == 4
    assert reference["path"] is None
    assert reference["models"]["Model/Test"]["agg_latency_ms"] == [5.0, 5.0]
    assert reference["models"]["Model/Test"]["afd_latency_ms"] == [8.0, 8.0]
    assert reference["models"]["Model/Test"]["deepep_agg_latency_ms"] == [9.0, 9.0]
    assert reference["models"]["Model/Test"]["deepep_afd_latency_ms"] == [12.0, 12.0]


def test_renderer_summarizes_mocker_accounting_without_case_artifacts(renderer_module, tmp_path):
    path = tmp_path / "mocker_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aic.afd-fixed-pool-mocker.v2",
                "dynamo_branch": "afd-moe-timing",
                "dynamo_commit": "a" * 40,
                "output_tokens_per_request": 128,
                "waves": 8,
                "results": [
                    {
                        "model": "qwen3_235b",
                        "workload": "8k",
                        "total_gpus": 72,
                        "nextn": 0,
                        "mean_tpot_error_pct": 0.0,
                        "finite_wave_efficiency_vs_aic_steady_state": 0.999,
                    },
                    {
                        "model": "qwen3_235b",
                        "workload": "8k",
                        "total_gpus": 72,
                        "nextn": 3,
                        "mean_tpot_error_pct": -0.2,
                        "finite_wave_efficiency_vs_aic_steady_state": 0.91,
                    },
                ],
            }
        )
    )

    summary = renderer_module.load_mocker_summaries([path])[0]

    assert summary["cases"] == 2
    assert summary["no_mtp_max_abs_tpot_error_pct"] == 0.0
    assert summary["mtp_max_abs_tpot_error_pct"] == 0.2
    assert summary["mtp_finite_efficiency"] == [0.91, 0.91]


def test_renderer_line_charts_use_nvidia_palette_without_point_labels(renderer_module):
    svg = renderer_module.line_svg(
        {"AGG": [(16, 1.0), (24, 1.1)], "AGG + AFD": [(16, 1.2), (24, 1.3)]},
        x_label="GPU count",
        y_label="Ratio",
        x_ticks=(16, 24),
        y_max=2.0,
    )

    assert renderer_module.COLORS["AGG + AFD"] == "#76B900"
    assert svg.count("<circle ") == 4
    assert 'class="value-label"' not in svg


def test_renderer_pareto_charts_do_not_label_points(renderer_module):
    svg = renderer_module.scatter_svg(
        {"AGG": [(10.0, 120.0)], "AGG + AFD": [(8.0, 150.0)]},
        x_max=50.0,
        y_max=200.0,
    )

    assert svg.count("<circle ") == 2
    assert 'class="value-label"' not in svg


def test_renderer_chart_data_table_lists_series_and_axis_semantics(renderer_module):
    data = renderer_module.chart_data_table(
        {"AGG": [(16.0, 1.234567890123)], "AGG + AFD": [(24.0, 2.5)]},
        x_label="Fixed GPU pool size (GPUs)",
        y_label="Output throughput (tokens/s/GPU)",
    )

    assert "<th>Series</th>" in data
    assert "<th>X — Fixed GPU pool size (GPUs)</th>" in data
    assert "<th>Y — Output throughput (tokens/s/GPU)</th>" in data
    assert "<td>AGG</td><td>16</td><td>1.23456789012</td>" in data
    assert "<td>AGG + AFD</td><td>24</td><td>2.5</td>" in data


def test_renderer_grouped_bar_charts_label_every_bar(renderer_module):
    svg = renderer_module.grouped_bar_svg(
        ["8K"],
        {"A path": [12.5], "F path": [15.0]},
        x_label="Case",
        y_label="Time (µs)",
        y_max=20.0,
    )

    assert svg.count('class="value-label"') == 2
    assert ">12.5</text>" in svg
    assert ">15.0</text>" in svg


def test_renderer_stacked_bar_charts_label_every_total(renderer_module):
    svg = renderer_module.stacked_bar_svg(
        ["8K", "16K"],
        [
            {"attention": 1.25, "MoE / shared expert": 2.75},
            {"attention": 2.0, "MoE / shared expert": 3.0},
        ],
        y_max=6.0,
    )

    assert svg.count('class="value-label"') == 2
    assert ">4.00</text>" in svg
    assert ">5.00</text>" in svg


def test_renderer_places_series_table_below_chart(renderer_module):
    data = renderer_module.chart_data_table(
        {"AGG": [(16.0, 1.25)]},
        x_label="GPUs",
        y_label="Ratio (AFD / AGG)",
    )

    rendered = renderer_module.figure("<svg></svg>", "Read me", data_table=data)

    assert rendered.index("</svg>") < rendered.index("<table>") < rendered.index("How to read")
