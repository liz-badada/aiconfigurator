# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
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
