# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiconfigurator.sdk.afd_moe_profile import AFDMoEStageKey, AFDMoEStageProfile


def _entry(**overrides):
    value = {
        "model_path": "Example/Model",
        "model_profile": "example_fp4",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 96,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 60,
        "moe_precision": "fp4",
        "latency_ms": 4.25,
        "validation": {
            "stable": True,
            "correctness": True,
            "matched_speedup": 1.4,
            "evidence": "same-point-colocated",
        },
        "source": {
            "commit": "abc123",
            "source_tree_sha256": "deadbeef",
            "result": "/measurements/example.json",
        },
    }
    value.update(overrides)
    return value


def _write(tmp_path, entries):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aic.afd-moe-stage-profile.v1",
                "lookup_policy": "exact-only",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def _key(**overrides):
    values = {
        "model_path": "Example/Model",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 96,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 60,
        "moe_precision": "fp4",
    }
    values.update(overrides)
    return AFDMoEStageKey(**values)


def test_profile_uses_exact_full_key(tmp_path):
    profile = AFDMoEStageProfile.load(_write(tmp_path, [_entry()]))

    assert profile.require(_key()).latency_ms == pytest.approx(4.25)
    assert profile.find(_key(system="b200_sxm")) is None
    assert profile.find(_key(logical_batch_per_source_rank=48)) is None
    assert profile.find(_key(mtp_nextn=1)) is None


def test_profile_accepts_stable_afd_entry_without_colocated_fields(tmp_path):
    afd_entry = _entry(
        stage="afd",
        topology="16A8F",
        microbatches=2,
        validation={
            "stable": True,
            "correctness": None,
            "matched_speedup": None,
            "evidence": "same-model-system-precision-colocated-plus-stable-split",
        },
    )
    profile = AFDMoEStageProfile.load(_write(tmp_path, [afd_entry]))

    measurement = profile.require(_key(stage="afd", topology="16A8F", microbatches=2))
    assert measurement.correctness is None


@pytest.mark.parametrize(
    "entry,match",
    [
        (
            _entry(
                validation={
                    "stable": False,
                    "correctness": True,
                    "matched_speedup": 1.4,
                    "evidence": "same-point-colocated",
                }
            ),
            "stable",
        ),
        (
            _entry(
                validation={
                    "stable": True,
                    "correctness": False,
                    "matched_speedup": 1.4,
                    "evidence": "same-point-colocated",
                }
            ),
            "correctness",
        ),
        (
            _entry(
                validation={
                    "stable": True,
                    "correctness": True,
                    "matched_speedup": 0.9,
                    "evidence": "same-point-colocated",
                }
            ),
            "matched_speedup",
        ),
        (_entry(latency_ms=float("nan")), "latency_ms"),
    ],
)
def test_profile_rejects_unvalidated_measurements(tmp_path, entry, match):
    with pytest.raises(ValueError, match=match):
        AFDMoEStageProfile.load(_write(tmp_path, [entry]))


def test_profile_rejects_duplicate_exact_key(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        AFDMoEStageProfile.load(_write(tmp_path, [_entry(), _entry(latency_ms=5.0)]))
