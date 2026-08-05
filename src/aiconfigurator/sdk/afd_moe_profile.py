# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact measured AFD MoE-stage latency profiles."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

PROFILE_SCHEMA = "aic.afd-moe-stage-profile.v3"
PROFILE_SCHEMA_V2 = "aic.afd-moe-stage-profile.v2"
LEGACY_PROFILE_SCHEMA = "aic.afd-moe-stage-profile.v1"
Stage = Literal["agg", "afd"]


@dataclass(frozen=True)
class AFDMoEStageKey:
    """Identity of one measured MoE-stage latency point."""

    model_path: str
    system: str
    stage: Stage
    topology: str
    logical_batch_per_source_rank: int
    mtp_nextn: int
    microbatches: int
    moe_layers: int
    routed_topk: int | None
    moe_precision: str
    moe_backend: str


@dataclass(frozen=True)
class AFDMoEStageMeasurement:
    """Validated latency and provenance for an exact stage key."""

    key: AFDMoEStageKey
    model_profile: str
    latency_ms: float
    evidence: str
    correctness: bool | None
    matched_speedup: float | None
    matched_speedup_lower_bound: float | None
    source_commit: str
    source_tree_sha256: str
    source_result: str


@dataclass(frozen=True)
class AFDMoEStageProjection:
    """Latency interpolated inside one measured per-F-rank load envelope."""

    key: AFDMoEStageKey
    model_profile: str
    latency_ms: float
    evidence: str
    source_system: str
    source_topology: str
    target_logical_tokens_per_f_rank_per_microbatch: float
    target_routed_assignments_per_f_rank_per_microbatch: float
    latency_scale: float
    lower_anchor_load: float
    upper_anchor_load: float
    lower_anchor: AFDMoEStageMeasurement
    upper_anchor: AFDMoEStageMeasurement

    @property
    def target_load_per_f_rank(self) -> float:
        """Backward-compatible alias for the routed-assignment load."""

        return self.target_routed_assignments_per_f_rank_per_microbatch


class AFDMoEStageProfile:
    """An immutable exact-only lookup table for measured stage latency."""

    def __init__(self, entries: tuple[AFDMoEStageMeasurement, ...], *, source: Path) -> None:
        self.entries = entries
        self.source = source
        self._by_key: dict[AFDMoEStageKey, AFDMoEStageMeasurement] = {}
        for entry in entries:
            if entry.key in self._by_key:
                raise ValueError(f"duplicate AFD MoE stage profile key: {entry.key}")
            self._by_key[entry.key] = entry

    @classmethod
    def load(cls, path: str | Path) -> AFDMoEStageProfile:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("AFD MoE stage profile root must be an object")
        schema = payload.get("schema")
        if schema not in (LEGACY_PROFILE_SCHEMA, PROFILE_SCHEMA_V2, PROFILE_SCHEMA):
            raise ValueError(f"unsupported AFD MoE stage profile schema: {payload.get('schema')!r}")
        if payload.get("lookup_policy") != "exact-only":
            raise ValueError("AFD MoE stage profile lookup_policy must be 'exact-only'")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise TypeError("AFD MoE stage profile entries must be a list")
        return cls(
            tuple(_parse_entry(value, index, schema=schema) for index, value in enumerate(raw_entries)),
            source=source,
        )

    def find(self, key: AFDMoEStageKey) -> AFDMoEStageMeasurement | None:
        """Return the exact point, without interpolation or topology conversion."""

        measurement = self._by_key.get(key)
        if measurement is not None or key.routed_topk is None:
            return measurement
        # v1/v2 profiles predate routed_topk. Their model_path still fixes the
        # routing contract, so exact lookup can safely use the legacy key. Load
        # projection below also requires the target key's explicit top-k.
        return self._by_key.get(replace(key, routed_topk=None))

    def require(self, key: AFDMoEStageKey) -> AFDMoEStageMeasurement:
        measurement = self.find(key)
        if measurement is None:
            raise KeyError(f"no exact AFD MoE stage measurement for {key}")
        return measurement

    def project_by_f_rank_load(
        self,
        key: AFDMoEStageKey,
        *,
        source_system: str,
        target_load_per_f_rank: float | None = None,
        latency_scale: float = 1.0,
        source_topology: str | None = None,
    ) -> AFDMoEStageProjection | None:
        """Interpolate latency inside one measured topology's load envelope.

        This method never extrapolates and never treats a projection as an
        exact measurement.  ``latency_scale`` must be supplied by the caller
        when transferring an envelope between systems.
        """

        target_logical_tokens, expected_target_load = f_rank_loads(key)
        target_load = expected_target_load
        if target_load_per_f_rank is not None:
            supplied_target_load = _finite_float(
                target_load_per_f_rank,
                "target_load_per_f_rank",
                positive=True,
            )
            if not math.isclose(supplied_target_load, expected_target_load, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "target_load_per_f_rank must equal batch * verify_width * routed_topk "
                    "after A/F and microbatch partitioning: "
                    f"supplied={supplied_target_load}, expected={expected_target_load}"
                )
        scale = _finite_float(latency_scale, "latency_scale", positive=True)
        groups: dict[str, list[tuple[float, AFDMoEStageMeasurement]]] = {}
        for entry in self.entries:
            candidate = entry.key
            if (
                candidate.model_path != key.model_path
                or candidate.system != source_system
                or candidate.stage != key.stage
                or candidate.mtp_nextn != key.mtp_nextn
                or candidate.microbatches != key.microbatches
                or candidate.moe_layers != key.moe_layers
                or (
                    candidate.routed_topk is not None
                    and key.routed_topk is not None
                    and candidate.routed_topk != key.routed_topk
                )
                or candidate.moe_precision != key.moe_precision
                or candidate.moe_backend != key.moe_backend
                or (source_topology is not None and candidate.topology != source_topology)
            ):
                continue
            groups.setdefault(candidate.topology, []).append(
                (f_rank_loads(candidate, routed_topk_fallback=key.routed_topk)[1], entry)
            )

        eligible: list[tuple[str, list[tuple[float, AFDMoEStageMeasurement]]]] = []
        for topology, raw_anchors in groups.items():
            anchors_by_load: dict[float, AFDMoEStageMeasurement] = {}
            for load, entry in raw_anchors:
                previous = anchors_by_load.get(load)
                if previous is None or entry.latency_ms > previous.latency_ms:
                    anchors_by_load[load] = entry
            anchors = sorted(anchors_by_load.items())
            if anchors and anchors[0][0] <= target_load <= anchors[-1][0]:
                eligible.append((topology, anchors))

        if not eligible:
            return None
        if len(eligible) != 1:
            topologies = sorted(topology for topology, _anchors in eligible)
            raise ValueError(
                "multiple measured topologies cover the requested F-rank load; "
                f"set source_topology explicitly: {topologies}"
            )

        topology, anchors = eligible[0]
        monotone: list[tuple[float, AFDMoEStageMeasurement, float]] = []
        latency_floor = 0.0
        for load, entry in anchors:
            latency_floor = max(latency_floor, entry.latency_ms)
            monotone.append((load, entry, latency_floor))

        lower_load, lower_entry, lower_latency = monotone[0]
        upper_load, upper_entry, upper_latency = monotone[-1]
        for index, anchor in enumerate(monotone):
            load, entry, latency = anchor
            if math.isclose(target_load, load, rel_tol=1e-12, abs_tol=1e-12):
                lower_load = upper_load = load
                lower_entry = upper_entry = entry
                lower_latency = upper_latency = latency
                break
            if load > target_load:
                lower_load, lower_entry, lower_latency = monotone[index - 1]
                upper_load, upper_entry, upper_latency = anchor
                break

        if math.isclose(lower_load, upper_load):
            interpolated = lower_latency
        else:
            fraction = (target_load - lower_load) / (upper_load - lower_load)
            interpolated = lower_latency + fraction * (upper_latency - lower_latency)
        return AFDMoEStageProjection(
            key=key,
            model_profile=lower_entry.model_profile,
            latency_ms=interpolated * scale,
            evidence=(
                "within-envelope monotone linear interpolation by routed expert assignments per F rank per microbatch"
            ),
            source_system=source_system,
            source_topology=topology,
            target_logical_tokens_per_f_rank_per_microbatch=target_logical_tokens,
            target_routed_assignments_per_f_rank_per_microbatch=target_load,
            latency_scale=scale,
            lower_anchor_load=lower_load,
            upper_anchor_load=upper_load,
            lower_anchor=lower_entry,
            upper_anchor=upper_entry,
        )


_AFD_TOPOLOGY = re.compile(r"^(?P<a>[1-9][0-9]*)A(?P<f>[1-9][0-9]*)F$")


def f_rank_loads(
    key: AFDMoEStageKey,
    *,
    routed_topk_fallback: int | None = None,
) -> tuple[float, float]:
    """Return logical tokens and routed assignments per F rank/microbatch."""

    logical_tokens = key.logical_batch_per_source_rank * (key.mtp_nextn + 1) / key.microbatches
    if key.stage == "afd":
        match = _AFD_TOPOLOGY.fullmatch(key.topology)
        if match is None:
            raise ValueError(f"AFD profile topology must use '<A>A<F>F': {key.topology!r}")
        logical_tokens *= int(match.group("a")) / int(match.group("f"))
    routed_topk = key.routed_topk if key.routed_topk is not None else routed_topk_fallback
    if routed_topk is None:
        raise ValueError("routed_topk is required to compute routed expert assignment load")
    return logical_tokens, logical_tokens * routed_topk


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and > 0" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def _parse_entry(value: Any, index: int, *, schema: str) -> AFDMoEStageMeasurement:
    prefix = f"entries[{index}]"
    raw = _object(value, prefix)
    stage = _string(raw.get("stage"), f"{prefix}.stage")
    if stage not in ("agg", "afd"):
        raise ValueError(f"{prefix}.stage must be 'agg' or 'afd'")
    validation = _object(raw.get("validation"), f"{prefix}.validation")
    if validation.get("stable") is not True:
        raise ValueError(f"{prefix}.validation.stable must be true")
    evidence = _string(validation.get("evidence"), f"{prefix}.validation.evidence")
    correctness = validation.get("correctness")
    speedup = validation.get("matched_speedup")
    speedup_lower_bound = validation.get("matched_speedup_lower_bound")
    if stage == "agg":
        if correctness is not True:
            raise ValueError(f"{prefix}.validation.correctness must be true for AGG")
        if speedup is not None:
            speedup = _finite_float(speedup, f"{prefix}.validation.matched_speedup", positive=True)
        if speedup_lower_bound is not None:
            speedup_lower_bound = _finite_float(
                speedup_lower_bound,
                f"{prefix}.validation.matched_speedup_lower_bound",
                positive=True,
            )
    else:
        if correctness is not None and not isinstance(correctness, bool):
            raise ValueError(f"{prefix}.validation.correctness must be boolean or null")
        if speedup is not None:
            speedup = _finite_float(speedup, f"{prefix}.validation.matched_speedup", positive=True)
        if speedup_lower_bound is not None:
            speedup_lower_bound = _finite_float(
                speedup_lower_bound,
                f"{prefix}.validation.matched_speedup_lower_bound",
                positive=True,
            )

    source = _object(raw.get("source"), f"{prefix}.source")
    key = AFDMoEStageKey(
        model_path=_string(raw.get("model_path"), f"{prefix}.model_path"),
        system=_string(raw.get("system"), f"{prefix}.system"),
        stage=stage,
        topology=_string(raw.get("topology"), f"{prefix}.topology"),
        logical_batch_per_source_rank=_integer(
            raw.get("logical_batch_per_source_rank"),
            f"{prefix}.logical_batch_per_source_rank",
            minimum=1,
        ),
        mtp_nextn=_integer(raw.get("mtp_nextn"), f"{prefix}.mtp_nextn", minimum=0),
        microbatches=_integer(raw.get("microbatches"), f"{prefix}.microbatches", minimum=1),
        moe_layers=_integer(raw.get("moe_layers"), f"{prefix}.moe_layers", minimum=1),
        routed_topk=(
            _integer(raw.get("routed_topk"), f"{prefix}.routed_topk", minimum=1)
            if schema == PROFILE_SCHEMA or raw.get("routed_topk") is not None
            else None
        ),
        moe_precision=_string(raw.get("moe_precision"), f"{prefix}.moe_precision"),
        moe_backend=(
            "megamoe" if schema == LEGACY_PROFILE_SCHEMA else _string(raw.get("moe_backend"), f"{prefix}.moe_backend")
        ),
    )
    return AFDMoEStageMeasurement(
        key=key,
        model_profile=_string(raw.get("model_profile"), f"{prefix}.model_profile"),
        latency_ms=_finite_float(raw.get("latency_ms"), f"{prefix}.latency_ms", positive=True),
        evidence=evidence,
        correctness=correctness,
        matched_speedup=speedup,
        matched_speedup_lower_bound=speedup_lower_bound,
        source_commit=_string(source.get("commit"), f"{prefix}.source.commit"),
        source_tree_sha256=_string(source.get("source_tree_sha256"), f"{prefix}.source.source_tree_sha256"),
        source_result=_string(source.get("result"), f"{prefix}.source.result"),
    )


__all__ = [
    "LEGACY_PROFILE_SCHEMA",
    "PROFILE_SCHEMA",
    "PROFILE_SCHEMA_V2",
    "AFDMoEStageKey",
    "AFDMoEStageMeasurement",
    "AFDMoEStageProfile",
    "AFDMoEStageProjection",
    "f_rank_loads",
]
