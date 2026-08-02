#!/usr/bin/env python3
"""Run a same-rack decode-only AFD/MTP comparison across supported MoE models."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.config import AFDConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession, InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile
from aiconfigurator.sdk.task_v2 import Task

SYSTEM = "gb200"
BACKEND = "sglang"
DATABASE_MODE = "HYBRID"
TOTAL_GPUS = 72
GPUS_PER_NODE = 4
TOTAL_NODES = TOTAL_GPUS // GPUS_PER_NODE
PIPELINE_MODEL = "conservative"
A_TPS = (1, 2, 4)
MICROBATCHES = (1, 2, 4)
F_NODE_GRID = (1, 2, 4, 8)
STATIC_WORLDS = (4, 8, 12, 18, 24, 36, 72)
STATIC_TPS = (1, 2, 4, 8)
STATIC_BATCH_CAP = 256

WORKLOADS = {
    "8k": {"context": 8192, "batch_per_a_gpu": (4, 8, 12, 16, 24, 32, 48, 64, 72, 96)},
    "16k": {"context": 16384, "batch_per_a_gpu": (2, 4, 6, 8, 12, 16, 24, 32, 36, 48)},
}
REFERENCE_AGG = {
    ("minimax_m25", 8192): {"world": 4, "tp": 1, "local_batch": 48},
    ("minimax_m25", 16384): {"world": 4, "tp": 1, "local_batch": 24},
}


@dataclass(frozen=True)
class Scenario:
    name: str
    nextn: int
    accepted_drafts: float | None
    acceptance_basis: str

    @property
    def q(self) -> int:
        return self.nextn + 1

    @property
    def progress(self) -> float:
        return 1.0 + float(self.accepted_drafts or 0.0)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_path: str
    backend_version: str
    moe_backend: str | None
    layers: int
    topk: int
    parameter_note: str
    attention_note: str
    attention_timing_note: str
    moe_note: str
    moe_shape_note: str
    precision_profiles: tuple[PrecisionProfile, ...]
    scenarios: tuple[Scenario, ...]
    f_node_grid: tuple[int, ...] = F_NODE_GRID


@dataclass(frozen=True)
class PrecisionProfile:
    key: str
    moe_quant_mode: str
    evidence: str
    timing_note: str
    primary: bool = False
    measured_complete_f: bool = False
    measured_microbatches: int | None = None
    measured_curve_key: str | None = None
    gemm_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None


def expected_accepted(nextn: int, conditional_rate: float) -> float:
    return sum(conditional_rate**position for position in range(1, nextn + 1))


NO_MTP = Scenario("no_mtp", 0, None, "disabled")
MODELS = (
    ModelSpec(
        key="qwen3_235b",
        label="Qwen3-235B-A22B",
        model_path="Qwen/Qwen3-235B-A22B-FP8",
        backend_version="0.5.14",
        moe_backend=None,
        layers=94,
        topk=8,
        parameter_note="235B total / 22B active",
        attention_note=(
            "full-context GQA, 64 query heads / 4 KV heads, head_dim=128; "
            "no sparse, sliding-window, or linear attention"
        ),
        attention_timing_note=(
            "SGLang 0.5.14 GQA table; single-token decode can resolve to silicon, while q=4 MTP verification "
            "uses the q-wide HYBRID estimate"
        ),
        moe_note=(
            "the primary AIC track uses same-shape NVFP4 expert data; the separate measured complete-F "
            "track remains FP8 because that is the kernel that was measured"
        ),
        moe_shape_note="hidden=4096, expert_inter=1536, 128 routed experts, top-8, power-law-1.01 routing",
        precision_profiles=(
            PrecisionProfile(
                key="nvfp4",
                moe_quant_mode="nvfp4",
                evidence="native AIC / NVFP4",
                timing_note=(
                    "same-shape GB200 SGLang 0.5.14 silicon rows, flashinfer TRT-LLM MoE; "
                    "dispatch/combine remain AIC communication models"
                ),
                primary=True,
            ),
            PrecisionProfile(
                key="fp8",
                moe_quant_mode="fp8_block",
                evidence="native AIC / FP8",
                timing_note=(
                    "same-shape GB200 SGLang 0.5.14 silicon rows, flashinfer TRT-LLM MoE; "
                    "used as a precision-control track"
                ),
            ),
            PrecisionProfile(
                key="measured_fp8_complete_f",
                moe_quant_mode="fp8_block",
                evidence="measured complete-F overlay / FP8",
                timing_note=(
                    "measured B200 FastAFD complete F stage: fused dispatch + persistent FP8 experts + combine; "
                    "interpolation is restricted to the measured assignment range"
                ),
                measured_complete_f=True,
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "eagle3_n3",
                3,
                1.28125,
                "NVIDIA Qwen Eagle3 mean committed output per step 2.28125; E[drafts]=1.28125",
            ),
        ),
    ),
    ModelSpec(
        key="minimax_m25",
        label="MiniMax-M2.5-FP8",
        model_path="MiniMaxAI/MiniMax-M2.5",
        backend_version="0.5.14",
        moe_backend=None,
        layers=62,
        topk=8,
        parameter_note="about 230B total / about 10B active",
        attention_note=(
            "full-context GQA, 48 query heads / 8 KV heads, head_dim=128; "
            "no sparse, sliding-window, or linear attention"
        ),
        attention_timing_note=(
            "SGLang 0.5.14 full-GQA HYBRID path with the FastAFD runtime contract fixed to BF16 FMHA and KV cache"
        ),
        moe_note=(
            "the reproduction track replaces AIC's generic F estimate with a measured complete MegaMoE F stage; "
            "the generic AIC FP8 track is retained as a diagnostic control"
        ),
        moe_shape_note="hidden=3072, expert_inter=1536, 256 routed experts, top-8, 62 layers",
        precision_profiles=(
            PrecisionProfile(
                key="calibrated_fp8_effective_f",
                moe_quant_mode="fp8_block",
                evidence="FastAFD-calibrated effective F / MiniMax-M2.5 FP8",
                timing_note=(
                    "effective F latency solved from the published GB200 NVL72 AFD/AGG ratio under the fixed "
                    "AIC baseline and pipeline equations; this is a calibration target, not an independent "
                    "F measurement"
                ),
                primary=True,
                measured_complete_f=True,
                measured_microbatches=2,
                measured_curve_key="fastafd_calibrated_effective_f",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="measured_fp8_complete_f",
                moe_quant_mode="fp8_block",
                evidence="measured complete-F overlay / MiniMax-M2.5 FP8",
                timing_note=(
                    "B200 real-kernel MegaMoE complete F stage at the exact 17A:1F load; fused dispatch, "
                    "persistent FP8 experts, and combine are timed together"
                ),
                measured_complete_f=True,
                measured_microbatches=2,
                measured_curve_key="b200_nvl8_measured_f",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="generic_fp8",
                moe_quant_mode="fp8_block",
                evidence="native AIC / generic MiniMax-M2.5 FP8",
                timing_note="AIC HYBRID generic FP8 MoE path; shown only to isolate the F-stage modeling gap",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
        ),
        scenarios=(NO_MTP,),
        f_node_grid=(1,),
    ),
    ModelSpec(
        key="minimax_m3",
        label="MiniMax-M3",
        model_path="MiniMaxAI/MiniMax-M3",
        backend_version="0.5.14",
        moe_backend=None,
        layers=60,
        topk=4,
        parameter_note="428B total / about 23B active",
        attention_note=(
            "MiniMax Sparse Attention (MSA): GQA 64Q/4KV, block indexer selects 16 x 128-token blocks "
            "for a 2,048-token sparse attention budget; not linear attention"
        ),
        attention_timing_note=(
            "no MSA silicon table; target-shape SOL transfers measured DSA utilization, and q-wide MTP "
            "verification is estimated"
        ),
        moe_note=(
            "no exact MiniMax-M3 MoE row exists at any precision; NVFP4, FP8, and BF16 are all AIC HYBRID "
            "shape projections, with NVFP4 selected as the deployment target"
        ),
        moe_shape_note="hidden=6144, expert_inter=3072, 128 routed experts, top-4, power-law-1.01 routing",
        precision_profiles=(
            PrecisionProfile(
                key="nvfp4_projected",
                moe_quant_mode="nvfp4",
                evidence="native AIC / NVFP4 projected",
                timing_note=(
                    "HYBRID projection at the exact MiniMax shape; no same-shape silicon row, so this is a "
                    "quantized-deployment sensitivity rather than an FP4 measurement"
                ),
                primary=True,
            ),
            PrecisionProfile(
                key="fp8_projected",
                moe_quant_mode="fp8_block",
                evidence="native AIC / FP8 projected",
                timing_note="HYBRID projection at the exact MiniMax shape; no same-shape silicon row",
            ),
            PrecisionProfile(
                key="bf16_projected",
                moe_quant_mode="bfloat16",
                evidence="native AIC / BF16 projected",
                timing_note="HYBRID projection at the exact MiniMax shape; no same-shape silicon row",
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario("mtp_n1_r70", 1, expected_accepted(1, 0.70), "scenario: conditional acceptance r=0.70"),
            Scenario("mtp_n3_r70", 3, expected_accepted(3, 0.70), "scenario: conditional acceptance r=0.70"),
            Scenario("mtp_n7_r70", 7, expected_accepted(7, 0.70), "scenario: conditional acceptance r=0.70"),
        ),
    ),
    ModelSpec(
        key="deepseek_v4_flash",
        label="DeepSeek-V4-Flash",
        model_path="deepseek-ai/DeepSeek-V4-Flash",
        backend_version="0.5.14",
        moe_backend=None,
        layers=43,
        topk=6,
        parameter_note="284B total / 13B active",
        attention_note=(
            "21 CSA layers (compression 4, top-512) + 20 HCA layers (compression 128) + 2 pure SWA layers, "
            "all with a 128-token local window and mHC; no linear attention"
        ),
        attention_timing_note=(
            "CSA/HCA and mHC use model-specific SGLang 0.5.14 tables; the 2 pure-SWA layers are approximated "
            "with HCA latency; q=3 MTP verification uses HYBRID estimation"
        ),
        moe_note=(
            "FP4 experts use the model's native MXFP4-weight/MXFP8-activation kernel; FP8 is retained only as "
            "a same-shape precision-control track"
        ),
        moe_shape_note="hidden=4096, expert_inter=2048, 256 routed experts, top-6, power-law-1.01 routing",
        precision_profiles=(
            PrecisionProfile(
                key="mxfp4_mxfp8",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                evidence="native AIC / MXFP4-MXFP8",
                timing_note=(
                    "same-shape GB200 SGLang 0.5.14 silicon rows, MXFP4 weights + MXFP8 activations, "
                    "flashinfer TRT-LLM MoE"
                ),
                primary=True,
            ),
            PrecisionProfile(
                key="fp8",
                moe_quant_mode="fp8_block",
                evidence="native AIC / FP8",
                timing_note="same-shape GB200 SGLang 0.5.14 silicon rows, fused Triton MoE",
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario("mtp_n2_r70", 2, expected_accepted(2, 0.70), "sensitivity anchor only: conditional r=0.70"),
        ),
    ),
    ModelSpec(
        key="deepseek_v4_pro",
        label="DeepSeek-V4-Pro",
        model_path="deepseek-ai/DeepSeek-V4-Pro",
        backend_version="0.5.12",
        moe_backend="megamoe",
        layers=61,
        topk=6,
        parameter_note="1.6T total / 49B active",
        attention_note=(
            "30 CSA layers (compression 4, top-1,024) + 31 HCA layers (compression 128), each with a "
            "128-token local window and mHC; no pure SWA-only or linear-attention layer"
        ),
        attention_timing_note=(
            "backend 0.5.12 declares reuse of model-specific CSA/HCA and mHC donors from SGLang 0.5.14; "
            "q=3 MTP verification uses HYBRID estimation"
        ),
        moe_note="measured FP4 MegaMoE module from the declared SGLang 0.5.10 donor",
        moe_shape_note="hidden=7168, expert_inter=3072, 384 routed experts, top-6, power-law-1.01 routing",
        precision_profiles=(
            PrecisionProfile(
                key="megamoe_fp4",
                moe_quant_mode="w4a8_mxfp4_mxfp8",
                evidence="native AIC / MegaMoE FP4",
                timing_note=(
                    "same-shape measured MegaMoE module, FP8 activations + FP4 experts; exact/interpolated up "
                    "to 512 local decode tokens and utilization-hold extrapolation above that range"
                ),
                primary=True,
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario("mtp_n2_r70", 2, expected_accepted(2, 0.70), "sensitivity anchor only: conditional r=0.70"),
        ),
        f_node_grid=(1, 2, 4, 8),
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}


def precision_for(model_key: str, precision_key: str) -> PrecisionProfile:
    spec = MODEL_BY_KEY[model_key]
    return next(profile for profile in spec.precision_profiles if profile.key == precision_key)


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def load_measured_f(path: Path | None) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    curves: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            curve_key = row.get("curve_key")
            if not curve_key and row.get("implementation") != "MegaMoE":
                continue
            model_key = row.get("model_key") or "qwen3_235b"
            curve_key = curve_key or "measured_fp8_complete_f"
            curves[(model_key, curve_key, int(row["context_tokens"]))].append(
                (float(row["assignments_per_f_gpu_per_microbatch"]), float(row["complete_stage_mean_ms"]))
            )
    return {context: sorted(set(values)) for context, values in curves.items()}


def interpolate_inside(curve: list[tuple[float, float]], x: float) -> tuple[float | None, str]:
    if not curve:
        return None, "no measured curve"
    if x < curve[0][0] or x > curve[-1][0]:
        return None, f"outside [{curve[0][0]:.0f}, {curve[-1][0]:.0f}]"
    for x0, y0 in curve:
        if math.isclose(x, x0):
            return y0, "exact measured load"
    for (x0, y0), (x1, y1) in pairwise(curve):
        if x0 <= x <= x1:
            alpha = (x - x0) / (x1 - x0)
            return y0 + alpha * (y1 - y0), f"interpolated inside [{x0:.0f}, {x1:.0f}]"
    raise AssertionError("in-range point was not interpolated")


def op_group(name: str) -> str:
    value = name.lower()
    if "attention" in value:
        return "attention"
    if "mhc" in value:
        return "mHC"
    if "router" in value:
        return "router"
    if "moe" in value:
        return "MoE"
    if "allgather" in value or "reducescatter" in value:
        return "F collective"
    if "combine" in value:
        return "A combine"
    if "transfer" in value:
        return "A-F transfer"
    if "gemm" in value:
        return "dense GEMM"
    return "norm / embedding / logits"


def operation_rows(side: str, values: dict[str, float], multiplier: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "side": side,
            "op": name,
            "group": op_group(name),
            "raw_round_ms": float(latency) * multiplier,
        }
        for name, latency in values.items()
    ]


@cache
def task_for(model_key: str, context: int, scenario_name: str, precision_key: str) -> Task:
    spec = MODEL_BY_KEY[model_key]
    scenario = next(value for value in spec.scenarios if value.name == scenario_name)
    precision = precision_for(model_key, precision_key)
    return Task(
        serving_mode="agg",
        model_path=spec.model_path,
        system_name=SYSTEM,
        backend_name=BACKEND,
        backend_version=spec.backend_version,
        database_mode=DATABASE_MODE,
        isl=context - 1,
        osl=2,
        nextn=scenario.nextn,
        nextn_accepted=scenario.accepted_drafts,
        moe_backend=spec.moe_backend,
        gemm_quant_mode=(
            common.GEMMQuantMode[precision.gemm_quant_mode] if precision.gemm_quant_mode is not None else None
        ),
        moe_quant_mode=common.MoEQuantMode[precision.moe_quant_mode],
        kvcache_quant_mode=(
            common.KVCacheQuantMode[precision.kvcache_quant_mode] if precision.kvcache_quant_mode is not None else None
        ),
        fmha_quant_mode=(
            common.FMHAQuantMode[precision.fmha_quant_mode] if precision.fmha_quant_mode is not None else None
        ),
    )


def configured_model(task: Task, *, tp: int, dp: int, moe_tp: int, moe_ep: int):
    model_config = task.build_model_config(role="agg")
    model_config.tp_size = tp
    model_config.pp_size = 1
    model_config.attention_dp_size = dp
    model_config.moe_tp_size = moe_tp
    model_config.moe_ep_size = moe_ep
    return model_config, get_model(task.model_path, model_config, task.backend_name)


@cache
def static_point(
    model_key: str,
    context: int,
    scenario_name: str,
    precision_key: str,
    world: int,
    tp: int,
    local_batch: int,
) -> dict[str, Any]:
    task = task_for(model_key, context, scenario_name, precision_key)
    scenario = next(value for value in MODEL_BY_KEY[model_key].scenarios if value.name == scenario_name)
    dp = world // tp
    model_config, model = configured_model(task, tp=tp, dp=dp, moe_tp=1, moe_ep=world)
    database = task._load_database(SYSTEM, BACKEND, task.backend_version)
    summary = InferenceSession(model, database, get_backend(BACKEND)).run_static(
        task.build_runtime_config(batch_size=local_batch),
        mode="static_gen",
        stride=1,
    )
    summary = SpeculativeDecodingProfile.from_inputs(scenario.nextn, scenario.accepted_drafts).project_summary(
        summary, role="decode"
    )
    generation = {name: float(value) for name, value in summary.get_generation_latency_dict().items()}
    raw_round_ms = sum(generation.values())
    result = summary.get_result_dict() or {}
    global_batch = local_batch * dp
    return {
        "world": world,
        "tp": tp,
        "dp": dp,
        "moe_tp": 1,
        "moe_ep": world,
        "local_batch": local_batch,
        "global_batch": global_batch,
        "raw_round_ms": raw_round_ms,
        "effective_tpot_ms": raw_round_ms / scenario.progress,
        "output_tokens_s_replica": global_batch * scenario.progress * 1000.0 / raw_round_ms,
        "memory_gb": float(result.get("memory", math.nan)),
        "oom": bool(summary.check_oom() or summary.check_kv_cache_oom()),
        "ops": operation_rows("AGG", generation),
        "op_sources": dict(summary.get_generation_source_dict()),
        "quant": {
            "gemm": model_config.gemm_quant_mode.name,
            "moe": model_config.moe_quant_mode.name,
            "kvcache": model_config.kvcache_quant_mode.name,
            "fmha": model_config.fmha_quant_mode.name,
        },
    }


@cache
def static_capacity(
    model_key: str,
    context: int,
    scenario_name: str,
    precision_key: str,
    world: int,
    tp: int,
) -> int:
    try:
        if static_point(model_key, context, scenario_name, precision_key, world, tp, 1)["oom"]:
            return 0
    except Exception:
        return 0
    low, high = 1, STATIC_BATCH_CAP
    while low < high:
        middle = (low + high + 1) // 2
        try:
            feasible = not static_point(model_key, context, scenario_name, precision_key, world, tp, middle)["oom"]
        except Exception:
            feasible = False
        if feasible:
            low = middle
        else:
            high = middle - 1
    return low


@cache
def static_cluster(
    model_key: str,
    context: int,
    scenario_name: str,
    precision_key: str,
    offered_requests: int,
) -> dict[str, Any]:
    scenario = next(value for value in MODEL_BY_KEY[model_key].scenarios if value.name == scenario_name)
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for world in STATIC_WORLDS:
        if TOTAL_GPUS % world:
            continue
        replicas = TOTAL_GPUS // world
        for tp in STATIC_TPS:
            if world % tp:
                continue
            dp = world // tp
            capacity = static_capacity(model_key, context, scenario_name, precision_key, world, tp)
            if not capacity:
                continue
            active = min(offered_requests, replicas * dp * capacity)
            local_batch = max(1, math.ceil(active / (replicas * dp)))
            try:
                point = static_point(model_key, context, scenario_name, precision_key, world, tp, local_batch)
            except Exception as error:
                errors.append(f"world={world},tp={tp}: {type(error).__name__}: {error}")
                continue
            output_tokens_s = active * scenario.progress * 1000.0 / point["raw_round_ms"]
            candidates.append(
                {
                    **point,
                    "replicas": replicas,
                    "offered_requests": offered_requests,
                    "active_requests": active,
                    "queued_requests": offered_requests - active,
                    "capacity_local_batch": capacity,
                    "cluster_output_tokens_s": output_tokens_s,
                    "cluster_output_tokens_s_gpu": output_tokens_s / TOTAL_GPUS,
                }
            )
    if not candidates:
        raise RuntimeError(f"no feasible AGG layout for {model_key}/{context}/{scenario_name}: {errors[:3]}")
    return max(candidates, key=lambda row: row["cluster_output_tokens_s_gpu"])


def baseline_cluster(
    model_key: str,
    context: int,
    scenario_name: str,
    precision_key: str,
    offered_requests: int,
) -> dict[str, Any]:
    contract = REFERENCE_AGG.get((model_key, context))
    if contract is None:
        return static_cluster(model_key, context, scenario_name, precision_key, offered_requests)
    point = static_point(
        model_key,
        context,
        scenario_name,
        precision_key,
        contract["world"],
        contract["tp"],
        contract["local_batch"],
    )
    if point["oom"]:
        raise RuntimeError(f"reference AGG contract is OOM: {model_key}/{context}")
    replicas = TOTAL_GPUS // contract["world"]
    active_requests = replicas * point["global_batch"]
    cluster_output_tokens_s = replicas * point["output_tokens_s_replica"]
    return {
        **point,
        "replicas": replicas,
        "offered_requests": active_requests,
        "active_requests": active_requests,
        "queued_requests": 0,
        "capacity_local_batch": contract["local_batch"],
        "cluster_output_tokens_s": cluster_output_tokens_s,
        "cluster_output_tokens_s_gpu": cluster_output_tokens_s / TOTAL_GPUS,
        "contract": "published fixed AGG layout and batch",
    }


def afd_point(
    spec: ModelSpec,
    scenario: Scenario,
    precision: PrecisionProfile,
    *,
    context: int,
    a_nodes: int,
    f_nodes: int,
    a_tp: int,
    batch_per_a_gpu: int,
    microbatches: int,
    measured_f_ms: float | None = None,
    measured_note: str = "",
) -> dict[str, Any]:
    task = task_for(spec.key, context, scenario.name, precision.key)
    database = task._load_database(SYSTEM, BACKEND, spec.backend_version)
    base_config = task.build_model_config(role="agg")
    a_config = copy.deepcopy(base_config)
    a_config.tp_size = a_tp
    a_config.pp_size = 1
    a_config.attention_dp_size = 1
    # The A pool never executes routed experts. Do not force MegaMoE's EP>1
    # construction contract onto a_tp=1; keep the A-only model's unused MoE
    # branch generic and satisfy the ordinary parallel-product invariant.
    if spec.moe_backend == "megamoe":
        a_config.moe_backend = None
    a_config.moe_tp_size = a_tp
    a_config.moe_ep_size = 1

    f_gpus = f_nodes * GPUS_PER_NODE
    f_config = copy.deepcopy(base_config)
    f_config.tp_size = f_gpus
    f_config.pp_size = 1
    f_config.attention_dp_size = 1
    f_config.moe_tp_size = 1
    f_config.moe_ep_size = f_gpus

    a_batch_size = batch_per_a_gpu * a_tp
    afd_config = AFDConfig(
        n_a_nodes=a_nodes,
        n_f_nodes=f_nodes,
        gpus_per_node=GPUS_PER_NODE,
        tp_a=a_tp,
        f_moe_ep_size=f_gpus,
        a_batch_size=a_batch_size,
        num_microbatches=microbatches,
        pipeline_model=PIPELINE_MODEL,
        phase="decode",
        combined_with_pd=False,
    )
    global_requests = a_nodes * GPUS_PER_NODE * batch_per_a_gpu
    runtime = task.build_runtime_config(batch_size=global_requests)
    summary = AFDInferenceSession(
        model_path=spec.model_path,
        a_model_config=a_config,
        f_model_config=f_config,
        database=database,
        backend=get_backend(BACKEND),
        afd_config=afd_config,
        afd_moe_time_ms=measured_f_ms,
    ).run_afd(
        runtime,
        phase="decode",
        speculative_profile=SpeculativeDecodingProfile.from_inputs(scenario.nextn, scenario.accepted_drafts),
    )
    if summary.check_oom() or summary.check_kv_cache_oom():
        raise RuntimeError("OOM")
    raw = dict(summary.get_result_dict() or {})
    per_ops = summary.get_per_ops_data() or {}
    a_ops = {name: float(value) for name, value in per_ops.get("decode_a_worker", {}).items()}
    f_ops = {name: float(value) for name, value in per_ops.get("decode_f_worker", {}).items()}
    comm_ops = {
        name: float(value) * spec.layers * microbatches
        for name, value in per_ops.get("comm", {}).items()
        if name.endswith("_a2f") or name.endswith("_f2a")
    }
    raw_round_ms = float(raw["decode_batch_service_time_ms"])
    manual_throughput = global_requests * scenario.progress * 1000.0 / raw_round_ms
    baseline = baseline_cluster(spec.key, context, scenario.name, precision.key, global_requests)
    modules = (
        operation_rows("A", a_ops, microbatches)
        + operation_rows("F", f_ops, microbatches)
        + operation_rows("fabric", comm_ops)
    )
    return {
        "model": spec.key,
        "scenario": scenario.name,
        "context": context,
        "evidence": precision.evidence,
        "precision_profile": precision.key,
        "attention_structure": spec.attention_note,
        "attention_timing_note": spec.attention_timing_note,
        "moe_shape_note": spec.moe_shape_note,
        "moe_timing_note": precision.timing_note,
        "measured_f_note": measured_note,
        "nextn": scenario.nextn,
        "q": scenario.q,
        "accepted_drafts": scenario.accepted_drafts,
        "progress": scenario.progress,
        "a_nodes": a_nodes,
        "f_nodes": f_nodes,
        "a_tp": a_tp,
        "f_ep": f_gpus,
        "batch_per_a_gpu": batch_per_a_gpu,
        "a_batch_size_per_worker": a_batch_size,
        "microbatches": microbatches,
        "global_requests": global_requests,
        "raw_round_ms": raw_round_ms,
        "effective_tpot_ms": raw_round_ms / scenario.progress,
        "output_tokens_s": manual_throughput,
        "output_tokens_s_gpu": manual_throughput / TOTAL_GPUS,
        "a_full_work_ms": sum(a_ops.values()) * microbatches,
        "f_full_work_ms": sum(f_ops.values()) * microbatches,
        "pipeline_bottleneck": "A" if sum(a_ops.values()) >= sum(f_ops.values()) else "F",
        "a_memory_gb": float(raw["(a)memory"]),
        "f_memory_gb": float(raw["(f)memory"]),
        "measured_f_ms": measured_f_ms,
        "quant": {
            "a_gemm": a_config.gemm_quant_mode.name,
            "a_fmha": a_config.fmha_quant_mode.name,
            "a_kvcache": a_config.kvcache_quant_mode.name,
            "f_moe": f_config.moe_quant_mode.name,
        },
        "modules": modules,
        "agg": baseline,
        "afd_over_agg": manual_throughput / baseline["cluster_output_tokens_s"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = [MODEL_BY_KEY[key] for key in args.models]
    measured_curves = load_measured_f(args.measured_f_csv)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_groups = sum(len(model.scenarios) * len(model.precision_profiles) * len(WORKLOADS) for model in selected)
    group_index = 0
    for spec in selected:
        for workload, workload_spec in WORKLOADS.items():
            context = int(workload_spec["context"])
            for scenario in spec.scenarios:
                for precision in spec.precision_profiles:
                    group_index += 1
                    print(
                        f"[{group_index}/{total_groups}] {spec.key} {workload} {scenario.name} {precision.key}",
                        flush=True,
                    )
                    for f_nodes in spec.f_node_grid:
                        a_nodes = TOTAL_NODES - f_nodes
                        for a_tp in A_TPS:
                            for batch_per_a_gpu in workload_spec["batch_per_a_gpu"]:
                                for microbatches in MICROBATCHES:
                                    key = {
                                        "model": spec.key,
                                        "workload": workload,
                                        "scenario": scenario.name,
                                        "precision_profile": precision.key,
                                        "a_nodes": a_nodes,
                                        "f_nodes": f_nodes,
                                        "a_tp": a_tp,
                                        "batch_per_a_gpu": batch_per_a_gpu,
                                        "microbatches": microbatches,
                                        "evidence": precision.evidence,
                                    }
                                    measured_ms = None
                                    measured_note = ""
                                    assignments = None
                                    if precision.measured_complete_f:
                                        if microbatches != precision.measured_microbatches:
                                            continue
                                        global_requests = a_nodes * GPUS_PER_NODE * int(batch_per_a_gpu)
                                        layer_token_factor = scenario.q + scenario.nextn / spec.layers
                                        assignments = (
                                            global_requests
                                            * layer_token_factor
                                            * spec.topk
                                            / (microbatches * f_nodes * GPUS_PER_NODE)
                                        )
                                        measured_ms, measured_note = interpolate_inside(
                                            measured_curves.get(
                                                (spec.key, precision.measured_curve_key or precision.key, context), []
                                            ),
                                            assignments,
                                        )
                                        if measured_ms is None:
                                            continue
                                    try:
                                        point = afd_point(
                                            spec,
                                            scenario,
                                            precision,
                                            context=context,
                                            a_nodes=a_nodes,
                                            f_nodes=f_nodes,
                                            a_tp=a_tp,
                                            batch_per_a_gpu=int(batch_per_a_gpu),
                                            microbatches=microbatches,
                                            measured_f_ms=measured_ms,
                                            measured_note=measured_note,
                                        )
                                        point["workload"] = workload
                                        if assignments is not None:
                                            point["assignments_per_f_gpu_per_microbatch"] = assignments
                                        rows.append(point)
                                    except Exception as error:
                                        failures.append(
                                            {
                                                **key,
                                                "error": f"{type(error).__name__}: {error}",
                                            }
                                        )

                    checkpoint = {
                        "schema": "aic.afd-multimodel-mtp.v2.partial",
                        "rows": rows,
                        "failures": failures,
                    }
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.with_suffix(".partial.json").write_text(
                        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
                    )

    source_counts = Counter()
    for row in rows:
        source_counts.update(row["agg"].get("op_sources", {}).values())
    return {
        "schema": "aic.afd-multimodel-mtp.v2",
        "code": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "contract": {
            "system": SYSTEM,
            "backend": BACKEND,
            "database_mode": DATABASE_MODE,
            "total_gpus": TOTAL_GPUS,
            "gpus_per_node": GPUS_PER_NODE,
            "total_nodes": TOTAL_NODES,
            "pipeline_model": PIPELINE_MODEL,
            "a_tp_grid": list(A_TPS),
            "microbatch_grid": list(MICROBATCHES),
            "f_node_grid": list(F_NODE_GRID),
            "static_world_grid": list(STATIC_WORLDS),
            "static_tp_grid": list(STATIC_TPS),
            "fixed_reference_agg": {
                f"{model_key}:{context}": contract for (model_key, context), contract in REFERENCE_AGG.items()
            },
            "batch_semantics": "batch_per_a_gpu; a_batch_size=batch_per_a_gpu*a_tp",
            "mtp_compute": "q=nextn+1 target tokens plus nextn draft-layer equivalents: q*L+nextn",
            "mtp_progress": "1+expected accepted draft tokens",
            "moe_precision_policy": (
                "Primary headline uses the lowest precision with a model-compatible AIC path: "
                "MXFP4/NVFP4, then FP8, then BF16. AGG and AFD use the same MoE precision. "
                "Alternate precision profiles are sensitivity controls, not mixed into the headline."
            ),
            "scope": "decode only; no prefill or request-arrival model in the AIC sweep",
        },
        "models": [
            {
                **asdict(spec),
                "scenarios": [
                    asdict(scenario) | {"q": scenario.q, "progress": scenario.progress} for scenario in spec.scenarios
                ],
            }
            for spec in selected
        ],
        "workloads": WORKLOADS,
        "measured_f_input": {
            "path": str(args.measured_f_csv) if args.measured_f_csv else None,
            "rule": "interpolate only inside measured assignment range; never extrapolate",
            "curves": {
                f"{model_key}:{curve_key}:{context}": [{"assignments": x, "latency_ms": y} for x, y in values]
                for (model_key, curve_key, context), values in measured_curves.items()
            },
        },
        "agg_source_counts": dict(source_counts),
        "rows": rows,
        "failures": failures,
        "unsupported": [
            {
                "model": "Kimi-K3",
                "reason": (
                    "AIC has no Kimi-K3 model. Its KDA/Gated-MLA, AttnRes, Stable LatentMoE, "
                    "896-expert/top-16 shape, and 2.8T/104B-active scale are not equivalent to Kimi-K2; "
                    "borrowing K2 timings would not be an auditable HYBRID estimate."
                ),
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-f-csv", type=Path)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_BY_KEY), default=sorted(MODEL_BY_KEY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.disable(logging.CRITICAL)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    partial = args.output.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(payload["rows"]),
                "failures": len(payload["failures"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
