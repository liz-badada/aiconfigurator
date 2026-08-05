#!/usr/bin/env python3
"""Sweep fixed-size GPU pools for AGG, AFD, and matched MTP variants."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.afd_moe_profile import (
    AFDMoEStageKey,
    AFDMoEStageMeasurement,
    AFDMoEStageProfile,
    AFDMoEStageProjection,
    f_rank_loads,
)
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.config import AFDConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession, InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import load_system_spec
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile
from aiconfigurator.sdk.task_v2 import Task

DEFAULT_SYSTEM = "gb200"
BACKEND = "sglang"
DATABASE_MODE = "HYBRID"
DEFAULT_GPUS_PER_NODE = 4
TOTAL_GPU_GRID = (16, 24, 36, 48, 72)
PIPELINE_MODEL = "conservative"
DECODE_STRIDE = 128
A_TPS = (1, 2, 4)
MICROBATCHES = (1, 2, 4)
STATIC_TPS = (1, 2, 4, 8)
SPEED_FLOORS = (20, 30, 40, 50, 60, 70, 100)

WORKLOADS = {
    "8k": {
        "isl": 8192,
        "osl": 1024,
        "afd_batch_per_a_gpu": (4, 8, 16, 24, 32, 48, 64, 72, 80, 88, 96, 128, 160, 192),
        "agg_local_batch": (1, 2, 4, 8, 16, 24, 32, 48, 64, 72, 80, 88, 96, 128, 160, 192, 224, 256),
    },
    "16k": {
        "isl": 16384,
        "osl": 1024,
        "afd_batch_per_a_gpu": (2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96),
        "agg_local_batch": (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 160),
    },
    "32k": {
        "isl": 32768,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64),
        "agg_local_batch": (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80),
    },
    "64k": {
        "isl": 65536,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4, 8, 12, 16, 20, 24, 32, 40),
        "agg_local_batch": (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48),
    },
    "128k": {
        "isl": 131072,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4, 8, 12, 16, 20),
        "agg_local_batch": (1, 2, 4, 8, 12, 16, 20, 24, 32, 40),
    },
    "256k": {
        "isl": 262144,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4, 8, 12, 16, 20),
        "agg_local_batch": (1, 2, 4, 8, 12, 16, 20),
    },
    "512k": {
        "isl": 524288,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4, 8),
        "agg_local_batch": (1, 2, 4, 8, 12, 16),
    },
    "1m": {
        # Keep ISL + OSL inside the 1,048,576-token model contract.
        "isl": 1047552,
        "osl": 1024,
        "afd_batch_per_a_gpu": (1, 2, 4),
        "agg_local_batch": (1, 2, 4, 8),
    },
}


@dataclass(frozen=True)
class Scenario:
    name: str
    nextn: int
    accepted_drafts: float | None
    acceptance_basis: str
    primary: bool = False
    total_gpu_grid: tuple[int, ...] = TOTAL_GPU_GRID

    @property
    def verification_width(self) -> int:
        return self.nextn + 1

    @property
    def progress(self) -> float:
        return 1.0 + float(self.accepted_drafts or 0.0)


@dataclass(frozen=True)
class PrecisionProfile:
    key: str
    moe_quant_mode: str
    moe_kernel: str
    evidence: str
    exact_shape_data: bool
    backend_family: str = "other"
    primary: bool = False
    total_gpu_grid: tuple[int, ...] = TOTAL_GPU_GRID
    measured_moe_precision: str | None = None
    measured_moe_backend: str | None = None
    gemm_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_path: str
    backend_version: str
    moe_backend: str | None
    max_sequence_length: int
    attention_heads: int
    layers: int
    moe_layers: int
    topk: int
    parameter_note: str
    attention_type: str
    attention_backend: str
    attention_evidence: str
    moe_structure: str
    precision_profiles: tuple[PrecisionProfile, ...]
    scenarios: tuple[Scenario, ...]


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
        max_sequence_length=40960,
        attention_heads=64,
        layers=94,
        moe_layers=94,
        topk=8,
        parameter_note="235B total / 22B active",
        attention_type="Full-context GQA, 64 query heads / 4 KV heads, head_dim=128",
        attention_backend="SGLang 0.5.14 generation attention",
        attention_evidence="same-shape silicon where available; HYBRID fallback for uncovered q-wide MTP shapes",
        moe_structure="hidden=4096, expert_inter=1536, 128 routed experts, top-8",
        precision_profiles=(
            PrecisionProfile(
                key="mxfp4_mxfp8_megamoe",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="megamoe_m2n",
                evidence="exact measured MegaMoE profile only; generic fallback is never labeled MegaMoE",
                exact_shape_data=False,
                backend_family="megamoe",
                primary=True,
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="megamoe",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_deepep_deepgemm",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="deepep_deepgemm",
                evidence="exact measured DeepEP dispatch/combine plus DeepGEMM profile only",
                exact_shape_data=False,
                backend_family="deepep_deepgemm",
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="deepep_deepgemm",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_trtllm_control",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="sglang_mxfp4_flashinfer_trtllm_moe",
                evidence="generic AIC SGLang FlashInfer/TensorRT-LLM MoE control",
                exact_shape_data=False,
                backend_family="trtllm",
            ),
            PrecisionProfile(
                key="nvfp4_control",
                moe_quant_mode="nvfp4",
                moe_kernel="sglang_flashinfer_trtllm_moe",
                evidence="generic AIC NVFP4 control only; no exact MegaMoE profile match",
                exact_shape_data=True,
                total_gpu_grid=(72,),
            ),
            PrecisionProfile(
                key="fp8",
                moe_quant_mode="fp8_block",
                moe_kernel="sglang_flashinfer_trtllm_moe",
                evidence="same-shape GB200 silicon; 72-GPU precision control",
                exact_shape_data=True,
                total_gpu_grid=(72,),
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "eagle3_n3",
                3,
                1.28125,
                "mean committed progress=2.28125 tokens/iteration",
                primary=True,
            ),
        ),
    ),
    ModelSpec(
        key="minimax_m25",
        label="MiniMax-M2.5",
        model_path="MiniMaxAI/MiniMax-M2.5",
        backend_version="0.5.14",
        moe_backend=None,
        max_sequence_length=196608,
        attention_heads=48,
        layers=62,
        moe_layers=62,
        topk=8,
        parameter_note="about 230B total / about 10B active",
        attention_type="Full-context GQA, 48 query heads / 8 KV heads, head_dim=128",
        attention_backend="SGLang 0.5.14 generation attention",
        attention_evidence="HYBRID full-GQA path with BF16 FMHA and KV cache",
        moe_structure="hidden=3072, expert_inter=1536, 256 routed experts, top-8",
        precision_profiles=(
            PrecisionProfile(
                key="mxfp4_mxfp8_megamoe",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="megamoe_m2n",
                evidence="exact measured MegaMoE profile only; generic fallback is never labeled MegaMoE",
                exact_shape_data=False,
                backend_family="megamoe",
                primary=True,
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="megamoe",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_deepep_deepgemm",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="deepep_deepgemm",
                evidence="exact measured DeepEP dispatch/combine plus DeepGEMM profile only",
                exact_shape_data=False,
                backend_family="deepep_deepgemm",
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="deepep_deepgemm",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_trtllm_control",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="sglang_mxfp4_flashinfer_trtllm_moe",
                evidence="generic AIC SGLang FlashInfer/TensorRT-LLM MoE control",
                exact_shape_data=False,
                backend_family="trtllm",
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="nvfp4_control",
                moe_quant_mode="nvfp4",
                moe_kernel="sglang_flashinfer_trtllm_moe",
                evidence="generic AIC NVFP4 control only; no exact MegaMoE profile match",
                exact_shape_data=True,
                total_gpu_grid=(72,),
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
            PrecisionProfile(
                key="fp8",
                moe_quant_mode="fp8_block",
                moe_kernel="sglang_fused_moe_triton",
                evidence="same-shape GB200 silicon; 72-GPU FP8 control",
                exact_shape_data=True,
                total_gpu_grid=(72,),
                gemm_quant_mode="fp8_block",
                kvcache_quant_mode="bfloat16",
                fmha_quant_mode="bfloat16",
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "mtp_n1_r70",
                1,
                expected_accepted(1, 0.70),
                "sensitivity only: conditional acceptance=70%",
                primary=True,
            ),
        ),
    ),
    ModelSpec(
        key="minimax_m3",
        label="MiniMax-M3",
        model_path="MiniMaxAI/MiniMax-M3",
        backend_version="0.5.14",
        moe_backend=None,
        max_sequence_length=1048576,
        attention_heads=64,
        layers=60,
        moe_layers=57,
        topk=4,
        parameter_note="428B total / about 23B active",
        attention_type="MiniMax Sparse Attention: 16 x 128-token selected blocks (2,048-token sparse budget)",
        attention_backend="SGLang 0.5.14 MSA model using DSA utilization transfer",
        attention_evidence="no native MSA silicon table; target-shape HYBRID projection",
        moe_structure="hidden=6144, expert_inter=3072, 128 routed experts, top-4",
        precision_profiles=(
            PrecisionProfile(
                key="mxfp4_mxfp8_megamoe",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="megamoe_m2n",
                evidence="exact measured MegaMoE profile only; generic fallback is never labeled MegaMoE",
                exact_shape_data=False,
                backend_family="megamoe",
                primary=True,
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="megamoe",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_deepep_deepgemm",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="deepep_deepgemm",
                evidence="exact measured DeepEP dispatch/combine plus DeepGEMM profile only",
                exact_shape_data=False,
                backend_family="deepep_deepgemm",
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="deepep_deepgemm",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_trtllm_control",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="sglang_mxfp4_flashinfer_trtllm_moe",
                evidence="generic AIC SGLang FlashInfer/TensorRT-LLM MoE control",
                exact_shape_data=False,
                backend_family="trtllm",
            ),
            PrecisionProfile(
                key="nvfp4_projected_control",
                moe_quant_mode="nvfp4",
                moe_kernel="sglang_flashinfer_trtllm_moe",
                evidence="generic AIC projected NVFP4 control only; no exact MegaMoE profile match",
                exact_shape_data=False,
                total_gpu_grid=(72,),
            ),
            PrecisionProfile(
                key="fp8_projected",
                moe_quant_mode="fp8_block",
                moe_kernel="sglang_flashinfer_trtllm_moe",
                evidence="exact target shape with cross-shape utilization transfer; 72-GPU control",
                exact_shape_data=False,
                total_gpu_grid=(72,),
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "mtp_n1_r70",
                1,
                expected_accepted(1, 0.70),
                "conditional acceptance=70%",
                primary=True,
            ),
            Scenario(
                "mtp_n2_r70",
                2,
                expected_accepted(2, 0.70),
                "conditional acceptance=70%; depth sensitivity",
                total_gpu_grid=(72,),
            ),
        ),
    ),
    ModelSpec(
        key="deepseek_v4_flash",
        label="DeepSeek-V4-Flash",
        model_path="deepseek-ai/DeepSeek-V4-Flash",
        backend_version="0.5.14",
        moe_backend=None,
        max_sequence_length=1048576,
        attention_heads=64,
        layers=43,
        moe_layers=43,
        topk=6,
        parameter_note="284B total / 13B active",
        attention_type="21 CSA + 20 HCA + 2 SWA layers, 128-token local window, mHC",
        attention_backend="SGLang 0.5.14 model-specific CSA/HCA modules",
        attention_evidence="CSA/HCA silicon tables; SWA reuses HCA timing; q-wide MTP is HYBRID",
        moe_structure="hidden=4096, expert_inter=2048, 256 routed experts, top-6",
        precision_profiles=(
            PrecisionProfile(
                key="mxfp4_mxfp8_megamoe",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="megamoe_m2n",
                evidence="exact measured MegaMoE profile only; generic fallback is never labeled MegaMoE",
                exact_shape_data=True,
                backend_family="megamoe",
                primary=True,
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="megamoe",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_deepep_deepgemm",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="deepep_deepgemm",
                evidence="exact measured DeepEP dispatch/combine plus DeepGEMM profile only",
                exact_shape_data=True,
                backend_family="deepep_deepgemm",
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="deepep_deepgemm",
            ),
            PrecisionProfile(
                key="mxfp4_mxfp8_trtllm_control",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="sglang_mxfp4_flashinfer_trtllm_moe",
                evidence="generic AIC SGLang FlashInfer/TensorRT-LLM MoE control",
                exact_shape_data=True,
                backend_family="trtllm",
            ),
            PrecisionProfile(
                key="fp8",
                moe_quant_mode="fp8_block",
                moe_kernel="sglang_fused_moe_triton",
                evidence="same-shape GB200 silicon; 72-GPU precision control",
                exact_shape_data=True,
                total_gpu_grid=(72,),
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "mtp_n2_r70",
                2,
                expected_accepted(2, 0.70),
                "sensitivity only: conditional acceptance=70%",
                primary=True,
            ),
        ),
    ),
    ModelSpec(
        key="deepseek_v4_pro",
        label="DeepSeek-V4-Pro",
        model_path="deepseek-ai/DeepSeek-V4-Pro",
        backend_version="0.5.12",
        moe_backend=None,
        max_sequence_length=1048576,
        attention_heads=128,
        layers=61,
        moe_layers=61,
        topk=6,
        parameter_note="1.6T total / 49B active",
        attention_type="30 CSA + 31 HCA layers, 128-token local window, mHC",
        attention_backend="SGLang 0.5.12 with declared 0.5.14 CSA/HCA donors",
        attention_evidence="model-specific donor tables; q-wide MTP is HYBRID",
        moe_structure="hidden=7168, expert_inter=3072, 384 routed experts, top-6",
        precision_profiles=(
            PrecisionProfile(
                key="megamoe_fp4",
                moe_quant_mode="w4a8_mxfp4_mxfp8",
                moe_kernel="dsv4_megamoe_module_perf",
                evidence="same-shape measured MegaMoE module; utilization-hold above measured token range",
                exact_shape_data=True,
                backend_family="megamoe",
                primary=True,
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="megamoe",
            ),
            PrecisionProfile(
                key="deepep_deepgemm_fp4",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="deepep_deepgemm",
                evidence="exact measured DeepEP dispatch/combine plus DeepGEMM profile only",
                exact_shape_data=False,
                backend_family="deepep_deepgemm",
                measured_moe_precision="w4a8_mxfp4_mxfp8",
                measured_moe_backend="deepep_deepgemm",
            ),
            PrecisionProfile(
                key="trtllm_fp4_control",
                moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
                moe_kernel="sglang_mxfp4_flashinfer_trtllm_moe",
                evidence="generic AIC SGLang FlashInfer/TensorRT-LLM MoE control",
                exact_shape_data=True,
                backend_family="trtllm",
            ),
        ),
        scenarios=(
            NO_MTP,
            Scenario(
                "mtp_n2_r70",
                2,
                expected_accepted(2, 0.70),
                "sensitivity only: conditional acceptance=70%",
                primary=True,
            ),
        ),
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}
MoEStageTiming = AFDMoEStageMeasurement | AFDMoEStageProjection


def supports_workload(spec: ModelSpec, workload: str) -> bool:
    """Return whether prompt plus generated tokens stay inside the model contract."""

    values = WORKLOADS[workload]
    return int(values["isl"]) + int(values["osl"]) <= spec.max_sequence_length


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def precision_for(model_key: str, precision_key: str) -> PrecisionProfile:
    return next(profile for profile in MODEL_BY_KEY[model_key].precision_profiles if profile.key == precision_key)


def scenario_for(model_key: str, scenario_name: str) -> Scenario:
    return next(scenario for scenario in MODEL_BY_KEY[model_key].scenarios if scenario.name == scenario_name)


@cache
def measured_profile(path: str) -> AFDMoEStageProfile:
    return AFDMoEStageProfile.load(path)


def measured_stage(
    profile_path: str | None,
    *,
    spec: ModelSpec,
    precision: PrecisionProfile,
    scenario: Scenario,
    stage: str,
    topology: str,
    logical_batch_per_source_rank: int,
    microbatches: int,
    profile_policy: str = "exact",
    profile_source_system: str | None = None,
    profile_latency_scale: float | None = None,
    system: str = DEFAULT_SYSTEM,
) -> tuple[AFDMoEStageKey | None, MoEStageTiming | None]:
    if profile_path is None or precision.measured_moe_precision is None or precision.measured_moe_backend is None:
        return None, None
    key = AFDMoEStageKey(
        model_path=spec.model_path,
        system=system,
        stage=stage,
        topology=topology,
        logical_batch_per_source_rank=logical_batch_per_source_rank,
        mtp_nextn=scenario.nextn,
        microbatches=microbatches,
        moe_layers=spec.moe_layers,
        routed_topk=spec.topk,
        moe_precision=precision.measured_moe_precision,
        moe_backend=precision.measured_moe_backend,
    )
    profile = measured_profile(profile_path)
    exact = profile.find(key)
    if exact is not None or profile_policy == "exact":
        return key, exact
    if profile_policy != "load-interpolate":
        raise ValueError(f"unsupported MoE profile policy: {profile_policy!r}")
    source_system = profile_source_system or system
    if source_system != system and profile_latency_scale is None:
        raise ValueError("cross-system MoE load projection requires an explicit profile latency scale")
    return key, profile.project_by_f_rank_load(
        key,
        source_system=source_system,
        latency_scale=1.0 if profile_latency_scale is None else profile_latency_scale,
    )


def unmeasured_moe_fraction(spec: ModelSpec, scenario: Scenario) -> float:
    """Fraction of AIC's MoE proxy work not covered by the measured stage.

    The measured stage executes ``q * moe_layers`` expert layers, where
    ``q=nextn+1``. AIC additionally approximates non-MoE decoder layers with
    the model's MoE op and adds ``nextn`` auxiliary layer-equivalents.
    """

    total_equivalents = scenario.verification_width * spec.layers + scenario.nextn
    measured_equivalents = scenario.verification_width * spec.moe_layers
    return max(total_equivalents - measured_equivalents, 0) / total_equivalents


def measurement_record(
    key: AFDMoEStageKey | None,
    measurement: MoEStageTiming | None,
    *,
    profile_path: str | None,
    generic_residual_ms: float = 0.0,
) -> dict[str, Any]:
    if key is None:
        return {"requested": False, "used": False, "reason": "precision has no measured profile contract"}
    logical_load, routed_load = f_rank_loads(key)
    load_contract = {
        "logical_tokens_per_f_rank_per_microbatch": logical_load,
        "routed_topk": key.routed_topk,
        "routed_assignments_per_f_rank_per_microbatch": routed_load,
        "formula": (
            "logical_batch_per_source_rank * (mtp_nextn + 1) * topology_factor / microbatches "
            "* routed_topk; topology_factor=A/F for AFD and 1 for AGG"
        ),
    }
    if measurement is None:
        return {
            "requested": True,
            "used": False,
            "reason": "no exact key",
            "profile": profile_path,
            "key": asdict(key),
            "load_contract": load_contract,
        }
    if isinstance(measurement, AFDMoEStageProjection):
        return {
            "requested": True,
            "used": True,
            "timing_source": "load-interpolated-profile",
            "profile": profile_path,
            "key": asdict(key),
            "load_contract": load_contract,
            "projected_latency_ms": measurement.latency_ms,
            "source_system": measurement.source_system,
            "source_topology": measurement.source_topology,
            "target_logical_tokens_per_f_rank_per_microbatch": (
                measurement.target_logical_tokens_per_f_rank_per_microbatch
            ),
            "target_routed_assignments_per_f_rank_per_microbatch": (
                measurement.target_routed_assignments_per_f_rank_per_microbatch
            ),
            "latency_scale": measurement.latency_scale,
            "generic_residual_ms": generic_residual_ms,
            "evidence": measurement.evidence,
            "anchors": [
                {
                    "routed_assignments_per_f_rank_per_microbatch": load,
                    "logical_batch_per_source_rank": anchor.key.logical_batch_per_source_rank,
                    "latency_ms": anchor.latency_ms,
                    "source_commit": anchor.source_commit,
                    "source_tree_sha256": anchor.source_tree_sha256,
                    "source_result": anchor.source_result,
                }
                for load, anchor in (
                    (measurement.lower_anchor_load, measurement.lower_anchor),
                    (measurement.upper_anchor_load, measurement.upper_anchor),
                )
            ],
        }
    return {
        "requested": True,
        "used": True,
        "timing_source": "exact-measured-profile",
        "profile": profile_path,
        "key": asdict(key),
        "load_contract": load_contract,
        "measured_latency_ms": measurement.latency_ms,
        "generic_residual_ms": generic_residual_ms,
        "source_commit": measurement.source_commit,
        "source_tree_sha256": measurement.source_tree_sha256,
        "source_result": measurement.source_result,
        "evidence": measurement.evidence,
        "matched_speedup": measurement.matched_speedup,
        "matched_speedup_lower_bound": measurement.matched_speedup_lower_bound,
    }


def measured_backend_label(measurement: MoEStageTiming) -> str:
    prefix = "projected" if isinstance(measurement, AFDMoEStageProjection) else "measured"
    return f"{prefix}-{measurement.key.moe_backend.replace('_', '-')}"


def moe_backend_contract(
    spec: ModelSpec,
    precision: PrecisionProfile,
    measurement: MoEStageTiming | None,
) -> dict[str, str]:
    if measurement is not None:
        return {
            "moe_backend": measured_backend_label(measurement),
            "moe_time_source": (
                "load-interpolated-profile"
                if isinstance(measurement, AFDMoEStageProjection)
                else "exact-measured-profile"
            ),
            "moe_kernel": measurement.key.moe_backend,
        }

    control = precision
    if precision.measured_moe_backend is not None:
        matches = tuple(
            candidate
            for candidate in spec.precision_profiles
            if candidate.backend_family == "trtllm" and candidate.moe_quant_mode == precision.moe_quant_mode
        )
        if len(matches) != 1:
            raise ValueError(
                f"{spec.key}/{precision.key} requires one generic TRT-LLM control for "
                f"{precision.moe_quant_mode}, found {len(matches)}"
            )
        control = matches[0]
    return {
        "moe_backend": "generic-trtllm",
        "moe_time_source": "aic-database",
        "moe_kernel": control.moe_kernel,
    }


def overlap_contract(
    system_kind: str,
    *,
    measurement: MoEStageTiming | None,
    microbatches: int = 1,
) -> dict[str, Any]:
    """Describe exactly where overlap is accounted for in one result row."""

    measured_stage = measurement is not None
    if system_kind == "agg":
        return {
            "outer_pipeline": "none-colocated",
            "a_f_compute_overlap": False,
            "backend_internal_overlap": (
                "included-in-complete-measured-moe-stage"
                if measured_stage
                else "embedded-in-aic-operation-latencies"
            ),
            "communication_accounting": (
                "quant-dispatch-expert-combine-included-once-in-measured-stage"
                if measured_stage
                else "generic-aic-colocated-operation-graph"
            ),
            "fully_hidden_comm_assumed": False,
        }
    if system_kind != "afd":
        raise ValueError(f"unsupported overlap-contract system kind: {system_kind!r}")
    return {
        "outer_pipeline": (
            "serial-for-one-microbatch"
            if microbatches < 2
            else "conservative-k2-max(a+a2f,f+f2a)"
        ),
        "a_f_compute_overlap": microbatches >= 2,
        "backend_internal_overlap": (
            "included-in-complete-measured-split-stage"
            if measured_stage
            else "generic-compute-and-communication-terms"
        ),
        "communication_accounting": (
            "dispatch-transfer-combine-in-measured-stage; generic-comm-zeroed-to-avoid-double-counting"
            if measured_stage
            else "a2f/f2a-explicit; f-allgather/reducescatter-and-a-combine-folded-into-pipeline-branches"
        ),
        "fully_hidden_comm_assumed": False,
        "comm_hidden_flag_note": (
            "false under the conservative K=2 model; this does not disable A/F overlap"
        ),
    }


def op_group(name: str) -> str:
    value = name.lower()
    if "attention" in value:
        return "attention"
    if "mhc" in value:
        return "mHC"
    if "router" in value:
        return "router"
    if "allgather" in value or "reducescatter" in value:
        return "F collective"
    if "combine" in value:
        return "A combine"
    if "transfer" in value:
        return "A-F transfer"
    if "moe" in value or "expert" in value:
        return "MoE / shared expert"
    if "gemm" in value:
        return "dense GEMM"
    return "norm / embedding / logits"


def operation_rows(side: str, values: dict[str, float], multiplier: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "side": side,
            "op": name,
            "group": op_group(name),
            "raw_work_ms": float(latency) * multiplier,
        }
        for name, latency in values.items()
    ]


def average_overlap_parts(model, database, runtime_config, op) -> tuple[float, float]:
    """Return serialized router and expert-only overlap latency per decode step."""

    groups = (getattr(op, "_group_a", ()), getattr(op, "_group_b", ()))
    if not any(groups):
        return 0.0, 0.0
    sequence_batch = int(runtime_config.batch_size)
    verification_width = int(getattr(model, "_nextn", 0) or 0) + 1
    token_batch = sequence_batch * verification_width
    beam_width = int(runtime_config.beam_width)
    decode_steps = max(int(runtime_config.osl or 1) - 1, 1)
    router_total = 0.0
    expert_total = 0.0
    repeat_total = 0
    for offset in range(0, decode_steps, DECODE_STRIDE):
        repeat = min(DECODE_STRIDE, decode_steps - offset)
        group_router: list[float] = []
        group_expert: list[float] = []
        for group in groups:
            router_ms = 0.0
            expert_ms = 0.0
            for inner in group:
                result = inner.query(
                    database,
                    x=token_batch * beam_width,
                    batch_size=token_batch,
                    beam_width=beam_width,
                    s=int(runtime_config.isl) + offset + 1,
                    gen_seq_imbalance_correction_scale=(runtime_config.gen_seq_imbalance_correction_scale),
                )
                if "router" in inner._name.lower():
                    router_ms += float(result)
                else:
                    expert_ms += float(result)
            group_router.append(router_ms)
            group_expert.append(expert_ms)
        router_total += sum(group_router) * repeat
        expert_total += max(group_expert, default=0.0) * repeat
        repeat_total += repeat
    denominator = max(repeat_total, 1)
    return router_total / denominator, expert_total / denominator


def replace_agg_moe_stage(
    generation: dict[str, float],
    sources: dict[str, str],
    *,
    model,
    database,
    runtime_config,
    spec: ModelSpec,
    scenario: Scenario,
    measurement: AFDMoEStageMeasurement,
) -> tuple[float, float]:
    """Replace generic expert work while retaining router and uncovered work."""

    moe_names = [name for name in generation if op_group(name) == "MoE / shared expert"]
    if not moe_names:
        raise RuntimeError("measured MoE profile matched, but the AGG graph has no MoE operation")
    generic_expert_ms = 0.0
    nested_router_ms = 0.0
    ops_by_name = {op._name: op for op in model.generation_ops}
    for name in moe_names:
        op = ops_by_name.get(name)
        if op is not None and (hasattr(op, "_group_a") or hasattr(op, "_group_b")):
            router_ms, expert_ms = average_overlap_parts(model, database, runtime_config, op)
            nested_router_ms += router_ms
            generic_expert_ms += expert_ms
        else:
            generic_expert_ms += generation[name]
        del generation[name]
        sources.pop(name, None)

    generic_residual_ms = generic_expert_ms * unmeasured_moe_fraction(spec, scenario)
    generation["generation_measured_moe_stage"] = measurement.latency_ms + generic_residual_ms
    sources["generation_measured_moe_stage"] = "measured-profile"
    if nested_router_ms:
        generation["generation_measured_moe_router"] = nested_router_ms
        sources["generation_measured_moe_router"] = "aic-generic-router"
    return generic_expert_ms, generic_residual_ms


def generic_afd_residual(
    summary,
    *,
    f_model,
    database,
    runtime_config,
    spec: ModelSpec,
    scenario: Scenario,
    microbatches: int,
) -> tuple[float, dict[str, float]]:
    """Compute the AIC work outside an exact measured AFD MoE boundary."""

    per_ops = summary.get_per_ops_data() or {}
    f_ops = {name: float(value) for name, value in per_ops.get("decode_f_worker", {}).items()}
    a_ops = {name: float(value) for name, value in per_ops.get("decode_a_worker", {}).items()}
    ops_by_name = {op._name: op for op in f_model.generation_ops}
    generic_expert_ms = 0.0
    for name, value in f_ops.items():
        if op_group(name) != "MoE / shared expert":
            continue
        op = ops_by_name.get(name)
        if op is not None and (hasattr(op, "_group_a") or hasattr(op, "_group_b")):
            _router_ms, expert_ms = average_overlap_parts(f_model, database, runtime_config, op)
            generic_expert_ms += expert_ms
        else:
            generic_expert_ms += value

    expert_residual_ms = generic_expert_ms * microbatches * unmeasured_moe_fraction(spec, scenario)
    f_collective_ms = sum(value for name, value in f_ops.items() if op_group(name) == "F collective")
    a_combine_ms = sum(value for name, value in a_ops.items() if op_group(name) == "A combine")
    raw = dict(summary.get_result_dict() or {})
    transfer_ms = (float(raw["decode_t_a2f_layer"]) + float(raw["decode_t_f2a_layer"])) * spec.layers
    generic_comm_ms = (f_collective_ms + a_combine_ms + transfer_ms) * microbatches
    uncovered_layer_fraction = max(spec.layers - spec.moe_layers, 0) / spec.layers
    comm_residual_ms = generic_comm_ms * uncovered_layer_fraction
    components = {
        "generic_expert_proxy_ms": generic_expert_ms * microbatches,
        "expert_residual_ms": expert_residual_ms,
        "generic_stage_comm_ms": generic_comm_ms,
        "comm_residual_ms": comm_residual_ms,
    }
    return expert_residual_ms + comm_residual_ms, components


@cache
def task_for(
    model_key: str,
    workload: str,
    scenario_name: str,
    precision_key: str,
    system: str = DEFAULT_SYSTEM,
) -> Task:
    spec = MODEL_BY_KEY[model_key]
    scenario = scenario_for(model_key, scenario_name)
    precision = precision_for(model_key, precision_key)
    workload_spec = WORKLOADS[workload]
    return Task(
        serving_mode="agg",
        model_path=spec.model_path,
        system_name=system,
        backend_name=BACKEND,
        backend_version=spec.backend_version,
        database_mode=DATABASE_MODE,
        isl=int(workload_spec["isl"]),
        osl=int(workload_spec["osl"]),
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


@cache
def database_for(
    model_key: str,
    workload: str,
    scenario_name: str,
    precision_key: str,
    system: str = DEFAULT_SYSTEM,
):
    task = task_for(model_key, workload, scenario_name, precision_key, system)
    return task._load_database(system, BACKEND, task.backend_version)


def configured_model(task: Task, *, tp: int, dp: int, moe_tp: int, moe_ep: int):
    model_config = task.build_model_config(role="agg")
    model_config.tp_size = tp
    model_config.pp_size = 1
    model_config.attention_dp_size = dp
    model_config.moe_tp_size = moe_tp
    model_config.moe_ep_size = moe_ep
    return model_config, get_model(task.model_path, model_config, task.backend_name)


def agg_source_batch_per_rank(local_batch: int, attention_tp: int) -> int | None:
    """Map per-DP-replica AGG batch to a uniform MoE source-rank batch."""

    if local_batch % attention_tp:
        return None
    return local_batch // attention_tp


@cache
def agg_point(
    model_key: str,
    workload: str,
    scenario_name: str,
    precision_key: str,
    world: int,
    tp: int,
    local_batch: int,
    measured_profile_path: str | None,
    system: str = DEFAULT_SYSTEM,
    profile_policy: str = "exact",
    profile_source_system: str | None = None,
    profile_latency_scale: float | None = None,
) -> dict[str, Any]:
    task = task_for(model_key, workload, scenario_name, precision_key, system)
    scenario = scenario_for(model_key, scenario_name)
    precision = precision_for(model_key, precision_key)
    dp = world // tp
    model_config, model = configured_model(task, tp=tp, dp=dp, moe_tp=1, moe_ep=world)
    session = InferenceSession(
        model,
        database_for(model_key, workload, scenario_name, precision_key, system),
        get_backend(BACKEND),
    )
    runtime_config = task.build_runtime_config(batch_size=local_batch)
    summary = session.run_static(
        runtime_config,
        mode="static_gen",
        stride=DECODE_STRIDE,
    )
    decode_steps = max(int(WORKLOADS[workload]["osl"]) - 1, 1)
    generation = {name: float(value) / decode_steps for name, value in summary.get_generation_latency_dict().items()}
    sources = dict(summary.get_generation_source_dict())
    measured_key = None
    measurement = None
    source_batch_per_rank = agg_source_batch_per_rank(local_batch, tp)
    if source_batch_per_rank is not None:
        measured_key, measurement = measured_stage(
            measured_profile_path,
            spec=MODEL_BY_KEY[model_key],
            precision=precision,
            scenario=scenario,
            stage="agg",
            topology=f"ep{world}",
            logical_batch_per_source_rank=source_batch_per_rank,
            microbatches=1,
            profile_policy=profile_policy,
            profile_source_system=profile_source_system,
            profile_latency_scale=profile_latency_scale,
            system=system,
        )
    generic_moe_ms = 0.0
    generic_residual_ms = 0.0
    if measurement is not None:
        generic_moe_ms, generic_residual_ms = replace_agg_moe_stage(
            generation,
            sources,
            model=model,
            database=database_for(model_key, workload, scenario_name, precision_key, system),
            runtime_config=runtime_config,
            spec=MODEL_BY_KEY[model_key],
            scenario=scenario,
            measurement=measurement,
        )
    raw_round_ms = sum(generation.values())
    result = summary.get_result_dict() or {}
    global_batch = local_batch * dp
    moe_measurement = measurement_record(
        measured_key,
        measurement,
        profile_path=measured_profile_path,
        generic_residual_ms=generic_residual_ms,
    ) | {"generic_expert_proxy_ms": generic_moe_ms}
    if (
        measured_profile_path is not None
        and precision.measured_moe_precision is not None
        and precision.measured_moe_backend is not None
        and source_batch_per_rank is None
    ):
        moe_measurement = {
            "requested": True,
            "used": False,
            "reason": "agg local batch is not divisible by attention TP; no exact uniform source-rank batch",
            "profile": measured_profile_path,
            "local_batch_per_attention_dp_replica": local_batch,
            "attention_tp": tp,
        }
    return {
        "system_kind": "agg",
        "model": model_key,
        "workload": workload,
        "scenario": scenario_name,
        "precision_profile": precision_key,
        "world": world,
        "tp": tp,
        "dp": dp,
        "moe_tp": 1,
        "moe_ep": world,
        "local_batch": local_batch,
        "moe_source_batch_per_rank": source_batch_per_rank,
        "global_batch_per_replica": global_batch,
        "raw_round_ms": raw_round_ms,
        "effective_tpot_ms": raw_round_ms / scenario.progress,
        "tokps_per_user": scenario.progress * 1000.0 / raw_round_ms,
        "output_tokens_s_replica": global_batch * scenario.progress * 1000.0 / raw_round_ms,
        "memory_gb": float(result.get("memory", math.nan)),
        "oom": bool(summary.check_oom() or summary.check_kv_cache_oom()),
        "modules": operation_rows("AGG", generation),
        "op_sources": sources,
        "moe_measurement": moe_measurement,
        "overlap_contract": overlap_contract("agg", measurement=measurement),
        "backend_contract": {
            "framework": f"SGLang {MODEL_BY_KEY[model_key].backend_version}",
            **moe_backend_contract(MODEL_BY_KEY[model_key], precision, measurement),
            "moe_precision": model_config.moe_quant_mode.name,
            "attention_backend": MODEL_BY_KEY[model_key].attention_backend,
        },
        "quant": {
            "gemm": model_config.gemm_quant_mode.name,
            "moe": model_config.moe_quant_mode.name,
            "kvcache": model_config.kvcache_quant_mode.name,
            "fmha": model_config.fmha_quant_mode.name,
        },
    }


def agg_cluster_rows(
    spec: ModelSpec,
    workload: str,
    scenario: Scenario,
    precision: PrecisionProfile,
    total_gpus: int,
    measured_profile_path: str | None,
    *,
    system: str = DEFAULT_SYSTEM,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    profile_policy: str = "exact",
    profile_source_system: str | None = None,
    profile_latency_scale: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for world in range(gpus_per_node, total_gpus + 1, gpus_per_node):
        replicas = total_gpus // world
        if replicas <= 0:
            continue
        used_gpus = replicas * world
        for tp in STATIC_TPS:
            if world % tp:
                continue
            for local_batch in WORKLOADS[workload]["agg_local_batch"]:
                try:
                    point = agg_point(
                        spec.key,
                        workload,
                        scenario.name,
                        precision.key,
                        world,
                        tp,
                        int(local_batch),
                        measured_profile_path,
                        system=system,
                        profile_policy=profile_policy,
                        profile_source_system=profile_source_system,
                        profile_latency_scale=profile_latency_scale,
                    )
                    if point["oom"]:
                        continue
                    cluster_output = replicas * point["output_tokens_s_replica"]
                    rows.append(
                        {
                            **point,
                            "total_gpus": total_gpus,
                            "used_gpus": used_gpus,
                            "idle_gpus": total_gpus - used_gpus,
                            "replicas": replicas,
                            "cluster_concurrency": replicas * point["global_batch_per_replica"],
                            "output_tokens_s": cluster_output,
                            "output_tokens_s_gpu": cluster_output / total_gpus,
                        }
                    )
                except Exception as error:
                    failures.append(
                        {
                            "system_kind": "agg",
                            "model": spec.key,
                            "workload": workload,
                            "scenario": scenario.name,
                            "precision_profile": precision.key,
                            "total_gpus": total_gpus,
                            "world": world,
                            "tp": tp,
                            "local_batch": local_batch,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    return rows, failures


def afd_point(
    spec: ModelSpec,
    workload: str,
    scenario: Scenario,
    precision: PrecisionProfile,
    *,
    total_gpus: int,
    a_nodes: int,
    f_nodes: int,
    a_tp: int,
    batch_per_a_gpu: int,
    microbatches: int,
    measured_profile_path: str | None,
    system: str = DEFAULT_SYSTEM,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    profile_policy: str = "exact",
    profile_source_system: str | None = None,
    profile_latency_scale: float | None = None,
) -> dict[str, Any]:
    task = task_for(spec.key, workload, scenario.name, precision.key, system)
    database = database_for(spec.key, workload, scenario.name, precision.key, system)
    base_config = task.build_model_config(role="agg")
    a_config = copy.deepcopy(base_config)
    a_config.tp_size = a_tp
    a_config.pp_size = 1
    a_config.attention_dp_size = 1
    if spec.moe_backend == "megamoe":
        a_config.moe_backend = None
    a_config.moe_tp_size = a_tp
    a_config.moe_ep_size = 1

    a_gpus = a_nodes * gpus_per_node
    f_gpus = f_nodes * gpus_per_node
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
        gpus_per_node=gpus_per_node,
        tp_a=a_tp,
        f_moe_ep_size=f_gpus,
        a_batch_size=a_batch_size,
        num_microbatches=microbatches,
        pipeline_model=PIPELINE_MODEL,
        phase="decode",
        combined_with_pd=False,
    )
    global_requests = a_gpus * batch_per_a_gpu
    measured_key, measurement = measured_stage(
        measured_profile_path,
        spec=spec,
        precision=precision,
        scenario=scenario,
        stage="afd",
        topology=f"{a_gpus}A{f_gpus}F",
        logical_batch_per_source_rank=batch_per_a_gpu,
        microbatches=microbatches,
        profile_policy=profile_policy,
        profile_source_system=profile_source_system,
        profile_latency_scale=profile_latency_scale,
        system=system,
    )
    runtime_config = task.build_runtime_config(batch_size=global_requests)
    speculative_profile = SpeculativeDecodingProfile.from_inputs(scenario.nextn, scenario.accepted_drafts)

    def simulate(afd_moe_time_ms: float | None):
        return AFDInferenceSession(
            model_path=spec.model_path,
            a_model_config=a_config,
            f_model_config=f_config,
            database=database,
            backend=get_backend(BACKEND),
            afd_config=afd_config,
            afd_moe_time_ms=afd_moe_time_ms,
            decode_stride=DECODE_STRIDE,
        ).run_afd(
            runtime_config,
            phase="decode",
            speculative_profile=speculative_profile,
        )

    generic_residual_ms = 0.0
    residual_components: dict[str, float] = {}
    measured_stage_ms = None
    if measurement is None:
        summary = simulate(None)
    else:
        generic_summary = simulate(None)
        if generic_summary.check_oom() or generic_summary.check_kv_cache_oom():
            raise RuntimeError("OOM")
        a_workers = a_gpus // a_tp
        a_micro_batch_size = math.ceil(a_batch_size / microbatches)
        f_micro_batch_size = a_workers * a_micro_batch_size
        f_model = get_model(spec.model_path, f_config, BACKEND)
        generic_residual_ms, residual_components = generic_afd_residual(
            generic_summary,
            f_model=f_model,
            database=database,
            runtime_config=task.build_runtime_config(batch_size=f_micro_batch_size),
            spec=spec,
            scenario=scenario,
            microbatches=microbatches,
        )
        measured_stage_ms = measurement.latency_ms + generic_residual_ms
        summary = simulate(measured_stage_ms)
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
    output_tokens_s = global_requests * scenario.progress * 1000.0 / raw_round_ms
    t_a_layer = float(raw["decode_t_a_layer"])
    t_f_layer = float(raw["decode_t_f_layer"])
    t_a2f_layer = float(raw["decode_t_a2f_layer"])
    t_f2a_layer = float(raw["decode_t_f2a_layer"])
    pipeline_fill_ms = t_a_layer + t_f_layer + t_a2f_layer + t_f2a_layer
    t_cycle_layer = pipeline_fill_ms if microbatches < 2 else max(t_a_layer + t_a2f_layer, t_f_layer + t_f2a_layer)
    modules = (
        operation_rows("A", a_ops, microbatches)
        + operation_rows("F", f_ops, microbatches)
        + operation_rows("fabric", comm_ops)
    )
    return {
        "system_kind": "afd",
        "model": spec.key,
        "workload": workload,
        "scenario": scenario.name,
        "precision_profile": precision.key,
        "total_gpus": total_gpus,
        "a_nodes": a_nodes,
        "f_nodes": f_nodes,
        "a_gpus": a_gpus,
        "f_gpus": f_gpus,
        "a_tp": a_tp,
        "f_ep": f_gpus,
        "batch_per_a_gpu": batch_per_a_gpu,
        "a_batch_size_per_worker": a_batch_size,
        "microbatches": microbatches,
        "global_requests": global_requests,
        "raw_round_ms": raw_round_ms,
        "effective_tpot_ms": raw_round_ms / scenario.progress,
        "tokps_per_user": scenario.progress * 1000.0 / raw_round_ms,
        "output_tokens_s": output_tokens_s,
        "output_tokens_s_gpu": output_tokens_s / total_gpus,
        "a_raw_work_ms": sum(a_ops.values()) * microbatches,
        "f_raw_work_ms": sum(f_ops.values()) * microbatches,
        "t_a_layer_ms": t_a_layer,
        "t_f_layer_ms": t_f_layer,
        "t_a2f_layer_ms": t_a2f_layer,
        "t_f2a_layer_ms": t_f2a_layer,
        "t_cycle_layer_ms": t_cycle_layer,
        "pipeline_fill_ms": pipeline_fill_ms,
        "pipeline_formula_ms": pipeline_fill_ms + t_cycle_layer * (microbatches * spec.layers - 1),
        "pipeline_bottleneck": "A" if t_a_layer + t_a2f_layer >= t_f_layer + t_f2a_layer else "F",
        "comm_hidden": bool(raw.get("decode_comm_hidden", False)),
        "overlap_contract": overlap_contract(
            "afd",
            measurement=measurement,
            microbatches=microbatches,
        ),
        "a_memory_gb": float(raw["(a)memory"]),
        "f_memory_gb": float(raw["(f)memory"]),
        "modules": modules,
        "moe_measurement": measurement_record(
            measured_key,
            measurement,
            profile_path=measured_profile_path,
            generic_residual_ms=generic_residual_ms,
        )
        | {
            "injected_stage_ms": measured_stage_ms,
            "residual_components": residual_components,
        },
        "backend_contract": {
            "framework": f"SGLang {spec.backend_version}",
            **moe_backend_contract(spec, precision, measurement),
            "moe_precision": f_config.moe_quant_mode.name,
            "attention_backend": spec.attention_backend,
        },
        "quant": {
            "a_gemm": a_config.gemm_quant_mode.name,
            "a_fmha": a_config.fmha_quant_mode.name,
            "a_kvcache": a_config.kvcache_quant_mode.name,
            "f_moe": f_config.moe_quant_mode.name,
        },
    }


def afd_cluster_rows(
    spec: ModelSpec,
    workload: str,
    scenario: Scenario,
    precision: PrecisionProfile,
    total_gpus: int,
    measured_profile_path: str | None,
    *,
    system: str = DEFAULT_SYSTEM,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    profile_policy: str = "exact",
    profile_source_system: str | None = None,
    profile_latency_scale: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_nodes = total_gpus // gpus_per_node
    for f_nodes in range(1, total_nodes):
        a_nodes = total_nodes - f_nodes
        a_gpus = a_nodes * gpus_per_node
        f_gpus = f_nodes * gpus_per_node
        if spec.attention_heads % f_gpus:
            continue
        for a_tp in A_TPS:
            if a_gpus % a_tp:
                continue
            for batch_per_a_gpu in WORKLOADS[workload]["afd_batch_per_a_gpu"]:
                for microbatches in MICROBATCHES:
                    try:
                        rows.append(
                            afd_point(
                                spec,
                                workload,
                                scenario,
                                precision,
                                total_gpus=total_gpus,
                                a_nodes=a_nodes,
                                f_nodes=f_nodes,
                                a_tp=a_tp,
                                batch_per_a_gpu=int(batch_per_a_gpu),
                                microbatches=microbatches,
                                measured_profile_path=measured_profile_path,
                                system=system,
                                gpus_per_node=gpus_per_node,
                                profile_policy=profile_policy,
                                profile_source_system=profile_source_system,
                                profile_latency_scale=profile_latency_scale,
                            )
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "system_kind": "afd",
                                "model": spec.key,
                                "workload": workload,
                                "scenario": scenario.name,
                                "precision_profile": precision.key,
                                "total_gpus": total_gpus,
                                "a_nodes": a_nodes,
                                "f_nodes": f_nodes,
                                "a_tp": a_tp,
                                "batch_per_a_gpu": batch_per_a_gpu,
                                "microbatches": microbatches,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
    return rows, failures


def selected_profiles(
    spec: ModelSpec,
    profile_scope: str,
    backend_families: set[str] | None = None,
) -> tuple[PrecisionProfile, ...]:
    profiles = (
        spec.precision_profiles
        if profile_scope == "all"
        else tuple(profile for profile in spec.precision_profiles if profile.primary)
    )
    if backend_families is None:
        return profiles
    return tuple(profile for profile in profiles if profile.backend_family in backend_families)


def afd_service_unit_grid(
    fixed_pool_sizes: tuple[int, ...],
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
) -> tuple[int, ...]:
    """Return every node-aligned AFD unit that can fit a selected fixed pool."""

    if not fixed_pool_sizes:
        return ()
    return tuple(range(2 * gpus_per_node, max(fixed_pool_sizes) + 1, gpus_per_node))


def system_gpus_per_node(system: str) -> int:
    """Return the node width from the selected AIC system specification."""

    return int(load_system_spec(system)["node"]["num_gpus_per_node"])


def require_profile_system(profile: AFDMoEStageProfile, system: str) -> None:
    """Reject a measured-only run when the profile targets other hardware."""

    available = sorted({entry.key.system for entry in profile.entries})
    if system not in available:
        raise ValueError(f"measured MoE profile has systems {available}, but this sweep requires {system!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    system = getattr(args, "system", DEFAULT_SYSTEM)
    gpus_per_node = system_gpus_per_node(system)
    profile_policy = getattr(args, "moe_profile_policy", "exact")
    profile_source_system = getattr(args, "moe_profile_source_system", None)
    profile_latency_scale = getattr(args, "moe_profile_latency_scale", None)
    require_profiled_moe = bool(
        getattr(args, "require_measured_moe", False) or getattr(args, "require_profiled_moe", False)
    )
    selected_models = [MODEL_BY_KEY[key] for key in args.models]
    selected_workloads = list(args.workloads)
    selected_totals = tuple(sorted(set(args.total_gpus)))
    selected_backend_families = None if not getattr(args, "moe_backends", None) else set(args.moe_backends)
    measured_profile_path = str(args.afd_moe_profile.resolve()) if args.afd_moe_profile is not None else None
    if measured_profile_path is not None:
        profile = measured_profile(measured_profile_path)
        if args.require_measured_moe:
            require_profile_system(profile, system)
        if (
            profile_policy == "load-interpolate"
            and profile_source_system is not None
            and profile_source_system != system
            and profile_latency_scale is None
        ):
            raise ValueError("cross-system MoE load projection requires --moe-profile-latency-scale")
    agg_rows: list[dict[str, Any]] = []
    afd_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    groups = [
        (spec, workload, scenario, precision, fixed_pool_sizes)
        for spec in selected_models
        for workload in selected_workloads
        if supports_workload(spec, workload)
        for scenario in spec.scenarios
        for precision in selected_profiles(spec, args.profile_scope, selected_backend_families)
        if (
            fixed_pool_sizes := tuple(
                total_gpus
                for total_gpus in selected_totals
                if total_gpus in precision.total_gpu_grid and total_gpus in scenario.total_gpu_grid
            )
        )
    ]
    for index, (spec, workload, scenario, precision, fixed_pool_sizes) in enumerate(groups, start=1):
        print(
            f"[{index}/{len(groups)}] {spec.key} {workload} {scenario.name} {precision.key} "
            f"fixed pools={fixed_pool_sizes}",
            flush=True,
        )
        new_agg: list[dict[str, Any]] = []
        agg_failures: list[dict[str, Any]] = []
        for fixed_pool_size in fixed_pool_sizes:
            rows, row_failures = agg_cluster_rows(
                spec,
                workload,
                scenario,
                precision,
                fixed_pool_size,
                measured_profile_path,
                system=system,
                gpus_per_node=gpus_per_node,
                profile_policy=profile_policy,
                profile_source_system=profile_source_system,
                profile_latency_scale=profile_latency_scale,
            )
            new_agg.extend(rows)
            agg_failures.extend(row_failures)

        new_afd: list[dict[str, Any]] = []
        afd_failures: list[dict[str, Any]] = []
        for unit_gpus in afd_service_unit_grid(fixed_pool_sizes, gpus_per_node):
            rows, row_failures = afd_cluster_rows(
                spec,
                workload,
                scenario,
                precision,
                unit_gpus,
                measured_profile_path,
                system=system,
                gpus_per_node=gpus_per_node,
                profile_policy=profile_policy,
                profile_source_system=profile_source_system,
                profile_latency_scale=profile_latency_scale,
            )
            new_afd.extend(rows)
            afd_failures.extend(row_failures)
        if require_profiled_moe:
            new_agg = [row for row in new_agg if row["moe_measurement"]["used"]]
            new_afd = [row for row in new_afd if row["moe_measurement"]["used"]]
        agg_rows.extend(new_agg)
        afd_rows.extend(new_afd)
        failures.extend(agg_failures)
        failures.extend(afd_failures)
        checkpoint = {
            "schema": "aic.afd-fixed-pool-sweep.v3.partial",
            "agg_rows": agg_rows,
            "afd_rows": afd_rows,
            "failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
        )

    source_counts = Counter()
    for row in agg_rows:
        source_counts.update(row.get("op_sources", {}).values())
    return {
        "schema": "aic.afd-fixed-pool-sweep.v3",
        "code": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "contract": {
            "system": system,
            "backend": BACKEND,
            "database_mode": DATABASE_MODE,
            "gpus_per_node": gpus_per_node,
            "total_gpu_grid": list(selected_totals),
            "pipeline_model": PIPELINE_MODEL,
            "overlap_accounting": (
                "AGG has no outer A/F pipeline. AFD uses serial execution for one microbatch and the "
                "conservative K=2 max(TA+TA2F, TF+TF2A) cadence for two or more microbatches. "
                "Measured MoE stages already include backend-internal quant/dispatch/compute/combine scheduling; "
                "generic AFD communication terms are zeroed when such a stage is injected."
            ),
            "comm_hidden_semantics": (
                "false in the conservative model means no optimistic fully hidden K=3 communication stage; "
                "it does not mean A/F compute overlap is disabled"
            ),
            "decode_stride": DECODE_STRIDE,
            "a_tp_grid": list(A_TPS),
            "microbatch_grid": list(MICROBATCHES),
            "f_node_rule": "all integer node splits: 1 <= F nodes < total nodes",
            "afd_row_scope": "one AFD service unit; fixed-pool analysis may pack identical units and charges idle GPUs",
            "afd_service_unit_gpu_grid": list(afd_service_unit_grid(selected_totals, gpus_per_node)),
            "agg_world_rule": (
                f"all {gpus_per_node}-GPU node increments; idle remainder is counted in the fixed total-GPU denominator"
            ),
            "static_tp_grid": list(STATIC_TPS),
            "speed_floors_tokps_per_user": list(SPEED_FLOORS),
            "afd_moe_profile": measured_profile_path,
            "moe_backends": (
                sorted(selected_backend_families) if selected_backend_families is not None else "profile-scope"
            ),
            "require_measured_moe": bool(args.require_measured_moe),
            "require_profiled_moe": require_profiled_moe,
            "moe_profile_policy": profile_policy,
            "moe_profile_source_system": profile_source_system,
            "moe_profile_latency_scale": profile_latency_scale,
            "afd_moe_profile_policy": (
                "exact full-key lookup, or explicit within-envelope monotone interpolation by routed expert "
                "assignments per F rank and microbatch; interpolation never extrapolates and cross-system scaling "
                "is explicit"
            ),
            "workload_support": (
                "a workload is swept only when ISL + OSL <= the model configuration's max_sequence_length"
            ),
            "batch_semantics": "batch_per_a_gpu; a_batch_size_per_worker=batch_per_a_gpu*a_tp",
            "mtp_compute": "verification width q=nextn+1; transformer work=q*L+nextn layer-equivalents",
            "mtp_progress": "committed output progress=1+expected accepted draft tokens",
            "comparison": (
                "At each user-speed floor, AGG and AFD are independently optimized under the same fixed-pool, "
                "model, precision, framework, and MoE kernel contract; identical service-unit replicas are allowed "
                "and idle GPUs remain in the denominator."
            ),
            "scope": "decode only; ISL fixed by workload; OSL=1024; no arrival-process model",
        },
        "models": [
            {
                **asdict(spec),
                "scenarios": [
                    asdict(scenario)
                    | {
                        "verification_width": scenario.verification_width,
                        "progress": scenario.progress,
                    }
                    for scenario in spec.scenarios
                ],
            }
            for spec in selected_models
        ],
        "workloads": {key: WORKLOADS[key] for key in selected_workloads},
        "agg_source_counts": dict(source_counts),
        "agg_rows": agg_rows,
        "afd_rows": afd_rows,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system", default=DEFAULT_SYSTEM, help="AIC system database name, for example gb200 or b200_sxm"
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_BY_KEY), default=sorted(MODEL_BY_KEY))
    parser.add_argument("--workloads", nargs="+", choices=sorted(WORKLOADS), default=sorted(WORKLOADS))
    parser.add_argument("--total-gpus", nargs="+", type=int, default=list(TOTAL_GPU_GRID))
    parser.add_argument("--profile-scope", choices=("primary", "all"), default="all")
    parser.add_argument(
        "--moe-backends",
        nargs="+",
        choices=("megamoe", "deepep_deepgemm", "trtllm"),
        help="Optional MoE backend-family filter; use with --profile-scope all for controls",
    )
    parser.add_argument(
        "--afd-moe-profile",
        type=Path,
        help="Optional exact-only measured AFD MoE-stage profile JSON",
    )
    parser.add_argument(
        "--moe-profile-policy",
        choices=("exact", "load-interpolate"),
        default="exact",
        help="Use exact keys only, or interpolate inside a measured per-F-rank load envelope",
    )
    parser.add_argument(
        "--moe-profile-source-system",
        help="Measured source system used by load interpolation; defaults to --system",
    )
    parser.add_argument(
        "--moe-profile-latency-scale",
        type=float,
        help="Explicit positive latency multiplier for a cross-system load projection",
    )
    parser.add_argument(
        "--require-measured-moe",
        action="store_true",
        help="Drop candidates without an exact measured MoE point instead of using generic AIC",
    )
    parser.add_argument(
        "--require-profiled-moe",
        action="store_true",
        help="Drop candidates without an exact or within-envelope profile-derived MoE timing",
    )
    args = parser.parse_args()
    if (args.require_measured_moe or args.require_profiled_moe) and args.afd_moe_profile is None:
        parser.error("--require-measured-moe/--require-profiled-moe requires --afd-moe-profile")
    if args.require_measured_moe and args.moe_profile_policy != "exact":
        parser.error("--require-measured-moe is exact-only; use --require-profiled-moe for interpolation")
    if args.moe_profile_policy == "load-interpolate" and args.afd_moe_profile is None:
        parser.error("--moe-profile-policy load-interpolate requires --afd-moe-profile")
    if args.moe_profile_latency_scale is not None and args.moe_profile_latency_scale <= 0:
        parser.error("--moe-profile-latency-scale must be positive")
    if (
        args.moe_profile_policy == "load-interpolate"
        and args.moe_profile_source_system is not None
        and args.moe_profile_source_system != args.system
        and args.moe_profile_latency_scale is None
    ):
        parser.error("cross-system load interpolation requires --moe-profile-latency-scale")
    try:
        gpus_per_node = system_gpus_per_node(args.system)
    except Exception as error:
        parser.error(f"cannot load system {args.system!r}: {error}")
    invalid = [value for value in args.total_gpus if value < 2 * gpus_per_node]
    if invalid:
        parser.error(f"--total-gpus values must be at least two {gpus_per_node}-GPU nodes: {invalid}")
    return args


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
                "agg_rows": len(payload["agg_rows"]),
                "afd_rows": len(payload["afd_rows"]),
                "failures": len(payload["failures"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
