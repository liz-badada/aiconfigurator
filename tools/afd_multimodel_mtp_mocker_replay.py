#!/usr/bin/env python3
"""Replay fixed-pool AIC winners with Dynamo Mocker."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from render_afd_multimodel_mtp_report import load_payload, primary_mtp, primary_profile, select_best

BLOCK_SIZE = 64
DEFAULT_OUTPUT_TOKENS = 64
DEFAULT_TOTAL_GPUS = (72,)
DEFAULT_WORKLOADS = ("8k", "16k")
CONTEXT_LENGTHS = {"8k": 8192, "16k": 16384}


def equal_conditional_rate(nextn: int, accepted_drafts: float) -> float:
    """Find a constant conditional accept rate with the requested mean progress."""
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        expected = sum(middle**position for position in range(1, nextn + 1))
        if expected < accepted_drafts:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def write_profile(
    path: Path,
    *,
    context: int,
    output_tokens: int,
    local_batch: int,
    raw_round_ms: float,
    metadata: dict[str, Any],
) -> None:
    max_context = context + output_tokens
    np.savez(
        path,
        prefill_isl=np.asarray([0.0, float(local_batch * context)]),
        prefill_ttft_ms=np.asarray([0.0, 1.0]),
        decode_active_kv_tokens=np.asarray([0.0, float(local_batch * max_context)]),
        decode_context_length=np.asarray([float(context), float(max_context)]),
        decode_itl=np.full((2, 2), raw_round_ms, dtype=np.float64),
        afd_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def compact_source(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "system_kind",
        "model",
        "workload",
        "scenario",
        "precision_profile",
        "total_gpus",
        "used_gpus",
        "idle_gpus",
        "world",
        "tp",
        "dp",
        "replicas",
        "local_batch",
        "moe_source_batch_per_rank",
        "global_batch_per_replica",
        "unit_gpus",
        "afd_replicas",
        "a_gpus",
        "f_gpus",
        "a_tp",
        "batch_per_a_gpu",
        "microbatches",
        "cluster_concurrency",
        "raw_round_ms",
        "effective_tpot_ms",
        "output_tokens_s",
        "backend_contract",
    )
    return {key: row[key] for key in keys if key in row}


def worker_shape(row: dict[str, Any]) -> tuple[int, int, int]:
    if row["system_kind"] == "agg":
        return int(row["replicas"]), int(row["global_batch_per_replica"]), int(row["cluster_concurrency"])
    return int(row["afd_replicas"]), int(row["global_requests"]), int(row["cluster_concurrency"])


def replay_case(
    *,
    dynamo: Path,
    output_dir: Path,
    row: dict[str, Any],
    context: int,
    nextn: int,
    accepted_drafts: float | None,
    output_tokens: int,
    waves: int,
) -> dict[str, Any]:
    workers, worker_batch, global_requests = worker_shape(row)
    case_id = f"{row['model']}_{row['workload']}_{row['total_gpus']}gpu_{row['scenario']}_{row['system_kind']}"
    profile = output_dir / f"{case_id}.npz"
    report = output_dir / f"{case_id}.json"
    source = compact_source(row)
    write_profile(
        profile,
        context=context,
        output_tokens=output_tokens,
        local_batch=worker_batch,
        raw_round_ms=float(row["raw_round_ms"]),
        metadata={
            "schema": "aic.afd-fixed-pool-mocker-profile.v2",
            "case_id": case_id,
            "service_contract": "one AIC service unit represented as one virtual worker",
            "source": source,
        },
    )
    blocks_per_sequence = math.ceil((context + output_tokens) / BLOCK_SIZE) + 2
    engine_args: dict[str, Any] = {
        "engine_type": "vllm",
        "num_gpu_blocks": worker_batch * blocks_per_sequence,
        "block_size": BLOCK_SIZE,
        "max_num_seqs": worker_batch,
        "max_num_batched_tokens": worker_batch * context,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "aic_mtp_seed": 42,
        "planner_profile_data": str(profile),
    }
    conditional_rate = None
    if nextn:
        conditional_rate = equal_conditional_rate(nextn, float(accepted_drafts))
        engine_args.update(
            {
                "aic_nextn": nextn,
                "aic_nextn_accept_rates": ",".join([f"{conditional_rate:.12f}"] * nextn),
            }
        )
    command = [
        str(dynamo / ".venv/bin/python"),
        "-m",
        "dynamo.replay",
        "--input-tokens",
        str(context),
        "--output-tokens",
        str(output_tokens),
        "--request-count",
        str(global_requests * waves),
        "--replay-concurrency",
        str(global_requests),
        "--num-workers",
        str(workers),
        "--replay-mode",
        "offline",
        "--router-mode",
        "round_robin",
        "--extra-engine-args",
        json.dumps(engine_args, separators=(",", ":")),
        "--report-json",
        str(report),
    ]
    completed = subprocess.run(
        command,
        cwd=dynamo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Mocker failed for {case_id}:\n{completed.stdout[-8000:]}")
    result = json.loads(report.read_text(encoding="utf-8"))
    expected_tpot_ms = float(row["effective_tpot_ms"])
    expected_output_tps = float(row["output_tokens_s"])
    finite_wave_efficiency = result["output_throughput_tok_s"] / expected_output_tps
    return {
        "case_id": case_id,
        "model": row["model"],
        "workload": row["workload"],
        "scenario": row["scenario"],
        "system_kind": row["system_kind"],
        "total_gpus": row["total_gpus"],
        "nextn": nextn,
        "conditional_rate_surrogate": conditional_rate,
        "accepted_drafts_target": accepted_drafts,
        "global_requests": global_requests,
        "workers": workers,
        "worker_batch": worker_batch,
        "raw_round_ms": row["raw_round_ms"],
        "profile": str(profile),
        "report": str(report),
        "command": command,
        "completed_requests": result["completed_requests"],
        "duration_ms": result["duration_ms"],
        "output_throughput_tok_s": result["output_throughput_tok_s"],
        "expected_output_throughput_tok_s": expected_output_tps,
        "finite_wave_efficiency_vs_aic_steady_state": finite_wave_efficiency,
        "output_throughput_error_pct": (finite_wave_efficiency - 1.0) * 100.0,
        "throughput_comparison_contract": (
            "Mocker output throughput includes finite-wave fill/drain and the stochastic final-wave tail; "
            "the AIC reference is a saturated steady-state rate. Increase --waves to check convergence."
        ),
        "mean_tpot_ms": result["mean_tpot_ms"],
        "expected_steady_state_tpot_ms": expected_tpot_ms,
        "mean_tpot_error_pct": (result["mean_tpot_ms"] / expected_tpot_ms - 1.0) * 100.0,
        "p90_tpot_ms": result["p90_tpot_ms"],
        "mean_itl_ms": result["mean_itl_ms"],
        "p90_itl_ms": result["p90_itl_ms"],
        "mean_e2e_latency_ms": result["mean_e2e_latency_ms"],
        "p90_e2e_latency_ms": result["p90_e2e_latency_ms"],
        "source": source,
    }


def selected_rows(
    payload: dict[str, Any],
    *,
    models: list[str],
    workloads: list[str],
    total_gpus: list[int],
    speed_floor: float,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    selected = []
    for model_key in models:
        model = payload["models"][model_key]
        precision = primary_profile(model)["key"]
        mtp = primary_mtp(model)
        scenarios = [
            next(scenario for scenario in model["scenarios"] if scenario["name"] == "no_mtp"),
            mtp,
        ]
        for workload in workloads:
            for total in total_gpus:
                for scenario in scenarios:
                    rows = []
                    for system_kind in ("agg", "afd"):
                        row = select_best(
                            payload,
                            model=model_key,
                            workload=workload,
                            scenario=scenario["name"],
                            precision=precision,
                            total_gpus=total,
                            system_kind=system_kind,
                            speed_floor=speed_floor,
                        )
                        if row is None:
                            raise ValueError(
                                f"no {system_kind} winner for {model_key}/{workload}/{scenario['name']}/{total} GPU"
                            )
                        rows.append(row)
                    selected.extend((model, scenario, row) for row in rows)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--dynamo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--workloads", nargs="+", default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--total-gpus", nargs="+", type=int, default=list(DEFAULT_TOTAL_GPUS))
    parser.add_argument("--speed-floor", type=float, default=30.0)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--waves", type=int, default=1)
    args = parser.parse_args()
    if args.waves < 1 or args.output_tokens < 1:
        parser.error("--waves and --output-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload = load_payload(args.sweep)
    models = args.models or sorted(payload["models"])
    unknown = sorted(set(models) - set(payload["models"]))
    if unknown:
        raise ValueError(f"models not present in sweep: {', '.join(unknown)}")
    dynamo = args.dynamo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model, scenario, row in selected_rows(
        payload,
        models=models,
        workloads=args.workloads,
        total_gpus=args.total_gpus,
        speed_floor=args.speed_floor,
    ):
        print(
            f"running {row['model']} {row['workload']} {row['total_gpus']} GPU {row['scenario']} {row['system_kind']}",
            flush=True,
        )
        results.append(
            replay_case(
                dynamo=dynamo,
                output_dir=output_dir,
                row=row,
                context=CONTEXT_LENGTHS[row["workload"]],
                nextn=int(scenario["nextn"]),
                accepted_drafts=scenario["accepted_drafts"],
                output_tokens=args.output_tokens,
                waves=args.waves,
            )
        )
    output = output_dir / "mocker_summary.json"
    output.write_text(
        json.dumps(
            {
                "schema": "aic.afd-fixed-pool-mocker.v2",
                "dynamo_branch": subprocess.check_output(
                    ("git", "branch", "--show-current"), cwd=dynamo, text=True
                ).strip(),
                "dynamo_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=dynamo, text=True).strip(),
                "speed_floor_tokps_per_user": args.speed_floor,
                "output_tokens_per_request": args.output_tokens,
                "waves": args.waves,
                "sweep_sources": payload["sources"],
                "profile_note": (
                    "AIC supplies one fixed service unit's raw decode-round time. Mocker validates unit replication, "
                    "round-robin routing, request lifecycle, finite-wave tails, and stochastic MTP burst accounting; "
                    "it does not re-estimate attention or MoE kernels. Mocker's reported output throughput is a "
                    "finite-wave measurement, while expected_output_throughput_tok_s is AIC's saturated "
                    "steady-state reference."
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "cases": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
