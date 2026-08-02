#!/usr/bin/env python3
"""Replay selected multi-model AFD/MTP service points with Dynamo Mocker."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

BLOCK_SIZE = 64
OUTPUT_TOKENS = 64
HEADLINE_MTP = {
    "qwen3_235b": "eagle3_n3",
    "minimax_m3": "mtp_n1_r70",
    "deepseek_v4_flash": "mtp_n2_r70",
    "deepseek_v4_pro": "mtp_n2_r70",
}


def equal_conditional_rate(nextn: int, accepted_drafts: float) -> float:
    """Solve sum(r**i, i=1..nextn) == accepted_drafts."""
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        expected = sum(middle**position for position in range(1, nextn + 1))
        if expected < accepted_drafts:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def evidence_for(model: str) -> str:
    return "measured complete-F overlay" if model == "qwen3_235b" else "native AIC"


def row_key(row: dict[str, Any]) -> tuple:
    return (
        row["model"],
        row["workload"],
        row["evidence"],
        row["a_nodes"],
        row["f_nodes"],
        row["a_tp"],
        row["batch_per_a_gpu"],
        row["microbatches"],
    )


def select_pairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload["rows"]
    selected: list[dict[str, Any]] = []
    for model, mtp_name in HEADLINE_MTP.items():
        evidence = evidence_for(model)
        for workload in ("8k", "16k"):
            mtp_rows = [
                row
                for row in rows
                if row["model"] == model
                and row["workload"] == workload
                and row["scenario"] == mtp_name
                and row["evidence"] == evidence
            ]
            mtp = max(mtp_rows, key=lambda row: row["output_tokens_s_gpu"])
            no_mtp_rows = [row for row in rows if row["scenario"] == "no_mtp" and row_key(row) == row_key(mtp)]
            if len(no_mtp_rows) != 1:
                raise ValueError(f"expected one matched no-MTP row for {model}/{workload}, got {len(no_mtp_rows)}")
            selected.append({"model": model, "workload": workload, "no_mtp": no_mtp_rows[0], "mtp": mtp})
    return selected


def write_profile(
    path: Path,
    *,
    context: int,
    local_batch: int,
    raw_round_ms: float,
    metadata: dict[str, Any],
) -> None:
    max_context = context + OUTPUT_TOKENS
    np.savez(
        path,
        prefill_isl=np.asarray([0.0, float(local_batch * context)]),
        prefill_ttft_ms=np.asarray([0.0, 1.0]),
        decode_active_kv_tokens=np.asarray([0.0, float(local_batch * max_context)]),
        decode_context_length=np.asarray([float(context), float(max_context)]),
        decode_itl=np.full((2, 2), raw_round_ms, dtype=np.float64),
        afd_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def replay_case(
    *,
    dynamo: Path,
    output_dir: Path,
    case_id: str,
    context: int,
    global_requests: int,
    workers: int,
    worker_batch: int,
    raw_round_ms: float,
    nextn: int,
    accepted_drafts: float | None,
    source: dict[str, Any],
    topology: str,
    waves: int,
) -> dict[str, Any]:
    profile = output_dir / f"{case_id}.npz"
    report = output_dir / f"{case_id}.json"
    write_profile(
        profile,
        context=context,
        local_batch=worker_batch,
        raw_round_ms=raw_round_ms,
        metadata={
            "schema": "aic.afd-multimodel-mtp-fixed-replay.v1",
            "case_id": case_id,
            "service_contract": "raw full-resident decode-round wall time",
            "topology": topology,
            "source": source,
        },
    )
    blocks_per_sequence = math.ceil((context + OUTPUT_TOKENS) / BLOCK_SIZE) + 2
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
        str(OUTPUT_TOKENS),
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
    return {
        "case_id": case_id,
        "model": source["model"],
        "workload": source["workload"],
        "scenario": source["scenario"],
        "topology": topology,
        "context": context,
        "nextn": nextn,
        "conditional_rate_surrogate": conditional_rate,
        "accepted_drafts_target": accepted_drafts,
        "global_requests": global_requests,
        "workers": workers,
        "worker_batch": worker_batch,
        "raw_round_ms": raw_round_ms,
        "profile": str(profile),
        "report": str(report),
        "command": command,
        "completed_requests": result["completed_requests"],
        "duration_ms": result["duration_ms"],
        "output_throughput_tok_s": result["output_throughput_tok_s"],
        "mean_tpot_ms": result["mean_tpot_ms"],
        "p90_tpot_ms": result["p90_tpot_ms"],
        "mean_itl_ms": result["mean_itl_ms"],
        "p90_itl_ms": result["p90_itl_ms"],
        "mean_e2e_latency_ms": result["mean_e2e_latency_ms"],
        "p90_e2e_latency_ms": result["p90_e2e_latency_ms"],
        "source": source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--dynamo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--waves", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep = json.loads(args.sweep.resolve().read_text(encoding="utf-8"))
    dynamo = args.dynamo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for pair in select_pairs(sweep):
        for scenario_name in ("no_mtp", "mtp"):
            afd = pair[scenario_name]
            context = int(afd["context"])
            global_requests = int(afd["global_requests"])
            nextn = int(afd["nextn"])
            accepted = afd["accepted_drafts"]
            agg = afd["agg"]
            variants = (
                (
                    "agg",
                    int(agg["replicas"]),
                    int(agg["global_batch"]),
                    float(agg["raw_round_ms"]),
                    agg,
                ),
                ("afd", 1, global_requests, float(afd["raw_round_ms"]), afd),
            )
            for topology, workers, worker_batch, raw_round_ms, source in variants:
                case_id = f"{pair['model']}_{pair['workload']}_{scenario_name}_{topology}"
                print(f"running {case_id}", flush=True)
                results.append(
                    replay_case(
                        dynamo=dynamo,
                        output_dir=output_dir,
                        case_id=case_id,
                        context=context,
                        global_requests=global_requests,
                        workers=workers,
                        worker_batch=worker_batch,
                        raw_round_ms=raw_round_ms,
                        nextn=nextn,
                        accepted_drafts=accepted,
                        source=afd,
                        topology=topology,
                        waves=args.waves,
                    )
                )
    output = output_dir / "mocker_summary.json"
    output.write_text(
        json.dumps(
            {
                "schema": "aic.afd-multimodel-mtp-mocker.v1",
                "dynamo_branch": subprocess.check_output(
                    ("git", "branch", "--show-current"), cwd=dynamo, text=True
                ).strip(),
                "dynamo_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=dynamo, text=True).strip(),
                "output_tokens_per_request": OUTPUT_TOKENS,
                "profile_note": (
                    "AIC supplies the fixed full-resident raw service time. Mocker validates routing, "
                    "request lifecycle, finite-wave tails, and stochastic MTP burst accounting; it does not "
                    "re-estimate attention or MoE kernels."
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
