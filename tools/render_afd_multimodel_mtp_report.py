#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Render self-contained HTML reports for the multi-model AFD/MTP sweep."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

TOTAL_GPUS = 72
CONTEXTS = ("8k", "16k")
MODEL_ORDER = ("qwen3_235b", "minimax_m3", "deepseek_v4_flash", "deepseek_v4_pro")
ALL_MODEL_LINK_ORDER = ("minimax_m25",) + MODEL_ORDER
HEADLINE_SCENARIO = {
    "qwen3_235b": "eagle3_n3",
    "minimax_m3": "mtp_n1_r70",
    "deepseek_v4_flash": "mtp_n2_r70",
    "deepseek_v4_pro": "mtp_n2_r70",
}
MODEL_LABELS = {
    "minimax_m25": "MiniMax-M2.5-FP8",
    "qwen3_235b": "Qwen3-235B-A22B",
    "minimax_m3": "MiniMax-M3",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek_v4_pro": "DeepSeek-V4-Pro",
}
MODEL_LINKS = {
    "minimax_m25": "https://huggingface.co/MiniMaxAI/MiniMax-M2.5",
    "qwen3_235b": "https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8",
    "minimax_m3": "https://huggingface.co/MiniMaxAI/MiniMax-M3",
    "deepseek_v4_flash": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
    "deepseek_v4_pro": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
}

COLORS = {
    "AGG": "#0072B2",
    "AGG + MTP": "#56B4E9",
    "AGG + AFD": "#D55E00",
    "AGG + AFD + MTP": "#E69F00",
    "No MTP": "#0072B2",
    "With MTP": "#E69F00",
    "8K": "#0072B2",
    "16K": "#D55E00",
    "Progress P": "#009E73",
    "Raw cost Tq/T1": "#CC79A7",
    "Final speedup": "#E69F00",
    "attention": "#0072B2",
    "mHC": "#56B4E9",
    "dense GEMM": "#009E73",
    "router": "#F0E442",
    "norm / embedding / logits": "#999999",
    "MoE": "#D55E00",
    "F collective": "#CC79A7",
    "A combine": "#E69F00",
    "A-F transfer": "#000000",
    "FastAFD silicon": "#009E73",
    "AIC calibrated F": "#E69F00",
    "NVL8 measured F": "#CC79A7",
    "FastAFD-required F": "#E69F00",
    "Generic AIC": "#0072B2",
    "AIC baseline": "#56B4E9",
    "A-side work": "#0072B2",
    "F-side work": "#D55E00",
    "E2E service": "#000000",
}

MINIMAX_M25_REFERENCE = {
    "8k": {
        "baseline_tps_gpu": 1516.0,
        "afd_tps_gpu": 2198.0,
        "speedup": 1.45,
        "baseline_step_ms": 31.662269,
        "afd_step_ms": 30.937216,
        "baseline_batch_gpu": 48,
        "afd_batch_a_gpu": 72,
    },
    "16k": {
        "baseline_tps_gpu": 745.0,
        "afd_tps_gpu": 1006.0,
        "speedup": 1.35,
        "baseline_step_ms": 32.214765,
        "afd_step_ms": 33.797217,
        "baseline_batch_gpu": 24,
        "afd_batch_a_gpu": 36,
    },
}

FASTAFD_REFERENCE = {
    "qwen3_235b": {
        "8k": {
            "baseline_tps_gpu": 1781.0,
            "afd_tps_gpu": 2518.0,
            "speedup": 1.41,
            "baseline_step_ms": 35.934868,
            "afd_step_ms": 33.359809,
            "baseline_batch_gpu": 64,
            "afd_batch_a_gpu": 96,
            "a_nodes": 7,
            "f_nodes": 1,
            "total_gpus": 32,
        },
        "16k": {
            "baseline_tps_gpu": 954.0,
            "afd_tps_gpu": 1377.0,
            "speedup": 1.44,
            "baseline_step_ms": 33.542977,
            "afd_step_ms": 31.953522,
            "baseline_batch_gpu": 32,
            "afd_batch_a_gpu": 48,
            "a_nodes": 11,
            "f_nodes": 1,
            "total_gpus": 48,
        },
    },
    "minimax_m25": {
        workload: {
            **reference,
            "a_nodes": 17,
            "f_nodes": 1,
            "total_gpus": 72,
        }
        for workload, reference in MINIMAX_M25_REFERENCE.items()
    },
}

ATTENTION_SHORT = {
    "minimax_m25": "Full GQA",
    "qwen3_235b": "Full GQA",
    "minimax_m3": "MSA",
    "deepseek_v4_flash": "CSA + HCA + SWA",
    "deepseek_v4_pro": "CSA + HCA",
}

TRACK_LABELS = {
    ("minimax_m25", "calibrated_fp8_effective_f"): "AIC calibrated reproduction",
    ("minimax_m25", "measured_fp8_complete_f"): "B200 NVL8 measured F injected",
    ("minimax_m25", "generic_fp8"): "Generic AIC FP8",
    ("qwen3_235b", "nvfp4"): "AIC native NVFP4",
    ("qwen3_235b", "fp8"): "Generic AIC FP8",
    ("qwen3_235b", "measured_fp8_complete_f"): "B200 NVL8 measured F injected",
    ("minimax_m3", "nvfp4_projected"): "AIC NVFP4 projection",
    ("minimax_m3", "fp8_projected"): "AIC FP8 projection",
    ("minimax_m3", "bf16_projected"): "AIC BF16 projection",
    ("deepseek_v4_flash", "mxfp4_mxfp8"): "AIC native MXFP4/MXFP8",
    ("deepseek_v4_flash", "fp8"): "Generic AIC FP8",
    ("deepseek_v4_pro", "megamoe_fp4"): "AIC + measured MegaMoE FP4",
}

CSS = """
:root{--ink:#17202a;--muted:#5f6b76;--line:#d8dee4;--panel:#f7f9fb;--blue:#0072B2;--orange:#D55E00}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.55 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:34px 34px 70px}h1{font-size:32px;line-height:1.15;margin:0 0 8px}h2{margin:38px 0 12px;padding-top:10px;border-top:2px solid var(--ink);font-size:22px}h3{font-size:17px;margin:22px 0 8px}.subtitle,.muted{color:var(--muted)}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.nav a,.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;text-decoration:none;color:var(--ink);background:#fff}.nav a:hover{border-color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}.card{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--panel)}.card .value{font-size:26px;font-weight:750}.card .label{color:var(--muted);font-size:13px}
.callout{border-left:5px solid var(--blue);background:#eef7fb;padding:12px 15px;margin:14px 0}.warn{border-left-color:var(--orange);background:#fff4ef}.ok{border-left-color:#009E73;background:#effaf6}
.figure{border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0;background:#fff}.figure svg{display:block;width:100%;height:auto}.comment{margin:10px 4px 2px;color:#34404b}.comment strong{color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:right;vertical-align:top}th{background:#f1f4f6}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}.good{color:#007a55;font-weight:700}.bad{color:#b43b20;font-weight:700}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.small{font-size:12px}.foot{margin-top:38px;color:var(--muted);font-size:12px}
.matrix table{min-width:2200px}.matrix th{position:sticky;top:0;z-index:1;white-space:nowrap}.matrix td{white-space:nowrap}.matrix tr.public{background:#effaf6}.matrix tr.calibrated{background:#fff8e8}.matrix tr.measured{background:#fff2fa}.matrix tr.projected{background:#f7f3ff}
svg text{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;fill:#24313d}.grid{stroke:#dfe5ea;stroke-width:1}.axis{stroke:#53606b;stroke-width:1.2}.legend{font-size:12px}.tick{font-size:11px}.value-label{font-size:10px;font-weight:650}
@media print{main{max-width:none;padding:18px}.figure{break-inside:avoid}a{color:inherit}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float, digits: int = 1) -> str:
    if not math.isfinite(float(value)):
        return "—"
    return f"{value:,.{digits}f}"


def merge_payload(base: dict[str, Any], overlays: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(base)
    merged["rows"] = list(base["rows"])
    merged["models"] = list(base.get("models", []))
    merged["failures"] = list(base.get("failures", []))
    merged["applied_overlays"] = []
    for overlay in overlays:
        model_keys = {model["key"] for model in overlay.get("models", [])}
        merged["rows"] = [row for row in merged["rows"] if row["model"] not in model_keys] + overlay["rows"]
        merged["models"] = [model for model in merged["models"] if model["key"] not in model_keys] + overlay.get(
            "models", []
        )
        merged["failures"] = [row for row in merged["failures"] if row.get("model") not in model_keys] + overlay.get(
            "failures", []
        )
        merged["applied_overlays"].append({"models": sorted(model_keys), "code": overlay.get("code", {})})
    return merged


def row_key(row: dict[str, Any]) -> tuple:
    return tuple(
        row[key]
        for key in (
            "model",
            "workload",
            "evidence",
            "a_nodes",
            "f_nodes",
            "a_tp",
            "batch_per_a_gpu",
            "microbatches",
        )
    )


def primary_precision(model_meta: dict[str, Any]) -> dict[str, Any]:
    profiles = [profile for profile in model_meta["precision_profiles"] if profile["primary"]]
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one primary precision for {model_meta['key']}, got {len(profiles)}")
    return profiles[0]


def select_pairs(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODEL_ORDER:
        scenario = HEADLINE_SCENARIO[model]
        model_meta = next(value for value in payload["models"] if value["key"] == model)
        evidence = primary_precision(model_meta)["evidence"]
        for workload in CONTEXTS:
            mtp = max(
                (
                    row
                    for row in payload["rows"]
                    if row["model"] == model
                    and row["workload"] == workload
                    and row["scenario"] == scenario
                    and row["evidence"] == evidence
                ),
                key=lambda row: row["output_tokens_s_gpu"],
            )
            no_mtp = next(
                row for row in payload["rows"] if row["scenario"] == "no_mtp" and row_key(row) == row_key(mtp)
            )
            selected[model][workload] = {"no_mtp": no_mtp, "mtp": mtp}
    return dict(selected)


def best_matched_pair(
    payload: dict[str, Any],
    *,
    model: str,
    workload: str,
    scenario: str,
    evidence: str,
) -> dict[str, dict[str, Any]] | None:
    candidates: list[dict[str, dict[str, Any]]] = []
    no_mtp_by_key = {
        row_key(row): row
        for row in payload["rows"]
        if row["model"] == model
        and row["workload"] == workload
        and row["scenario"] == "no_mtp"
        and row["evidence"] == evidence
    }
    for mtp in payload["rows"]:
        if not (
            mtp["model"] == model
            and mtp["workload"] == workload
            and mtp["scenario"] == scenario
            and mtp["evidence"] == evidence
        ):
            continue
        no_mtp = no_mtp_by_key.get(row_key(mtp))
        if no_mtp is not None:
            candidates.append({"no_mtp": no_mtp, "mtp": mtp})
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair["mtp"]["output_tokens_s_gpu"])


def four_way(pair: dict[str, dict[str, Any]]) -> dict[str, float]:
    no_mtp, mtp = pair["no_mtp"], pair["mtp"]
    return {
        "AGG": no_mtp["agg"]["cluster_output_tokens_s_gpu"],
        "AGG + MTP": mtp["agg"]["cluster_output_tokens_s_gpu"],
        "AGG + AFD": no_mtp["output_tokens_s_gpu"],
        "AGG + AFD + MTP": mtp["output_tokens_s_gpu"],
    }


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    raw = value * 1.12
    power = 10 ** math.floor(math.log10(raw))
    scaled = raw / power
    step = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return step * power


def svg_grouped_bars(
    groups: list[tuple[str, dict[str, float]]],
    series: list[str],
    *,
    y_label: str,
    x_label: str,
    y_max: float | None = None,
    value_digits: int = 2,
) -> str:
    width, height = 1040, 500
    left, right, top, bottom = 84, 28, 58, 126
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = y_max or nice_max(max(value for _, values in groups for value in values.values()))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h - plot_h * tick / 5
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 9}" y="{y + 4:.1f}" text-anchor="end">{esc(fmt(value, 1))}</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    group_w = plot_w / max(len(groups), 1)
    inner = group_w * 0.82
    bar_w = inner / max(len(series), 1)
    for group_index, (label, values) in enumerate(groups):
        x0 = left + group_index * group_w + (group_w - inner) / 2
        for series_index, name in enumerate(series):
            value = float(values[name])
            bar_h = max(0.0, plot_h * value / y_max)
            x = x0 + series_index * bar_w + 1
            y = top + plot_h - bar_h
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" height="{bar_h:.1f}" '
                f'fill="{COLORS[name]}"><title>{esc(label)} — {esc(name)}: {fmt(value, value_digits)}</title></rect>'
            )
            if bar_h > 18:
                out.append(
                    f'<text class="value-label" x="{x + bar_w / 2:.1f}" y="{max(y - 4, top + 10):.1f}" '
                    f'text-anchor="middle">{esc(fmt(value, value_digits))}</text>'
                )
        cx = left + (group_index + 0.5) * group_w
        out.append(
            f'<text class="tick" transform="translate({cx:.1f},{top + plot_h + 18}) rotate(-24)" '
            f'text-anchor="end">{esc(label)}</text>'
        )
    legend_x = left
    for name in series:
        out.append(f'<rect x="{legend_x}" y="18" width="13" height="13" fill="{COLORS[name]}"/>')
        out.append(f'<text class="legend" x="{legend_x + 18}" y="29">{esc(name)}</text>')
        legend_x += 28 + 8.0 * len(name)
    out.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{esc(y_label)}</text>'
    )
    out.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle">{esc(x_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_lines(
    series_points: list[tuple[str, list[tuple[float, float, str]]]],
    *,
    x_label: str,
    y_label: str,
    x_max: float | None = None,
    y_max: float | None = None,
) -> str:
    width, height = 1040, 470
    left, right, top, bottom = 84, 30, 55, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for _, points in series_points for point in points]
    x_max = x_max or nice_max(max(point[0] for point in all_points))
    y_max = y_max or nice_max(max(point[1] for point in all_points))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(6):
        xv, yv = x_max * tick / 5, y_max * tick / 5
        x = left + plot_w * tick / 5
        y = top + plot_h - plot_h * tick / 5
        out.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>',
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>',
                f'<text class="tick" x="{left - 9}" y="{y + 4:.1f}" text-anchor="end">{esc(fmt(yv, 1))}</text>',
                f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{esc(fmt(xv, 1))}</text>',
            ]
        )
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    legend_x = left
    for name, points in series_points:
        color = COLORS[name]
        coords = [(left + plot_w * x / x_max, top + plot_h - plot_h * y / y_max, tooltip) for x, y, tooltip in points]
        out.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
            + '"/>'
        )
        for (x, y, tooltip), (_, raw_y, _) in zip(coords, points, strict=True):
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5">'
                f"<title>{esc(tooltip)}</title></circle>"
            )
            if raw_y == max(point[1] for point in points):
                out.append(f'<text class="value-label" x="{x + 7:.1f}" y="{y - 7:.1f}">{esc(fmt(raw_y, 1))}</text>')
        out.append(f'<line x1="{legend_x}" y1="25" x2="{legend_x + 18}" y2="25" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text class="legend" x="{legend_x + 24}" y="29">{esc(name)}</text>')
        legend_x += 58 + 8 * len(name)
    out.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{esc(y_label)}</text>'
    )
    out.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle">{esc(x_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def module_totals(row: dict[str, Any], side: str) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for module in row["modules"]:
        if module["side"] == side:
            values[module["group"]] += float(module["raw_round_ms"])
    return dict(values)


def svg_module_stacks(model_pairs: dict[str, dict[str, dict[str, Any]]]) -> str:
    categories = [
        "attention",
        "mHC",
        "dense GEMM",
        "router",
        "norm / embedding / logits",
        "MoE",
        "F collective",
        "A combine",
        "A-F transfer",
    ]
    bars: list[tuple[str, dict[str, float]]] = []
    for workload in CONTEXTS:
        for scenario in ("no_mtp", "mtp"):
            row = model_pairs[workload][scenario]
            for side in ("A", "F", "fabric"):
                bars.append(
                    (f"{workload.upper()} {scenario.replace('_', ' ').upper()} {side}", module_totals(row, side))
                )
    width, height = 1120, 550
    left, right, top, bottom = 84, 25, 72, 138
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = nice_max(max(sum(values.values()) for _, values in bars))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(6):
        value, y = y_max * tick / 5, top + plot_h - plot_h * tick / 5
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 9}" y="{y + 4:.1f}" text-anchor="end">{esc(fmt(value, 1))}</text>')
    bar_slot = plot_w / len(bars)
    for index, (label, values) in enumerate(bars):
        x = left + index * bar_slot + bar_slot * 0.15
        bar_w = bar_slot * 0.7
        cursor = top + plot_h
        for category in categories:
            value = values.get(category, 0.0)
            if value <= 0:
                continue
            segment_h = plot_h * value / y_max
            cursor -= segment_h
            out.append(
                f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" height="{segment_h:.1f}" '
                f'fill="{COLORS[category]}"><title>{esc(label)} — {esc(category)}: {fmt(value, 3)} ms</title></rect>'
            )
        out.append(
            f'<text class="tick" transform="translate({x + bar_w / 2:.1f},{top + plot_h + 18}) rotate(-35)" '
            f'text-anchor="end">{esc(label)}</text>'
        )
    legend_x, legend_y = left, 18
    for category in categories:
        if legend_x + 22 + 8 * len(category) > width - right:
            legend_x, legend_y = left, legend_y + 22
        out.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{COLORS[category]}"/>')
        out.append(f'<text class="legend" x="{legend_x + 17}" y="{legend_y + 11}">{esc(category)}</text>')
        legend_x += 32 + 7.3 * len(category)
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    out.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">Accumulated module work (ms / raw round)</text>'
    )
    out.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle">Context, MTP mode, and worker side</text>'
    )
    out.append("</svg>")
    return "".join(out)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (value["effective_tpot_ms"], -value["output_tokens_s_gpu"])):
        if not front or row["output_tokens_s_gpu"] > front[-1]["output_tokens_s_gpu"]:
            front.append(row)
    return front


def solve_break_even_rate(nextn: int, raw_ratio: float) -> float | None:
    if raw_ratio <= 1:
        return 0.0
    if raw_ratio >= nextn + 1:
        return None
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        progress = 1 + sum(middle**position for position in range(1, nextn + 1))
        if progress < raw_ratio:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def table(
    headers: list[str],
    rows: list[list[object]],
    classes: list[str] | None = None,
    wrapper_class: str = "",
) -> str:
    wrapper_classes = "scroll" + (f" {wrapper_class}" if wrapper_class else "")
    out = [f'<div class="{wrapper_classes}"><table><thead><tr>']
    out.extend(f"<th>{esc(header)}</th>" for header in headers)
    out.append("</tr></thead><tbody>")
    for row_index, row in enumerate(rows):
        class_name = f' class="{classes[row_index]}"' if classes else ""
        out.append(f"<tr{class_name}>")
        out.extend(f"<td>{value}</td>" for value in row)
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def figure(svg: str, comment: str) -> str:
    return f'<div class="figure">{svg}<p class="comment"><strong>Interpretation.</strong> {comment}</p></div>'


def document(title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p>{body}<p class="foot">Self-contained HTML: all charts are inline SVG; no external image, CSS, or JavaScript dependency.</p></main></body></html>"""


def evidence_note(model: str) -> tuple[str, str]:
    notes = {
        "qwen3_235b": (
            "NVFP4 same-shape MoE / HYBRID system",
            "The headline uses the exact Qwen expert shape in the GB200 NVFP4 MoE table for both AGG and AFD. Full GQA attention and the remaining system pieces are AIC HYBRID. The measured fused FP8 complete-F result is retained as a separate calibration track and is never relabeled as FP4.",
        ),
        "minimax_m3": (
            "NVFP4 projection / HYBRID",
            "MSA has no native silicon table and transfers utilization from measured DSA. No same-shape MiniMax MoE row exists at BF16, FP8, or FP4, so the primary NVFP4 result is a target-shape projection, not an FP4 measurement. The 70% conditional acceptance is a PoC scenario.",
        ),
        "deepseek_v4_flash": (
            "Model-specific sparse attention / FP4 MoE",
            "Twenty-one CSA and twenty HCA layers use model-specific tables; two pure-SWA layers reuse HCA timing. The same-shape MoE table uses MXFP4 weights and MXFP8 activations. Single-token attention is silicon-backed; q=3 verification is estimated. The 70% acceptance is sensitivity-only.",
        ),
        "deepseek_v4_pro": (
            "Measured MegaMoE hybrid",
            "Thirty CSA layers plus thirty-one HCA layers use declared 0.5.14 donors, and the F path uses the declared measured FP4 MegaMoE module from 0.5.10. The AFD path queries rank-local tokens; loads above 512 local decode tokens use utilization-hold extrapolation. q=3 attention and acceptance are sensitivity estimates.",
        ),
    }
    return notes[model]


def model_comment(model: str, pairs: dict[str, dict[str, dict[str, Any]]]) -> str:
    values = {workload: four_way(pairs[workload]) for workload in CONTEXTS}
    ratios = {workload: values[workload]["AGG + AFD + MTP"] / values[workload]["AGG + MTP"] for workload in CONTEXTS}
    if model == "qwen3_235b":
        return (
            f"With the same-shape NVFP4 MoE track on both sides, AFD+MTP is {ratios['8k']:.2f}× AGG+MTP "
            f"at 8K and {ratios['16k']:.2f}× at 16K. The separate FP8 complete-F measurement is shown in "
            "the precision/evidence section and is not mixed into these four bars."
        )
    if model == "minimax_m3":
        return (
            "This is the primary NVFP4 projection, not a MiniMax silicon measurement. The selected N=1 AFD speedup is "
            f"{values['8k']['AGG + AFD + MTP'] / values['8k']['AGG + AFD']:.2f}× at 8K and "
            f"{values['16k']['AGG + AFD + MTP'] / values['16k']['AGG + AFD']:.2f}× at 16K because F-side "
            "work grows relative to accepted-token progress. Treat the direction as a calibration target until the "
            "MSA and same-shape MoE kernels are measured."
        )
    if model == "deepseek_v4_flash":
        return (
            f"At the 70% sensitivity point, the MXFP4/MXFP8 track gives AFD+MTP / AGG+MTP of "
            f"{ratios['8k']:.2f}× at 8K and {ratios['16k']:.2f}× at 16K. Its no-MTP CSA/HCA kernels are "
            "silicon-backed, but q=3 attention is estimated and the two pure-SWA layers use an HCA proxy."
        )
    return (
        f"The measured FP4 MegaMoE track gives AFD+MTP / AGG+MTP of {ratios['8k']:.2f}× at 8K and "
        f"{ratios['16k']:.2f}× at 16K. Rank-local token accounting is corrected, but q=3 attention and MegaMoE "
        "loads above the measured local-token range remain extrapolated."
    )


def render_model(
    model: str,
    payload: dict[str, Any],
    pairs: dict[str, dict[str, dict[str, Any]]],
    mocker: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    label = MODEL_LABELS[model]
    model_meta = next(value for value in payload["models"] if value["key"] == model)
    scenario = HEADLINE_SCENARIO[model]
    primary = primary_precision(model_meta)
    evidence = primary["evidence"]
    four_groups = [(workload.upper(), four_way(pairs[workload])) for workload in CONTEXTS]
    evidence_title, evidence_text = evidence_note(model)
    nav = (
        '<div class="nav"><a href="index.html">Cross-model summary</a>'
        + "".join(f'<a href="{esc(key)}.html">{esc(MODEL_LABELS[key])}</a>' for key in ALL_MODEL_LINK_ORDER)
        + "</div>"
    )

    cards = []
    for workload in CONTEXTS:
        values = four_way(pairs[workload])
        ratio = values["AGG + AFD + MTP"] / values["AGG + MTP"]
        cards.append(
            f'<div class="card"><div class="value {"good" if ratio > 1 else "bad"}">{ratio:.2f}×</div>'
            f'<div class="label">{workload.upper()} · (AGG+AFD+MTP) / (AGG+MTP)</div></div>'
        )
    body = nav + '<div class="cards">' + "".join(cards) + "</div>"
    body += f'<div class="callout"><strong>{esc(evidence_title)}.</strong> {esc(evidence_text)}</div>'

    body += "<h2>1. Exact comparison contract</h2>"
    body += table(
        ["Item", "Setting"],
        [
            ["Hardware", "GB200 NVL72 · 72 GPUs · 18 nodes × 4 GPUs · one NVLink domain"],
            ["Scope", "Decode only; ISL 8,192 / 16,384; OSL=2 for one representative decode round"],
            ["Compared systems", "AGG, AGG+MTP, AGG+AFD, AGG+AFD+MTP on the same 72-GPU budget"],
            [
                "Concurrency control",
                "For each context, all four bars use the same offered requests; AFD no-MTP is matched to the MTP-selected topology",
            ],
            ["AFD search", "A:F nodes {17:1, 16:2, 14:4, 10:8}; A-TP {1,2,4}; microbatches {1,2,4}"],
            ["Batch semantics", "batch_per_a_gpu; CLI-equivalent a_batch_size = batch_per_a_gpu × A-TP"],
            ["Database", f"SGLang {esc(model_meta['backend_version'])}, HYBRID; headline evidence: {esc(evidence)}"],
            ["Attention structure", esc(model_meta["attention_note"])],
            ["Attention timing", esc(model_meta["attention_timing_note"])],
            ["F-side MoE shape", esc(model_meta["moe_shape_note"])],
            ["F-side MoE precision", esc(primary["moe_quant_mode"])],
            ["F-side MoE timing", esc(primary["timing_note"])],
            [
                "Model",
                f'<a href="{esc(MODEL_LINKS[model])}">{esc(model_meta["model_path"])}</a> · {esc(model_meta["parameter_note"])}',
            ],
        ],
    )

    body += "<h2>2. MoE precision and evidence sensitivity</h2>"
    precision_groups = []
    precision_rows = []
    for profile in model_meta["precision_profiles"]:
        ratios: dict[str, float] = {"8K": 0.0, "16K": 0.0}
        cells: dict[str, tuple[float, float, str]] = {}
        for workload in CONTEXTS:
            pair = best_matched_pair(
                payload,
                model=model,
                workload=workload,
                scenario=scenario,
                evidence=profile["evidence"],
            )
            if pair is None:
                continue
            values = four_way(pair)
            ratio = values["AGG + AFD + MTP"] / values["AGG + MTP"]
            ratios[workload.upper()] = ratio
            cells[workload] = (
                ratio,
                values["AGG + AFD + MTP"],
                f"{pair['mtp']['a_nodes']}A:{pair['mtp']['f_nodes']}F",
            )
        if not cells:
            continue
        label_suffix = " · primary" if profile["primary"] else ""
        precision_groups.append((profile["key"] + label_suffix, ratios))
        precision_rows.append(
            [
                esc(profile["key"] + label_suffix),
                esc(profile["moe_quant_mode"]),
                esc(profile["timing_note"]),
                "—" if "8k" not in cells else fmt(cells["8k"][0], 3) + "×",
                "—" if "8k" not in cells else fmt(cells["8k"][1], 1),
                "—" if "8k" not in cells else cells["8k"][2],
                "—" if "16k" not in cells else fmt(cells["16k"][0], 3) + "×",
                "—" if "16k" not in cells else fmt(cells["16k"][1], 1),
                "—" if "16k" not in cells else cells["16k"][2],
            ]
        )
    precision_comment = (
        "Each bar compares AGG and AFD at the same MoE precision, then independently re-optimizes the A:F "
        "split, A-TP, batch, and microbatch count. The primary profile follows FP4 first, then FP8, then "
        "BF16 only when a lower-precision path is unavailable. A measured FP8 stage is kept FP8 rather than "
        "rescaled or relabeled as FP4. The ratio and absolute AFD throughput answer different questions: a "
        "lower precision can accelerate both systems yet reduce AFD/AGG when AGG benefits more; use the table's "
        "tok/s/GPU columns to judge absolute AFD performance."
    )
    if model == "qwen3_235b":
        precision_comment += (
            " For Qwen, the measured complete-F FP8 profile includes fused dispatch, expert execution, combine, "
            "and their overlap; it can therefore beat the generic NVFP4 module even though its arithmetic "
            "precision is higher. That comparison diagnoses the implementation boundary, not an FP8-over-FP4 "
            "kernel claim."
        )
    elif model == "minimax_m3":
        precision_comment += (
            " All MiniMax precision profiles are target-shape projections rather than same-shape silicon rows, "
            "so their ordering is a sensitivity result, not a calibrated kernel ranking."
        )
    body += figure(
        svg_grouped_bars(
            precision_groups,
            ["8K", "16K"],
            y_label="(AGG+AFD+MTP) / (AGG+MTP) at matched MoE precision (×)",
            x_label="MoE precision / evidence profile",
            y_max=2.0,
            value_digits=2,
        ),
        precision_comment,
    )
    body += table(
        [
            "Profile",
            "F MoE precision",
            "Timing evidence",
            "8K AFD/AGG",
            "8K AFD tok/s/GPU",
            "8K A:F",
            "16K AFD/AGG",
            "16K AFD tok/s/GPU",
            "16K A:F",
        ],
        precision_rows,
    )

    body += "<h2>3. Four-way system result</h2>"
    body += figure(
        svg_grouped_bars(
            four_groups,
            ["AGG", "AGG + MTP", "AGG + AFD", "AGG + AFD + MTP"],
            y_label="Committed output tokens / second / GPU",
            x_label="Input context length",
        ),
        model_comment(model, pairs),
    )
    result_rows = []
    for workload in CONTEXTS:
        pair, values = pairs[workload], four_way(pairs[workload])
        mtp = pair["mtp"]
        result_rows.append(
            [
                workload.upper(),
                fmt(values["AGG"]),
                fmt(values["AGG + MTP"]),
                fmt(values["AGG + AFD"]),
                fmt(values["AGG + AFD + MTP"]),
                fmt(values["AGG + AFD"] / values["AGG"], 3) + "×",
                fmt(values["AGG + AFD + MTP"] / values["AGG + MTP"], 3) + "×",
                f"{mtp['a_nodes']}A:{mtp['f_nodes']}F",
                str(mtp["a_tp"]),
                str(mtp["batch_per_a_gpu"]),
                str(mtp["microbatches"]),
            ]
        )
    body += table(
        [
            "ISL",
            "AGG",
            "AGG+MTP",
            "AGG+AFD",
            "AGG+AFD+MTP",
            "AFD/AGG no MTP",
            "AFD/AGG with MTP",
            "A:F",
            "A-TP",
            "Batch/A-GPU",
            "MB",
        ],
        result_rows,
    )

    body += "<h2>4. MTP accounting: cost, acceptance, and final gain</h2>"
    first_mtp = pairs["8k"]["mtp"]
    nextn, progress = int(first_mtp["nextn"]), float(first_mtp["progress"])
    body += (
        '<div class="callout"><span class="mono">q=N+1</span> target tokens are verified per round; '
        '<span class="mono">P=1+E[accepted drafts]</span> committed tokens advance per round; '
        '<span class="mono">effective TPOT=Traw(q)/P</span>; '
        '<span class="mono">MTP speedup=P·Traw(1)/Traw(q)</span>. '
        "The draft-layer overhead is included once as q·L+N layer-token equivalents.</div>"
    )
    if model == "qwen3_235b":
        body += (
            '<div class="callout warn"><strong>2.28125 is not a 76% acceptance rate.</strong> It is the expected '
            "committed output tokens per step. Therefore E[accepted drafts]=1.28125. For Mocker only, an equal "
            "conditional per-position surrogate r=0.63125 solves r+r²+r³=1.28125; accepted draft slots divided "
            "by three would be 42.7%, not 76%.</div>"
        )
    component_groups = []
    equation_rows = []
    for workload in CONTEXTS:
        no_mtp, mtp = pairs[workload]["no_mtp"], pairs[workload]["mtp"]
        raw_ratio = mtp["raw_round_ms"] / no_mtp["raw_round_ms"]
        final_speedup = mtp["output_tokens_s_gpu"] / no_mtp["output_tokens_s_gpu"]
        agg_raw_ratio = mtp["agg"]["raw_round_ms"] / no_mtp["agg"]["raw_round_ms"]
        agg_speedup = mtp["agg"]["cluster_output_tokens_s_gpu"] / no_mtp["agg"]["cluster_output_tokens_s_gpu"]
        component_groups.append(
            (workload.upper(), {"Progress P": progress, "Raw cost Tq/T1": raw_ratio, "Final speedup": final_speedup})
        )
        break_even = solve_break_even_rate(nextn, raw_ratio)
        equation_rows.append(
            [
                workload.upper(),
                str(nextn),
                fmt(progress, 4),
                fmt(agg_raw_ratio, 3) + "×",
                fmt(agg_speedup, 3) + "×",
                fmt(raw_ratio, 3) + "×",
                fmt(final_speedup, 3) + "×",
                "unreachable" if break_even is None else fmt(100 * break_even, 1) + "% conditional",
            ]
        )
    body += figure(
        svg_grouped_bars(
            component_groups,
            ["Progress P", "Raw cost Tq/T1", "Final speedup"],
            y_label="Multiplier versus no-MTP AFD",
            x_label="Input context length",
            y_max=nice_max(max(progress, 3.0)),
        ),
        "MTP wins only when accepted-token progress exceeds the extra raw verification cost. The final orange bar is exactly P divided by the pink raw-cost bar; it is not an independently fitted parameter.",
    )
    body += table(
        [
            "ISL",
            "N",
            "Progress P",
            "AGG raw cost",
            "AGG MTP speedup",
            "AFD raw cost",
            "AFD MTP speedup",
            "AFD break-even rate",
        ],
        equation_rows,
    )

    body += "<h2>5. Attention, router, MoE, collective, and transfer work</h2>"
    source_rows = []
    for workload in CONTEXTS:
        no_mtp_source = (
            pairs[workload]["no_mtp"]["agg"]
            .get("op_sources", {})
            .get("generation_attention", "not separately reported")
        )
        mtp_source = (
            pairs[workload]["mtp"]["agg"].get("op_sources", {}).get("generation_attention", "not separately reported")
        )
        source_rows.append([workload.upper(), esc(no_mtp_source), esc(mtp_source)])
    body += table(
        ["Operator contract", "Value", "Evidence / caveat"],
        [
            ["A attention", esc(model_meta["attention_note"]), esc(model_meta["attention_timing_note"])],
            ["F routed MoE", esc(model_meta["moe_shape_note"]), esc(primary["timing_note"])],
            [
                "Precision",
                f"A GEMM={esc(pairs['8k']['mtp']['quant']['a_gemm'])}; A FMHA={esc(pairs['8k']['mtp']['quant']['a_fmha'])}; "
                f"KV={esc(pairs['8k']['mtp']['quant']['a_kvcache'])}; F MoE={esc(pairs['8k']['mtp']['quant']['f_moe'])}",
                "AGG and AFD use the same F-side MoE precision in every ratio",
            ],
        ],
    )
    body += table(["ISL", "No-MTP attention source", "MTP attention source"], source_rows)
    body += figure(
        svg_module_stacks(pairs),
        f"These bars use {esc(model_meta['attention_note'])} on A and {esc(primary['moe_quant_mode'])} routed experts on F. They are accumulated worker-side module work, not additive E2E latency: A and F execute as a layer pipeline, so raw service is calculated by the pipeline recurrence rather than by summing every bar.",
    )
    module_rows = []
    for workload in CONTEXTS:
        for mode in ("no_mtp", "mtp"):
            row = pairs[workload][mode]
            a, f, fabric = module_totals(row, "A"), module_totals(row, "F"), module_totals(row, "fabric")
            module_rows.append(
                [
                    f"{workload.upper()} / {'MTP' if mode == 'mtp' else 'no MTP'}",
                    fmt(row["raw_round_ms"], 3),
                    fmt(a.get("attention", 0) + a.get("mHC", 0), 3),
                    fmt(a.get("dense GEMM", 0), 3),
                    fmt(a.get("router", 0), 3),
                    fmt(a.get("norm / embedding / logits", 0) + a.get("A combine", 0), 3),
                    fmt(f.get("MoE", 0), 3),
                    fmt(f.get("F collective", 0), 3),
                    fmt(fabric.get("A-F transfer", 0), 3),
                    row["pipeline_bottleneck"],
                ]
            )
    body += table(
        [
            "Case",
            "Raw service",
            "A attention/mHC",
            "A dense",
            "A router",
            "A other/combine",
            "F MoE",
            "F collective",
            "A-F transfer",
            "F precision",
            "Bottleneck",
        ],
        [row[:-1] + [esc(primary["moe_quant_mode"]), row[-1]] for row in module_rows],
    )

    body += "<h2>6. Throughput–latency Pareto frontier</h2>"
    front_series = []
    fronts = {}
    for workload in CONTEXTS:
        rows = [
            row
            for row in payload["rows"]
            if row["model"] == model
            and row["workload"] == workload
            and row["scenario"] == scenario
            and row["evidence"] == evidence
        ]
        front = pareto_front(rows)
        fronts[workload] = front
        front_series.append(
            (
                workload.upper(),
                [
                    (
                        row["effective_tpot_ms"],
                        row["output_tokens_s_gpu"],
                        f"{workload.upper()} {row['a_nodes']}A:{row['f_nodes']}F, A-TP={row['a_tp']}, batch/A-GPU={row['batch_per_a_gpu']}, MB={row['microbatches']}",
                    )
                    for row in front
                ],
            )
        )
    body += figure(
        svg_lines(
            front_series,
            x_label="Effective TPOT (ms / committed output token; lower is better)",
            y_label="Committed output tokens / second / GPU (higher is better)",
        ),
        "Each point is non-dominated: moving right accepts more latency to obtain more throughput. Both contexts share exactly the same x and y scales in this chart; the axes are derived from the combined 8K and 16K frontiers, not independently stretched panels.",
    )

    body += "<h2>7. A:F hardware-ratio sensitivity</h2>"
    ratio_series = []
    for workload in CONTEXTS:
        selected = pairs[workload]["mtp"]
        points = []
        for f_nodes in (1, 2, 4, 8):
            matches = [
                row
                for row in payload["rows"]
                if row["model"] == model
                and row["workload"] == workload
                and row["scenario"] == scenario
                and row["evidence"] == evidence
                and row["f_nodes"] == f_nodes
                and row["a_tp"] == selected["a_tp"]
                and row["batch_per_a_gpu"] == selected["batch_per_a_gpu"]
                and row["microbatches"] == selected["microbatches"]
            ]
            if matches:
                row = matches[0]
                points.append((float(f_nodes), row["output_tokens_s_gpu"], f"{row['a_nodes']}A:{f_nodes}F"))
        ratio_series.append((workload.upper(), points))
    body += figure(
        svg_lines(
            ratio_series,
            x_label="F nodes (A nodes = 18 − F nodes; total = 72 GPUs)",
            y_label="Committed output tokens / second / GPU",
            x_max=8,
        ),
        "Only F-node count changes along each line; A-TP, batch per A GPU, and microbatch count stay fixed at that context's selected MTP point. This isolates the hardware split from batch tuning and explains why a single fixed 16A:2F ratio is not generally valid.",
    )

    body += "<h2>8. Dynamo Mocker replay check</h2>"
    mock_rows = [result for result in mocker["results"] if result["model"] == model]
    error_groups = [
        (
            f"{result['workload'].upper()} {'MTP' if result['nextn'] else 'no MTP'} {result['topology'].upper()}",
            {
                "No MTP": abs(result["mean_tpot_error_pct"]) if not result["nextn"] else 0.0,
                "With MTP": abs(result["mean_tpot_error_pct"]) if result["nextn"] else 0.0,
            },
        )
        for result in mock_rows
    ]
    body += figure(
        svg_grouped_bars(
            error_groups,
            ["No MTP", "With MTP"],
            y_label="Absolute mean-TPOT error versus AIC steady state (%)",
            x_label="Context, MTP mode, and topology",
            y_max=1.0,
            value_digits=2,
        ),
        "Mocker reuses AIC's raw full-resident service time, then independently samples MTP bursts and request completion. Mean TPOT agrees within 0.9% in all cases, confirming that q-wide compute and committed-token progress are applied once. One-wave aggregate throughput is intentionally not used as steady state because the slowest stochastic acceptance chain creates a finite-wave tail.",
    )

    body += "<h2>9. MTP-depth capacity sensitivity</h2>"
    depth_series = []
    depth_rows = []
    for workload in CONTEXTS:
        points = []
        for scenario_meta in model_meta["scenarios"]:
            scenario_name = scenario_meta["name"]
            candidates = [
                row
                for row in payload["rows"]
                if row["model"] == model
                and row["workload"] == workload
                and row["scenario"] == scenario_name
                and row["evidence"] == evidence
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: row["output_tokens_s_gpu"])
            points.append(
                (
                    float(best["nextn"]),
                    best["output_tokens_s_gpu"],
                    f"{workload.upper()} N={best['nextn']}: {best['a_nodes']}A:{best['f_nodes']}F, "
                    f"A-TP={best['a_tp']}, batch/A-GPU={best['batch_per_a_gpu']}, MB={best['microbatches']}",
                )
            )
            depth_rows.append(
                [
                    workload.upper(),
                    str(best["nextn"]),
                    fmt(best["progress"], 4),
                    fmt(best["output_tokens_s_gpu"], 1),
                    fmt(best["afd_over_agg"], 3) + "×",
                    f"{best['a_nodes']}A:{best['f_nodes']}F",
                    str(best["a_tp"]),
                    str(best["batch_per_a_gpu"]),
                    str(best["microbatches"]),
                    esc(scenario_meta["acceptance_basis"]),
                ]
            )
        depth_series.append((workload.upper(), points))
    depth_comment = (
        "Unlike the controlled four-way comparison, each N here may choose a different A:F split, A-TP, batch, "
        "and microbatch count. It answers whether extra draft depth still pays after hardware is re-optimized."
    )
    if model == "minimax_m3":
        depth_comment += (
            " N=1 is the useful MTP setting in this grid; N=3 and N=7 add more F work than their assumed "
            "acceptance can repay."
        )
    body += figure(
        svg_lines(
            depth_series,
            x_label="Draft depth N (each point independently re-optimized)",
            y_label="Best committed output tokens / second / GPU",
            x_max=max(point[0] for _, points in depth_series for point in points) or 1,
        ),
        depth_comment,
    )
    body += table(
        [
            "ISL",
            "N",
            "Progress P",
            "Best AFD tok/s/GPU",
            "AFD/AGG",
            "A:F",
            "A-TP",
            "Batch/A-GPU",
            "MB",
            "Acceptance basis",
        ],
        depth_rows,
    )

    body += "<h2>10. What is supported, assumed, and not claimed</h2>"
    selected_scenario_meta = next(value for value in model_meta["scenarios"] if value["name"] == scenario)
    body += table(
        ["Layer", "Status", "Meaning"],
        [
            ["Model graph", "Supported", esc(model_meta["attention_note"] + "; " + model_meta["moe_shape_note"])],
            [
                "Attention latency",
                "Partial" if model in {"minimax_m3", "deepseek_v4_flash"} else "Calibrated / extrapolated",
                esc(model_meta["attention_timing_note"]),
            ],
            [
                "MoE latency",
                "Projected" if model == "minimax_m3" else "Silicon-backed module",
                esc(primary["timing_note"]),
            ],
            [
                "Quantization contract",
                "Controlled",
                f"Primary F MoE={esc(primary['moe_quant_mode'])}; AGG and AFD use the same precision. "
                "Alternate profiles are reported separately.",
            ],
            [
                "MTP depth / acceptance",
                "Measured" if model == "qwen3_235b" else "Scenario",
                esc(selected_scenario_meta["acceptance_basis"]),
            ],
            [
                "AIC sweep",
                "Executed",
                f"HYBRID, {len([row for row in payload['rows'] if row['model'] == model]):,} feasible rows across 8K/16K and all configured scenarios",
            ],
            [
                "Dynamo Mocker",
                "Executed",
                f"{len(mock_rows)} fixed-profile cases; lifecycle/routing/tails only, not kernel re-estimation",
            ],
            [
                "Silicon E2E",
                "Not claimed",
                "The output is a calibrated simulator result. Only explicitly labeled module/stage inputs are measured.",
            ],
        ],
    )

    title = f"{label}: GB200 NVL72 AFD × MTP Simulation"
    subtitle = "Controlled decode-only comparison of AGG, AGG+MTP, AGG+AFD, and AGG+AFD+MTP"
    summary = {
        "model": model,
        "label": label,
        "scenario": scenario,
        "evidence": evidence,
        "primary_precision": primary,
        "attention_structure": model_meta["attention_note"],
        "attention_timing": model_meta["attention_timing_note"],
        "moe_shape": model_meta["moe_shape_note"],
        "results": {
            workload: {"four_way": four_way(pairs[workload]), "selected_afd": pairs[workload]["mtp"]}
            for workload in CONTEXTS
        },
        "mocker_max_abs_mean_tpot_error_pct": max(abs(result["mean_tpot_error_pct"]) for result in mock_rows),
    }
    return document(title, subtitle, body), summary


def minimax_m25_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contracts = {
        "8k": {"a_nodes": 17, "f_nodes": 1, "a_tp": 1, "batch_per_a_gpu": 72, "microbatches": 2},
        "16k": {"a_nodes": 17, "f_nodes": 1, "a_tp": 1, "batch_per_a_gpu": 36, "microbatches": 2},
    }
    selected: dict[str, dict[str, Any]] = {}
    for workload, contract in contracts.items():
        calibrated = [
            row
            for row in payload["rows"]
            if row["model"] == "minimax_m25"
            and row["workload"] == workload
            and row["scenario"] == "no_mtp"
            and row["precision_profile"] == "calibrated_fp8_effective_f"
            and all(row[key] == value for key, value in contract.items())
        ]
        measured = [
            row
            for row in payload["rows"]
            if row["model"] == "minimax_m25"
            and row["workload"] == workload
            and row["scenario"] == "no_mtp"
            and row["precision_profile"] == "measured_fp8_complete_f"
            and all(row[key] == value for key, value in contract.items())
        ]
        generic = [
            row
            for row in payload["rows"]
            if row["model"] == "minimax_m25"
            and row["workload"] == workload
            and row["scenario"] == "no_mtp"
            and row["precision_profile"] == "generic_fp8"
            and all(row[key] == value for key, value in contract.items())
        ]
        if len(calibrated) != 1 or len(measured) != 1 or len(generic) != 1:
            raise ValueError(
                f"expected one calibrated, measured, and generic MiniMax-M2.5 row for {workload}, "
                f"got calibrated={len(calibrated)}, measured={len(measured)}, generic={len(generic)}"
            )
        selected[workload] = {"calibrated": calibrated[0], "measured": measured[0], "generic": generic[0]}
    return selected


def render_minimax_m25(
    payload: dict[str, Any],
    mocker: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    selected = minimax_m25_rows(payload)
    model_meta = next(value for value in payload["models"] if value["key"] == "minimax_m25")
    model_code = next(
        (
            overlay["code"]
            for overlay in payload.get("applied_overlays", [])
            if "minimax_m25" in overlay.get("models", [])
        ),
        payload["code"],
    )
    nav = (
        '<div class="nav"><a href="index.html">Cross-model summary</a>'
        + "".join(f'<a href="{esc(model)}.html">{esc(MODEL_LABELS[model])}</a>' for model in ALL_MODEL_LINK_ORDER)
        + "</div>"
    )
    ratio_groups = []
    timing_groups = []
    diagnostic_groups = []
    result_rows = []
    module_rows = []
    ratio_errors = []
    independent_ratio_errors = []
    formula_rows = []
    for workload in CONTEXTS:
        reference = MINIMAX_M25_REFERENCE[workload]
        calibrated = selected[workload]["calibrated"]
        measured = selected[workload]["measured"]
        generic = selected[workload]["generic"]
        calibrated_ratio = float(calibrated["afd_over_agg"])
        measured_ratio = float(measured["afd_over_agg"])
        generic_ratio = float(generic["afd_over_agg"])
        ratio_error = (calibrated_ratio / reference["speedup"] - 1.0) * 100.0
        independent_ratio_error = (measured_ratio / reference["speedup"] - 1.0) * 100.0
        ratio_errors.append(abs(ratio_error))
        independent_ratio_errors.append(abs(independent_ratio_error))
        ratio_groups.append(
            (
                workload.upper(),
                {
                    "FastAFD silicon": reference["speedup"],
                    "AIC calibrated F": calibrated_ratio,
                    "NVL8 measured F": measured_ratio,
                    "Generic AIC": generic_ratio,
                },
            )
        )
        timing_groups.append(
            (
                workload.upper(),
                {
                    "AIC baseline": float(calibrated["agg"]["raw_round_ms"]),
                    "A-side work": float(calibrated["a_full_work_ms"]),
                    "F-side work": float(calibrated["f_full_work_ms"]),
                    "E2E service": float(calibrated["raw_round_ms"]),
                },
            )
        )
        diagnostic_groups.append(
            (
                workload.upper(),
                {
                    "NVL8 measured F": float(measured["f_full_work_ms"]),
                    "FastAFD-required F": float(calibrated["f_full_work_ms"]),
                    "Generic AIC": float(generic["f_full_work_ms"]),
                },
            )
        )
        result_rows.append(
            [
                workload.upper(),
                fmt(reference["speedup"], 3) + "×",
                fmt(calibrated_ratio, 3) + "×",
                f'<span class="{"good" if abs(ratio_error) <= 5 else "bad"}">{ratio_error:+.2f}%</span>',
                fmt(measured_ratio, 3) + "×",
                f'<span class="{"good" if abs(independent_ratio_error) <= 5 else "bad"}">{independent_ratio_error:+.2f}%</span>',
                fmt(generic_ratio, 3) + "×",
                fmt(reference["baseline_step_ms"], 3),
                fmt(calibrated["agg"]["raw_round_ms"], 3),
                fmt(reference["afd_step_ms"], 3),
                fmt(calibrated["raw_round_ms"], 3),
            ]
        )
        baseline_batch = float(reference["baseline_batch_gpu"])
        afd_batch_cluster = float(calibrated["global_requests"])
        aic_baseline_tps = baseline_batch * 1000.0 / float(calibrated["agg"]["raw_round_ms"])
        aic_afd_tps = afd_batch_cluster * 1000.0 / (TOTAL_GPUS * float(calibrated["raw_round_ms"]))
        formula_rows.append(
            [
                workload.upper(),
                f"{int(baseline_batch)} × 1000 / {float(calibrated['agg']['raw_round_ms']):.3f}",
                fmt(aic_baseline_tps, 2),
                f"{int(afd_batch_cluster)} × 1000 / (72 × {float(calibrated['raw_round_ms']):.3f})",
                fmt(aic_afd_tps, 2),
                f"{aic_afd_tps:.2f} / {aic_baseline_tps:.2f} = {calibrated_ratio:.3f}×",
            ]
        )
        module_rows.append(
            [
                workload.upper(),
                fmt(calibrated["a_full_work_ms"], 3),
                fmt(measured["f_full_work_ms"], 3),
                fmt(calibrated["f_full_work_ms"], 3),
                fmt(calibrated["raw_round_ms"], 3),
                esc(calibrated["pipeline_bottleneck"]),
                fmt(generic["f_full_work_ms"], 3),
                fmt(float(calibrated["f_full_work_ms"]) / float(measured["f_full_work_ms"]), 2) + "×",
                fmt(float(generic["f_full_work_ms"]) / float(calibrated["f_full_work_ms"]), 2) + "×",
            ]
        )

    reproduced = max(ratio_errors) <= 5.0
    independently_reproduced = max(independent_ratio_errors) <= 5.0
    body = nav
    body += (
        f'<div class="callout {"ok" if reproduced else "warn"}"><strong>Result.</strong> '
        f"The FastAFD-calibrated AIC track reproduces the published ratios with a maximum numerical error of "
        f"{max(ratio_errors):.3f}%, by construction. The independent B200 NVL8 load-preserving measurement "
        f"{'also passes' if independently_reproduced else 'does not pass'} the 5% test: its maximum ratio error is "
        f"{max(independent_ratio_errors):.2f}%. It preserves MoE work and kernel tiling, but not the 68-source "
        "NVL72 fan-in. The report keeps calibrated, measured-surrogate, and generic-AIC tracks separate.</div>"
    )
    body += "<h2>1. Exact workload contract</h2>"
    body += table(
        ["Parameter", "8K", "16K"],
        [
            ["Model", "MiniMax-M2.5 FP8", "MiniMax-M2.5 FP8"],
            ["Attention", "Full GQA, 48Q/8KV, BF16 KV", "Full GQA, 48Q/8KV, BF16 KV"],
            ["F-side experts", "FP8, H=3072, I=1536, 256 experts, top-8", "FP8, H=3072, I=1536, 256 experts, top-8"],
            ["Cluster", "72 GPUs = 68A + 4F", "72 GPUs = 68A + 4F"],
            ["A tensor parallel", "1", "1"],
            ["Batch / A GPU", "72", "36"],
            ["Global AFD concurrency", "4,896", "2,448"],
            ["Microbatches", "2", "2"],
            ["Assignments / F GPU / lane", "4,896", "2,448"],
            ["Published AGG batch / GPU", "48", "24"],
        ],
    )
    body += (
        '<div class="callout"><strong>Quantization is explicit.</strong> Dense GEMM and routed experts use '
        "FP8 block quantization; FMHA and KV cache use BF16. AIC's model ID is "
        '<span class="mono">MiniMaxAI/MiniMax-M2.5</span>; the checkpoint itself carries the FP8 contract.</div>'
    )

    body += "<h2>2. Published speedup versus reproduced speedup</h2>"
    body += figure(
        svg_grouped_bars(
            ratio_groups,
            ["FastAFD silicon", "AIC calibrated F", "NVL8 measured F", "Generic AIC"],
            y_label="AFD throughput / AGG throughput (×)",
            x_label="Input context; all bars use the same 0–1.8× axis",
            y_max=1.8,
            value_digits=3,
        ),
        "Green is the published GB200 NVL72 result. Orange uses the effective F time solved from that result and therefore validates AIC's downstream service and throughput arithmetic, not an independent kernel prediction. Purple uses the real B200 NVL8 reduced-topology MegaMoE stage and is an independent compute/load check, but it lacks 68-source fan-in. Blue is unchanged generic AIC.",
    )
    body += table(
        [
            "ISL",
            "FastAFD AFD/AGG",
            "AIC calibrated",
            "Calibrated error",
            "NVL8 measured",
            "Independent error",
            "Generic AIC",
            "Fast AGG step (ms)",
            "AIC AGG step (ms)",
            "Fast AFD E2E (ms)",
            "AIC AFD service (ms)",
        ],
        result_rows,
    )
    body += table(
        ["ISL", "AGG tok/s/GPU formula", "AGG tok/s/GPU", "AFD tok/s/GPU formula", "AFD tok/s/GPU", "Final ratio"],
        formula_rows,
    )

    body += "<h2>3. From modules to end-to-end service</h2>"
    body += figure(
        svg_grouped_bars(
            timing_groups,
            ["AIC baseline", "A-side work", "F-side work", "E2E service"],
            y_label="Milliseconds per decode step",
            x_label="Input context; identical time axis",
            value_digits=2,
        ),
        "A-side work and the calibrated effective F-side work are full-stage totals, not values to add. AFD pipelines them; the slower side sets the steady-state cadence, while AIC's conservative pipeline model produces E2E service. The AGG baseline is fixed to the published 4-GPU EP4 layout and batch, not re-optimized at a different concurrency.",
    )
    body += table(
        [
            "ISL",
            "A work (ms)",
            "NVL8 measured F (ms)",
            "FastAFD-required F (ms)",
            "AFD service (ms)",
            "Bottleneck",
            "Generic F (ms)",
            "Required/measured",
            "Generic/required",
        ],
        module_rows,
    )

    body += "<h2>4. Why the generic AIC result misses</h2>"
    body += figure(
        svg_grouped_bars(
            diagnostic_groups,
            ["NVL8 measured F", "FastAFD-required F", "Generic AIC"],
            y_label="Complete F-stage time (ms / decode step)",
            x_label="Input context; same model/workload, different F timing evidence",
            value_digits=2,
        ),
        "The public-result-required F time lies between the two available estimates. Generic AIC is too slow and predicts a loss; the 3A+4F NVL8 surrogate is too fast because it removes most of the 68-source fan-in. Attention, batch, A:F split, TP, precision, and pipeline equations are identical, so this bracket isolates the unresolved term to the effective F/communication boundary. Assigning the whole gap to NVLink is still an inference until a topology-exact NVL72 trace is available.",
    )

    body += "<h2>5. Real-kernel measurement boundary</h2>"
    measured_rows = [selected[workload]["measured"] for workload in CONTEXTS]
    calibrated_rows = [selected[workload]["calibrated"] for workload in CONTEXTS]
    body += table(
        ["Item", "Value"],
        [
            ["Measured hardware", "One B200 NVL8 node, seven active ranks (3 measured A + 4 F)"],
            [
                "Load preservation",
                "Exact assignments per F GPU and lane; 816/408 tokens per measured A rank for 8K/16K",
            ],
            [
                "Kernel tiling hint",
                "Preserve production 24 × 68 A ranks as 544 × 3 measured A ranks; expected-row product is unchanged",
            ],
            ["Timed boundary", "62 layers × 2 lanes: fused quant/dispatch + persistent FP8 experts + combine"],
            [
                "Routing input",
                "Deterministic balanced top-8 routes; exact total F load, without live-request expert skew",
            ],
            ["8K measured F", fmt(measured_rows[0]["measured_f_ms"], 3) + " ms"],
            ["16K measured F", fmt(measured_rows[1]["measured_f_ms"], 3) + " ms"],
            [
                "8K FastAFD-required effective F",
                f"{fmt(calibrated_rows[0]['measured_f_ms'], 3)} ms; "
                f"{fmt(float(calibrated_rows[0]['measured_f_ms']) - float(measured_rows[0]['measured_f_ms']), 3)} ms above NVL8",
            ],
            [
                "16K FastAFD-required effective F",
                f"{fmt(calibrated_rows[1]['measured_f_ms'], 3)} ms; "
                f"{fmt(float(calibrated_rows[1]['measured_f_ms']) - float(measured_rows[1]['measured_f_ms']), 3)} ms above NVL8",
            ],
            [
                "Topology limitation",
                "F load and kernel tiling are preserved, but sender count is not a topology-exact NVL72 measurement",
            ],
            ["Published reference", '<a href="https://haoailab.com/blogs/fastafd/">FastAFD GB200 NVL72 results</a>'],
        ],
    )

    body += "<h2>6. Dynamo Mocker replay</h2>"
    mock_rows = [row for row in mocker["results"] if row["model"] == "minimax_m25"]
    body += table(
        ["ISL", "Topology", "AIC TPOT (ms)", "Mocker mean TPOT (ms)", "Error (%)", "Completed requests"],
        [
            [
                row["workload"].upper(),
                row["topology"].upper(),
                fmt(row["expected_steady_state_tpot_ms"], 6),
                fmt(row["mean_tpot_ms"], 6),
                f"{row['mean_tpot_error_pct']:+.6f}",
                str(row["completed_requests"]),
            ]
            for row in sorted(mock_rows, key=lambda value: (value["workload"], value["topology"]))
        ],
    )
    body += (
        '<div class="callout"><strong>Interpretation.</strong> Mocker consumes the fixed calibrated AIC service '
        "profiles and validates worker routing, request lifecycle, and token accounting. It does not re-estimate "
        "attention or MoE, so this checks orchestration arithmetic only; it is neither an independent FastAFD "
        "reproduction nor a second kernel measurement.</div>"
    )

    body += "<h2>7. Reproduction identity and claim boundary</h2>"
    body += table(
        ["Layer", "Identity / claim"],
        [
            ["AIC", f"branch={esc(model_code['branch'])}; commit={esc(model_code['commit'])}"],
            ["Dynamo", f"branch={esc(mocker['dynamo_branch'])}; commit={esc(mocker['dynamo_commit'])}"],
            ["AIC model support", esc(model_meta["attention_note"] + "; " + model_meta["moe_shape_note"])],
            [
                "Calibrated reproduction",
                "Published AFD/AGG ratio after solving the AIC-equivalent F time under the locked FP8/BF16, 17A:1F, batch, and MB=2 contract",
            ],
            [
                "Independent check",
                f"B200 NVL8 load/tiling-preserving MegaMoE; {'passes' if independently_reproduced else 'does not pass'} the 5% ratio criterion",
            ],
            [
                "Not claimed",
                "An independent topology-exact NVL72 E2E rerun or proof that the entire calibrated-minus-NVL8 gap is fabric fan-in",
            ],
        ],
    )
    summary = {
        "model": "minimax_m25",
        "label": MODEL_LABELS["minimax_m25"],
        "aic_code": model_code,
        "calibrated_reproduced_within_5pct": reproduced,
        "independent_nvl8_reproduced_within_5pct": independently_reproduced,
        "max_abs_calibrated_speedup_error_pct": max(ratio_errors),
        "max_abs_independent_speedup_error_pct": max(independent_ratio_errors),
        "results": {
            workload: {
                "fastafd": MINIMAX_M25_REFERENCE[workload],
                "aic_calibrated_f": selected[workload]["calibrated"],
                "aic_measured_f": selected[workload]["measured"],
                "aic_generic": selected[workload]["generic"],
            }
            for workload in CONTEXTS
        },
    }
    return document(
        "MiniMax-M2.5-FP8: FastAFD AFD Reproduction Audit",
        "GB200 NVL72 calibrated target, B200 NVL8 measured F bracket, generic AIC control, and Dynamo Mocker replay",
        body,
    ), summary


def minimax_m25_index_section(payload: dict[str, Any]) -> str:
    selected = minimax_m25_rows(payload)
    groups = []
    rows = []
    for workload in CONTEXTS:
        reference = MINIMAX_M25_REFERENCE[workload]
        calibrated = selected[workload]["calibrated"]
        measured = selected[workload]["measured"]
        generic = selected[workload]["generic"]
        groups.append(
            (
                workload.upper(),
                {
                    "FastAFD silicon": reference["speedup"],
                    "AIC calibrated F": calibrated["afd_over_agg"],
                    "NVL8 measured F": measured["afd_over_agg"],
                    "Generic AIC": generic["afd_over_agg"],
                },
            )
        )
        rows.append(
            [
                workload.upper(),
                fmt(reference["speedup"], 3) + "×",
                fmt(calibrated["afd_over_agg"], 3) + "×",
                fmt(measured["afd_over_agg"], 3) + "×",
                fmt(generic["afd_over_agg"], 3) + "×",
                f"17A:1F / TP1 / b{reference['afd_batch_a_gpu']} / MB2",
            ]
        )
    return (
        "<h2>MiniMax-M2.5-FP8 FastAFD reproduction</h2>"
        + figure(
            svg_grouped_bars(
                groups,
                ["FastAFD silicon", "AIC calibrated F", "NVL8 measured F", "Generic AIC"],
                y_label="AFD throughput / AGG throughput (×)",
                x_label="Input context; common 0–1.8× scale",
                y_max=1.8,
                value_digits=3,
            ),
            "This locked no-MTP case is separate from the four-way MTP study below. The calibrated track reproduces FastAFD by solving the effective F time; the NVL8 track independently checks MegaMoE compute/load but lacks NVL72 sender fan-in; the generic track exposes AIC's current F-stage overestimate.",
        )
        + table(["ISL", "FastAFD", "AIC calibrated", "NVL8 measured", "Generic AIC", "Contract"], rows)
        + '<p><a href="minimax_m25.html">Open the MiniMax-M2.5 module-by-module reproduction report →</a></p>'
    )


def matrix_evidence_class(model: str, profile: dict[str, Any]) -> tuple[str, str]:
    key = profile["key"]
    if key == "calibrated_fp8_effective_f":
        return "calibrated", "FastAFD E2E-derived effective F calibration"
    if profile["measured_complete_f"]:
        return "measured", "B200 NVL8 measured complete F + AIC system"
    if model == "deepseek_v4_pro":
        return "measured", "Measured MegaMoE module + AIC HYBRID system"
    if "projected" in key:
        return "projected", "AIC HYBRID target-shape projection"
    return "native", "Native AIC / same-shape table HYBRID"


def matrix_aic_record(row: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    row_class, evidence_class = matrix_evidence_class(row["model"], profile)
    nextn = int(row["nextn"])
    mtp = "Off" if nextn == 0 else f"N={nextn}; accepted={row['accepted_drafts']:.3f}; P={row['progress']:.3f}"
    return {
        "model": row["model"],
        "model_label": MODEL_LABELS[row["model"]],
        "workload": row["workload"],
        "scenario": row["scenario"],
        "mtp": mtp,
        "nextn": nextn,
        "progress": float(row["progress"]),
        "track": TRACK_LABELS[(row["model"], profile["key"])],
        "profile": profile["key"],
        "primary": bool(profile["primary"]),
        "evidence": row["evidence"],
        "evidence_class": evidence_class,
        "row_class": row_class,
        "attention": ATTENTION_SHORT[row["model"]],
        "f_precision": row["quant"]["f_moe"],
        "total_gpus": TOTAL_GPUS,
        "a_nodes": int(row["a_nodes"]),
        "f_nodes": int(row["f_nodes"]),
        "a_tp": int(row["a_tp"]),
        "batch_per_a_gpu": int(row["batch_per_a_gpu"]),
        "microbatches": int(row["microbatches"]),
        "a_ms": float(row["a_full_work_ms"]),
        "f_ms": float(row["f_full_work_ms"]),
        "agg_round_ms": float(row["agg"]["raw_round_ms"]),
        "afd_round_ms": float(row["raw_round_ms"]),
        "effective_tpot_ms": float(row["effective_tpot_ms"]),
        "agg_tps_gpu": float(row["agg"]["cluster_output_tokens_s_gpu"]),
        "afd_tps_gpu": float(row["output_tokens_s_gpu"]),
        "afd_over_agg": float(row["afd_over_agg"]),
        "bottleneck": row["pipeline_bottleneck"],
    }


def matrix_public_record(model: str, workload: str, reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "workload": workload,
        "scenario": "no_mtp",
        "mtp": "Off",
        "nextn": 0,
        "progress": 1.0,
        "track": "FastAFD silicon",
        "profile": "published_fastafd",
        "primary": False,
        "evidence": "FastAFD published silicon throughput",
        "evidence_class": "Published FastAFD silicon",
        "row_class": "public",
        "attention": ATTENTION_SHORT[model],
        "f_precision": "FP8 block",
        "total_gpus": int(reference["total_gpus"]),
        "a_nodes": int(reference["a_nodes"]),
        "f_nodes": int(reference["f_nodes"]),
        "a_tp": 1,
        "batch_per_a_gpu": int(reference["afd_batch_a_gpu"]),
        "microbatches": 2,
        "a_ms": None,
        "f_ms": None,
        "agg_round_ms": float(reference["baseline_step_ms"]),
        "afd_round_ms": float(reference["afd_step_ms"]),
        "effective_tpot_ms": float(reference["afd_step_ms"]),
        "agg_tps_gpu": float(reference["baseline_tps_gpu"]),
        "afd_tps_gpu": float(reference["afd_tps_gpu"]),
        "afd_over_agg": float(reference["speedup"]),
        "bottleneck": "not published",
    }


def complete_matrix_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_meta = {model["key"]: model for model in payload["models"]}
    minimax_locked = minimax_m25_rows(payload)
    minimax_profile_rows = {
        "calibrated_fp8_effective_f": "calibrated",
        "measured_fp8_complete_f": "measured",
        "generic_fp8": "generic",
    }
    records: list[dict[str, Any]] = []
    for model in ALL_MODEL_LINK_ORDER:
        meta = model_meta[model]
        for workload in CONTEXTS:
            reference = FASTAFD_REFERENCE.get(model, {}).get(workload)
            if reference is not None:
                records.append(matrix_public_record(model, workload, reference))
            if model == "minimax_m25":
                for profile in meta["precision_profiles"]:
                    row = minimax_locked[workload][minimax_profile_rows[profile["key"]]]
                    records.append(matrix_aic_record(row, profile))
                continue
            for scenario in meta["scenarios"]:
                for profile in meta["precision_profiles"]:
                    candidates = [
                        row
                        for row in payload["rows"]
                        if row["model"] == model
                        and row["workload"] == workload
                        and row["scenario"] == scenario["name"]
                        and row["precision_profile"] == profile["key"]
                    ]
                    if not candidates:
                        continue
                    row = max(candidates, key=lambda value: value["output_tokens_s_gpu"])
                    records.append(matrix_aic_record(row, profile))
    return records


def render_complete_matrix(records: list[dict[str, Any]]) -> str:
    rows = []
    classes = []
    for record in records:
        ratio = record["afd_over_agg"]
        config = (
            f"{record['a_nodes']}A:{record['f_nodes']}F / TP{record['a_tp']} / "
            f"b{record['batch_per_a_gpu']} / MB{record['microbatches']}"
        )
        rows.append(
            [
                esc(record["model_label"]),
                record["workload"].upper(),
                esc(record["mtp"]),
                esc(record["track"] + (" ★" if record["primary"] else "")),
                esc(record["profile"]),
                esc(record["evidence_class"]),
                esc(record["attention"]),
                esc(record["f_precision"]),
                str(record["total_gpus"]),
                esc(config),
                fmt(record["a_ms"], 3) if record["a_ms"] is not None else "—",
                fmt(record["f_ms"], 3) if record["f_ms"] is not None else "—",
                fmt(record["agg_round_ms"], 3),
                fmt(record["afd_round_ms"], 3),
                fmt(record["effective_tpot_ms"], 3),
                fmt(record["agg_tps_gpu"], 1),
                fmt(record["afd_tps_gpu"], 1),
                f'<span class="{"good" if ratio > 1 else "bad"}">{ratio:.3f}×</span>',
                esc(record["bottleneck"]),
            ]
        )
        classes.append(record["row_class"])
    return (
        "<h2>Complete simulation and measurement matrix</h2>"
        f'<div class="callout"><strong>{len(records)} comparable result rows.</strong> Each AIC row is the '
        "highest-throughput "
        "feasible grid point for one exact model × context × MTP scenario × precision/evidence profile. "
        "MiniMax-M2.5 is the exception: its three AIC rows stay locked to the public 17A:1F workload so the "
        "calibrated, NVL8-measured-F, and generic tracks remain directly comparable. FastAFD rows are published "
        "silicon points, not AIC grid winners. The GPU column matters: public Qwen uses 32/48 GPUs, while the AIC "
        "cross-model study and all MiniMax-M2.5 rows use 72 GPUs.</div>"
        + '<p class="small"><strong>Reading the times.</strong> A and F are full raw-round stage work and are '
        "pipelined, not added. For MTP, raw-round time includes q=N+1 verification work; effective TPOT divides "
        "that round by expected committed-token progress P. On an MTP row, the AGG and AFD throughput columns "
        "mean AGG+MTP and AGG+AFD+MTP. Green rows are public silicon, amber is calibrated, pink contains measured "
        "kernel/stage input, and violet is projected.</p>"
        + table(
            [
                "Model",
                "ISL",
                "MTP / progress",
                "Result track",
                "Exact profile",
                "Evidence class",
                "Attention",
                "F precision",
                "GPUs",
                "AFD config",
                "A work (ms)",
                "F work (ms)",
                "AGG round (ms)",
                "AFD round (ms)",
                "AFD effective TPOT (ms)",
                "AGG tok/s/GPU",
                "AFD tok/s/GPU",
                "AFD/AGG",
                "AFD bottleneck",
            ],
            rows,
            classes=classes,
            wrapper_class="matrix",
        )
    )


def render_index(
    payload: dict[str, Any],
    selected: dict[str, dict[str, dict[str, dict[str, Any]]]],
    mocker: dict[str, Any],
    complete_matrix: list[dict[str, Any]],
) -> str:
    nav = (
        '<div class="nav"><span class="pill">Cross-model summary</span>'
        + "".join(f'<a href="{esc(model)}.html">{esc(MODEL_LABELS[model])}</a>' for model in ALL_MODEL_LINK_ORDER)
        + "</div>"
    )
    body = nav
    body += (
        '<div class="callout"><strong>Question answered.</strong> For each model and context, compare '
        "AGG+AFD against AGG, then compare AGG+AFD+MTP against AGG+MTP. All values use the same 72-GPU "
        "NVL72 budget, offered concurrency, and MoE precision within a four-way group. The headline precision "
        "policy is FP4 first, then FP8, then BF16; a lower-precision result is never compared against a "
        "higher-precision baseline.</div>"
    )
    body += render_complete_matrix(complete_matrix)
    if any(model["key"] == "minimax_m25" for model in payload.get("models", [])):
        body += minimax_m25_index_section(payload)
    body += "<h2>1. Cross-model AFD result</h2>"
    ratio_groups = []
    speedup_groups = []
    summary_rows = []
    for model in MODEL_ORDER:
        for workload in CONTEXTS:
            values = four_way(selected[model][workload])
            no_ratio = values["AGG + AFD"] / values["AGG"]
            mtp_ratio = values["AGG + AFD + MTP"] / values["AGG + MTP"]
            agg_mtp = values["AGG + MTP"] / values["AGG"]
            afd_mtp = values["AGG + AFD + MTP"] / values["AGG + AFD"]
            label = f"{MODEL_LABELS[model]} {workload.upper()}"
            ratio_groups.append((label, {"No MTP": no_ratio, "With MTP": mtp_ratio}))
            speedup_groups.append((label, {"AGG": agg_mtp, "AGG + AFD": afd_mtp}))
            mtp = selected[model][workload]["mtp"]
            summary_rows.append(
                [
                    f'<a href="{model}.html">{esc(MODEL_LABELS[model])}</a> {workload.upper()}',
                    fmt(values["AGG"], 1),
                    fmt(values["AGG + MTP"], 1),
                    fmt(values["AGG + AFD"], 1),
                    fmt(values["AGG + AFD + MTP"], 1),
                    f'<span class="{"good" if no_ratio > 1 else "bad"}">{no_ratio:.3f}×</span>',
                    f'<span class="{"good" if mtp_ratio > 1 else "bad"}">{mtp_ratio:.3f}×</span>',
                    f"{mtp['a_nodes']}A:{mtp['f_nodes']}F / TP{mtp['a_tp']} / b{mtp['batch_per_a_gpu']} / MB{mtp['microbatches']}",
                ]
            )
    winning_no_mtp = [label for label, values in ratio_groups if values["No MTP"] > 1]
    winning_mtp = [label for label, values in ratio_groups if values["With MTP"] > 1]
    body += figure(
        svg_grouped_bars(
            ratio_groups,
            ["No MTP", "With MTP"],
            y_label="AFD throughput / AGG throughput (×)",
            x_label="Model and input context",
            y_max=2.0,
            value_digits=2,
        ),
        "The horizontal decision boundary is 1×. No-MTP AFD is above that boundary for "
        + (", ".join(winning_no_mtp) if winning_no_mtp else "none of the cases")
        + "; AFD+MTP is above it for "
        + (", ".join(winning_mtp) if winning_mtp else "none of the cases")
        + ". These are simulator conclusions under the evidence and precision contract in Section 4, not blanket hardware claims.",
    )
    body += table(
        ["Case", "AGG", "AGG+MTP", "AGG+AFD", "AGG+AFD+MTP", "AFD/AGG", "AFD+MTP / AGG+MTP", "Selected AFD config"],
        summary_rows,
    )

    body += "<h2>2. MTP gain inside each topology</h2>"
    body += figure(
        svg_grouped_bars(
            speedup_groups,
            ["AGG", "AGG + AFD"],
            y_label="MTP throughput / no-MTP throughput (×)",
            x_label="Model and input context",
            y_max=2.5,
            value_digits=2,
        ),
        "This chart separates MTP benefit from AFD benefit. Verification is a cost; multiple committed tokens are the benefit. Qwen AFD keeps raw verification growth small, while V4-Flash's AFD raw cost nearly equals its 2.19× progress and therefore produces almost no net MTP gain.",
    )

    body += "<h2>3. Acceptance semantics</h2>"
    body += (
        '<div class="callout warn"><strong>For N=3, q=4; 2.28125/3 is not the acceptance rate.</strong> '
        "2.28125 is expected committed output tokens per target verification round, so accepted drafts are 1.28125. "
        "An equal conditional-rate surrogate is r=63.1%, because 1+r+r²+r³=2.28125. The report never multiplies q "
        "into output tokens; q affects compute, while P affects committed output progress.</div>"
    )
    acceptance_rows = []
    for model in MODEL_ORDER:
        row = selected[model]["8k"]["mtp"]
        basis = next(
            scenario["acceptance_basis"]
            for value in payload["models"]
            if value["key"] == model
            for scenario in value["scenarios"]
            if scenario["name"] == row["scenario"]
        )
        acceptance_rows.append(
            [
                f'<a href="{model}.html">{esc(MODEL_LABELS[model])}</a>',
                str(row["nextn"]),
                str(row["q"]),
                fmt(row["accepted_drafts"], 5),
                fmt(row["progress"], 5),
                esc(basis),
            ]
        )
    body += table(["Model", "N", "q=N+1", "E[accepted drafts]", "Progress P", "Evidence / assumption"], acceptance_rows)

    body += "<h2>4. Evidence strength and model support</h2>"
    evidence_rows = []
    for model in MODEL_ORDER:
        title, note = evidence_note(model)
        evidence_rows.append(
            [
                f'<a href="{model}.html">{esc(MODEL_LABELS[model])}</a>',
                esc(title),
                esc(note),
            ]
        )
    evidence_rows.append(
        [
            "Kimi-K3",
            "Excluded",
            "AIC has no Kimi-K3 model. Borrowing Kimi-K2 timings would conflate KDA/Gated-MLA, AttnRes, Stable LatentMoE, 896 experts/top-16, and a different active parameter scale; no auditable simulation is reported.",
        ]
    )
    body += table(["Model", "Evidence class", "What the result means"], evidence_rows)

    body += "<h2>5. Attention structure and F-side MoE precision</h2>"
    contract_rows = []
    for model in MODEL_ORDER:
        model_meta = next(value for value in payload["models"] if value["key"] == model)
        primary = primary_precision(model_meta)
        contract_rows.append(
            [
                f'<a href="{model}.html">{esc(MODEL_LABELS[model])}</a>',
                esc(model_meta["attention_note"]),
                esc(model_meta["attention_timing_note"]),
                esc(model_meta["moe_shape_note"]),
                esc(primary["moe_quant_mode"]),
                esc(primary["timing_note"]),
            ]
        )
    body += table(
        ["Model", "A-side attention", "Attention timing", "F-side MoE shape", "F precision", "MoE timing"],
        contract_rows,
    )
    body += (
        '<div class="callout warn"><strong>Structural support is not the same as silicon calibration.</strong> '
        "MiniMax-M3 has neither an MSA silicon table nor a same-shape MoE row. DeepSeek-V4 Flash uses an HCA "
        "timing proxy for two pure-SWA layers. MTP attention is q-wide HYBRID estimation for all sparse-attention "
        "models in this report.</div>"
    )

    body += "<h2>6. Dynamo Mocker token-accounting validation</h2>"
    mock_groups = []
    for model in MODEL_ORDER:
        model_rows = [row for row in mocker["results"] if row["model"] == model]
        for workload in CONTEXTS:
            context_rows = [row for row in model_rows if row["workload"] == workload]
            mock_groups.append(
                (
                    f"{MODEL_LABELS[model]} {workload.upper()}",
                    {
                        "No MTP": max(abs(row["mean_tpot_error_pct"]) for row in context_rows if not row["nextn"]),
                        "With MTP": max(abs(row["mean_tpot_error_pct"]) for row in context_rows if row["nextn"]),
                    },
                )
            )
    body += figure(
        svg_grouped_bars(
            mock_groups,
            ["No MTP", "With MTP"],
            y_label="Maximum absolute mean-TPOT error versus AIC (%)",
            x_label="Model and input context",
            y_max=1.0,
            value_digits=2,
        ),
        "All 32 replay cases are below 0.9% mean-TPOT error. This validates the request lifecycle and stochastic accepted-token accounting against the AIC steady-state equation. Finite one-wave throughput is lower for MTP because completion waits for the slowest sampled acceptance chain; it is not used as the capacity metric.",
    )

    body += "<h2>7. Reproduction identity</h2>"
    body += table(
        ["Artifact", "Identity"],
        [
            ["Base sweep", f"branch={esc(payload['code']['branch'])}; commit={esc(payload['code']['commit'])}"],
            ["Corrected overlay", esc(json.dumps(payload.get("applied_overlays", []), sort_keys=True))],
            ["Dynamo Mocker", f"branch={esc(mocker['dynamo_branch'])}; commit={esc(mocker['dynamo_commit'])}"],
            ["Mocker scope", esc(mocker["profile_note"])],
        ],
    )
    return document(
        "GB200 NVL72 Multi-Model AFD × MTP Simulation",
        "MiniMax-M2.5 FastAFD reproduction plus Qwen3-235B, MiniMax-M3, and DeepSeek-V4 AFD × MTP studies",
        body,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--overlay-sweep", type=Path, action="append", default=[])
    parser.add_argument("--mocker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.sweep.resolve().read_text(encoding="utf-8"))
    overlays = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in args.overlay_sweep]
    payload = merge_payload(base, overlays)
    mocker = json.loads(args.mocker.resolve().read_text(encoding="utf-8"))
    selected = select_pairs(payload)
    complete_matrix = complete_matrix_records(payload)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    if any(model["key"] == "minimax_m25" for model in payload.get("models", [])):
        report, summary = render_minimax_m25(payload, mocker)
        (output_dir / "minimax_m25.html").write_text(report, encoding="utf-8")
        summaries.append(summary)
    for model in MODEL_ORDER:
        report, summary = render_model(model, payload, selected[model], mocker)
        (output_dir / f"{model}.html").write_text(report, encoding="utf-8")
        summaries.append(summary)
    (output_dir / "index.html").write_text(render_index(payload, selected, mocker, complete_matrix), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "aic.afd-multimodel-mtp-report.v1",
                "base_code": payload["code"],
                "overlays": payload.get("applied_overlays", []),
                "dynamo": {"branch": mocker["dynamo_branch"], "commit": mocker["dynamo_commit"]},
                "complete_matrix": complete_matrix,
                "models": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "reports": len(summaries) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
