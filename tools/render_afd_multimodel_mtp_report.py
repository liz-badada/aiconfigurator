#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Render self-contained HTML reports for fixed-pool AGG/AFD/MTP sweeps."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CONTEXTS = ("8k", "16k")
TOTAL_GPU_GRID = (16, 24, 36, 48, 72)
MODEL_ORDER = (
    "qwen3_235b",
    "minimax_m25",
    "minimax_m3",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
)
SYSTEM_LABELS = {
    "agg_no_mtp": "AGG",
    "afd_no_mtp": "AGG + AFD",
    "agg_mtp": "AGG + MTP",
    "afd_mtp": "AGG + AFD + MTP",
}
COLORS = {
    "AGG": "#0072B2",
    "AGG + AFD": "#D55E00",
    "AGG + MTP": "#56B4E9",
    "AGG + AFD + MTP": "#E69F00",
    "No MTP": "#0072B2",
    "With MTP": "#E69F00",
    "A path": "#0072B2",
    "F path": "#D55E00",
    "Pipeline cycle": "#111111",
    "attention": "#0072B2",
    "mHC": "#56B4E9",
    "dense GEMM": "#009E73",
    "router": "#F0E442",
    "MoE / shared expert": "#D55E00",
    "F collective": "#CC79A7",
    "A combine": "#E69F00",
    "A-F transfer": "#111111",
    "norm / embedding / logits": "#999999",
}
MODULE_ORDER = (
    "attention",
    "mHC",
    "dense GEMM",
    "router",
    "MoE / shared expert",
    "F collective",
    "A combine",
    "A-F transfer",
    "norm / embedding / logits",
)

CSS = """
:root{--ink:#17202a;--muted:#5f6b76;--line:#d8dee4;--panel:#f7f9fb;--blue:#0072B2;--orange:#D55E00;--green:#007a55}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.52 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1280px;margin:0 auto;padding:34px 34px 72px}h1{font-size:32px;line-height:1.15;margin:0 0 8px}h2{margin:38px 0 12px;padding-top:10px;border-top:2px solid var(--ink);font-size:22px}h3{font-size:17px;margin:22px 0 8px}.subtitle,.muted{color:var(--muted)}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.nav a,.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;text-decoration:none;color:var(--ink);background:#fff}.nav a:hover{border-color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}.card{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--panel)}.card .value{font-size:25px;font-weight:750}.card .label{color:var(--muted);font-size:13px}
.callout{border-left:5px solid var(--blue);background:#eef7fb;padding:12px 15px;margin:14px 0}.warn{border-left-color:var(--orange);background:#fff4ef}.ok{border-left-color:var(--green);background:#effaf6}
.figure{border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0;background:#fff}.figure svg{display:block;width:100%;height:auto}.comment{margin:10px 4px 2px;color:#34404b}.comment strong{color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:right;vertical-align:top}th{background:#f1f4f6;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}.wide table{min-width:1500px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.small{font-size:12px}.good{color:var(--green);font-weight:750}.bad{color:#b43b20;font-weight:750}.neutral{color:#5f6b76;font-weight:700}.foot{margin-top:38px;color:var(--muted);font-size:12px}
svg text{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;fill:#24313d}.grid{stroke:#dfe5ea;stroke-width:1}.axis{stroke:#53606b;stroke-width:1.2}.legend{font-size:12px}.tick{font-size:11px}.value-label{font-size:10px;font-weight:650}.ref{stroke:#777;stroke-width:1.2;stroke-dasharray:5 4}
@media print{main{max-width:none;padding:18px}.figure{break-inside:avoid}a{color:inherit}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def ratio_html(value: float | None) -> str:
    if value is None:
        return '<span class="neutral">—</span>'
    css = "good" if value > 1.005 else "bad" if value < 0.995 else "neutral"
    return f'<span class="{css}">{value:.3f}×</span>'


def table(headers: list[str], rows: Iterable[list[object]], *, css: str = "") -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
    return (
        f'<div class="scroll {css}"><table><thead><tr>'
        + "".join(f"<th>{esc(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def figure(svg: str, comment: str, contract_note: str | None = None) -> str:
    contract = (
        f'<p class="small muted"><strong>Backend contract:</strong> {contract_note}</p>' if contract_note else ""
    )
    return f'<div class="figure">{svg}{contract}<p class="comment"><strong>How to read:</strong> {comment}</p></div>'


def document(title: str, subtitle: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{esc(title)}</h1><p class=\"subtitle\">{subtitle}</p>{body}</main></body></html>"
    )


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    common = (
        row["system_kind"],
        row["model"],
        row["workload"],
        row["scenario"],
        row["precision_profile"],
        row["total_gpus"],
    )
    if row["system_kind"] == "agg":
        return common + (row["world"], row["tp"], row["local_batch"])
    return common + (
        row["a_gpus"],
        row["f_gpus"],
        row["a_tp"],
        row["batch_per_a_gpu"],
        row["microbatches"],
    )


def failure_key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def load_payload(paths: list[Path]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    contracts: list[dict[str, Any]] = []
    code: list[dict[str, Any]] = []
    sources = []
    for path in paths:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        if payload.get("schema") != "aic.afd-fixed-pool-sweep.v3":
            raise ValueError(f"unsupported sweep schema in {path}: {payload.get('schema')}")
        sources.append(str(path.resolve()))
        contracts.append(payload["contract"])
        code.append(payload["code"])
        for model in payload["models"]:
            previous = models.setdefault(model["key"], model)
            if previous != model:
                raise ValueError(f"inconsistent model metadata for {model['key']}")
        for row in payload["agg_rows"] + payload["afd_rows"]:
            rows[row_key(row)] = row
        for failure in payload.get("failures", []):
            failures[failure_key(failure)] = failure
    if not contracts:
        raise ValueError("at least one sweep is required")
    invariant_fields = (
        "system",
        "backend",
        "database_mode",
        "gpus_per_node",
        "pipeline_model",
        "decode_stride",
        "batch_semantics",
        "mtp_compute",
        "mtp_progress",
    )
    for contract in contracts[1:]:
        for field in invariant_fields:
            if contract[field] != contracts[0][field]:
                raise ValueError(f"inconsistent sweep contract: {field}")
    return {
        "schema": "aic.afd-fixed-pool-report-input.v1",
        "models": models,
        "rows": list(rows.values()),
        "failures": list(failures.values()),
        "contract": contracts[0],
        "code": code,
        "sources": sources,
    }


def primary_profile(model: dict[str, Any]) -> dict[str, Any]:
    profiles = [profile for profile in model["precision_profiles"] if profile["primary"]]
    if len(profiles) != 1:
        raise ValueError(f"expected one primary precision profile for {model['key']}")
    return profiles[0]


def primary_mtp(model: dict[str, Any]) -> dict[str, Any]:
    scenarios = [scenario for scenario in model["scenarios"] if scenario["primary"]]
    if len(scenarios) != 1:
        raise ValueError(f"expected one primary MTP scenario for {model['key']}")
    return scenarios[0]


def moe_backend_label(model: dict[str, Any]) -> str:
    if model.get("moe_backend") == "megamoe":
        return "MegaMoE"
    return f"SGLang {model['backend_version']}"


def select_best(
    payload: dict[str, Any],
    *,
    model: str,
    workload: str,
    scenario: str,
    precision: str,
    total_gpus: int,
    system_kind: str,
    speed_floor: float,
) -> dict[str, Any] | None:
    base = [
        row
        for row in payload["rows"]
        if row["model"] == model
        and row["workload"] == workload
        and row["scenario"] == scenario
        and row["precision_profile"] == precision
        and row["system_kind"] == system_kind
        and row["tokps_per_user"] >= speed_floor
    ]
    if system_kind == "agg":
        candidates = [row for row in base if row["total_gpus"] == total_gpus]
    else:
        candidates = [
            materialize_afd_cluster(row, total_gpus)
            for row in base
            if row["total_gpus"] <= total_gpus
        ]
    return max(candidates, key=lambda row: row["output_tokens_s_gpu"], default=None)


def materialize_afd_cluster(row: dict[str, Any], total_gpus: int) -> dict[str, Any]:
    """Pack identical AFD units into a fixed pool and charge idle GPUs."""
    unit_gpus = int(row["total_gpus"])
    replicas = total_gpus // unit_gpus
    if replicas < 1:
        raise ValueError(f"AFD unit {unit_gpus} does not fit in {total_gpus} GPUs")
    used_gpus = replicas * unit_gpus
    result = dict(row)
    result.update(
        {
            "unit_gpus": unit_gpus,
            "afd_replicas": replicas,
            "total_gpus": total_gpus,
            "used_gpus": used_gpus,
            "idle_gpus": total_gpus - used_gpus,
            "cluster_a_gpus": row["a_gpus"] * replicas,
            "cluster_f_gpus": row["f_gpus"] * replicas,
            "cluster_concurrency": row["global_requests"] * replicas,
            "output_tokens_s": row["output_tokens_s"] * replicas,
            "output_tokens_s_gpu": row["output_tokens_s"] * replicas / total_gpus,
        }
    )
    return result


def paired_winner(
    payload: dict[str, Any],
    *,
    model: str,
    workload: str,
    scenario: str,
    precision: str,
    total_gpus: int,
    speed_floor: float,
) -> dict[str, Any]:
    agg = select_best(
        payload,
        model=model,
        workload=workload,
        scenario=scenario,
        precision=precision,
        total_gpus=total_gpus,
        system_kind="agg",
        speed_floor=speed_floor,
    )
    afd = select_best(
        payload,
        model=model,
        workload=workload,
        scenario=scenario,
        precision=precision,
        total_gpus=total_gpus,
        system_kind="afd",
        speed_floor=speed_floor,
    )
    ratio = None
    if agg is not None and afd is not None:
        comparable = ("framework", "moe_backend", "moe_kernel", "moe_precision", "attention_backend")
        mismatch = [
            field
            for field in comparable
            if agg["backend_contract"].get(field) != afd["backend_contract"].get(field)
        ]
        if mismatch:
            raise ValueError(
                f"backend mismatch for {model}/{workload}/{scenario}/{total_gpus}: {', '.join(mismatch)}"
            )
        ratio = afd["output_tokens_s_gpu"] / agg["output_tokens_s_gpu"]
    return {"agg": agg, "afd": afd, "ratio": ratio}


def winners_for_model(payload: dict[str, Any], model: dict[str, Any], speed_floor: float) -> dict[tuple, dict]:
    profile = primary_profile(model)["key"]
    mtp = primary_mtp(model)["name"]
    result = {}
    for workload in CONTEXTS:
        for total in TOTAL_GPU_GRID:
            for scenario in ("no_mtp", mtp):
                result[(workload, total, scenario)] = paired_winner(
                    payload,
                    model=model["key"],
                    workload=workload,
                    scenario=scenario,
                    precision=profile,
                    total_gpus=total,
                    speed_floor=speed_floor,
                )
    return result


def agg_config(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    return (
        f"unit {row['world']} GPU; TP{row['tp']}/DP{row['dp']}; "
        f"MoE EP{row['moe_ep']}; batch {row['local_batch']}/DP worker; ×{row['replicas']} replicas"
    )


def afd_config(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    replicas = int(row.get("afd_replicas", 1))
    unit_gpus = int(row.get("unit_gpus", row["a_gpus"] + row["f_gpus"]))
    packing = (
        f"unit {row['a_gpus']}A:{row['f_gpus']}F ({unit_gpus} GPU) x{replicas}; "
        f"cluster {row.get('cluster_a_gpus', row['a_gpus'])}A:{row.get('cluster_f_gpus', row['f_gpus'])}F"
    )
    if row.get("idle_gpus", 0):
        packing += f" + {row['idle_gpus']} idle"
    return (
        f"{packing}; A-TP{row['a_tp']}; "
        f"batch {row['batch_per_a_gpu']}/A-GPU ({row['a_batch_size_per_worker']}/A worker); "
        f"microbatch {row['microbatches']}"
    )


def nice_max(value: float, minimum: float = 1.0) -> float:
    value = max(value, minimum)
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    step = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return step * magnitude


def line_svg(
    series: dict[str, list[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
    x_ticks: tuple[int, ...] = TOTAL_GPU_GRID,
    y_min: float = 0.0,
    y_max: float | None = None,
    reference_y: float | None = None,
    width: int = 900,
    height: int = 390,
) -> str:
    left, right, top, bottom = 82, 28, 34, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for points in series.values() for _, value in points if math.isfinite(value)]
    if y_max is None:
        y_max = nice_max(max(values, default=1.0) * 1.08)
    if y_max <= y_min:
        y_max = y_min + 1
    x_min, x_max = min(x_ticks), max(x_ticks)

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = y_pos(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for value in x_ticks:
        x = x_pos(value)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle">{value}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}"/>')
    if reference_y is not None and y_min <= reference_y <= y_max:
        y = y_pos(reference_y)
        parts.append(f'<line class="ref" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
    for label, points in series.items():
        color = COLORS.get(label, "#666")
        ordered = sorted(points)
        coords = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in ordered)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in ordered:
            parts.append(f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="4" fill="{color}"/>')
    legend_x = left
    for label in series:
        color = COLORS.get(label, "#666")
        parts.append(f'<rect x="{legend_x}" y="8" width="13" height="4" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x+18}" y="14">{esc(label)}</text>')
        legend_x += 28 + len(label) * 7
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(
        f'<text transform="translate(18 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_svg(
    categories: list[str],
    series: dict[str, list[float]],
    *,
    x_label: str,
    y_label: str,
    y_max: float | None = None,
    width: int = 980,
    height: int = 420,
) -> str:
    left, right, top, bottom = 86, 24, 42, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [value for values in series.values() for value in values if math.isfinite(value)]
    if y_max is None:
        y_max = nice_max(max(all_values, default=1.0) * 1.08)
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(30.0, group_w * 0.78 / max(len(series), 1))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index in range(6):
        value = y_max * index / 5
        y = top + plot_h - value / y_max * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>')
    labels = list(series)
    for group_index, category in enumerate(categories):
        center = left + (group_index + 0.5) * group_w
        start = center - len(labels) * bar_w / 2
        for series_index, label in enumerate(labels):
            value = series[label][group_index]
            bar_h = max(value, 0) / y_max * plot_h
            x = start + series_index * bar_w
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{bar_h:.1f}" fill="{COLORS.get(label, "#666")}"/>'
            )
        parts.append(
            f'<text class="tick" x="{center:.1f}" y="{top+plot_h+22}" text-anchor="middle">{esc(category)}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}"/>')
    legend_x = left
    for label in labels:
        parts.append(f'<rect x="{legend_x}" y="13" width="12" height="12" fill="{COLORS.get(label, "#666")}"/>')
        parts.append(f'<text class="legend" x="{legend_x+17}" y="23">{esc(label)}</text>')
        legend_x += 30 + len(label) * 7
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(
        f'<text transform="translate(18 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def stacked_bar_svg(
    categories: list[str],
    values: list[dict[str, float]],
    *,
    y_max: float,
    width: int = 980,
    height: int = 430,
) -> str:
    left, right, top, bottom = 86, 24, 58, 102
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(76.0, group_w * 0.62)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index in range(6):
        value = y_max * index / 5
        y = top + plot_h - value / y_max * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>')
    for index, (category, totals) in enumerate(zip(categories, values, strict=True)):
        x = left + (index + 0.5) * group_w - bar_w / 2
        cursor = top + plot_h
        for group in MODULE_ORDER:
            value = totals.get(group, 0.0)
            bar_h = value / y_max * plot_h
            cursor -= bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{COLORS[group]}"/>'
            )
        parts.append(
            f'<text class="tick" x="{x+bar_w/2:.1f}" y="{top+plot_h+22}" text-anchor="middle">{esc(category)}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}"/>')
    legend_x, legend_y = left, 14
    for group in MODULE_ORDER:
        width_guess = 34 + len(group) * 6.5
        if legend_x + width_guess > width - right:
            legend_x = left
            legend_y += 20
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="11" height="11" fill="{COLORS[group]}"/>')
        parts.append(f'<text class="legend" x="{legend_x+15}" y="{legend_y+10}">{esc(group)}</text>')
        legend_x += width_guess
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle">System / context</text>')
    parts.append(
        f'<text transform="translate(18 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">Raw module work (ms / decode round)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def scatter_svg(
    series: dict[str, list[tuple[float, float]]],
    *,
    x_max: float,
    y_max: float,
    width: int = 900,
    height: int = 400,
) -> str:
    left, right, top, bottom = 82, 26, 38, 68
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(value: float) -> float:
        return left + value / x_max * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index in range(6):
        x_value = x_max * index / 5
        y_value = y_max * index / 5
        x = x_pos(x_value)
        y = y_pos(y_value)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle">{x_value:.0f}</text>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{y_value:.0f}</text>')
    for label, points in series.items():
        color = COLORS[label]
        ordered = sorted(points)
        coords = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in ordered)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in ordered:
            parts.append(f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3.6" fill="{color}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}"/>')
    legend_x = left
    for label in series:
        parts.append(f'<rect x="{legend_x}" y="12" width="12" height="5" fill="{COLORS[label]}"/>')
        parts.append(f'<text class="legend" x="{legend_x+17}" y="18">{esc(label)}</text>')
        legend_x += 30 + len(label) * 7
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle">Effective TPOT (ms / committed token)</text>')
    parts.append(
        f'<text transform="translate(18 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">Output throughput (tokens/s/GPU)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def module_totals(row: dict[str, Any]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for module in row.get("modules", []):
        totals[module["group"]] += float(module["raw_work_ms"])
    return dict(totals)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    best_throughput = -math.inf
    for row in sorted(rows, key=lambda value: (value["effective_tpot_ms"], -value["output_tokens_s_gpu"])):
        throughput = row["output_tokens_s_gpu"]
        if throughput > best_throughput:
            front.append(row)
            best_throughput = throughput
    return front


def ratio_series(winners: dict[tuple, dict], workload: str) -> dict[str, list[tuple[float, float]]]:
    scenarios = sorted({key[2] for key in winners})
    no_mtp = "no_mtp"
    mtp = next(name for name in scenarios if name != no_mtp)
    return {
        "No MTP": [(total, winners[(workload, total, no_mtp)]["ratio"]) for total in TOTAL_GPU_GRID],
        "With MTP": [(total, winners[(workload, total, mtp)]["ratio"]) for total in TOTAL_GPU_GRID],
    }


def throughput_series(
    winners: dict[tuple, dict], workload: str, mtp_name: str
) -> dict[str, list[tuple[float, float]]]:
    mapping = {
        "AGG": ("no_mtp", "agg"),
        "AGG + AFD": ("no_mtp", "afd"),
        "AGG + MTP": (mtp_name, "agg"),
        "AGG + AFD + MTP": (mtp_name, "afd"),
    }
    result = {}
    for label, (scenario, side) in mapping.items():
        points = []
        for total in TOTAL_GPU_GRID:
            row = winners[(workload, total, scenario)][side]
            if row is not None:
                points.append((total, row["output_tokens_s_gpu"]))
        result[label] = points
    return result


def model_navigation(payload: dict[str, Any], current: str | None = None) -> str:
    links = ['<a href="index.html">Cross-model index</a>']
    for key in MODEL_ORDER:
        if key not in payload["models"]:
            continue
        label = payload["models"][key]["label"]
        if key == current:
            links.append(f'<span class="pill">{esc(label)}</span>')
        else:
            links.append(f'<a href="{key}.html">{esc(label)}</a>')
    return '<div class="nav">' + "".join(links) + "</div>"


def contract_section(model: dict[str, Any], contract: dict[str, Any]) -> str:
    profile = primary_profile(model)
    mtp = primary_mtp(model)
    exact = "same-shape data" if profile["exact_shape_data"] else "projected / transferred utilization"
    return (
        "<h2>1. Compared contract</h2>"
        '<div class="callout"><strong>Matched-backend rule.</strong> AGG and AFD use the same framework, '
        "MoE kernel, MoE precision, model and fixed GPU budget. The ratio therefore measures the serving layout, "
        "not a backend substitution.</div>"
        + table(
            ["Item", "Value", "Evidence / interpretation"],
            [
                ["Model", esc(model["label"]), esc(model["parameter_note"])],
                ["Attention structure", esc(model["attention_type"]), esc(model["attention_evidence"])],
                ["Attention backend", esc(model["attention_backend"]), "Used by both AGG and AFD A-side"],
                ["MoE structure", esc(model["moe_structure"]), f"Top-{model['topk']}, {model['layers']} layers"],
                [
                    "MoE backend / kernel",
                    esc(f"{moe_backend_label(model)} / {profile['moe_kernel']}"),
                    esc(profile["evidence"]),
                ],
                ["MoE precision", esc(profile["moe_quant_mode"]), exact],
                [
                    "MTP",
                    f"nextN={mtp['nextn']}; verify width q={mtp['verification_width']}; progress P={mtp['progress']:.4f}",
                    esc(mtp["acceptance_basis"]),
                ],
                [
                    "System",
                    "GB200; 16/24/36/48/72 fixed GPUs",
                    "4 GPUs/node; every 8–72 GPU service-unit size and every integer-node A:F split; "
                    "identical units packed, idle remainder charged",
                ],
                ["Database", esc(contract["database_mode"]), f"decode stride={contract['decode_stride']}"],
            ],
        )
        + '<p class="small muted">MTP accounting: verification executes q=N+1 token positions; committed progress is '
        "P=1+E[accepted drafts]. Effective TPOT=T<sub>round</sub>/P and throughput=concurrency×P/T<sub>round</sub>. "
        "Acceptance assumptions change only P; they do not erase the q-wide attention/MoE work.</p>"
    )


def render_model(payload: dict[str, Any], model: dict[str, Any], speed_floor: float) -> tuple[str, dict[str, Any]]:
    winners = winners_for_model(payload, model, speed_floor)
    profile = primary_profile(model)
    mtp = primary_mtp(model)
    chart_contract = (
        f"Attention: {esc(model['attention_type'])}; {esc(model['attention_backend'])}. "
        f"MoE: {esc(model['moe_structure'])}; backend={esc(moe_backend_label(model))}; "
        f"kernel={esc(profile['moe_kernel'])}; "
        f"precision={esc(profile['moe_quant_mode'])}."
    )
    title = f"{model['label']} — fixed-pool AGG / AFD / MTP sweep"
    body = model_navigation(payload, model["key"])
    body += (
        f'<div class="callout"><strong>Selection objective:</strong> maximize output tokens/s/GPU while each arm '
        f"independently satisfies ≥{speed_floor:g} committed tokens/s/user (effective TPOT ≤{1000/speed_floor:.2f} ms). "
        "Blank entries mean that arm has no feasible configuration.</div>"
    )
    cards = []
    for workload in CONTEXTS:
        no_ratio = winners[(workload, 72, "no_mtp")]["ratio"]
        mtp_ratio = winners[(workload, 72, mtp["name"])]["ratio"]
        cards.extend(
            [
                (f"{workload.upper()} · 72 GPU · no MTP", ratio_html(no_ratio)),
                (f"{workload.upper()} · 72 GPU · with MTP", ratio_html(mtp_ratio)),
            ]
        )
    body += '<div class="cards">' + "".join(
        f'<div class="card"><div class="value">{value}</div><div class="label">{esc(label)} · AFD / AGG</div></div>'
        for label, value in cards
    ) + "</div>"
    body += contract_section(model, payload["contract"])

    body += "<h2>2. End-to-end fixed-pool performance</h2>"
    throughput = {workload: throughput_series(winners, workload, mtp["name"]) for workload in CONTEXTS}
    common_throughput_max = nice_max(
        max(value for by_context in throughput.values() for points in by_context.values() for _, value in points) * 1.05
    )
    for workload in CONTEXTS:
        body += f"<h3>{workload.upper()} input</h3>"
        body += figure(
            line_svg(
                throughput[workload],
                x_label="Fixed GPU pool size (GPUs)",
                y_label="Output throughput (tokens/s/GPU)",
                y_max=common_throughput_max,
            ),
            "Each curve is independently optimized at the same user-speed floor. The y-axis is per total allocated GPU, "
            "so idle GPUs in either AGG or AFD unit packing remain in the denominator. Both context panels use the same y scale.",
            chart_contract,
        )
        ratios = ratio_series(winners, workload)
        ratio_max = nice_max(max(value for points in ratios.values() for _, value in points) * 1.05)
        body += figure(
            line_svg(
                ratios,
                x_label="Fixed GPU pool size (GPUs)",
                y_label="Throughput ratio: AFD / AGG",
                y_max=max(ratio_max, 1.2),
                reference_y=1.0,
            ),
            "The dashed 1.0 line is break-even. Above it, the best matched-backend AFD layout beats the best AGG layout; "
            "below it, partitioning and transfer overhead outweigh A/F overlap at that pool size.",
            chart_contract,
        )

    body += "<h2>3. A:F ratio, stage balance, and end-to-end formula</h2>"
    winner_rows = []
    for workload in CONTEXTS:
        for total in TOTAL_GPU_GRID:
            for scenario, label in (("no_mtp", "No MTP"), (mtp["name"], "With MTP")):
                pair = winners[(workload, total, scenario)]
                agg, afd = pair["agg"], pair["afd"]
                winner_rows.append(
                    [
                        workload.upper(),
                        total,
                        label,
                        agg_config(agg),
                        fmt(agg and agg["output_tokens_s_gpu"]),
                        afd_config(afd),
                        fmt(afd and afd["output_tokens_s_gpu"]),
                        ratio_html(pair["ratio"]),
                        esc(afd["pipeline_bottleneck"] if afd else "—"),
                    ]
                )
    body += table(
        ["ISL", "GPUs", "Mode", "Best AGG", "AGG tok/s/GPU", "Best AFD", "AFD tok/s/GPU", "AFD/AGG", "AFD bottleneck"],
        winner_rows,
        css="wide",
    )

    selected_72 = []
    for workload in CONTEXTS:
        for scenario, label in (("no_mtp", "No MTP"), (mtp["name"], "MTP")):
            pair = winners[(workload, 72, scenario)]
            if pair["afd"]:
                selected_72.append((f"{workload.upper()} {label}", pair["afd"]))
    stage_series = {
        "A path": [(row["t_a_layer_ms"] + row["t_a2f_layer_ms"]) * 1000 for _, row in selected_72],
        "F path": [(row["t_f_layer_ms"] + row["t_f2a_layer_ms"]) * 1000 for _, row in selected_72],
        "Pipeline cycle": [row["t_cycle_layer_ms"] * 1000 for _, row in selected_72],
    }
    body += figure(
        grouped_bar_svg(
            [label for label, _ in selected_72],
            stage_series,
            x_label="72-GPU selected AFD point",
            y_label="Per-layer branch / cycle time (µs)",
        ),
        "A path=A compute+A→F transfer; F path=F compute+F→A transfer. With at least two microbatches, the cycle is "
        "max(A path,F path). It is not A+F. The taller branch is the pipeline bottleneck.",
        chart_contract,
    )
    stage_rows = []
    for label, row in selected_72:
        stage_rows.append(
            [
                esc(label),
                afd_config(row),
                fmt(row["t_a_layer_ms"] * 1000, 1),
                fmt(row["t_f_layer_ms"] * 1000, 1),
                fmt(row["t_a2f_layer_ms"] * 1000, 1),
                fmt(row["t_f2a_layer_ms"] * 1000, 1),
                fmt(row["t_cycle_layer_ms"] * 1000, 1),
                fmt(row["raw_round_ms"], 2),
                fmt(row["effective_tpot_ms"], 2),
            ]
        )
    body += table(
        ["Case", "AFD config", "A µs/layer", "F µs/layer", "A→F µs/layer", "F→A µs/layer", "Cycle µs/layer", "Raw round ms", "Effective TPOT ms"],
        stage_rows,
        css="wide",
    )
    body += (
        '<div class="callout"><strong>AFD service formula.</strong> Fill=A+F+A→F+F→A. For M≥2 microbatches, '
        "cycle=max(A+A→F,F+F→A), and T<sub>round</sub>=fill+(M×L−1)×cycle. For M=1 there is no inter-microbatch "
        "pipeline, so cycle=fill. Effective TPOT divides the raw round by committed MTP progress P.</div>"
    )

    body += "<h2>4. Module work versus overlapped service</h2>"
    module_categories = []
    module_values = []
    for workload in CONTEXTS:
        for scenario, suffix in (("no_mtp", "no MTP"), (mtp["name"], "MTP")):
            pair = winners[(workload, 72, scenario)]
            for side, prefix in (("agg", "AGG"), ("afd", "AFD")):
                row = pair[side]
                if row is not None:
                    module_categories.append(f"{workload.upper()} {prefix} {suffix}")
                    module_values.append(module_totals(row))
    module_max = nice_max(max(sum(values.values()) for values in module_values) * 1.05)
    body += figure(
        stacked_bar_svg(module_categories, module_values, y_max=module_max),
        "These are serial-equivalent raw module work counters, not stacked wall-clock E2E latency. AFD executes A and F "
        "on different GPU pools and overlaps them across microbatches; compare this chart with the stage-cycle and raw-round table above.",
        chart_contract,
    )
    composition_rows = []
    for category, values in zip(module_categories, module_values, strict=True):
        composition_rows.append(
            [esc(category)] + [fmt(values.get(group, 0.0), 3) for group in MODULE_ORDER] + [fmt(sum(values.values()), 3)]
        )
    body += table(
        ["Case", *[f"{group} ms" for group in MODULE_ORDER], "Raw work total ms"],
        composition_rows,
        css="wide",
    )

    body += "<h2>5. Throughput–latency Pareto front</h2>"
    pareto_by_context = {}
    y_values = []
    for workload in CONTEXTS:
        series = {}
        for label, (scenario, side) in {
            "AGG": ("no_mtp", "agg"),
            "AGG + AFD": ("no_mtp", "afd"),
            "AGG + MTP": (mtp["name"], "agg"),
            "AGG + AFD + MTP": (mtp["name"], "afd"),
        }.items():
            raw_candidates = [
                row
                for row in payload["rows"]
                if row["model"] == model["key"]
                and row["workload"] == workload
                and row["scenario"] == scenario
                and row["precision_profile"] == profile["key"]
                and row["system_kind"] == side
                and row["effective_tpot_ms"] <= 50
            ]
            candidates = (
                [row for row in raw_candidates if row["total_gpus"] == 72]
                if side == "agg"
                else [materialize_afd_cluster(row, 72) for row in raw_candidates if row["total_gpus"] <= 72]
            )
            points = [
                (row["effective_tpot_ms"], row["output_tokens_s_gpu"]) for row in pareto_front(candidates)
            ]
            series[label] = points
            y_values.extend(value for _, value in points)
        pareto_by_context[workload] = series
    pareto_y_max = nice_max(max(y_values, default=1.0) * 1.05)
    for workload in CONTEXTS:
        body += f"<h3>{workload.upper()} input · 72 GPUs</h3>"
        body += figure(
            scatter_svg(pareto_by_context[workload], x_max=50, y_max=pareto_y_max),
            "Left is lower effective TPOT; up is higher tokens/s/GPU. Only non-dominated points with TPOT≤50 ms are shown. "
            "The 8K and 16K charts use identical x and y intervals, so visual distances are directly comparable.",
            chart_contract,
        )

    body += "<h2>6. Precision and kernel controls at 72 GPUs</h2>"
    control_rows = []
    available_profiles = sorted(
        {row["precision_profile"] for row in payload["rows"] if row["model"] == model["key"]}
    )
    for precision_key in available_profiles:
        precision = next(value for value in model["precision_profiles"] if value["key"] == precision_key)
        for workload in CONTEXTS:
            for scenario, label in (("no_mtp", "No MTP"), (mtp["name"], "With MTP")):
                pair = paired_winner(
                    payload,
                    model=model["key"],
                    workload=workload,
                    scenario=scenario,
                    precision=precision_key,
                    total_gpus=72,
                    speed_floor=speed_floor,
                )
                control_rows.append(
                    [
                        esc(precision_key),
                        esc(precision["moe_quant_mode"]),
                        esc(precision["moe_kernel"]),
                        workload.upper(),
                        label,
                        fmt(pair["agg"] and pair["agg"]["output_tokens_s_gpu"]),
                        fmt(pair["afd"] and pair["afd"]["output_tokens_s_gpu"]),
                        ratio_html(pair["ratio"]),
                        esc(precision["evidence"]),
                    ]
                )
    body += table(
        ["Profile", "MoE precision", "MoE kernel", "ISL", "Mode", "AGG tok/s/GPU", "AFD tok/s/GPU", "AFD/AGG", "Evidence"],
        control_rows,
        css="wide",
    )
    body += (
        '<p class="small muted">Rows with different precision or kernels are sensitivity controls, not direct AFD '
        "speedups. Within every row, AGG and AFD remain matched. Secondary profiles are exact 72-GPU-unit controls; "
        "the primary profile alone receives the complete 8–72 GPU service-unit packing search.</p>"
    )

    model_rows = [row for row in payload["rows"] if row["model"] == model["key"]]
    model_failures = [row for row in payload["failures"] if row.get("model") == model["key"]]
    failure_counts = Counter(failure["error"].split(":", 1)[0] for failure in model_failures)
    body += "<h2>7. Sweep coverage and confidence</h2>"
    body += table(
        ["Metric", "Value"],
        [
            ["Valid AGG points", f"{sum(row['system_kind']=='agg' for row in model_rows):,}"],
            ["Valid AFD points", f"{sum(row['system_kind']=='afd' for row in model_rows):,}"],
            ["Rejected / unsupported points", f"{len(model_failures):,}"],
            ["Top rejection classes", esc(", ".join(f"{key}={value}" for key, value in failure_counts.most_common(4)))],
            ["Primary attention evidence", esc(model["attention_evidence"])],
            ["Primary MoE evidence", esc(profile["evidence"])],
        ],
    )
    if not profile["exact_shape_data"]:
        body += '<div class="callout warn"><strong>Projection warning.</strong> This primary profile lacks a native target-shape silicon row. Treat absolute throughput and the A:F optimum as a calibrated hypothesis until measured on the target kernel.</div>'
    else:
        body += '<div class="callout ok"><strong>Primary MoE shape is covered.</strong> HYBRID may still estimate uncovered attention, GEMM, communication, or q-wide MTP shapes; the per-operation source mix remains part of the evidence boundary.</div>'
    body += '<p class="foot">All figures are inline SVG. No image assets or network access are required to view this file.</p>'

    summary_rows = []
    for workload in CONTEXTS:
        for total in TOTAL_GPU_GRID:
            record = {"workload": workload, "total_gpus": total}
            for scenario, prefix in (("no_mtp", "no_mtp"), (mtp["name"], "mtp")):
                pair = winners[(workload, total, scenario)]
                record[prefix] = {
                    "ratio": pair["ratio"],
                    "agg_tps_gpu": pair["agg"] and pair["agg"]["output_tokens_s_gpu"],
                    "afd_tps_gpu": pair["afd"] and pair["afd"]["output_tokens_s_gpu"],
                    "agg_config": agg_config(pair["agg"]),
                    "afd_config": afd_config(pair["afd"]),
                }
            summary_rows.append(record)
    summary = {
        "model": model["key"],
        "label": model["label"],
        "primary_precision": profile,
        "primary_mtp": mtp,
        "attention_type": model["attention_type"],
        "attention_backend": model["attention_backend"],
        "moe_backend": moe_backend_label(model),
        "moe_structure": model["moe_structure"],
        "winners": summary_rows,
    }
    return document(title, f"GB200 · decode-only · ISL 8K/16K · OSL 1024 · speed floor {speed_floor:g} tok/s/user", body), summary


def render_index(payload: dict[str, Any], summaries: list[dict[str, Any]], speed_floor: float) -> str:
    body = model_navigation(payload)
    body += (
        '<div class="callout"><strong>Question answered:</strong> for each fixed 16/24/36/48/72-GPU pool, what are '
        "the best matched-backend AGG and AFD layouts, before and after MTP? Ratios compare independently optimized "
        f"arms at ≥{speed_floor:g} committed tokens/s/user.</div>"
    )
    body += "<h2>1. Backend and model contract</h2>"
    backend_rows = []
    for summary in summaries:
        profile = summary["primary_precision"]
        backend_rows.append(
            [
                f'<a href="{summary["model"]}.html">{esc(summary["label"])}</a>',
                esc(summary["attention_type"]),
                esc(summary["attention_backend"]),
                esc(summary["moe_structure"]),
                esc(summary["moe_backend"]),
                esc(profile["moe_quant_mode"]),
                esc(profile["moe_kernel"]),
                esc(profile["evidence"]),
            ]
        )
    body += table(
        [
            "Model",
            "Attention structure",
            "Attention backend",
            "MoE structure",
            "MoE backend",
            "MoE precision",
            "MoE kernel",
            "Evidence",
        ],
        backend_rows,
        css="wide",
    )
    body += (
        '<div class="callout"><strong>Backend policy.</strong> MegaMoE is used for both AGG and AFD only for '
        "DeepSeek-V4-Pro, where AIC has a model-specific measured MegaMoE contract. All other primary comparisons "
        "use the same SGLang backend and kernel on both arms; no unsupported MegaMoE proxy is substituted.</div>"
    )

    body += "<h2>2. 72-GPU headline</h2>"
    headline_rows = []
    for summary in summaries:
        mtp = summary["primary_mtp"]
        for workload in CONTEXTS:
            record = next(
                row for row in summary["winners"] if row["workload"] == workload and row["total_gpus"] == 72
            )
            headline_rows.append(
                [
                    f'<a href="{summary["model"]}.html">{esc(summary["label"])}</a>',
                    workload.upper(),
                    fmt(record["no_mtp"]["agg_tps_gpu"]),
                    fmt(record["no_mtp"]["afd_tps_gpu"]),
                    ratio_html(record["no_mtp"]["ratio"]),
                    esc(record["no_mtp"]["afd_config"]),
                    fmt(record["mtp"]["agg_tps_gpu"]),
                    fmt(record["mtp"]["afd_tps_gpu"]),
                    ratio_html(record["mtp"]["ratio"]),
                    esc(record["mtp"]["afd_config"]),
                    f"N={mtp['nextn']}, P={mtp['progress']:.3f}",
                ]
            )
    body += table(
        ["Model", "ISL", "AGG", "AGG+AFD", "AFD/AGG", "Best A:F no MTP", "AGG+MTP", "AGG+AFD+MTP", "AFD/AGG with MTP", "Best A:F with MTP", "MTP"],
        headline_rows,
        css="wide",
    )

    body += "<h2>3. Scale trend across fixed GPU pools</h2>"
    for workload in CONTEXTS:
        for mode, key in (("No MTP", "no_mtp"), ("With MTP", "mtp")):
            series = {}
            for summary in summaries:
                points = [
                    (row["total_gpus"], row[key]["ratio"])
                    for row in summary["winners"]
                    if row["workload"] == workload and row[key]["ratio"] is not None
                ]
                series[summary["label"]] = points
                COLORS.setdefault(summary["label"], ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")[len(series)-1])
            max_ratio = max(value for points in series.values() for _, value in points)
            body += f"<h3>{workload.upper()} · {mode}</h3>"
            body += figure(
                line_svg(
                    series,
                    x_label="Fixed GPU pool size (GPUs)",
                    y_label="Throughput ratio: AFD / AGG",
                    y_max=max(1.2, nice_max(max_ratio * 1.05)),
                    reference_y=1.0,
                ),
                "Every point uses the model's primary matched backend shown above. The dashed line is break-even. "
                "This is a scale trend, not a cross-model absolute-throughput comparison.",
                "Model-specific attention structure/backend and MoE structure/kernel/precision are listed in section 1; "
                "each AGG/AFD pair is matched within its model.",
            )

    body += "<h2>4. Complete fixed-pool winner matrix</h2>"
    matrix_rows = []
    for summary in summaries:
        profile = summary["primary_precision"]
        for record in summary["winners"]:
            matrix_rows.append(
                [
                    f'<a href="{summary["model"]}.html">{esc(summary["label"])}</a>',
                    record["workload"].upper(),
                    record["total_gpus"],
                    fmt(record["no_mtp"]["agg_tps_gpu"]),
                    fmt(record["no_mtp"]["afd_tps_gpu"]),
                    ratio_html(record["no_mtp"]["ratio"]),
                    esc(record["no_mtp"]["afd_config"]),
                    fmt(record["mtp"]["agg_tps_gpu"]),
                    fmt(record["mtp"]["afd_tps_gpu"]),
                    ratio_html(record["mtp"]["ratio"]),
                    esc(record["mtp"]["afd_config"]),
                    esc(profile["moe_kernel"]),
                    esc(profile["moe_quant_mode"]),
                ]
            )
    body += table(
        ["Model", "ISL", "GPUs", "AGG", "AGG+AFD", "AFD/AGG", "Best A:F no MTP", "AGG+MTP", "AGG+AFD+MTP", "AFD/AGG with MTP", "Best A:F with MTP", "MoE kernel", "MoE precision"],
        matrix_rows,
        css="wide",
    )

    body += "<h2>5. Interpretation boundary</h2>"
    body += table(
        ["Term", "Meaning"],
        [
            ["Ratio", "Best AGG+AFD tokens/s/GPU divided by best AGG tokens/s/GPU at the same model, context, MTP mode, precision, kernel, total GPUs, and user-speed floor."],
            ["AFD batch", "Requests per physical A GPU. AIC a_batch_size_per_worker=batch_per_A_GPU×A_TP."],
            ["Module work", "Serial-equivalent operator counters. It is not additive E2E latency when A/F pipeline overlap is active."],
            ["E2E", "Full resident decode-round wall time from the AIC session; MTP effective TPOT divides this time by committed progress P."],
            ["HYBRID", "Uses silicon rows when present and estimates uncovered shapes. Model pages identify projected attention/MoE contracts explicitly."],
        ],
    )
    body += '<p class="foot">All pages and figures are self-contained; copy the whole report directory or any individual HTML file.</p>'
    return document(
        "AFD × MTP fixed-pool simulation index",
        f"GB200 · 16/24/36/48/72 GPUs · matched backend · speed floor {speed_floor:g} tok/s/user",
        body,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--control-sweep", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed-floor", type=float, default=30.0)
    args = parser.parse_args()
    if args.speed_floor <= 0:
        parser.error("--speed-floor must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload = load_payload(args.sweep + args.control_sweep)
    missing = [model for model in MODEL_ORDER if model not in payload["models"]]
    if missing:
        raise ValueError(f"missing model sweeps: {', '.join(missing)}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_key in MODEL_ORDER:
        report, summary = render_model(payload, payload["models"][model_key], args.speed_floor)
        (output_dir / f"{model_key}.html").write_text(report, encoding="utf-8")
        summaries.append(summary)
    (output_dir / "index.html").write_text(render_index(payload, summaries, args.speed_floor), encoding="utf-8")
    summary = {
        "schema": "aic.afd-fixed-pool-report.v3",
        "speed_floor_tokps_per_user": args.speed_floor,
        "contract": payload["contract"],
        "sources": payload["sources"],
        "code": payload["code"],
        "models": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "html_reports": len(summaries) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
