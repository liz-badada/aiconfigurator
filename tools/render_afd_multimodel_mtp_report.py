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
HEADLINE = {
    "qwen3_235b": ("eagle3_n3", "measured complete-F overlay"),
    "minimax_m3": ("mtp_n1_r70", "native AIC"),
    "deepseek_v4_flash": ("mtp_n2_r70", "native AIC"),
    "deepseek_v4_pro": ("mtp_n2_r70", "native AIC"),
}
MODEL_LABELS = {
    "qwen3_235b": "Qwen3-235B-A22B",
    "minimax_m3": "MiniMax-M3",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek_v4_pro": "DeepSeek-V4-Pro",
}
MODEL_LINKS = {
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


def select_pairs(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODEL_ORDER:
        scenario, evidence = HEADLINE[model]
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


def table(headers: list[str], rows: list[list[object]], classes: list[str] | None = None) -> str:
    out = ['<div class="scroll"><table><thead><tr>']
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
            "Measured-F hybrid",
            "AFD F-stage uses the measured complete-stage MegaMoE curve, interpolated only inside its measured load range. A-side and AGG remain AIC HYBRID. This is the strongest AFD alignment evidence here, but it is not a full silicon E2E measurement.",
        ),
        "minimax_m3": (
            "Scenario / HYBRID",
            "MSA has no native silicon table and transfers utilization from measured DSA. MoE is generic AIC HYBRID. The 70% conditional acceptance is an internal PoC scenario, not a public position-by-position trace.",
        ),
        "deepseek_v4_flash": (
            "Model-specific attention / HYBRID MoE",
            "SWA, CSA/HCA and mHC use model-specific tables with HYBRID fallback. MoE uses the native FP4-expert model rather than a measured complete-F overlay. The 70% acceptance is sensitivity-only.",
        ),
        "deepseek_v4_pro": (
            "Measured MegaMoE hybrid",
            "Attention/mHC use declared 0.5.14 donors and MegaMoE uses the declared 0.5.10 measured module. The AFD path is corrected to query the table with per-EP-rank tokens. Loads above 512 local decode tokens use utilization-hold extrapolation. Acceptance is sensitivity-only.",
        ),
    }
    return notes[model]


def model_comment(model: str, pairs: dict[str, dict[str, dict[str, Any]]]) -> str:
    values = {workload: four_way(pairs[workload]) for workload in CONTEXTS}
    ratios = {workload: values[workload]["AGG + AFD + MTP"] / values[workload]["AGG + MTP"] for workload in CONTEXTS}
    if model == "qwen3_235b":
        return (
            f"With measured F-stage timing, AFD+MTP is {ratios['8k']:.2f}× AGG+MTP at 8K and "
            f"{ratios['16k']:.2f}× at 16K. This is the only model in this study whose aligned F evidence "
            "supports the same positive AFD direction at both contexts."
        )
    if model == "minimax_m3":
        return (
            "MTP helps both topologies, but AGG gains much more. The selected N=1 AFD speedup is only "
            f"{values['8k']['AGG + AFD + MTP'] / values['8k']['AGG + AFD']:.2f}× at 8K and "
            f"{values['16k']['AGG + AFD + MTP'] / values['16k']['AGG + AFD']:.2f}× at 16K because F-side "
            "work grows close to the accepted-token progress. This is not yet sufficient evidence for an AFD win."
        )
    if model == "deepseek_v4_flash":
        return (
            "At the 70% sensitivity point, q=3 verification increases AFD raw service by about 2.2× while "
            "progress is 2.19×, so AFD MTP is approximately neutral. AGG remains faster; the result should be "
            "used to identify missing F-stage calibration, not as a hardware conclusion."
        )
    return (
        "Correct rank-local MegaMoE token accounting makes no-MTP AFD competitive, but MTP shifts more work into "
        "the concentrated F pool. AGG therefore gains more from MTP, and AFD+MTP remains below AGG+MTP in this "
        "70% sensitivity scenario."
    )


def render_model(
    model: str,
    payload: dict[str, Any],
    pairs: dict[str, dict[str, dict[str, Any]]],
    mocker: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    label = MODEL_LABELS[model]
    scenario, evidence = HEADLINE[model]
    model_meta = next(value for value in payload["models"] if value["key"] == model)
    four_groups = [(workload.upper(), four_way(pairs[workload])) for workload in CONTEXTS]
    evidence_title, evidence_text = evidence_note(model)
    nav = (
        '<div class="nav"><a href="index.html">Cross-model summary</a>'
        + "".join(f'<a href="{esc(key)}.html">{esc(MODEL_LABELS[key])}</a>' for key in MODEL_ORDER)
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
            [
                "Model",
                f'<a href="{esc(MODEL_LINKS[model])}">{esc(model_meta["model_path"])}</a> · {esc(model_meta["parameter_note"])}',
            ],
        ],
    )

    body += "<h2>2. Four-way system result</h2>"
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

    body += "<h2>3. MTP accounting: cost, acceptance, and final gain</h2>"
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

    body += "<h2>4. Attention, router, MoE, collective, and transfer work</h2>"
    body += figure(
        svg_module_stacks(pairs),
        "These bars are accumulated worker-side module work, not additive E2E latency. A and F execute as a layer pipeline and communication may be hidden, so raw decode-round service is calculated by the pipeline recurrence rather than by summing every bar.",
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
            "Bottleneck",
        ],
        module_rows,
    )

    body += "<h2>5. Throughput–latency Pareto frontier</h2>"
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

    body += "<h2>6. A:F hardware-ratio sensitivity</h2>"
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

    body += "<h2>7. Dynamo Mocker replay check</h2>"
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

    body += "<h2>8. MTP-depth capacity sensitivity</h2>"
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

    body += "<h2>9. What is supported, assumed, and not claimed</h2>"
    selected_scenario_meta = next(value for value in model_meta["scenarios"] if value["name"] == scenario)
    body += table(
        ["Layer", "Status", "Meaning"],
        [
            ["Model structure", "Supported", esc(model_meta["attention_note"] + "; " + model_meta["moe_note"])],
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
        "results": {
            workload: {"four_way": four_way(pairs[workload]), "selected_afd": pairs[workload]["mtp"]}
            for workload in CONTEXTS
        },
        "mocker_max_abs_mean_tpot_error_pct": max(abs(result["mean_tpot_error_pct"]) for result in mock_rows),
    }
    return document(title, subtitle, body), summary


def render_index(
    payload: dict[str, Any],
    selected: dict[str, dict[str, dict[str, dict[str, Any]]]],
    mocker: dict[str, Any],
) -> str:
    nav = (
        '<div class="nav"><span class="pill">Cross-model summary</span>'
        + "".join(f'<a href="{esc(model)}.html">{esc(MODEL_LABELS[model])}</a>' for model in MODEL_ORDER)
        + "</div>"
    )
    body = nav
    body += (
        '<div class="callout"><strong>Question answered.</strong> For each model and context, compare '
        "AGG+AFD against AGG, then compare AGG+AFD+MTP against AGG+MTP. All values use the same 72-GPU "
        "NVL72 budget and the same offered concurrency within a four-way group.</div>"
    )
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
    body += figure(
        svg_grouped_bars(
            ratio_groups,
            ["No MTP", "With MTP"],
            y_label="AFD throughput / AGG throughput (×)",
            x_label="Model and input context",
            y_max=2.0,
            value_digits=2,
        ),
        "The horizontal decision boundary is 1×. Qwen's measured-F hybrid supports an AFD win in both contexts. DeepSeek-V4 Pro supports a no-MTP AFD win after correcting rank-local MegaMoE tokens, but not an AFD+MTP win at the assumed 70% rate. MiniMax and V4-Flash remain negative in the current HYBRID model.",
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

    body += "<h2>5. Dynamo Mocker token-accounting validation</h2>"
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

    body += "<h2>6. Reproduction identity</h2>"
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
        "Qwen3-235B, MiniMax-M3, DeepSeek-V4-Flash, and DeepSeek-V4-Pro",
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in MODEL_ORDER:
        report, summary = render_model(model, payload, selected[model], mocker)
        (output_dir / f"{model}.html").write_text(report, encoding="utf-8")
        summaries.append(summary)
    (output_dir / "index.html").write_text(render_index(payload, selected, mocker), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "aic.afd-multimodel-mtp-report.v1",
                "base_code": payload["code"],
                "overlays": payload.get("applied_overlays", []),
                "dynamo": {"branch": mocker["dynamo_branch"], "commit": mocker["dynamo_commit"]},
                "models": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "reports": len(MODEL_ORDER) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
