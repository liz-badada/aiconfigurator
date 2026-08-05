#!/usr/bin/env python3
"""Render a self-contained comparison of matched-backend AFD sweep ratios."""

from __future__ import annotations

import argparse
from pathlib import Path

import render_afd_multimodel_mtp_report as report

PALETTE = ("#76B900", "#0072B2", "#666666", "#E69F00", "#56B4E9")


def parse_sweep(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("sweep must use LABEL=/path/to/sweep.json")
    return label.strip(), Path(raw_path).expanduser()


def parse_detail_report(value: str) -> tuple[str, str]:
    label, separator, link = value.partition("=")
    if not separator or not label.strip() or not link.strip():
        raise argparse.ArgumentTypeError("detail report must use LABEL=relative/path/index.html")
    return label.strip(), link.strip()


def load_sweeps(specs: list[tuple[str, Path]]) -> list[tuple[str, dict]]:
    labels = [label for label, _ in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("backend comparison labels must be unique")
    sweeps = [(label, report.load_payload([path])) for label, path in specs]
    reference = sweeps[0][1]
    required_models = set(report.MODEL_ORDER)
    for label, payload in sweeps:
        missing = required_models - payload["models"].keys()
        if missing:
            raise ValueError(f"{label} is missing models: {', '.join(sorted(missing))}")
        for field in ("system", "backend", "database_mode", "gpus_per_node"):
            if payload["contract"][field] != reference["contract"][field]:
                raise ValueError(f"inconsistent {field} for {label}")
    return sweeps


def ratio_points(
    sweeps: list[tuple[str, dict]],
    *,
    model_key: str,
    workload: str,
    with_mtp: bool,
    speed_floor: float,
) -> tuple[dict[str, list[tuple[float, float]]], list[list[object]]]:
    series = {}
    rows = []
    for label, payload in sweeps:
        model = payload["models"][model_key]
        mtp_name = report.primary_mtp(model)["name"]
        scenario = mtp_name if with_mtp else "no_mtp"
        winners = report.winners_for_model(payload, model, speed_floor)
        points = []
        for total in report.TOTAL_GPU_GRID:
            pair = winners[(workload, total, scenario)]
            ratio = pair["ratio"]
            if ratio is not None:
                points.append((total, ratio))
            rows.append(
                [
                    report.esc(label),
                    total,
                    report.fmt(pair["agg"] and pair["agg"]["output_tokens_s_gpu"]),
                    report.fmt(pair["afd"] and pair["afd"]["output_tokens_s_gpu"]),
                    report.ratio_html(ratio),
                    report.esc(report.agg_config(pair["agg"])),
                    report.esc(report.afd_config(pair["afd"])),
                ]
            )
        series[label] = points
    return series, rows


def backend_contract_rows(sweeps: list[tuple[str, dict]], model_key: str) -> list[list[object]]:
    rows = []
    for label, payload in sweeps:
        contracts = report.primary_arm_backend_contracts(payload, payload["models"][model_key])
        contract = contracts["AGG"]
        rows.append(
            [
                report.esc(label),
                report.esc(report.backend_display_name(contract["moe_backend"])),
                report.esc(contract["moe_kernel"]),
                report.esc(contract["moe_precision"]),
                report.esc(contract["attention_backend"]),
                '<span class="good">matched in all four arms</span>',
            ]
        )
    return rows


def render(sweeps: list[tuple[str, dict]], speed_floor: float, detail_reports: dict[str, str] | None = None) -> str:
    for index, (label, _) in enumerate(sweeps):
        report.COLORS[label] = PALETTE[index % len(PALETTE)]

    all_series = []
    prepared = {}
    for model_key in report.MODEL_ORDER:
        for workload in report.CONTEXTS:
            for with_mtp in (False, True):
                series, rows = ratio_points(
                    sweeps,
                    model_key=model_key,
                    workload=workload,
                    with_mtp=with_mtp,
                    speed_floor=speed_floor,
                )
                prepared[(model_key, workload, with_mtp)] = (series, rows)
                all_series.extend(value for points in series.values() for _, value in points)
    common_y_max = max(report.nice_max(max(all_series, default=1.0) * 1.05), 1.2)

    body = ""
    if detail_reports:
        body += (
            '<div class="nav">'
            + "".join(
                f'<a href="{report.esc(link)}">{report.esc(label)} detailed report</a>'
                for label, link in detail_reports.items()
            )
            + "</div>"
        )
    body += (
        '<div class="callout"><strong>Comparison rule.</strong> Each line compares AGG+AFD against AGG while '
        "holding the named MoE backend fixed in all four arms. A point is shown only when both arms satisfy the "
        f"{speed_floor:g} committed tokens/s/user floor and lie inside the available measured-load envelope.</div>"
        '<div class="callout"><strong>Scheduling boundary.</strong> Backend identity is matched, but topology is not: '
        "AGG has no split A/F pipeline and uses graph/backend internal overlap only; AFD uses the conservative "
        "microbatch schedule (serial at M=1, otherwise max(A+A→F, F+F→A)) with no optimistic communication "
        "hiding.</div>"
        '<div class="callout warn"><strong>Evidence boundary.</strong> Profile-derived curves use B200 MoE-stage '
        "measurements as a load-matched GB200 projection with scale 1.0. They are not GB200 silicon measurements. "
        "Missing points are not extrapolated. AIC multi-axis performance-grid interpolation uses the PR #1479 "
        "joint-log2 kNN4 implementation; that fix is distinct from measured-stage load interpolation and is not a "
        "universal ≤20% error guarantee.</div>"
    )
    body += "<h2>1. Backend identities</h2>"
    for model_key in report.MODEL_ORDER:
        model = sweeps[0][1]["models"][model_key]
        body += f"<h3>{report.esc(model['label'])}</h3>"
        body += report.table(
            ["Curve", "MoE backend", "MoE kernel", "MoE precision", "Attention backend", "Four-arm check"],
            backend_contract_rows(sweeps, model_key),
            css="wide",
        )

    body += "<h2>2. AFD/AGG throughput-ratio curves</h2>"
    body += (
        '<p class="small muted">All charts use the same x-axis and y-axis ranges. The dashed 1.0 line is break-even. '
        "Line points are intentionally unlabeled; exact values and missing configurations are listed below each "
        "chart.</p>"
    )
    for model_key in report.MODEL_ORDER:
        model = sweeps[0][1]["models"][model_key]
        body += f"<h3>{report.esc(model['label'])}</h3>"
        body += (
            f'<p class="small muted">Attention: {report.esc(model["attention_type"])}; '
            f"{report.esc(model['attention_backend'])}. MoE: {report.esc(model['moe_structure'])}.</p>"
        )
        for workload in report.CONTEXTS:
            for with_mtp, mode in ((False, "No MTP"), (True, "With MTP")):
                series, rows = prepared[(model_key, workload, with_mtp)]
                body += f"<h3>{workload.upper()} input · {mode}</h3>"
                body += report.figure(
                    report.line_svg(
                        series,
                        x_label="Fixed GPU pool size (GPUs)",
                        y_label="Throughput ratio: AFD / AGG",
                        y_max=common_y_max,
                        reference_y=1.0,
                    ),
                    "Above 1.0 favors AFD; below 1.0 favors AGG. Every backend is optimized independently, but its "
                    "AGG and AFD arms use the same backend contract.",
                    data_table=report.table(
                        [
                            "Backend curve",
                            "GPUs",
                            "AGG tok/s/GPU",
                            "AFD tok/s/GPU",
                            "AFD/AGG",
                            "Best AGG",
                            "Best AFD",
                        ],
                        rows,
                        css="wide",
                    ),
                )
    body += '<p class="foot">Self-contained HTML: charts are inline SVG and require no image files.</p>'
    system = sweeps[0][1]["contract"]["system"]
    return report.document(
        "AFD matched-backend comparison",
        f"{report.esc(system)} · fixed 16/24/36/48/72-GPU pools · B200 measured-load sensitivity",
        body,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="append", type=parse_sweep, required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--detail-report",
        action="append",
        type=parse_detail_report,
        default=[],
        metavar="LABEL=RELATIVE_INDEX",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed-floor", type=float, default=30.0)
    args = parser.parse_args()
    if len(args.sweep) < 2:
        parser.error("at least two --sweep values are required")
    if args.speed_floor <= 0:
        parser.error("--speed-floor must be positive")
    return args


def main() -> int:
    args = parse_args()
    sweeps = load_sweeps(args.sweep)
    detail_reports = dict(args.detail_report)
    unknown_details = detail_reports.keys() - {label for label, _ in sweeps}
    if unknown_details:
        raise ValueError(f"detail reports have no matching sweep: {', '.join(sorted(unknown_details))}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(sweeps, args.speed_floor, detail_reports), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
