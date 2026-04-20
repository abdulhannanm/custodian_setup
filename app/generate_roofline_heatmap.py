"""
Generate roofline heatmap payload from a DCGM metrics .dat file.

Output JSON includes:
- heatmap bin centers and values (raw counts + log-density display)
- per-metric point counts
- roofline curves for FP64/FP32/INT/Tensor
- metadata for plotting
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GPU_PEAKS = {
    "GH200": {
        "hbm_tbps": 4.0,
        "fp64_tfps": 34.0,
        "fp32_tfps": 67.0,
        "tensor_tfps": 989.0,
        "int_tops": 30.0,
    },
    "A100": {
        "hbm_tbps": 1.6,
        "fp64_tfps": 9.7,
        "fp32_tfps": 19.5,
        "tensor_tfps": 312.0,
        "int_tops": 624.0,
    },
}

# Backward-compatible default profile.
PEAKS = GPU_PEAKS["GH200"]


def parse_gpu_name_from_dat(dat_path: Path) -> str:
    try:
        with dat_path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().strip()
    except Exception:
        return "GH200"

    if not first_line:
        return "GH200"
    if not first_line.lower().startswith("gpu_name"):
        return "GH200"

    parts = first_line.split("=", maxsplit=1)
    if len(parts) != 2:
        return "GH200"
    parsed = parts[1].strip().upper()
    return parsed if parsed else "GH200"


def get_gpu_peaks(gpu_name: str) -> Dict[str, float]:
    return GPU_PEAKS.get(gpu_name.upper(), GPU_PEAKS["GH200"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate roofline heatmap from DCGM metrics .dat")
    parser.add_argument("dat_path", type=Path, help="Path to dcgm_metrics*.dat")
    parser.add_argument("--x-min", type=float, default=0.01, help="Minimum arithmetic intensity")
    parser.add_argument("--x-max", type=float, default=1000.0, help="Maximum arithmetic intensity")
    parser.add_argument("--y-min-log10", type=float, default=-2.0, help="Minimum log10 throughput")
    parser.add_argument("--y-max-log10", type=float, default=3.0, help="Maximum log10 throughput")
    parser.add_argument("--x-bins", type=int, default=40, help="Number of x-axis bins")
    parser.add_argument("--y-bins", type=int, default=40, help="Number of y-axis bins")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    parser.add_argument("--png-output", type=Path, help="Optional output PNG path for a rendered roofline heatmap figure")
    parser.add_argument(
        "--png-renderer",
        choices=("plotly",),
        default="plotly",
        help="PNG renderer backend (forced to plotly for report-match output)",
    )
    return parser.parse_args()


def parse_number(raw: str) -> Optional[float]:
    token = raw.strip()
    if not token or token.upper() == "N/A":
        return 0.000
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_dcgm_rows(dat_path: Path) -> List[Dict[str, Optional[float]]]:
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    metrics: List[str] = []
    rows: List[Dict[str, Optional[float]]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("gpu_name"):
            continue
        if stripped.startswith("#Entity"):
            parts = stripped.split()
            metrics = [x.upper() for x in parts[1:]]
            continue
        if stripped == "ID" or stripped.startswith("ID "):
            continue
        if not stripped.startswith("GPU"):
            continue
        if not metrics:
            continue

        tokens = stripped.split()
        if len(tokens) < 2 + len(metrics):
            continue

        values = tokens[2 : 2 + len(metrics)]
        row: Dict[str, Optional[float]] = {}
        for idx, metric in enumerate(metrics):
            row[metric] = parse_number(values[idx])
        rows.append(row)

    return rows


def convert_to_df(str_file_path: str) -> pd.DataFrame:
    dat_path = Path(str_file_path)
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    metrics: List[str] = []
    rows: List[List[int]] = []
    metrics_row_counter = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("gpu_name"):
            continue
        if stripped.startswith("#Entity"):
            if metrics_row_counter == 0:
                parts = stripped.split()
                metrics = [x.upper() for x in parts[1:]]
            metrics_row_counter += 1
            continue
        if stripped == "ID" or stripped.startswith("ID "):
            continue
        if not stripped.startswith("GPU"):
            continue
        if not metrics:
            continue

        tokens = stripped.split()
        if len(tokens) < 2 + len(metrics):
            continue

        row = tokens[2 : 2 + len(metrics)]
        row = [parse_number(item) for item in row]
        rows.append(row)
    df = pd.DataFrame(rows, columns=metrics)

    return df


def parse_csv_rows(csv_path: Path) -> List[Dict[str, Optional[float]]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.upper() for c in df.columns]
    rows: List[Dict[str, Optional[float]]] = []
    for _, row in df.iterrows():
        rows.append({col: parse_number(str(row[col])) for col in df.columns})
    return rows


def build_points(rows: List[Dict[str, Optional[float]]], metric: str, compute_peak: float, hbm_peak: float) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for row in rows:
        activity = row.get(metric)
        drama = row.get("DRAMA")
        if activity is None or drama is None:
            continue
        throughput = activity * compute_peak
        bandwidth = drama * hbm_peak
        if throughput <= 0 or bandwidth <= 0:
            continue
        ai = throughput / bandwidth
        if ai <= 0:
            continue
        points.append((ai, throughput))
    return points


def build_roofline_curve(peak_compute: float, hbm_tbps: float, x_min: float, x_max: float) -> Dict[str, List[float]]:
    knee_ai = peak_compute / hbm_tbps
    if knee_ai <= x_min or knee_ai >= x_max:
        return {
            "x": [x_min, x_max],
            "y": [min(peak_compute, hbm_tbps * x_min), min(peak_compute, hbm_tbps * x_max)],
        }
    return {"x": [x_min, knee_ai, x_max], "y": [hbm_tbps * x_min, peak_compute, peak_compute]}


def render_png(
    png_path: Path,
    z_normalized_category: List[List[float]],
    x_min: float,
    x_max: float,
    y_min_log10: float,
    y_max_log10: float,
    roofline_curves: Dict[str, Dict[str, List[float]]],
    gpu_name: str,
) -> None:
    render_png_with_matplotlib(
        png_path=png_path,
        z_normalized_category=z_normalized_category,
        x_min=x_min,
        x_max=x_max,
        y_min_log10=y_min_log10,
        y_max_log10=y_max_log10,
        roofline_curves=roofline_curves,
        gpu_name=gpu_name,
    )


def render_png_with_matplotlib(
    png_path: Path,
    z_normalized_category: List[List[float]],
    x_min: float,
    x_max: float,
    y_min_log10: float,
    y_max_log10: float,
    roofline_curves: Dict[str, Dict[str, List[float]]],
    gpu_name: str,
) -> None:
    z_array = np.array(z_normalized_category, dtype=float)
    y_min = 10 ** y_min_log10
    y_max = 10 ** y_max_log10

    cmap = plt.matplotlib.colors.ListedColormap(
        ["#FFFFFF", "#1F77B4", "#FF7F0E", "#D62728"]
    )
    bounds = [0.0, 1.0, 2.0, 3.0, 4.1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    mesh = ax.pcolormesh(
        np.geomspace(x_min, x_max, z_array.shape[1] + 1),
        np.geomspace(y_min, y_max, z_array.shape[0] + 1),
        np.floor(z_array),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    curve_colors = {"FP64": "#1F77B4", "FP32": "#2CA02C", "INT": "#FF7F0E", "Tensor": "#D62728"}
    for name, curve in roofline_curves.items():
        ax.plot(
            curve["x"],
            curve["y"],
            color=curve_colors.get(name, "#333333"),
            linewidth=2,
            linestyle="--",
            label=f"{name} Roofline",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{gpu_name.upper()} Roofline Heatmap from DCGM Samples")
    ax.set_xlabel("Arithmetic Intensity (Ops/Byte)")
    ax.set_ylabel("Throughput (TF/s or TOps/s)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, framealpha=0.9)

    cbar = fig.colorbar(mesh, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5], pad=0.08)
    cbar.ax.set_yticklabels(["None", "FP64", "INT", "Tensor"])
    cbar.set_label("Dominant Type")

    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dat_path.exists():
        raise SystemExit(f"File not found: {args.dat_path}")

    gpu_name = parse_gpu_name_from_dat(args.dat_path)
    peaks = get_gpu_peaks(gpu_name)

    rows = parse_dcgm_rows(args.dat_path)
    if not rows:
        raise SystemExit(f"No GPU data rows parsed from: {args.dat_path}")

    per_metric_points = {
        "FP64A": build_points(rows, "FP64A", peaks["fp64_tfps"], peaks["hbm_tbps"]),
        "FP32A": build_points(rows, "FP32A", peaks["fp32_tfps"], peaks["hbm_tbps"]),
        "INTAC": build_points(rows, "INTAC", peaks["int_tops"], peaks["hbm_tbps"]),
        "TENSO": build_points(rows, "TENSO", peaks["tensor_tfps"], peaks["hbm_tbps"]),
    }

    x_min_log = math.log10(args.x_min)
    x_max_log = math.log10(args.x_max)
    y_min_log = args.y_min_log10
    y_max_log = args.y_max_log10

    fp64_bins = [[0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    int_bins = [[0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    tensor_bins = [[0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]

    def add_point(ai: float, throughput: float, bins: List[List[int]]) -> None:
        if ai < args.x_min or ai > args.x_max or throughput <= 0:
            return
        y_log = math.log10(throughput)
        if y_log < y_min_log or y_log > y_max_log:
            return
        x_norm = (math.log10(ai) - x_min_log) / (x_max_log - x_min_log)
        y_norm = (y_log - y_min_log) / (y_max_log - y_min_log)
        x_idx = min(args.x_bins - 1, max(0, int(math.floor(x_norm * args.x_bins))))
        y_idx = min(args.y_bins - 1, max(0, int(math.floor(y_norm * args.y_bins))))
        bins[y_idx][x_idx] += 1

    for ai, throughput in per_metric_points["FP64A"]:
        add_point(ai, throughput, fp64_bins)
    for ai, throughput in per_metric_points["INTAC"]:
        add_point(ai, throughput, int_bins)
    for ai, throughput in per_metric_points["TENSO"]:
        add_point(ai, throughput, tensor_bins)

    category_heatmap = [[0.0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    category_count = [[0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    category_label = [["None" for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    total_points_all_metrics = max(1, len(per_metric_points["FP64A"]) + len(per_metric_points["INTAC"]) + len(per_metric_points["TENSO"]))
    global_max_norm = 0.0
    for y in range(args.y_bins):
        for x in range(args.x_bins):
            fp64_count = fp64_bins[y][x]
            int_count = int_bins[y][x]
            tensor_count = tensor_bins[y][x]
            fp64_norm = fp64_count / total_points_all_metrics
            int_norm = int_count / total_points_all_metrics
            tensor_norm = tensor_count / total_points_all_metrics
            max_norm = max(fp64_norm, int_norm, tensor_norm)
            category_count[y][x] = fp64_count + int_count + tensor_count
            global_max_norm = max(global_max_norm, max_norm)
            if max_norm == 0:
                category_heatmap[y][x] = 0.0
                category_label[y][x] = "None"
            elif fp64_norm >= int_norm and fp64_norm >= tensor_norm:
                category_heatmap[y][x] = 1.0 + max_norm
                category_label[y][x] = "FP64"
            elif int_norm >= tensor_norm:
                category_heatmap[y][x] = 2.0 + max_norm
                category_label[y][x] = "INT"
            else:
                category_heatmap[y][x] = 3.0 + max_norm
                category_label[y][x] = "Tensor"

    normalized_category = [[0.0 for _ in range(args.x_bins)] for _ in range(args.y_bins)]
    for y in range(args.y_bins):
        for x in range(args.x_bins):
            value = category_heatmap[y][x]
            if value == 0.0 or global_max_norm == 0.0:
                normalized_category[y][x] = 0.0
                continue
            base = math.floor(value)
            norm = value - base
            scaled_norm = norm / global_max_norm
            normalized_category[y][x] = base + scaled_norm

    x_centers = []
    for i in range(args.x_bins):
        x_center_log = x_min_log + ((i + 0.5) / args.x_bins) * (x_max_log - x_min_log)
        x_centers.append(10 ** x_center_log)

    y_centers = []
    for i in range(args.y_bins):
        y_center_log = y_min_log + ((i + 0.5) / args.y_bins) * (y_max_log - y_min_log)
        y_centers.append(10 ** y_center_log)

    output = {
        "source_file": str(args.dat_path),
        "sample_count": len(rows),
        "binning": {
            "x_bins": args.x_bins,
            "y_bins": args.y_bins,
            "x_range_ai": [args.x_min, args.x_max],
            "y_range_log10": [args.y_min_log10, args.y_max_log10],
        },
        "gpu_name": gpu_name,
        "peaks": peaks,
        "per_metric_point_count": {k: len(v) for k, v in per_metric_points.items()},
        "heatmap": {
            "x_centers": x_centers,
            "y_centers": y_centers,
            "z_normalized_category": normalized_category,
            "z_category_count": category_count,
            "z_category_label": category_label,
            "total_plotted_points": total_points_all_metrics,
        },
        "roofline_curves": {
            "FP64": build_roofline_curve(peaks["fp64_tfps"], peaks["hbm_tbps"], args.x_min, args.x_max),
            "FP32": build_roofline_curve(peaks["fp32_tfps"], peaks["hbm_tbps"], args.x_min, args.x_max),
            "INT": build_roofline_curve(peaks["int_tops"], peaks["hbm_tbps"], args.x_min, args.x_max),
            "Tensor": build_roofline_curve(peaks["tensor_tfps"], peaks["hbm_tbps"], args.x_min, args.x_max),
        },
    }

    if args.png_output:
        render_png(
            png_path=args.png_output,
            z_normalized_category=normalized_category,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min_log10=args.y_min_log10,
            y_max_log10=args.y_max_log10,
            roofline_curves=output["roofline_curves"],
            gpu_name=gpu_name,
        )

    payload = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
