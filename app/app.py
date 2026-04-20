#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

# Ensure the app/ directory is on sys.path so that sibling modules
# (generate_roofline_heatmap, models/) resolve correctly regardless
# of the working directory the process was launched from.
sys.path.insert(0, str(Path(__file__).parent))

from models.dataObject import dataObject


def main():
    parser = argparse.ArgumentParser(
        description='Process cluster data and generate plots'
    )
    parser.add_argument(
        '--inputFile',
        type=str,
        help='Path to the input CSV data file (optional, auto-discovers from data_samples/ if not provided)'
    )
    parser.add_argument(
        '--functionName',
        type=str,
        required=True,
        choices=['get_cluster_analysis', 'generate_radar_plot', 'plot_signal_smoothing', 'generate_roofline_heatmap'],
        help='Function to call on the dataObject'
    )
    parser.add_argument(
        '--time',
        type=int,
        help='Time point for generate_radar_plot (time = index + 1)'
    )
    parser.add_argument(
        '--label',
        type=str,
        help='Label for plot_signal_smoothing (e.g., TENSO, DRAMA, FP64A)'
    )
    parser.add_argument(
        '--xmin',
        type=int,
        help='Minimum x value for plot_signal_smoothing'
    )
    parser.add_argument(
        '--xmax',
        type=int,
        help='Maximum x value for plot_signal_smoothing'
    )
    parser.add_argument('--x-min', type=float, default=0.01, help='Roofline: minimum arithmetic intensity')
    parser.add_argument('--x-max', type=float, default=1000.0, help='Roofline: maximum arithmetic intensity')
    parser.add_argument('--y-min-log10', type=float, default=-2.0, help='Roofline: minimum log10 throughput')
    parser.add_argument('--y-max-log10', type=float, default=3.0, help='Roofline: maximum log10 throughput')
    parser.add_argument('--x-bins', type=int, default=40, help='Roofline: number of x-axis bins')
    parser.add_argument('--y-bins', type=int, default=40, help='Roofline: number of y-axis bins')
    parser.add_argument('--png-output', type=str, help='Roofline: optional path to write PNG output')

    args = parser.parse_args()

    cluster_error_result: dict[str, Any] = {
        "error": None,
        "classification_map": None,
        "cluster_activity_times": None,
        "cluster_roofline_plot_paths": [],
        "cluster_compute_metric_means": {},
    }

    plot_error_result: dict[str, Any] = {
        "error": None,
        "file_path": None,
        "description": None
    }

    plots_dir = Path(os.environ.get("PLOTS_DIR", Path(__file__).parent / "plots")).resolve()
    plot_registry_url = os.environ.get("PLOT_REGISTRY_URL", "").strip()
    session_id = os.environ.get("SESSION_ID", "").strip() or "unknown-session"

    def register_plot(file_path: str, tool_name: str, title: str, description: Optional[str] = None):
        if not plot_registry_url:
            return None
        if not file_path:
            return None

        candidate_path = Path(file_path).resolve()
        try:
            rel_path = str(candidate_path.relative_to(plots_dir)).replace("\\", "/")
        except ValueError:
            return None

        payload = {
            "session_id": session_id,
            "tool_name": tool_name,
            "rel_path": rel_path,
            "title": title,
            "description": description,
        }

        try:
            req = urllib.request.Request(
                plot_registry_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.5) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("plot_id")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return None

    def get_input_file():
        if args.inputFile:
            try:
                return os.path.expanduser(args.inputFile)
            except Exception as e:
                return e
        else:
            return None

    try:
        input_file = get_input_file()
        if input_file is Exception:
            error_msg = input_file
            if args.functionName == 'get_cluster_analysis':
                cluster_error_result["error"] = error_msg
                print(json.dumps(cluster_error_result))
            else:
                plot_error_result["error"] = error_msg
                print(json.dumps(plot_error_result))
            sys.exit(1)

        if input_file is None:
            error_msg = f"Data file not found: {input_file}"
            if args.functionName == 'get_cluster_analysis':
                cluster_error_result["error"] = error_msg
                print(json.dumps(cluster_error_result))
            else:
                plot_error_result["error"] = error_msg
                print(json.dumps(plot_error_result))
            sys.exit(1)

        obj = dataObject(input_file)

        if args.functionName != 'generate_roofline_heatmap':
            preprocessing_result = obj.csv_data_preprocessing()
            if preprocessing_result == "No optimal k":
                error_msg = "Could not find optimal k for clustering"
                if args.functionName == 'get_cluster_analysis':
                    cluster_error_result["error"] = error_msg
                    print(json.dumps(cluster_error_result))
                else:
                    plot_error_result["error"] = error_msg
                    print(json.dumps(plot_error_result))
                sys.exit(1)

        if args.functionName == 'get_cluster_analysis':
            analysis_data = obj.get_cluster_analysis()
            roofline_data = obj.generate_cluster_roofline_plots()
            phase_unique_tag = uuid.uuid4().hex[:10]
            phase_plot_path = Path(__file__).parent / "plots" / "phase_plot" / f"{obj.cleaned_name}_{phase_unique_tag}_phase_plot.png"
            phase_plot_path.parent.mkdir(parents=True, exist_ok=True)
            obj.create_colored_phase_plot(str(phase_plot_path))

            registered_plots = []
            phase_plot_id = register_plot(
                file_path=str(phase_plot_path),
                tool_name='get_cluster_analysis',
                title='Phase progression plot',
                description='Cluster phase progression over time',
            )
            if phase_plot_id:
                registered_plots.append(phase_plot_id)

            roofline_paths = roofline_data.get("cluster_roofline_plot_paths", [])
            if not isinstance(roofline_paths, list):
                roofline_paths = []

            for i, roofline_path in enumerate(roofline_paths, start=1):
                match = re.search(r"_cluster_(\d+)_", Path(roofline_path).name)
                cluster_label = f"Cluster {match.group(1)}" if match else f"Cluster {i}"
                roofline_plot_id = register_plot(
                    file_path=roofline_path,
                    tool_name='get_cluster_analysis',
                    title=f'{cluster_label} roofline plot',
                    description=f'Roofline heatmap for {cluster_label}',
                )
                if roofline_plot_id:
                    registered_plots.append(roofline_plot_id)

            class_dict_serializable = {
                k: list(v) if isinstance(v, frozenset) else v
                for k, v in analysis_data["classification_dict"].items()
            }

            cluster_result = {
                "error": None,
                "classification_map": class_dict_serializable,
                "cluster_activity_times": analysis_data["cluster_activity_times"],
                "cluster_roofline_plot_paths": roofline_data["cluster_roofline_plot_paths"],
                "cluster_compute_metric_means": roofline_data["cluster_compute_metric_means"],
                "plot_ids": registered_plots,
                "possible_metrics_for_investigative_tools": roofline_data["possible_metrics_for_investigative_tools"]
            }
            print(json.dumps(cluster_result))

        elif args.functionName == 'generate_radar_plot':
            if args.time is None:
                plot_error_result["error"] = "time parameter is required for generate_radar_plot"
                print(json.dumps(plot_error_result))
                sys.exit(1)

            radar_data = obj.generate_radar_plot(args.time)

            radar_result = {
                "error": None,
                "file_path": radar_data["file_path"],
                "metric_values": radar_data["metric_values"],
                "description": f"Radar plot showing feature values at time point {args.time}",
                "plot_id": register_plot(
                    file_path=radar_data["file_path"],
                    tool_name='generate_radar_plot',
                    title=f'Radar plot at time {args.time}',
                    description=f'Radar plot for timestamp {args.time}',
                )
            }
            print(json.dumps(radar_result))

        elif args.functionName == 'plot_signal_smoothing':
            if not args.label:
                plot_error_result["error"] = "label parameter is required for plot_signal_smoothing"
                print(json.dumps(plot_error_result))
                sys.exit(1)

            if args.xmin is None or args.xmax is None:
                plot_error_result["error"] = "xmin and xmax parameters are required for plot_signal_smoothing"
                print(json.dumps(plot_error_result))
                sys.exit(1)

            output_dir = Path(__file__).parent / "plots"
            line_path = obj.generate_line_plot(
                label=args.label,
                xmin=args.xmin,
                xmax=args.xmax,
                output_dir=str(output_dir)
            )

            line_result = {
                "error": None,
                "file_path": line_path,
                "description": f"Smoothed line plot for {args.label} from time {args.xmin} to {args.xmax}",
                "plot_id": register_plot(
                    file_path=line_path,
                    tool_name='plot_signal_smoothing',
                    title=f'{args.label} smoothing {args.xmin}-{args.xmax}',
                    description=f'Smoothed {args.label} signal from {args.xmin} to {args.xmax}',
                )
            }
            print(json.dumps(line_result))

        elif args.functionName == 'generate_roofline_heatmap':
            png_path = obj.generate_roofline_heatmap(
                x_min=args.x_min,
                x_max=args.x_max,
                y_min_log10=args.y_min_log10,
                y_max_log10=args.y_max_log10,
                x_bins=args.x_bins,
                y_bins=args.y_bins,
            )
            roofline_result = {
                "error": None,
                "file_path": png_path,
                "description": f"Roofline heatmap PNG for {input_file}",
                "plot_id": register_plot(
                    file_path=png_path,
                    tool_name='generate_roofline_heatmap',
                    title='Roofline heatmap',
                    description=f'Roofline heatmap generated from {input_file}',
                )
            }
            print(json.dumps(roofline_result))

    except Exception as e:
        error_msg = str(e)
        if args.functionName == 'get_cluster_analysis':
            cluster_error_result["error"] = error_msg
            print(json.dumps(cluster_error_result))
        else:
            plot_error_result["error"] = error_msg
            print(json.dumps(plot_error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
