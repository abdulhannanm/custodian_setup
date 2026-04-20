from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from generate_roofline_heatmap import convert_to_df

DEFAULT_FILE_READING_FEATURES = ["TENSO", "DRAMA", "FP64A", "PCIE_LOAD", "PCIETX", "PCIERX"]
PCIE_MODIFIED_FEATURES = ["TENSO", "DRAMA", "FP64A", "PCIE_LOAD", "PCIE_DIRECTION"]
ROOFLINE_TRIGGER_METRICS = ("INTAC", "TENSO", "FP64A", "FP32A")
PLOT_WIDTH_PX = 1200
PLOT_HEIGHT_PX = 800
PLOT_DPI = 100


def save_radar_png_with_matplotlib(
    output_path: Path,
    features: List[str],
    values: List[float],
    time: int,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    wrapped_values = values + values[:1]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI), subplot_kw={"projection": "polar"})
    ax.plot(angles, wrapped_values, linewidth=3, color="#1f77b4")
    ax.fill(angles, wrapped_values, color="#1f77b4", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f"Radar Plot at t={time} (100ms Sampling)", pad=24)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def perform_kmeans(perform_df, n_clusters: int) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(perform_df)

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    kmeans.fit(scaled_matrix)

    perform_df["Cluster_Label"] = kmeans.labels_ + 1

    return perform_df


def apply_error_correction(df, threshold: float = 5e-2):
    matrix = df.to_numpy(copy=True)
    matrix[np.abs(matrix) < threshold] = 0
    df.loc[:, :] = matrix
    return df


def interpret_clusters(df: pd.DataFrame) -> dict:
    classification_dict = {}
    feature_cols = [c for c in df.columns if c != "Cluster_Label"]
    has_pcie = "PCIE_LOAD" in feature_cols and "PCIE_DIRECTION" in feature_cols

    if has_pcie:
        compute_cols = [c for c in feature_cols if c not in ("PCIE_LOAD", "PCIE_DIRECTION")]
        summary = df.groupby("Cluster_Label")[compute_cols].median()
        pcie_load = df.groupby("Cluster_Label")["PCIE_LOAD"].median()

        pcie_direction_active = (
            df[df["PCIE_LOAD"] == 1]
            .groupby("Cluster_Label")["PCIE_DIRECTION"]
            .mean()
            .reindex(summary.index, fill_value=0.0)
        )

        special_cases = pd.concat(
            [
                pcie_load.rename("PCIE_LOAD"),
                pcie_direction_active.rename("PCIE_DIRECTION"),
            ],
            axis=1,
        )
        final_df = pd.concat([summary, special_cases], axis=1)
    else:
        compute_cols = feature_cols
        summary = df.groupby("Cluster_Label")[compute_cols].median()
        final_df = summary

    all_cols = final_df.columns.tolist()
    clusters = final_df.index.values

    for cluster in clusters:
        classification_list = []
        pcie_active = False
        for col in all_cols:
            val = final_df.loc[cluster, col]
            if col == "PCIE_LOAD":
                if val >= 0.5:
                    classification_list.append("PCIE_LOAD HIGH")
                    pcie_active = True
            elif col == "PCIE_DIRECTION":
                if pcie_active:
                    if val >= 0.2:
                        classification_list.append("PCIETX INTENSIVE")
                    elif val <= -0.2:
                        classification_list.append("PCIERX INTENSIVE")
                    elif val >= 0.05:
                        classification_list.append("Extremely Low PCIETX")
                    elif val <= -0.05:
                        classification_list.append("Extremely Low PCIERX")
                    else:
                        classification_list.append("PCIETX = PCIERX")
            else:
                if val >= 0.5:
                    classification_list.append(f"High {col}")
                elif val >= 0.1:
                    classification_list.append(f"Low {col}")
                elif val > 0:
                    classification_list.append(f"Extremely Low {col}")
        if not classification_list:
            classification_list.append("IDLE")
        classification_dict[f"Cluster {cluster}"] = frozenset(classification_list)

    return classification_dict


def find_optimal_k(df, min_val: int = 2, max_val: int = 11):
    error_corrected_df = apply_error_correction(df)
    k_range = range(min_val, max_val)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(error_corrected_df)

    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(scaled_matrix)
        inertias.append(kmeans.inertia_)

    # Scan from higher k toward lower k to find where the elbow starts to
    # flatten (plateau), then validate cluster label uniqueness by decrementing.
    optimal_k = list(k_range)[-1]
    for i in range(len(inertias) - 2, 0, -1):
        left_drop = inertias[i - 1] - inertias[i] #positive, big
        right_drop = inertias[i] - inertias[i + 1]   #positive, small WARNING: BANKING HARD ON THE ASSUMPTION THAT DATA GIVES TXTBOOK ELBOW PLOT
        if left_drop > 0 and right_drop / left_drop < 0.5:
            optimal_k = list(k_range)[i]
            break
    else:
        optimal_k = list(k_range)[-1]

    print("optimal k: ", optimal_k)
    print("lowest score: ", inertias[list(k_range).index(optimal_k)])
    found_optimal_k = False
    while not found_optimal_k and optimal_k != 0:
        kmean_df = perform_kmeans(error_corrected_df, optimal_k)
        classification_dict = interpret_clusters(kmean_df)
        dict_values = classification_dict.values()
        set_values = set(dict_values) #removes all duplicates
        found_optimal_k = len(dict_values) == len(set_values) #checks to see if nonduplicated version == original
        if found_optimal_k:
            print(f"FOUND OPTIMAL K! {optimal_k}")
            return kmean_df, classification_dict
        optimal_k -= 1
    return "No optimal k"


@dataclass
class dataObject:
    filepath: str = ""
    cleaned_name: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    phase_plot_path: str = ""
    line_plot_path: str = ""
    radar_plot_path: str = ""
    classification_dict: Dict = field(default_factory=dict)
    cluster_activity_times: Dict = field(default_factory=dict)

    def __init__(self, filepath):
        path_obj = Path(filepath)
        self.cleaned_name = path_obj.stem
        self.filepath = filepath
        if filepath.endswith(".dat"):
            self.df = convert_to_df(path_obj)
            self.df = self.df.apply(pd.to_numeric, errors="coerce")
        else:
            self.df = pd.read_csv(filepath)

        # Keep clustering features on comparable scales.
        # FBFRE and SMCLK are not normalized to [0, 1] like activity features.
        self.df.drop(labels=["FBFRE", "SMCLK"], axis=1, inplace=True, errors="ignore")

    def csv_data_preprocessing(self, gap_threshold: int = 20) -> str:
        """
        Preprocesses the data by applying PCIE filtering (if PCITX/PCIERX present)
        and K-means clustering. Must be called before any plot generation functions.
        """
        has_pcie_raw = "PCITX" in self.df.columns and "PCIRX" in self.df.columns
        if has_pcie_raw:
            obj_with_pcie = self.pcie_filter()
            if obj_with_pcie is None:
                raise ValueError("pcie_filter failed unexpectedly.")
        else:
            if "TIMESTAMP" in self.df.columns:
                self.df.drop(labels=["TIMESTAMP"], axis=1, inplace=True)

        copy_df = self.df.copy()
        optimal_k_results = find_optimal_k(copy_df)
        if optimal_k_results == "No optimal k":
            return optimal_k_results
        finished_df, class_dict = optimal_k_results
        self.df = finished_df
        self.classification_dict = class_dict
        self.cluster_activity_times = self.analyze_cluster_peaks(gap_threshold=gap_threshold)
        return "success"

    def pcie_filter(self, threshold: float = 7):
        if self.df.get("PCITX").empty or self.df.get("PCIRX").empty:
            print("This file does not have the right column labels")
            return None
        raw_sum = self.df["PCITX"] + self.df["PCIRX"] + (1e-12)
        log_sum = np.log10(raw_sum)
        pcie_load = np.where(log_sum >= threshold, 1.0, 0.0)
        self.df["PCIE_LOAD"] = pcie_load

        numerator = self.df["PCITX"] - self.df["PCIRX"]
        quotient = numerator / (raw_sum.to_numpy(copy=False))
        conditions = [
            (self.df["PCIE_LOAD"] == 1) & (quotient >= 0.2),
            (self.df["PCIE_LOAD"] == 1) & (quotient <= -0.2),
        ]
        choices = [1, -1]
        self.df["PCIE_DIRECTION"] = np.select(condlist=conditions, choicelist=choices, default=0)
        self.df.drop(labels=["PCITX", "PCIRX", "TIMESTAMP"], axis=1, inplace=True, errors="ignore")
        return self

    def get_cluster_analysis(self) -> dict:
        """Get cluster analysis results. Must call data_preprocessing() first."""
        if not self.classification_dict:
            raise ValueError("data_preprocessing() must be called before get_cluster_analysis()")
        return {
            "classification_dict": self.classification_dict,
            "cluster_activity_times": self.cluster_activity_times,
        }

    def generate_cluster_roofline_plots(self, trigger_metrics=ROOFLINE_TRIGGER_METRICS) -> Dict[str, object]:
        """Generate roofline plots for clusters with non-zero mean compute trigger metrics."""
        if "Cluster_Label" not in self.df.columns or not self.classification_dict:
            raise ValueError("data_preprocessing() must be called before generate_cluster_roofline_plots()")

        roofline_paths: List[str] = []
        cluster_compute_metric_means: Dict[str, Dict[str, float]] = {}
        output_dir = Path(__file__).parents[1] / "plots" / "roofline_by_cluster_auto"
        output_dir.mkdir(parents=True, exist_ok=True)

        source_gpu_name: Optional[str] = None
        source_input_path = Path(self.filepath)
        if source_input_path.suffix.lower() == ".dat":
            from generate_roofline_heatmap import parse_gpu_name_from_dat

            source_gpu_name = parse_gpu_name_from_dat(source_input_path)

        for cluster_name in self.classification_dict.keys():

            try:
                cluster_id = int(cluster_name.split(" ")[-1])
            except Exception:
                continue

            cluster_rows = self.df[self.df["Cluster_Label"] == cluster_id].copy()
            if cluster_rows.empty:
                continue

            non_zero_means: Dict[str, float] = {}
            for metric in trigger_metrics:
                if metric not in cluster_rows.columns:
                    continue
                metric_mean = float(cluster_rows[metric].mean())
                if metric_mean > 0:
                    non_zero_means[metric] = metric_mean

            if not non_zero_means:
                continue

            cluster_compute_metric_means[cluster_name] = non_zero_means

            unique_tag = uuid.uuid4().hex[:10]
            csv_path = output_dir / f"{self.cleaned_name}_cluster_{cluster_id}_{unique_tag}.csv"
            cluster_rows.to_csv(csv_path, index=False)

            cluster_obj = dataObject(str(csv_path))
            roofline_png = cluster_obj.generate_roofline_heatmap(gpu_name_override=source_gpu_name)
            target_png = output_dir / f"{self.cleaned_name}_cluster_{cluster_id}_{unique_tag}_roofline.png"
            Path(roofline_png).replace(target_png)
            roofline_paths.append(str(target_png))

        return {
            "possible_metrics_for_investigative_tools" : list(self.df.columns),
            "cluster_roofline_plot_paths": roofline_paths,
            "cluster_compute_metric_means": cluster_compute_metric_means,
        }

    def create_colored_phase_plot(self, output_path: str):
        if "Cluster_Label" not in self.df.columns:
            raise ValueError("data_preprocessing() must be called before create_colored_phase_plot()")

        idx = self.df.index

        num_clusters = len(self.df["Cluster_Label"].unique())
        x = idx + 1
        y = self.df["Cluster_Label"]

        plt.figure(figsize=(PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI))

        cluster_colors = {
            1: "#1f77b4",
            2: "#ff7f0e",
            3: "#2ca02c",
            4: "#d62728",
            5: "#9467bd",
            6: "#8c564b",
            7: "#e377c2",
            8: "#333333",
            9: "#bcbd22",
            10: "#17becf",
        }
        yticks = []

        for i in range(1, num_clusters + 1):
            legend_label = self.classification_dict[f"Cluster {i}"]
            str_label = ",".join(legend_label)
            plt.fill_between(x, y, step="post", where=(y == i), color=cluster_colors[i], label=str_label)
            yticks.append(i)

        plt.yticks(yticks)
        plt.xlabel("TIME (1 step = 100ms)")
        plt.ylabel("PHASE")
        plt.title("Phase Progression by Cluster (100ms Sampling)")
        plt.legend(loc="upper right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_DPI)
        plt.close()

        return output_path

    def analyze_cluster_peaks(self, gap_threshold: int = 5) -> Dict[str, List[List[int]]]:
        """Analyze cluster peaks and return time ranges for each cluster."""
        cluster_periods: Dict[str, List[List[int]]] = {}

        for cluster in self.df["Cluster_Label"].unique():
            cluster_mask = self.df["Cluster_Label"] == cluster
            cluster_indices = self.df[cluster_mask].index + 1

            if len(cluster_indices) == 0:
                continue
            ranges: List[List[int]] = []
            start_range = cluster_indices[0]
            end_range = cluster_indices[0]
            for i in range(1, len(cluster_indices)):
                gap = cluster_indices[i] - end_range
                if gap > gap_threshold:
                    ranges.append([int(start_range), int(end_range)])
                    start_range = cluster_indices[i]

                end_range = cluster_indices[i]
            ranges.append([int(start_range), int(end_range)])
            cp_key = self.classification_dict[f"Cluster {cluster}"]
            cp_key = ",".join(cp_key)
            cluster_periods[cp_key] = ranges
        return cluster_periods

    def generate_line_plot(self, label: str, xmin: int, xmax: int, output_dir: str = "../plots") -> str:
        """Plot smoothed signal for a specific column and time range."""
        df_copy = self.df.copy()

        mask = ((df_copy.index + 1) >= xmin) & ((df_copy.index + 1) <= xmax)
        df_slice = df_copy[mask]

        if label not in df_copy.columns:
            raise ValueError(f"LABEL {label} NOT IN DF")

        x = df_slice.index + 1
        y = df_slice[label].values
        smoothed_y = savgol_filter(y, window_length=5, polyorder=4)
        smoothed_y[np.abs(smoothed_y) < 5e-2] = 0

        corrected_y = medfilt(smoothed_y, kernel_size=7).astype(float)

        try:
            plt.figure(figsize=(PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI))
            plt.clf()
            plt.plot(x, corrected_y, label=f"Error Corrected {label}")
            plt.ylim(0, 1)
            plt.legend()
            plt.xlabel("TIME (1 step = 100ms)")
            plt.ylabel(label)
            plt.title(f"Line Plot at {label} ({xmin}-{xmax}, 100ms Sampling)")

            unique_tag = uuid.uuid4().hex[:10]
            save_path = Path(output_dir) / "line_plot" / f"{self.cleaned_name}_{xmin}-{xmax}_{label}_{unique_tag}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=PLOT_DPI)
            print(f"Saved plot to {save_path}")
            self.line_plot_path = str(save_path)
            return str(save_path)

        except Exception as e:
            print(f"Error plotting {label}: {e}")
            raise
        finally:
            plt.close()

    def generate_roofline_heatmap(
        self,
        x_min: float = 0.01,
        x_max: float = 1000.0,
        y_min_log10: float = -2.0,
        y_max_log10: float = 3.0,
        x_bins: int = 40,
        y_bins: int = 40,
        gpu_name_override: Optional[str] = None,
    ) -> str:
        """Generate roofline heatmap PNG from a .dat or .csv file and return the path."""
        import math
        import uuid

        from generate_roofline_heatmap import (
            build_points,
            build_roofline_curve,
            get_gpu_peaks,
            parse_csv_rows,
            parse_dcgm_rows,
            parse_gpu_name_from_dat,
            render_png,
        )

        input_path = Path(self.filepath)
        gpu_name = (gpu_name_override or parse_gpu_name_from_dat(input_path)).upper()
        peaks = get_gpu_peaks(gpu_name)
        if input_path.suffix.lower() == ".csv":
            rows = parse_csv_rows(input_path)
        else:
            rows = parse_dcgm_rows(input_path)
        if not rows:
            raise ValueError(f"No GPU data rows parsed from: {self.filepath}")

        per_metric_points = {
            "FP64A": build_points(rows, "FP64A", peaks["fp64_tfps"], peaks["hbm_tbps"]),
            "FP32A": build_points(rows, "FP32A", peaks["fp32_tfps"], peaks["hbm_tbps"]),
            "INTAC": build_points(rows, "INTAC", peaks["int_tops"], peaks["hbm_tbps"]),
            "TENSO": build_points(rows, "TENSO", peaks["tensor_tfps"], peaks["hbm_tbps"]),
        }

        x_min_log = math.log10(x_min)
        x_max_log = math.log10(x_max)

        fp64_bins = [[0] * x_bins for _ in range(y_bins)]
        int_bins = [[0] * x_bins for _ in range(y_bins)]
        tensor_bins = [[0] * x_bins for _ in range(y_bins)]

        def add_point(ai: float, throughput: float, bins) -> None:
            if ai < x_min or ai > x_max or throughput <= 0:
                return
            y_log = math.log10(throughput)
            if y_log < y_min_log10 or y_log > y_max_log10:
                return
            x_norm = (math.log10(ai) - x_min_log) / (x_max_log - x_min_log)
            y_norm = (y_log - y_min_log10) / (y_max_log10 - y_min_log10)
            x_idx = min(x_bins - 1, max(0, int(math.floor(x_norm * x_bins))))
            y_idx = min(y_bins - 1, max(0, int(math.floor(y_norm * y_bins))))
            bins[y_idx][x_idx] += 1

        for ai, throughput in per_metric_points["FP64A"]:
            add_point(ai, throughput, fp64_bins)
        for ai, throughput in per_metric_points["INTAC"]:
            add_point(ai, throughput, int_bins)
        for ai, throughput in per_metric_points["TENSO"]:
            add_point(ai, throughput, tensor_bins)

        total_points = max(1, len(per_metric_points["FP64A"]) + len(per_metric_points["INTAC"]) + len(per_metric_points["TENSO"]))
        category_heatmap = [[0.0] * x_bins for _ in range(y_bins)]
        global_max_norm = 0.0

        for y in range(y_bins):
            for x in range(x_bins):
                fp64_norm = fp64_bins[y][x] / total_points
                int_norm = int_bins[y][x] / total_points
                tensor_norm = tensor_bins[y][x] / total_points
                max_norm = max(fp64_norm, int_norm, tensor_norm)
                global_max_norm = max(global_max_norm, max_norm)
                if max_norm == 0:
                    category_heatmap[y][x] = 0.0
                elif fp64_norm >= int_norm and fp64_norm >= tensor_norm:
                    category_heatmap[y][x] = 1.0 + max_norm
                elif int_norm >= tensor_norm:
                    category_heatmap[y][x] = 2.0 + max_norm
                else:
                    category_heatmap[y][x] = 3.0 + max_norm

        normalized_category = [[0.0] * x_bins for _ in range(y_bins)]
        for y in range(y_bins):
            for x in range(x_bins):
                value = category_heatmap[y][x]
                if value == 0.0 or global_max_norm == 0.0:
                    continue
                base = math.floor(value)
                normalized_category[y][x] = base + (value - base) / global_max_norm

        roofline_curves = {
            "FP64": build_roofline_curve(peaks["fp64_tfps"], peaks["hbm_tbps"], x_min, x_max),
            "FP32": build_roofline_curve(peaks["fp32_tfps"], peaks["hbm_tbps"], x_min, x_max),
            "INT": build_roofline_curve(peaks["int_tops"], peaks["hbm_tbps"], x_min, x_max),
            "Tensor": build_roofline_curve(peaks["tensor_tfps"], peaks["hbm_tbps"], x_min, x_max),
        }

        png_path = Path(__file__).parents[1] / "plots" / "roofline_heatmap" / f"roofline_heatmap_{uuid.uuid4().hex[:10]}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        render_png(
            png_path=png_path,
            z_normalized_category=normalized_category,
            x_min=x_min,
            x_max=x_max,
            y_min_log10=y_min_log10,
            y_max_log10=y_max_log10,
            roofline_curves=roofline_curves,
            gpu_name=gpu_name,
        )
        return str(png_path)

    def generate_radar_plot(self, time: int, features: List[str] = None):
        if features is None:
            features = [c for c in self.df.columns if c != "Cluster_Label"]
        required_cols = set(features)
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        row = self.df.loc[(self.df.index + 1) == time]
        if row.empty:
            raise ValueError(f"No data found for timestamp {time}")
        row = row.iloc[0]

        values_array = row[features].astype(float).to_numpy()
        if np.isnan(values_array).any():
            raise ValueError("Non-numeric data encountered in feature columns")
        values = values_array.tolist()
        metric_values = {feature: float(value) for feature, value in zip(features, values)}
        unique_tag = uuid.uuid4().hex[:10]
        output_path = Path(__file__).parents[2] / "app" / "plots" / "radar_plot" / f"{self.cleaned_name}_radar_plot_{time}_{unique_tag}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_radar_png_with_matplotlib(
            output_path=output_path,
            features=features,
            values=values,
            time=time,
        )
        self.radar_plot_path = output_path
        return {
            "file_path": str(output_path),
            "metric_values": metric_values,
        }


if __name__ == "__main__":
    out_name = Path(__file__).parents[2] / "plots" / "phase_plots/phase_plot.png"
    filename = Path(__file__).parents[2] / "data_samples" / "cleaned_metrics.csv"

    new_object = dataObject(filepath=filename)
    return_value = new_object.data_preprocessing()
    if return_value:
        radar_plot_path = new_object.generate_line_plot(label="DRAMA", xmin=1100, xmax=1165)
