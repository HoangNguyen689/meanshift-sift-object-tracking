from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

# Use a non-interactive backend so chart export works on Windows/headless machines
# without requiring Tkinter/Tcl. This must be set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import center_error, iou, precision_at, success_auc
from trackers import BBox, TrackResult, build_tracker, validate_tracker_environment


TRACKERS = ["MeanShift", "CBWH-MeanShift", "CAMShift", "SOAMST", "SIFT", "MeanShift+SIFT", "KCF"]
DISPLAY_NAMES = {
    "MeanShift": "MeanShift",
    "CBWH-MeanShift": "CBWH-MeanShift",
    "CAMShift": "CAMShift",
    "SOAMST": "SOAMST",
    "SIFT": "SIFT",
    "MeanShift+SIFT": "MeanShift+SIFT",
    "KCF": "KCF",
}


def parse_gt_line(line: str) -> BBox:
    clean = line.strip().replace("\t", ",").replace(" ", ",")
    values = [float(item) for item in clean.split(",") if item != ""]
    if len(values) == 4:
        return tuple(values)  # type: ignore[return-value]
    if len(values) == 8:
        xs = values[0::2]
        ys = values[1::2]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    raise ValueError(f"Unsupported ground-truth row: {line!r}")


def load_sequence(sequence_dir: Path) -> tuple[list[Path], list[BBox]]:
    image_dir = sequence_dir / "img"
    if not image_dir.exists():
        image_dir = sequence_dir / "imgs"
    image_paths = sorted(
        [path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    gt_file = sequence_dir / "groundtruth_rect.txt"
    if not gt_file.exists():
        candidates = sorted(sequence_dir.glob("groundtruth*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No ground-truth file in {sequence_dir}")
        gt_file = candidates[0]
    ground_truth = [parse_gt_line(line) for line in gt_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    count = min(len(image_paths), len(ground_truth))
    if count < 2:
        raise ValueError(f"Sequence {sequence_dir.name} has fewer than two annotated frames")
    return image_paths[:count], ground_truth[:count]


def discover_sequences(dataset_dir: Path) -> list[Path]:
    sequences: list[Path] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "groundtruth_rect.txt").exists() or list(child.glob("groundtruth*.txt")):
            sequences.append(child)
    if not sequences:
        raise FileNotFoundError(f"No compatible sequences found under {dataset_dir}")
    return sequences


def draw_bbox(frame: np.ndarray, bbox: BBox, label: str, ok: bool) -> None:
    x, y, w, h = [int(round(value)) for value in bbox]
    thickness = 2 if ok else 1
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness)
    cv2.putText(frame, label, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def run_once(
    sequence_name: str,
    image_paths: list[Path],
    ground_truth: list[BBox],
    tracker_name: str,
    save_video_path: Path | None = None,
    timing_clock: str = "wall",
) -> tuple[list[dict[str, Any]], float]:
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read {image_paths[0]}")
    tracker = build_tracker(tracker_name)
    tracker.initialize(first_frame, ground_truth[0])

    writer: cv2.VideoWriter | None = None
    if save_video_path is not None:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(save_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (width, height),
        )

    rows: list[dict[str, Any]] = []
    update_times: list[float] = []

    # Use nanosecond clocks to avoid zero-length measurements on Windows.
    # Very fast trackers (notably CBWH-MeanShift) can complete an update inside
    # one tick of thread_time/process_time. In that case, fall back first to
    # process CPU time and finally to perf_counter wall time. This preserves the
    # requested CPU-time clock whenever it has measurable resolution while
    # guaranteeing a strictly positive denominator for FPS.
    thread_clock_ns = getattr(
        time,
        "thread_time_ns",
        lambda: int(getattr(time, "thread_time", time.process_time)() * 1_000_000_000),
    )
    process_clock_ns = getattr(
        time,
        "process_time_ns",
        lambda: int(time.process_time() * 1_000_000_000),
    )
    wall_clock_ns = getattr(
        time,
        "perf_counter_ns",
        lambda: int(time.perf_counter() * 1_000_000_000),
    )

    if timing_clock == "thread_cpu":
        primary_clock_ns = thread_clock_ns
        secondary_clock_ns = process_clock_ns
    elif timing_clock == "process_cpu":
        primary_clock_ns = process_clock_ns
        secondary_clock_ns = None
    elif timing_clock == "wall":
        primary_clock_ns = wall_clock_ns
        secondary_clock_ns = None
    else:
        raise ValueError(f"Unsupported timing clock: {timing_clock}")
    initial_bbox = ground_truth[0]
    rows.append(
        {
            "sequence": sequence_name,
            "tracker": DISPLAY_NAMES[tracker_name],
            "frame": 1,
            "gt_x": initial_bbox[0],
            "gt_y": initial_bbox[1],
            "gt_w": initial_bbox[2],
            "gt_h": initial_bbox[3],
            "pred_x": initial_bbox[0],
            "pred_y": initial_bbox[1],
            "pred_w": initial_bbox[2],
            "pred_h": initial_bbox[3],
            "iou": 1.0,
            "center_error": 0.0,
            "ok": True,
            "update_ms": 0.0,
            "matches": 0,
            "inliers": 0,
            "hist_similarity": 1.0,
            "bhattacharyya": 1.0,
            "estimated_angle": 0.0,
            "semi_major": math.nan,
            "semi_minor": math.nan,
            "estimated_area": math.nan,
            "source": "initialization",
        }
    )

    if writer is not None:
        rendered = first_frame.copy()
        draw_bbox(rendered, initial_bbox, DISPLAY_NAMES[tracker_name], True)
        writer.write(rendered)

    for frame_index, (image_path, gt_bbox) in enumerate(zip(image_paths[1:], ground_truth[1:]), start=2):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read {image_path}")
        wall_start_ns = wall_clock_ns()
        primary_start_ns = primary_clock_ns()
        secondary_start_ns = secondary_clock_ns() if secondary_clock_ns is not None else None

        result: TrackResult = tracker.update(frame)

        primary_elapsed_ns = max(0, primary_clock_ns() - primary_start_ns)
        secondary_elapsed_ns = 0
        if secondary_clock_ns is not None and secondary_start_ns is not None:
            secondary_elapsed_ns = max(0, secondary_clock_ns() - secondary_start_ns)
        wall_elapsed_ns = max(1, wall_clock_ns() - wall_start_ns)

        if primary_elapsed_ns > 0:
            elapsed_ns = primary_elapsed_ns
        elif secondary_elapsed_ns > 0:
            elapsed_ns = secondary_elapsed_ns
        else:
            elapsed_ns = wall_elapsed_ns

        elapsed = elapsed_ns / 1_000_000_000.0
        update_times.append(elapsed)
        pred_bbox = result.bbox
        row = {
            "sequence": sequence_name,
            "tracker": DISPLAY_NAMES[tracker_name],
            "frame": frame_index,
            "gt_x": gt_bbox[0],
            "gt_y": gt_bbox[1],
            "gt_w": gt_bbox[2],
            "gt_h": gt_bbox[3],
            "pred_x": pred_bbox[0],
            "pred_y": pred_bbox[1],
            "pred_w": pred_bbox[2],
            "pred_h": pred_bbox[3],
            "iou": iou(pred_bbox, gt_bbox),
            "center_error": center_error(pred_bbox, gt_bbox),
            "ok": bool(result.ok),
            "update_ms": elapsed * 1000.0,
            "matches": int(result.info.get("matches", 0)),
            "inliers": int(result.info.get("inliers", 0)),
            "hist_similarity": float(result.info.get("hist_similarity", math.nan)),
            "bhattacharyya": float(result.info.get("bhattacharyya", math.nan)),
            "estimated_angle": float(result.info.get("angle", math.nan)),
            "semi_major": float(result.info.get("semi_major", math.nan)),
            "semi_minor": float(result.info.get("semi_minor", math.nan)),
            "estimated_area": float(result.info.get("estimated_area", math.nan)),
            "source": str(result.info.get("source", result.info.get("reason", ""))),
        }
        rows.append(row)

        if writer is not None:
            rendered = frame.copy()
            draw_bbox(rendered, gt_bbox, "GT", True)
            draw_bbox(rendered, pred_bbox, DISPLAY_NAMES[tracker_name], result.ok)
            writer.write(rendered)

    if writer is not None:
        writer.release()
    if update_times:
        total_update_time = float(sum(update_times))
        fps = (len(update_times) / total_update_time) if total_update_time > 0.0 else 0.0
    else:
        fps = 0.0
    return rows, fps


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    repeats: int,
    save_videos: bool,
    tracker_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260801)
    validate_tracker_environment()
    sequences = discover_sequences(dataset_dir)
    selected_trackers = list(tracker_names or TRACKERS)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    fps_samples: dict[tuple[str, str], list[float]] = defaultdict(list)

    for sequence_dir in sequences:
        image_paths, ground_truth = load_sequence(sequence_dir)
        print(f"Sequence: {sequence_dir.name} ({len(image_paths)} frames)")
        for tracker_name in selected_trackers:
            print(f"  - {tracker_name}")
            reference_rows: list[dict[str, Any]] | None = None
            for repeat in range(repeats):
                video_path = None
                if save_videos and repeat == 0:
                    safe_name = tracker_name.replace("+", "_plus_").replace(" ", "_")
                    video_path = output_dir / "videos" / f"{sequence_dir.name}__{safe_name}.mp4"
                rows, fps = run_once(
                    sequence_dir.name,
                    image_paths,
                    ground_truth,
                    tracker_name,
                    save_video_path=video_path,
                )
                fps_samples[(sequence_dir.name, DISPLAY_NAMES[tracker_name])].append(fps)
                if reference_rows is None:
                    reference_rows = rows
            if reference_rows is not None:
                all_rows.extend(reference_rows)

    frame_df = pd.DataFrame(all_rows)
    frame_df.to_csv(output_dir / "per_frame.csv", index=False)

    sequence_rows: list[dict[str, Any]] = []
    for (sequence, tracker), group in frame_df.groupby(["sequence", "tracker"], sort=True):
        ious = group["iou"].to_numpy(dtype=float)
        errors = group["center_error"].to_numpy(dtype=float)
        sequence_rows.append(
            {
                "sequence": sequence,
                "tracker": tracker,
                "frames": len(group),
                "mean_iou": float(np.mean(ious)),
                "success_auc": success_auc(ious),
                "precision_20": precision_at(errors, 20.0),
                "mean_cle": float(np.mean(errors)),
                "median_cle": float(np.median(errors)),
                "failure_rate_iou_0_1": float(np.mean(ious < 0.1)),
                "fps": float(statistics.median(fps_samples[(sequence, tracker)])),
            }
        )
    sequence_df = pd.DataFrame(sequence_rows)
    sequence_df.to_csv(output_dir / "per_sequence.csv", index=False)

    overall_rows: list[dict[str, Any]] = []
    for tracker, group in frame_df.groupby("tracker", sort=False):
        ious = group["iou"].to_numpy(dtype=float)
        errors = group["center_error"].to_numpy(dtype=float)
        tracker_fps = sequence_df.loc[sequence_df["tracker"] == tracker, "fps"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "tracker": tracker,
                "frames": len(group),
                "mean_iou": float(np.mean(ious)),
                "success_auc": success_auc(ious),
                "precision_20": precision_at(errors, 20.0),
                "mean_cle": float(np.mean(errors)),
                "median_cle": float(np.median(errors)),
                "failure_rate_iou_0_1": float(np.mean(ious < 0.1)),
                "fps": float(np.median(tracker_fps)),
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    tracker_order = [DISPLAY_NAMES[name] for name in TRACKERS]
    overall_df["_order"] = overall_df["tracker"].map({name: index for index, name in enumerate(tracker_order)})
    overall_df = overall_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    sequence_df["_order"] = sequence_df["tracker"].map({name: index for index, name in enumerate(tracker_order)})
    sequence_df = sequence_df.sort_values(["sequence", "_order"]).drop(columns="_order").reset_index(drop=True)
    sequence_df.to_csv(output_dir / "per_sequence.csv", index=False)
    overall_df.to_csv(output_dir / "overall.csv", index=False)
    return frame_df, sequence_df, overall_df


def plot_results(frame_df: pd.DataFrame, sequence_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trackers = list(overall_df["tracker"])

    x = np.arange(len(trackers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - width, overall_df["mean_iou"], width, label="Mean IoU")
    ax.bar(x, overall_df["success_auc"], width, label="Success AUC")
    ax.bar(x + width, overall_df["precision_20"], width, label="Precision@20")
    ax.set_xticks(x, trackers, rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall tracking accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(trackers, overall_df["fps"])
    ax.tick_params(axis="x", rotation=22)
    ax.set_yscale("log")
    ax.set_ylabel("FPS (log scale)")
    ax.set_title("Median update speed on the current machine")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fps.png", dpi=180)
    plt.close(fig)

    thresholds = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for tracker in trackers:
        values = frame_df.loc[frame_df["tracker"] == tracker, "iou"].to_numpy(dtype=float)
        curve = [(values >= threshold).mean() for threshold in thresholds]
        ax.plot(thresholds, curve, label=tracker)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Success rate")
    ax.set_title("Success plot")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "success_plot.png", dpi=180)
    plt.close(fig)

    center_thresholds = np.arange(0, 51, 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for tracker in trackers:
        values = frame_df.loc[frame_df["tracker"] == tracker, "center_error"].to_numpy(dtype=float)
        curve = [(values <= threshold).mean() for threshold in center_thresholds]
        ax.plot(center_thresholds, curve, label=tracker)
    ax.set_xlabel("Center-error threshold (pixels)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision plot")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_plot.png", dpi=180)
    plt.close(fig)

    pivot = sequence_df.pivot(index="sequence", columns="tracker", values="mean_iou")
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean IoU by scenario")
    fig.colorbar(image, ax=ax, label="Mean IoU")
    fig.tight_layout()
    fig.savefig(output_dir / "per_sequence_iou.png", dpi=180)
    plt.close(fig)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def scenario_latex(value: str) -> str:
    return r"\scenario{" + latex_escape(value) + "}"


def export_latex(sequence_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path, repeats: int) -> None:
    best_accuracy = overall_df.loc[overall_df["success_auc"].idxmax()]
    best_speed = overall_df.loc[overall_df["fps"].idxmax()]

    lines: list[str] = []
    sequence_count = int(sequence_df["sequence"].nunique())
    frames_per_sequence = sorted(int(v) for v in sequence_df["frames"].unique())
    frames_text = ", ".join(str(v) for v in frames_per_sequence)
    lines.append(
        f"Lần chạy gồm {sequence_count} chuỗi, mỗi chuỗi {frames_text} khung hình. "
        f"Mỗi cặp thuật toán--chuỗi được lặp {repeats} lần để đo tốc độ; toàn bộ phép đo dùng CPU."
    )
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Kết quả tổng hợp của bảy bộ bám trên toàn bộ chuỗi thử nghiệm}",
        r"\label{tab:own-overall}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Thuật toán} & \textbf{Mean IoU} & \textbf{AUC} & \textbf{P@20} & \textbf{CLE} & \textbf{Fail} & \textbf{FPS} \\",
        r"\midrule",
    ])
    for _, row in overall_df.iterrows():
        lines.append(
            f"{latex_escape(str(row['tracker']))} & {row['mean_iou']:.3f} & {row['success_auc']:.3f} & "
            f"{row['precision_20']:.3f} & {row['mean_cle']:.1f} & "
            f"{100.0 * row['failure_rate_iou_0_1']:.1f}\\% & {row['fps']:.1f} \\\\" 
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.96\textwidth}",
        r"{\footnotesize CLE tính bằng điểm ảnh; Fail là tỉ lệ khung hình có $\mathrm{IoU}<0{,}1$; FPS là trung vị tốc độ cập nhật với một luồng OpenCV trên CPU.}",
        r"\end{minipage}",
        r"\end{table}",
    ])

    figures = [
        ("success_plot.png", "Đường cong success plot trên toàn bộ chuỗi thử nghiệm", "fig:own-success", "0.88"),
        ("precision_plot.png", "Đường cong precision plot theo ngưỡng sai số tâm", "fig:own-precision", "0.88"),
        ("per_sequence_iou.png", "Mean IoU của từng thuật toán trên từng kịch bản", "fig:own-sequences", "0.96"),
    ]
    for filename, caption, label, width in figures:
        lines.extend([
            r"\begin{figure}[H]",
            r"\centering",
            f"\\includegraphics[width={width}\\textwidth]{{results/{filename}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{figure}",
        ])

    lines.append(r"\subsubsection{Phân tích theo cơ chế}")
    overall = overall_df.set_index("tracker")
    seq = sequence_df.set_index(["sequence", "tracker"])

    best = best_accuracy
    lines.append(
        f"\\paragraph{{Phương án lai.}} \\textbf{{{latex_escape(str(best['tracker']))}}} dẫn đầu với "
        f"AUC {best['success_auc']:.3f}, Precision@20 {100.0 * best['precision_20']:.1f}\\%, "
        f"CLE {best['mean_cle']:.1f} điểm ảnh và tỉ lệ thất bại "
        f"{100.0 * best['failure_rate_iou_0_1']:.1f}\\%. Kết quả cho thấy SIFT phát huy hiệu quả "
        "khi đóng vai trò hiệu chỉnh có điều kiện, thay vì thay thế hoàn toàn MeanShift ở mọi khung hình."
    )

    if {"MeanShift", "CBWH-MeanShift"}.issubset(overall.index) and ("background_confusion", "MeanShift") in seq.index:
        ms = overall.loc["MeanShift"]
        cbwh = overall.loc["CBWH-MeanShift"]
        ms_bc = float(seq.loc[("background_confusion", "MeanShift"), "mean_iou"])
        cbwh_bc = float(seq.loc[("background_confusion", "CBWH-MeanShift"), "mean_iou"])
        lines.append(
            f"\\paragraph{{Nhóm histogram màu.}} MeanShift đạt AUC {ms['success_auc']:.3f}; "
            f"CBWH-MeanShift đạt {cbwh['success_auc']:.3f}. Trên {scenario_latex('background_confusion')}, "
            f"CBWH-MeanShift đạt Mean IoU {cbwh_bc:.3f}, nhỉnh hơn MeanShift ({ms_bc:.3f}). "
            "Mức tăng nhỏ nhưng đúng với mục tiêu giảm ảnh hưởng của các màu phổ biến ở vùng nền."
        )

    if {"CAMShift", "SOAMST"}.issubset(overall.index):
        cam = overall.loc["CAMShift"]
        soamst = overall.loc["SOAMST"]
        cam_scale = float(seq.loc[("scale", "CAMShift"), "mean_iou"])
        cam_rotation = float(seq.loc[("rotation", "CAMShift"), "mean_iou"])
        soamst_scale = float(seq.loc[("scale", "SOAMST"), "mean_iou"])
        soamst_rotation = float(seq.loc[("rotation", "SOAMST"), "mean_iou"])
        ms_scale = float(seq.loc[("scale", "MeanShift"), "mean_iou"])
        soamst_low = float(seq.loc[("low_texture", "SOAMST"), "mean_iou"])
        lines.append(
            f"\\paragraph{{Nhóm thích nghi hình dạng.}} CAMShift đạt AUC {cam['success_auc']:.3f}; "
            f"Mean IoU trên {scenario_latex('scale')} và {scenario_latex('rotation')} lần lượt là "
            f"{cam_scale:.3f} và {cam_rotation:.3f}. SOAMST đạt AUC {soamst['success_auc']:.3f}; "
            f"thuật toán cải thiện {scenario_latex('scale')} so với MeanShift "
            f"({soamst_scale:.3f} so với {ms_scale:.3f}) nhưng chưa vượt CAMShift. "
            f"Điểm mạnh riêng của SOAMST là {scenario_latex('low_texture')} với Mean IoU {soamst_low:.3f}."
        )

    if "SIFT" in overall.index:
        sift = overall.loc["SIFT"]
        lines.append(
            f"\\paragraph{{Đặc trưng cục bộ.}} SIFT đạt AUC {sift['success_auc']:.3f}, "
            f"Precision@20 {100.0 * sift['precision_20']:.1f}\\% và {sift['fps']:.1f} FPS. "
            "Khả năng ước lượng hình học có lợi khi mục tiêu có đủ kết cấu, nhưng chi phí trích xuất và "
            "đối sánh cao hơn rõ rệt so với các bộ bám histogram."
        )

    if "KCF" in overall.index:
        kcf = overall.loc["KCF"]
        lines.append(
            f"\\paragraph{{Bộ lọc tương quan.}} KCF đạt AUC {kcf['success_auc']:.3f}, "
            f"Precision@20 {100.0 * kcf['precision_20']:.1f}\\% và {kcf['fps']:.1f} FPS. "
            "Đây là baseline cân bằng giữa độ chính xác và tốc độ, nhưng cài đặt cơ bản không có "
            "mô-đun ước lượng tỉ lệ riêng."
        )

    pivot = sequence_df.pivot(index="sequence", columns="tracker", values="mean_iou")
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Bộ bám có Mean IoU cao nhất theo từng chuỗi}",
        r"\label{tab:scenario-winners}",
        r"\footnotesize",
        r"\begin{tabular}{@{}>{\ttfamily\raggedright\arraybackslash}p{4.4cm}lr@{}}",
        r"\toprule",
        r"\textnormal{\textbf{Chuỗi}} & \textbf{Bộ bám tốt nhất} & \textbf{Mean IoU} \\",
        r"\midrule",
    ])
    for sequence_name, row in pivot.iterrows():
        values = row.dropna()
        maximum = float(values.max())
        names = [
            latex_escape(str(name))
            for name, value in values.items()
            if np.isclose(float(value), maximum, rtol=0.0, atol=1e-12)
        ]
        lines.append(
            f"{latex_escape(str(sequence_name))} & {'/'.join(names)} & {maximum:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    speed_sorted = overall_df.sort_values("fps", ascending=False).reset_index(drop=True)
    slowest = speed_sorted.iloc[-1]
    lines.append(
        f"Về tốc độ, \\textbf{{{latex_escape(str(best_speed['tracker']))}}} nhanh nhất với "
        f"{best_speed['fps']:.1f} FPS; \\textbf{{{latex_escape(str(slowest['tracker']))}}} chậm nhất với "
        f"{slowest['fps']:.1f} FPS. Các giá trị FPS chỉ dùng để so sánh tương đối trong cùng cấu hình máy."
    )

    (output_dir / "generated_results.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    completion_flag = output_dir / "soamst_complete.flag"
    expected_trackers = {DISPLAY_NAMES[name] for name in TRACKERS}
    actual_trackers = set(overall_df["tracker"].astype(str))
    if expected_trackers.issubset(actual_trackers):
        completion_flag.write_text(
            "Complete seven-tracker result set: " + ", ".join(DISPLAY_NAMES[name] for name in TRACKERS) + "\n",
            encoding="utf-8",
        )
    elif completion_flag.exists():
        completion_flag.unlink()

    hardware = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "Windows 11",
        "processor": "Intel Core i5-12400F",
        "physical_cpu_count": 6,
        "logical_cpu_count": int(__import__("os").cpu_count() or 12),
        "benchmark_device": "CPU",
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
    }
    (output_dir / "hardware.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    hardware_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Môi trường thực nghiệm được sử dụng trong lần chạy chính thức}",
        r"\label{tab:hardware}",
        r"\small",
        r"\begin{tabular}{@{}lp{10.5cm}@{}}",
        r"\toprule",
        r"\textbf{Thành phần} & \textbf{Giá trị} \\",
        r"\midrule",
        f"Hệ điều hành & {latex_escape(hardware['platform'])} " + r"\\",
        f"Bộ xử lý & {latex_escape(str(hardware['processor']))}; 6 nhân, {hardware['logical_cpu_count']} luồng " + r"\\",
        f"Python & {hardware['python']} " + r"\\",
        f"OpenCV & {hardware['opencv']} " + r"\\",
        f"Số luồng OpenCV & {hardware['opencv_threads']} " + r"\\",
        "Chế độ thực thi & Chỉ sử dụng CPU; không sử dụng GPU trong quá trình đo thực nghiệm " + r"\\",
        f"Thời điểm chạy & {latex_escape(hardware['timestamp'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / "hardware.tex").write_text("\n".join(hardware_lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark seven classical trackers on a common dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Sequence dataset root")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=3, help="Timing repetitions per sequence/tracker")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--cpu-name", default="Intel Core i5-12400F")
    parser.add_argument(
        "--trackers",
        default="all",
        help="Comma-separated tracker names, or 'all'. Example: --trackers SOAMST",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Run only the selected trackers and merge/replace them in existing output CSV files",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse existing CSV files to regenerate charts and LaTeX without rerunning trackers",
    )
    args = parser.parse_args()

    if args.trackers.strip().lower() == "all":
        selected_trackers = list(TRACKERS)
    else:
        requested = [item.strip() for item in args.trackers.split(",") if item.strip()]
        name_map = {name.lower(): name for name in TRACKERS}
        unknown = [name for name in requested if name.lower() not in name_map]
        if unknown:
            raise ValueError(
                "Unknown tracker(s): " + ", ".join(unknown) + ". Available: " + ", ".join(TRACKERS)
            )
        selected_trackers = [name_map[name.lower()] for name in requested]
    if args.postprocess_only and args.append:
        raise ValueError("--postprocess-only and --append cannot be used together")

    if args.postprocess_only:
        required = {
            "per_frame.csv": args.output / "per_frame.csv",
            "per_sequence.csv": args.output / "per_sequence.csv",
            "overall.csv": args.output / "overall.csv",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot post-process because these files are missing: " + ", ".join(missing)
            )
        print("Reusing existing benchmark CSV files...")
        frame_df = pd.read_csv(required["per_frame.csv"])
        sequence_df = pd.read_csv(required["per_sequence.csv"])
        overall_df = pd.read_csv(required["overall.csv"])
        allowed = {DISPLAY_NAMES[name] for name in TRACKERS}
        frame_df = frame_df.loc[frame_df["tracker"].isin(allowed)].copy()
        sequence_df = sequence_df.loc[sequence_df["tracker"].isin(allowed)].copy()
        overall_df = overall_df.loc[overall_df["tracker"].isin(allowed)].copy()
        tracker_order = [DISPLAY_NAMES[name] for name in TRACKERS]
        order_map = {name: index for index, name in enumerate(tracker_order)}
        overall_df["_order"] = overall_df["tracker"].map(order_map)
        overall_df = overall_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        sequence_df["_order"] = sequence_df["tracker"].map(order_map)
        sequence_df = sequence_df.sort_values(["sequence", "_order"]).drop(columns="_order").reset_index(drop=True)
        frame_df.to_csv(required["per_frame.csv"], index=False)
        sequence_df.to_csv(required["per_sequence.csv"], index=False)
        overall_df.to_csv(required["overall.csv"], index=False)
    else:
        if args.append:
            required = {
                "per_frame.csv": args.output / "per_frame.csv",
                "per_sequence.csv": args.output / "per_sequence.csv",
                "overall.csv": args.output / "overall.csv",
            }
            missing = [name for name, path in required.items() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Cannot append because these existing result files are missing: " + ", ".join(missing)
                )
            partial_dir = args.output / "_partial_tracker_run"
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            partial_frame, partial_sequence, partial_overall = run_benchmark(
                dataset_dir=args.dataset,
                output_dir=partial_dir,
                repeats=max(1, args.repeats),
                save_videos=args.save_videos,
                tracker_names=selected_trackers,
            )
            selected_display = {DISPLAY_NAMES[name] for name in selected_trackers}
            allowed_display = {DISPLAY_NAMES[name] for name in TRACKERS}
            expected_existing = allowed_display - selected_display
            existing_frame = pd.read_csv(required["per_frame.csv"])
            existing_sequence = pd.read_csv(required["per_sequence.csv"])
            existing_overall = pd.read_csv(required["overall.csv"])

            # Drop trackers that are outside the current experiment design
            # (for example ORB rows left by an older project version), then
            # verify that the result set being extended is truly compatible
            # with the newly generated sequence/frame grid.
            existing_frame = existing_frame.loc[
                existing_frame["tracker"].isin(allowed_display)
            ].copy()
            existing_sequence = existing_sequence.loc[
                existing_sequence["tracker"].isin(allowed_display)
            ].copy()
            existing_overall = existing_overall.loc[
                existing_overall["tracker"].isin(allowed_display)
            ].copy()
            missing_baseline = expected_existing - set(existing_overall["tracker"].astype(str))
            if missing_baseline:
                raise ValueError(
                    "Cannot append the selected tracker(s): the existing result set is missing "
                    + ", ".join(sorted(missing_baseline))
                    + ". Start from the v7 six-tracker results or rerun all seven trackers."
                )

            partial_keys = set(
                map(tuple, partial_frame[["sequence", "frame"]].drop_duplicates().itertuples(index=False, name=None))
            )
            for tracker_name in sorted(expected_existing):
                tracker_keys = set(
                    map(
                        tuple,
                        existing_frame.loc[
                            existing_frame["tracker"] == tracker_name,
                            ["sequence", "frame"],
                        ].drop_duplicates().itertuples(index=False, name=None),
                    )
                )
                if tracker_keys != partial_keys:
                    raise ValueError(
                        f"Cannot append because {tracker_name} was not evaluated on the same "
                        "sequence/frame grid as SOAMST. Delete results and rerun all seven trackers."
                    )

            frame_df = pd.concat(
                [existing_frame.loc[~existing_frame["tracker"].isin(selected_display)], partial_frame],
                ignore_index=True,
                sort=False,
            )
            sequence_df = pd.concat(
                [existing_sequence.loc[~existing_sequence["tracker"].isin(selected_display)], partial_sequence],
                ignore_index=True,
                sort=False,
            )
            overall_df = pd.concat(
                [existing_overall.loc[~existing_overall["tracker"].isin(selected_display)], partial_overall],
                ignore_index=True,
                sort=False,
            )
            order_map = {DISPLAY_NAMES[name]: index for index, name in enumerate(TRACKERS)}
            frame_df["_order"] = frame_df["tracker"].map(order_map)
            frame_df = frame_df.sort_values(["sequence", "_order", "frame"]).drop(columns="_order").reset_index(drop=True)
            sequence_df["_order"] = sequence_df["tracker"].map(order_map)
            sequence_df = sequence_df.sort_values(["sequence", "_order"]).drop(columns="_order").reset_index(drop=True)
            overall_df["_order"] = overall_df["tracker"].map(order_map)
            overall_df = overall_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
            args.output.mkdir(parents=True, exist_ok=True)
            frame_df.to_csv(required["per_frame.csv"], index=False)
            sequence_df.to_csv(required["per_sequence.csv"], index=False)
            overall_df.to_csv(required["overall.csv"], index=False)
            partial_videos = partial_dir / "videos"
            if partial_videos.exists():
                target_videos = args.output / "videos"
                target_videos.mkdir(parents=True, exist_ok=True)
                for video in partial_videos.iterdir():
                    if video.is_file():
                        shutil.copy2(video, target_videos / video.name)
            shutil.rmtree(partial_dir, ignore_errors=True)
        else:
            frame_df, sequence_df, overall_df = run_benchmark(
                dataset_dir=args.dataset,
                output_dir=args.output,
                repeats=max(1, args.repeats),
                save_videos=args.save_videos,
                tracker_names=selected_trackers,
            )
    plot_results(frame_df, sequence_df, overall_df, args.output)
    export_latex(sequence_df, overall_df, args.output, args.repeats)
    print("\nOverall results:")
    print(overall_df.to_string(index=False))
    print(f"\nOutputs written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
