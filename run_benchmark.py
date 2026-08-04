from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from tqdm import tqdm

from metrics import center_error, iou, precision_at, success_auc
from trackers import BBox, DISPLAY_NAMES, TRACKERS, TrackResult, build_tracker, validate_tracker_environment


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
        for codec in ("mp4v", "avc1", "XVID"):
            writer = cv2.VideoWriter(
                str(save_video_path),
                cv2.VideoWriter_fourcc(*codec),
                25.0,
                (width, height),
            )
            if writer.isOpened():
                break
        else:
            print(f"Warning: could not open video writer for {save_video_path}; skipping video export")
            writer = None

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

    for frame_index, (image_path, gt_bbox) in enumerate(
        tqdm(zip(image_paths[1:], ground_truth[1:]), total=len(image_paths) - 1, desc="    Frames", unit="frame", leave=False),
        start=2,
    ):
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

    for sequence_dir in tqdm(sequences, desc="Sequences", unit="seq"):
        image_paths, ground_truth = load_sequence(sequence_dir)
        for tracker_name in tqdm(selected_trackers, desc=f"  {sequence_dir.name}", unit="tracker", leave=False):
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



from reporting import export_latex, plot_results

def run_benchmark_main(
    dataset: Path,
    output: Path,
    repeats: int = 3,
    save_videos: bool = False,
    cpu_name: str = "Intel Core i5-12400F",
    trackers_str: str = "all",
    postprocess_only: bool = False,
    append: bool = False,
) -> None:
    if trackers_str.strip().lower() == "all":
        selected_trackers = list(TRACKERS)
    else:
        requested = [item.strip() for item in trackers_str.split(",") if item.strip()]
        name_map = {name.lower(): name for name in TRACKERS}
        unknown = [name for name in requested if name.lower() not in name_map]
        if unknown:
            raise ValueError(
                "Unknown tracker(s): " + ", ".join(unknown) + ". Available: " + ", ".join(TRACKERS)
            )
        selected_trackers = [name_map[name.lower()] for name in requested]
    if postprocess_only and append:
        raise ValueError("--postprocess-only and --append cannot be used together")

    if postprocess_only:
        required = {
            "per_frame.csv": output / "per_frame.csv",
            "per_sequence.csv": output / "per_sequence.csv",
            "overall.csv": output / "overall.csv",
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
        if append:
            required = {
                "per_frame.csv": output / "per_frame.csv",
                "per_sequence.csv": output / "per_sequence.csv",
                "overall.csv": output / "overall.csv",
            }
            missing = [name for name, path in required.items() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Cannot append because these existing result files are missing: " + ", ".join(missing)
                )
            partial_dir = output / "_partial_tracker_run"
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            partial_frame, partial_sequence, partial_overall = run_benchmark(
                dataset_dir=dataset,
                output_dir=partial_dir,
                repeats=max(1, repeats),
                save_videos=save_videos,
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
            output.mkdir(parents=True, exist_ok=True)
            frame_df.to_csv(required["per_frame.csv"], index=False)
            sequence_df.to_csv(required["per_sequence.csv"], index=False)
            overall_df.to_csv(required["overall.csv"], index=False)
            partial_videos = partial_dir / "videos"
            if partial_videos.exists():
                target_videos = output / "videos"
                target_videos.mkdir(parents=True, exist_ok=True)
                for video in partial_videos.iterdir():
                    if video.is_file():
                        shutil.copy2(video, target_videos / video.name)
            shutil.rmtree(partial_dir, ignore_errors=True)
        else:
            frame_df, sequence_df, overall_df = run_benchmark(
                dataset_dir=dataset,
                output_dir=output,
                repeats=max(1, repeats),
                save_videos=save_videos,
                tracker_names=selected_trackers,
            )
    plot_results(frame_df, sequence_df, overall_df, output)
    export_latex(sequence_df, overall_df, output, repeats, cpu_name)
    print("\nOverall results:")
    print(overall_df.to_string(index=False))
    print(f"\nOutputs written to: {output.resolve()}")


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
    run_benchmark_main(
        dataset=args.dataset,
        output=args.output,
        repeats=args.repeats,
        save_videos=args.save_videos,
        cpu_name=args.cpu_name,
        trackers_str=args.trackers,
        postprocess_only=args.postprocess_only,
        append=args.append,
    )


if __name__ == "__main__":
    main()
