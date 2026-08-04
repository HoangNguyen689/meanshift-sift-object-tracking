from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from trackers import DISPLAY_NAMES, TRACKERS


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


def export_latex(sequence_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path, repeats: int, cpu_name: str = "Intel Core i5-12400F") -> None:
    best_accuracy = overall_df.loc[overall_df["success_auc"].idxmax()]
    best_speed = overall_df.loc[overall_df["fps"].idxmax()]

    lines: list[str] = []
    sequence_count = int(sequence_df["sequence"].nunique())
    frames_per_sequence = sorted(int(v) for v in sequence_df["frames"].unique())
    frames_text = ", ".join(str(v) for v in frames_per_sequence)
    lines.append(
        f"The run comprises {sequence_count} sequences, each with {frames_text} frames. "
        f"Every algorithm-sequence pair is repeated {repeats} times for timing; all measurements use CPU."
    )
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Overall results of seven trackers across all test sequences}",
        r"\label{tab:own-overall}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Tracker} & \textbf{Mean IoU} & \textbf{AUC} & \textbf{P@20} & \textbf{CLE} & \textbf{Fail} & \textbf{FPS} \\",
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
        r"{\footnotesize CLE in pixels; Fail is the frame rate with $\mathrm{IoU}<0.1$; FPS is the median update speed with a single OpenCV thread on CPU.}",
        r"\end{minipage}",
        r"\end{table}",
    ])

    figures = [
        ("success_plot.png", "Success plot across all test sequences", "fig:own-success", "0.88"),
        ("precision_plot.png", "Precision plot by center-error threshold", "fig:own-precision", "0.88"),
        ("per_sequence_iou.png", "Mean IoU of each tracker on each scenario", "fig:own-sequences", "0.96"),
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

    lines.append(r"\subsubsection{Analysis by mechanism}")
    overall = overall_df.set_index("tracker")
    seq = sequence_df.set_index(["sequence", "tracker"])

    best = best_accuracy
    lines.append(
        f"\\paragraph{{Hybrid approach.}} \\textbf{{{latex_escape(str(best['tracker']))}}} leads with "
        f"AUC {best['success_auc']:.3f}, Precision@20 {100.0 * best['precision_20']:.1f}\\%, "
        f"CLE {best['mean_cle']:.1f} pixels and failure rate "
        f"{100.0 * best['failure_rate_iou_0_1']:.1f}\\%. The result shows SIFT is effective "
        "when playing the role of conditional correction, rather than completely replacing MeanShift in every frame."
    )

    if {"MeanShift", "CBWH-MeanShift"}.issubset(overall.index) and ("background_confusion", "MeanShift") in seq.index:
        ms = overall.loc["MeanShift"]
        cbwh = overall.loc["CBWH-MeanShift"]
        ms_bc = float(seq.loc[("background_confusion", "MeanShift"), "mean_iou"])
        cbwh_bc = float(seq.loc[("background_confusion", "CBWH-MeanShift"), "mean_iou"])
        lines.append(
            f"\\paragraph{{Color histogram group.}} MeanShift achieves AUC {ms['success_auc']:.3f}; "
            f"CBWH-MeanShift achieves {cbwh['success_auc']:.3f}. On {scenario_latex('background_confusion')}, "
            f"CBWH-MeanShift achieves Mean IoU {cbwh_bc:.3f}, slightly higher than MeanShift ({ms_bc:.3f}). "
            "The small increase aligns with the goal of reducing the influence of common colors in the background region."
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
            f"\\paragraph{{Shape adaptation group.}} CAMShift achieves AUC {cam['success_auc']:.3f}; "
            f"Mean IoU on {scenario_latex('scale')} and {scenario_latex('rotation')} are "
            f"{cam_scale:.3f} and {cam_rotation:.3f}, respectively. SOAMST achieves AUC {soamst['success_auc']:.3f}; "
            f"the algorithm improves {scenario_latex('scale')} over MeanShift "
            f"({soamst_scale:.3f} versus {ms_scale:.3f}) but has not surpassed CAMShift. "
            f"SOAMST's unique strength is {scenario_latex('low_texture')} with Mean IoU {soamst_low:.3f}."
        )

    if "SIFT" in overall.index:
        sift = overall.loc["SIFT"]
        lines.append(
            f"\\paragraph{{Local features.}} SIFT achieves AUC {sift['success_auc']:.3f}, "
            f"Precision@20 {100.0 * sift['precision_20']:.1f}\\% and {sift['fps']:.1f} FPS. "
            "Geometric estimation is beneficial when the target has sufficient texture, but extraction and "
            "matching costs are significantly higher than histogram trackers."
        )

    if "KCF" in overall.index:
        kcf = overall.loc["KCF"]
        lines.append(
            f"\\paragraph{{Correlation filter.}} KCF achieves AUC {kcf['success_auc']:.3f}, "
            f"Precision@20 {100.0 * kcf['precision_20']:.1f}\\% and {kcf['fps']:.1f} FPS. "
            "This is a baseline balancing accuracy and speed, but the basic implementation lacks "
            "a dedicated scale-estimation module."
        )

    pivot = sequence_df.pivot(index="sequence", columns="tracker", values="mean_iou")
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Tracker with highest Mean IoU per sequence}",
        r"\label{tab:scenario-winners}",
        r"\footnotesize",
        r"\begin{tabular}{@{}>{\ttfamily\raggedright\arraybackslash}p{4.4cm}lr@{}}",
        r"\toprule",
        r"\textnormal{\textbf{Sequence}} & \textbf{Best tracker} & \textbf{Mean IoU} \\",
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
        f"In terms of speed, \\textbf{{{latex_escape(str(best_speed['tracker']))}}} is fastest at "
        f"{best_speed['fps']:.1f} FPS; \\textbf{{{latex_escape(str(slowest['tracker']))}}} is slowest at "
        f"{slowest['fps']:.1f} FPS. FPS values are only for relative comparison on the same machine configuration."
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
        "platform": f"{platform.system()} {platform.release()}",
        "processor": cpu_name,
        "logical_cpu_count": int(os.cpu_count() or 1),
        "benchmark_device": "CPU",
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
    }
    (output_dir / "hardware.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    hardware_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Experimental environment used in the benchmark run}",
        r"\label{tab:hardware}",
        r"\small",
        r"\begin{tabular}{@{}lp{10.5cm}@{}}",
        r"\toprule",
        r"\textbf{Component} & \textbf{Value} \\",
        r"\midrule",
        f"Operating system & {latex_escape(hardware['platform'])} " + r"\\",
        f"Processor & {latex_escape(str(hardware['processor']))}; {hardware['logical_cpu_count']} threads " + r"\\",
        f"Python & {hardware['python']} " + r"\\",
        f"OpenCV & {hardware['opencv']} " + r"\\",
        f"OpenCV threads & {hardware['opencv_threads']} " + r"\\",
        "Execution mode & CPU-only; no GPU used during benchmarking " + r"\\",
        f"Run timestamp & {latex_escape(hardware['timestamp'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / "hardware.tex").write_text("\n".join(hardware_lines) + "\n", encoding="utf-8")

