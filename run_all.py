from __future__ import annotations

import argparse
from pathlib import Path

from run_benchmark import run_benchmark_main
from synthetic_dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data, benchmark trackers and export report assets.")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--cpu-name", default="Intel Core i5-12400F")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--only-soamst",
        action="store_true",
        help="Run only SOAMST and append/replace it in an existing six-tracker result set",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse results CSV files and only regenerate charts/LaTeX",
    )
    args = parser.parse_args()
    if args.postprocess_only and args.only_soamst:
        raise ValueError("--postprocess-only and --only-soamst cannot be used together")

    if args.postprocess_only:
        print("[1/1] Regenerating charts and LaTeX from existing CSV files...")
    else:
        if not args.skip_generate:
            print("[1/2] Generating controlled synthetic sequences...")
            generate_dataset(args.dataset, num_frames=args.frames)
        print("[2/2] Running benchmark...")
    run_benchmark_main(
        dataset=args.dataset,
        output=args.output,
        repeats=args.repeats,
        save_videos=args.save_videos,
        cpu_name=args.cpu_name,
        trackers_str="SOAMST" if args.only_soamst else "all",
        postprocess_only=args.postprocess_only,
        append=args.only_soamst,
    )


if __name__ == "__main__":
    main()
