# Classical Object-Tracking Benchmark

This repo compares seven classical object trackers on a controlled synthetic dataset.

## Trackers

| Tracker | Implementation details | Role |
|---|---|---|
| MeanShift | Hue histogram, back-projection, fixed window | Baseline histogram |
| CBWH-MeanShift | Corrects target model with background histogram, then MeanShift | Reduces background clutter |
| CAMShift | Hue histogram + moments, bounded scale/orientation changes | Checks scale/rotation adaptation |
| SOAMST | RGB 16×16×16, `sqrt(q/p)` weight, moments 0/1/2 and Bhattacharyya | Adapts position, scale and orientation |
| SIFT | SIFT keypoints, ratio test and RANSAC partial affine | Local feature baseline |
| MeanShift+SIFT | MeanShift runs continuously, SIFT corrects periodically/by confidence | Hybrid approach |
| KCF | Grayscale KCF, Gaussian kernel and Fourier update | Correlation-filter baseline |

## Requirements

- Python >= 3.9
- OpenCV >= 4.8 (SIFT is included in the main `opencv-python` package since 4.4.2+)
- Other dependencies listed in `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
```

## Running experiments

### Quick sanity check (20 frames, 1 repeat)

```bash
python run_all.py --frames 20 --repeats 1
```

### Full benchmark

```bash
python run_all.py --repeats 3 --save-videos
```

Synthetic data is written to `data/synthetic/` by default. Benchmark results go to `results/`.

### Useful CLI flags

| Flag | Meaning |
|---|---|
| `--dataset <path>` | Change the synthetic-data output directory (default: `data/synthetic`) |
| `--output <path>` | Change the results directory (default: `results`) |
| `--frames <n>` | Number of frames per sequence (default: 100) |
| `--repeats <n>` | Timing repetitions per sequence/tracker (default: 3) |
| `--save-videos` | Export illustration videos for each tracker |
| `--skip-generate` | Skip dataset generation if it already exists |
| `--only-soamst` | Run only SOAMST and merge/replace into existing results |
| `--cpu-name <name>` | Override the CPU label in the report (default: `Intel Core i5-12400F`) |
| `--postprocess-only` | Regenerate charts/LaTeX from existing CSV files only |

### Run a subset of trackers directly

If you want to bypass `run_all.py`:

```bash
python run_benchmark.py --dataset data/synthetic --output results --trackers MeanShift,CAMShift
```

| Flag | Meaning |
|---|---|
| `--dataset <path>` | **Required.** Dataset root directory |
| `--trackers <names>` | Comma-separated tracker names, or `all` (default) |
| `--append` | Run selected trackers and merge/replace them in existing CSV outputs |

## Test sequences

Eight controlled sequences: `translation`, `scale`, `rotation`, `occlusion`, `background_confusion`, `fast_motion`, `illumination`, `low_texture`.

## Outputs

After running, `results/` contains:

| File | Description |
|---|---|
| `overall.csv` | Average accuracy and FPS per tracker |
| `per_sequence.csv` | Accuracy and FPS per tracker per sequence |
| `per_frame.csv` | IoU and center error per frame (for fine-grained analysis) |
| `generated_results.tex` | Summary tables in LaTeX |
| `hardware.json` | Hardware info in JSON |
| `hardware.tex` | Hardware info in LaTeX |
| `success_plot.png` | Success plot (IoU threshold) |
| `precision_plot.png` | Precision plot (pixel threshold) |
| `per_sequence_iou.png` | Average IoU per sequence |
| `overall_accuracy.png` | Overall accuracy comparison |
| `fps.png` | Speed comparison (FPS) |

## Regenerate charts from CSV

```bash
python run_all.py --postprocess-only
```

Or call `run_benchmark.py` directly:

```bash
python run_benchmark.py --dataset data/synthetic --output results --postprocess-only
```
