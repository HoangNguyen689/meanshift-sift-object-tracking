from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


FRAME_SIZE = (480, 270)  # width, height
NUM_FRAMES = 100
OBJECT_SIZE = (104, 78)  # width, height


def make_background(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    gradient_x = np.linspace(25, 95, width, dtype=np.float32)
    gradient_y = np.linspace(0, 45, height, dtype=np.float32)[:, None]
    base = gradient_y + gradient_x[None, :]
    background[..., 0] = np.clip(base + 20, 0, 255)
    background[..., 1] = np.clip(base * 0.85 + 30, 0, 255)
    background[..., 2] = np.clip(base * 0.65 + 35, 0, 255)

    for _ in range(45):
        x1 = int(rng.integers(0, width - 20))
        y1 = int(rng.integers(0, height - 20))
        x2 = min(width - 1, x1 + int(rng.integers(8, 70)))
        y2 = min(height - 1, y1 + int(rng.integers(8, 55)))
        color = tuple(int(v) for v in rng.integers(15, 135, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(background, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.circle(background, (x1, y1), max(4, (x2 - x1) // 2), color, -1)
    noise = rng.normal(0, 5, background.shape).astype(np.int16)
    return np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_object(low_texture: bool = False) -> tuple[np.ndarray, np.ndarray]:
    width, height = OBJECT_SIZE
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (width - 3, height - 3), 255, -1)

    if low_texture:
        image[:] = (35, 45, 225)
        cv2.rectangle(image, (2, 2), (width - 3, height - 3), (30, 35, 210), 3)
        return image, mask

    image[:] = (38, 52, 205)
    block = 14
    palette = [(40, 55, 220), (220, 75, 45), (45, 205, 225), (210, 210, 40)]
    for row, y in enumerate(range(0, height, block)):
        for col, x in enumerate(range(0, width, block)):
            color = palette[(row + col) % len(palette)]
            cv2.rectangle(image, (x, y), (min(x + block - 1, width - 1), min(y + block - 1, height - 1)), color, -1)
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), (245, 245, 245), 3)
    cv2.line(image, (8, 10), (width - 12, height - 12), (10, 10, 10), 3)
    cv2.line(image, (width - 12, 10), (8, height - 12), (255, 255, 255), 2)
    cv2.circle(image, (26, 24), 12, (20, 20, 20), 3)
    cv2.circle(image, (width - 25, height - 22), 10, (250, 250, 250), 3)
    cv2.putText(image, "SIFT", (19, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (5, 5, 5), 2, cv2.LINE_AA)
    return image, mask


def transform_object(
    object_image: np.ndarray,
    object_mask: np.ndarray,
    center: tuple[float, float],
    scale: float,
    angle_deg: float,
    frame_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    obj_h, obj_w = object_image.shape[:2]
    matrix = cv2.getRotationMatrix2D((obj_w / 2.0, obj_h / 2.0), angle_deg, scale)
    matrix[0, 2] += center[0] - obj_w / 2.0
    matrix[1, 2] += center[1] - obj_h / 2.0

    frame_h, frame_w = frame_shape[:2]
    warped_image = cv2.warpAffine(
        object_image,
        matrix,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        object_mask,
        matrix,
        (frame_w, frame_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0:
        bbox = (0.0, 0.0, 1.0, 1.0)
    else:
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
        bbox = (x1, y1, x2 - x1, y2 - y1)
    return warped_image, warped_mask, bbox


def alpha_compose(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = background.copy()
    selector = mask > 0
    result[selector] = foreground[selector]
    return result


def scenario_state(name: str, index: int, total: int) -> dict[str, float | tuple[float, float]]:
    width, height = FRAME_SIZE
    t = index / max(total - 1, 1)
    center_x = 80 + (width - 160) * t
    center_y = height / 2 + 55 * math.sin(2 * math.pi * t)
    scale = 1.0
    angle = 0.0
    illumination = 1.0

    if name == "translation":
        center_y = height / 2 + 35 * math.sin(4 * math.pi * t)
    elif name == "scale":
        scale = 0.62 + 0.88 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
        center_y = height / 2
    elif name == "rotation":
        angle = -55 + 110 * t
        center_y = height / 2
    elif name == "occlusion":
        center_y = height / 2 + 30 * math.sin(2 * math.pi * t)
    elif name == "background_confusion":
        center_y = height / 2 + 25 * math.sin(2 * math.pi * t)
    elif name == "fast_motion":
        segment = (index // 7) % 4
        targets = [(82, 68), (398, 78), (90, 205), (390, 198)]
        center_x, center_y = targets[segment]
    elif name == "illumination":
        illumination = 0.38 + 0.95 * (0.5 + 0.5 * math.sin(2 * math.pi * t))
        center_y = height / 2
    elif name == "low_texture":
        center_y = height / 2 + 35 * math.sin(4 * math.pi * t)
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return {
        "center": (float(center_x), float(center_y)),
        "scale": float(scale),
        "angle": float(angle),
        "illumination": float(illumination),
    }


def generate_sequence(root: Path, name: str, seed: int, num_frames: int) -> None:
    sequence_dir = root / name
    image_dir = sequence_dir / "img"
    image_dir.mkdir(parents=True, exist_ok=True)
    background = make_background(*FRAME_SIZE, seed=seed)
    low_texture = name == "low_texture"
    object_image, object_mask = make_object(low_texture=low_texture)
    gt_rows: list[str] = []

    for index in range(num_frames):
        state = scenario_state(name, index, num_frames)
        frame = background.copy()

        if name == "background_confusion":
            distractor_x = int(FRAME_SIZE[0] - 85 - 285 * index / max(num_frames - 1, 1))
            distractor_y = 80 + int(120 * math.sin(2 * math.pi * index / max(num_frames - 1, 1)))
            cv2.rectangle(frame, (distractor_x, distractor_y), (distractor_x + 105, distractor_y + 78), (35, 45, 225), -1)
            cv2.rectangle(frame, (distractor_x + 8, distractor_y + 8), (distractor_x + 97, distractor_y + 70), (45, 55, 215), 3)

        transformed, mask, bbox = transform_object(
            object_image,
            object_mask,
            center=state["center"],
            scale=float(state["scale"]),
            angle_deg=float(state["angle"]),
            frame_shape=frame.shape,
        )
        illumination = float(state["illumination"])
        if illumination != 1.0:
            transformed = np.clip(transformed.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
        frame = alpha_compose(frame, transformed, mask)

        if name == "occlusion" and int(num_frames * 0.38) <= index <= int(num_frames * 0.62):
            x, y, w, h = bbox
            occ_x = int(x + w * 0.12)
            occ_y = int(y - h * 0.1)
            occ_w = int(w * 0.76)
            occ_h = int(h * 1.2)
            cv2.rectangle(frame, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), (72, 92, 78), -1)
            cv2.line(frame, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), (105, 125, 110), 5)

        cv2.imwrite(str(image_dir / f"{index + 1:04d}.png"), frame)
        gt_rows.append(",".join(f"{value:.3f}" for value in bbox))

    (sequence_dir / "groundtruth_rect.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
    metadata = {
        "name": name,
        "frames": num_frames,
        "frame_size": {"width": FRAME_SIZE[0], "height": FRAME_SIZE[1]},
        "initial_bbox": [float(v) for v in gt_rows[0].split(",")],
        "description": {
            "translation": "Tịnh tiến trơn, tỉ lệ và góc gần như không đổi.",
            "scale": "Tỉ lệ thay đổi liên tục.",
            "rotation": "Đối tượng xoay từ -55 đến 55 độ.",
            "occlusion": "Che khuất phần lớn đối tượng ở giữa chuỗi.",
            "background_confusion": "Có vật gây nhiễu màu gần giống đối tượng.",
            "fast_motion": "Dịch chuyển lớn giữa các nhóm khung hình.",
            "illumination": "Cường độ sáng của đối tượng thay đổi mạnh.",
            "low_texture": "Đối tượng gần như không có điểm đặc trưng cục bộ.",
        }[name],
    }
    (sequence_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_dataset(output: Path, num_frames: int = NUM_FRAMES) -> None:
    scenarios = [
        "translation",
        "scale",
        "rotation",
        "occlusion",
        "background_confusion",
        "fast_motion",
        "illumination",
        "low_texture",
    ]
    output.mkdir(parents=True, exist_ok=True)
    for offset, scenario in enumerate(scenarios):
        generate_sequence(output, scenario, seed=100 + offset, num_frames=num_frames)
    manifest = {"format": "sequence-folder", "sequences": scenarios, "frames_per_sequence": num_frames}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a controlled object-tracking benchmark.")
    parser.add_argument("--output", type=Path, default=Path("experiment/data/synthetic"))
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()
    generate_dataset(args.output, num_frames=args.frames)
    print(f"Generated synthetic dataset at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
