"""阶段 A：用 locate-anything 逐帧检测目标物/框的中心，存成小坐标文件。

对每条 SYNC mcap，取"会进数据集的抽帧"(与阶段 B 相同的 decimate)，在头相机原图
(1280x960)上分别 ground 出 target 和 bin 各一个实例，取面积最大框的中心，按 head-frame
时间戳写 detections/<episode>.parquet（漏检记 NaN）。阶段 B 据此画点并对漏检做 ZOH。

只跑抽帧(约 14k)、坐标只有 KB，检测是最慢的 GPU 环节，一次跑完可反复复用。

用法：
    uv run python tools/detect_markers.py \\
        --data-dir /home/galbot/1105_1696 \\
        --out-dir /home/galbot/moon-galbot/detections \\
        --target-label "cola bottle" --bin-label "basket"
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # galbot_mcap
from galbot_mcap import decimate_indices, read_episode  # noqa: E402

SOURCE_FPS = 30.0
XY = tuple[float, float] | None


def _largest_box_center(boxes: list[dict]) -> XY:
    """从 parse_boxes 的框列表里取面积最大框的中心；空列表返回 None。"""
    if not boxes:
        return None
    b = max(boxes, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def _detect_center(worker, parse_boxes, img: Image.Image, label: str) -> XY:
    """ground 单个 label，返回最大框中心(像素，基于 img 尺寸)或 None。"""
    ans = worker.ground_single(img, label)["answer"]
    return _largest_box_center(parse_boxes(ans, img.width, img.height))


def run(
    data_dir: Path,
    out_dir: Path,
    target_label: str,
    bin_label: str,
    fps: float,
    model: str,
    la_repo: Path,
    max_episodes: int | None,
) -> None:
    sys.path.insert(0, str(la_repo))  # locateanything_worker 在 repo 根，不在安装包里
    from locateanything_worker import LocateAnythingWorker

    worker = LocateAnythingWorker(model)
    parse_boxes = LocateAnythingWorker.parse_boxes

    episodes = sorted(data_dir.glob("*.SYNC.mcap"))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"no *.SYNC.mcap under {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_path in episodes:
        name = ep_path.name.replace(".SYNC.mcap", "")
        streams = read_episode(ep_path)
        sel = decimate_indices(len(streams.head_ts), SOURCE_FPS, fps)

        rows = []
        miss_t = miss_b = 0
        for idx in sel:
            img = Image.open(io.BytesIO(streams.head_jpeg[idx])).convert("RGB")
            t = _detect_center(worker, parse_boxes, img, target_label)
            b = _detect_center(worker, parse_boxes, img, bin_label)
            miss_t += t is None
            miss_b += b is None
            rows.append(
                {
                    "head_ts": int(streams.head_ts[idx]),
                    "target_x": np.nan if t is None else t[0],
                    "target_y": np.nan if t is None else t[1],
                    "bin_x": np.nan if b is None else b[0],
                    "bin_y": np.nan if b is None else b[1],
                }
            )
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(
            f"{name}: {len(sel)} frames, "
            f"target miss {miss_t}/{len(sel)}, bin miss {miss_b}/{len(sel)}"
        )

    print(f"\ndone: {len(episodes)} episodes -> {out_dir}")


def _selfcheck() -> None:
    # 面积最大框中心（不依赖模型）
    boxes = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 100, "y1": 100, "x2": 200, "y2": 300},  # 更大
    ]
    assert _largest_box_center(boxes) == (150.0, 200.0), _largest_box_center(boxes)
    assert _largest_box_center([]) is None
    print("detect_markers selfcheck OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--target-label")
    p.add_argument("--bin-label")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--model", default="nvidia/LocateAnything-3B")
    p.add_argument("--la-repo", type=Path, default=Path("/home/galbot/eagle/Embodied"))
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--selfcheck", action="store_true", help="只跑离线自检，不加载模型")
    args = p.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    missing = [n for n in ("data_dir", "out_dir", "target_label", "bin_label")
               if getattr(args, n) is None]
    if missing:
        p.error("missing required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_label=args.target_label,
        bin_label=args.bin_label,
        fps=args.fps,
        model=args.model,
        la_repo=args.la_repo,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
