"""阶段 B：把 SYNC MCAP + 阶段 A 的检测坐标转成 lerobot 原生 ACT 数据集。

- observation.images.head : 头相机 RGB，画上 target/bin 中心点(来自 detections)，resize 640x480。
- observation.images.wrist: 右腕相机 RGB，原始 640x360（保留抓取时机线索，不画标记）。
- observation.state : 8 维 = 7 右臂关节(rad) + 夹爪开度[0,1]。
- action           : 8 维 = 遥操作关节指令的零阶保持，同布局。
- task             : 固定语言 prompt。

标记：按 head-frame 时间戳从 detections/<episode>.parquet 取中心，漏检(NaN)用 ZOH 沿用
上一次成功检测（复现部署低频刷新分布）。检测坐标基于头相机原始分辨率，先在原图画点再 resize。

依赖：galbot_mcap（读取/对齐/组装，复用）、markers（画点/ZOH）。

用法：
    uv run python tools/convert_mcap_to_lerobot.py \\
        --data-dir /home/galbot/1105_1696 \\
        --detections-dir /home/galbot/moon-galbot/detections \\
        --output-root $HF_LEROBOT_HOME --repo-id galbot_g1_marked --fps 15 --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from galbot_mcap import (  # noqa: E402
    ARM_JOINT_NAMES,
    EpisodeStreams,
    assemble_state_action,
    decimate_indices,
    nearest_indices,
    read_episode,
    zoh_indices,
)
from markers import draw_markers, zoh_fill  # noqa: E402

SOURCE_FPS = 30.0
HEAD_SIZE_WH = (640, 480)
STATE_NAMES = [*ARM_JOINT_NAMES, "right_gripper_open"]

MAX_WRIST_MISMATCH_NS = 70_000_000
MAX_SENSOR_MISMATCH_NS = 70_000_000
MAX_FRAME_GAP_FACTOR = 2.5

XY = tuple[float, float] | None


def _decode_rgb(jpeg: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode jpeg frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    import cv2

    if (img.shape[1], img.shape[0]) == size_wh:
        return img
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)


def _load_centers(
    det_path: Path, frame_ts: np.ndarray
) -> tuple[list[XY], list[XY]]:
    """按 frame 时间戳取 target/bin 中心，缺失记 None，再各自 ZOH 填补。"""
    df = pd.read_parquet(det_path).set_index("head_ts")
    targets: list[XY] = []
    bins: list[XY] = []
    for ts in frame_ts:
        if ts in df.index:
            r = df.loc[ts]
            tx, ty, bx, by = r["target_x"], r["target_y"], r["bin_x"], r["bin_y"]
            targets.append(None if np.isnan(tx) else (float(tx), float(ty)))
            bins.append(None if np.isnan(bx) else (float(bx), float(by)))
        else:
            targets.append(None)
            bins.append(None)
    return zoh_fill(targets), zoh_fill(bins)


def _episode_qa(
    name: str, streams: EpisodeStreams, sel: np.ndarray, fps: float
) -> list[str]:
    warnings = []
    frame_ts = streams.head_ts[sel]

    gaps = np.diff(frame_ts)
    max_gap_ns = 1e9 / fps * MAX_FRAME_GAP_FACTOR
    if len(gaps) and gaps.max() > max_gap_ns:
        warnings.append(f"{name}: head-frame gap {gaps.max() / 1e6:.0f} ms")

    wi = nearest_indices(frame_ts, streams.wrist_ts)
    wrist_err = np.abs(streams.wrist_ts[wi] - frame_ts).max()
    if wrist_err > MAX_WRIST_MISMATCH_NS:
        warnings.append(f"{name}: wrist-head mismatch {wrist_err / 1e6:.0f} ms")

    si = nearest_indices(frame_ts, streams.sensor_ts)
    sensor_err = np.abs(streams.sensor_ts[si] - frame_ts).max()
    if sensor_err > MAX_SENSOR_MISMATCH_NS:
        warnings.append(f"{name}: sensor-frame mismatch {sensor_err / 1e6:.0f} ms")
    return warnings


def convert(
    data_dir: Path,
    detections_dir: Path,
    output_root: Path,
    repo_id: str,
    fps: float,
    prompt: str,
    vcodec: str,
    max_episodes: int | None,
    overwrite: bool,
) -> None:
    import functools

    from lerobot.common.datasets import lerobot_dataset as lds
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # 默认 libsvtav1(AV1) 在本机 FFmpeg 4.4 上解不了(cv2/torchcodec 均失败)；
    # 用 h264 保证训练读取与通用工具都能解码。
    lds.encode_video_frames = functools.partial(lds.encode_video_frames, vcodec=vcodec)

    episodes = sorted(data_dir.glob("*.SYNC.mcap"))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"no *.SYNC.mcap files under {data_dir}")

    root = output_root / repo_id
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite")
        shutil.rmtree(root)

    probe = _decode_rgb(read_episode(episodes[0]).wrist_jpeg[0])
    wrist_h, wrist_w = probe.shape[:2]

    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": (HEAD_SIZE_WH[1], HEAD_SIZE_WH[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (wrist_h, wrist_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=root,
        robot_type="galbot_g1",
        features=features,
        image_writer_threads=8,
    )

    all_warnings: list[str] = []
    total_frames = 0
    for ep_path in episodes:
        name = ep_path.name.replace(".SYNC.mcap", "")
        det_path = detections_dir / f"{name}.parquet"
        if not det_path.exists():
            raise FileNotFoundError(f"missing detections for {name}: {det_path}")

        streams = read_episode(ep_path)
        sel = decimate_indices(len(streams.head_ts), SOURCE_FPS, fps)
        frame_ts = streams.head_ts[sel]

        state, actions = assemble_state_action(
            frame_ts,
            streams.sensor_ts,
            streams.sensor_arm,
            streams.sensor_gripper,
            streams.arm_target_ts,
            streams.arm_targets,
            streams.gripper_target_ts,
            streams.gripper_targets,
        )
        wrist_sel = nearest_indices(frame_ts, streams.wrist_ts)
        targets, bins = _load_centers(det_path, frame_ts)

        warnings = _episode_qa(name, streams, sel, fps)
        all_warnings.extend(warnings)

        for i, src_idx in enumerate(sel):
            head = draw_markers(
                _decode_rgb(streams.head_jpeg[src_idx]), targets[i], bins[i]
            )
            dataset.add_frame(
                {
                    "observation.images.head": _resize(head, HEAD_SIZE_WH),
                    "observation.images.wrist": _decode_rgb(
                        streams.wrist_jpeg[wrist_sel[i]]
                    ),
                    "observation.state": state[i],
                    "action": actions[i],
                    "task": prompt,
                }
            )
        dataset.save_episode()

        total_frames += len(sel)
        dur = (frame_ts[-1] - frame_ts[0]) / 1e9
        n_tmiss = sum(t is None for t in targets)
        status = " | ".join(warnings) if warnings else "ok"
        print(
            f"{name}: {len(sel)} frames, {dur:.1f}s, "
            f"target-marker missing(lead) {n_tmiss} [{status}]"
        )

    print(f"\ndone: {len(episodes)} episodes, {total_frames} frames -> {root}")
    if all_warnings:
        print(f"{len(all_warnings)} QA warning(s) above — inspect before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="galbot_g1_marked")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--prompt", default="put the cola bottle into the basket")
    parser.add_argument(
        "--vcodec", default="h264", choices=["h264", "hevc", "libsvtav1"],
        help="视频编码；默认 h264（本机 AV1 解不了）",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert(
        data_dir=args.data_dir,
        detections_dir=args.detections_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        prompt=args.prompt,
        vcodec=args.vcodec,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
