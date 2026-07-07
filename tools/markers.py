"""视觉标记：在头相机图像上画目标物/框的中心点，并对漏检做零阶保持(ZOH)。

这份代码训练(阶段 B 转换)和部署共用同一份，保证标记像素级一致——这是此类
方案的头号翻车点。只依赖 numpy + cv2，不碰 locate-anything。

约定：坐标是**头相机原始分辨率(1280x960)**下的 (x, y) 像素；先在原图上画点，
再由调用方 resize 到 640x480。检测缺失(遮挡)用 None 表示。
"""

from __future__ import annotations

import numpy as np

# 训练/部署必须用同一组常量。改这里就等于改标记外观，改完须重跑阶段 B。
TARGET_COLOR = (0, 255, 0)   # 目标物：绿
BIN_COLOR = (0, 0, 255)      # 框：蓝
RADIUS = 14                  # 原分辨率下半径(px)；0.5x resize 后约 7px

XY = tuple[float, float] | None


def draw_markers(
    rgb: np.ndarray,
    target_xy: XY,
    bin_xy: XY,
    *,
    target_color: tuple[int, int, int] = TARGET_COLOR,
    bin_color: tuple[int, int, int] = BIN_COLOR,
    radius: int = RADIUS,
) -> np.ndarray:
    """在 rgb 上画目标/框中心实心圆点，返回新图(不改分辨率)。None 的那个不画。"""
    import cv2

    out = rgb.copy()
    for xy, color in ((target_xy, target_color), (bin_xy, bin_color)):
        if xy is None:
            continue
        cx, cy = int(round(xy[0])), int(round(xy[1]))
        cv2.circle(out, (cx, cy), radius, color, thickness=-1)
    return out


def zoh_fill(centers: list[XY]) -> list[XY]:
    """零阶保持：把漏检(None)填成上一次成功检测的坐标。

    首次成功检测之前的 None 保持 None(该帧不画标记)。这复现部署时低频 VLM
    刷新之间标记不变的行为，使训练/部署分布一致。
    """
    out: list[XY] = []
    last: XY = None
    for c in centers:
        if c is not None:
            last = c
        out.append(last)
    return out


def _selfcheck() -> None:
    # draw: 圆心像素应被染成对应颜色，None 的一路不画
    img = np.zeros((960, 1280, 3), dtype=np.uint8)
    out = draw_markers(img, (100.4, 200.6), None, radius=5)
    assert tuple(out[201, 100]) == TARGET_COLOR, out[201, 100]
    assert out[:, :, 2].sum() == 0, "bin=None 不该出现蓝色"
    assert img.sum() == 0, "不得就地修改输入"

    # zoh: 漏检沿用上一次；首个 None 保持 None
    got = zoh_fill([None, (1, 1), None, (2, 2), None])
    assert got == [None, (1, 1), (1, 1), (2, 2), (2, 2)], got
    print("markers selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
