# 数据处理 Runbook

把 `/home/galbot/1105_1696` 的 G1 遥操作 mcap 转成带视觉标记的 LeRobot 数据集，喂给 ACT。
设计背景见 [../PLAN.md](../PLAN.md)；这里只讲**怎么用、按什么顺序**。

## 0. 环境（uv，单个 venv）

lerobot 和 locate-anything 共处一个 venv，`pyproject.toml` 已配好，直接同步即可：
```bash
cd /home/galbot/moon-galbot
uv sync            # 按 pyproject.toml + uv.lock 建 .venv 并装全部依赖
```

关键点（都已写在 `pyproject.toml` 里，供理解，不用手动做）：
- **带 ACT 的 lerobot 只在 git 上**。PyPI 的 `lerobot==0.1.0` 是残缺老版，`policies` 里只有
  `tdmpc`、没有 ACT。故用 `[tool.uv.sources]` 从 git commit `0cf8648` 装。
- 该 git 版依赖声明诚实（`torch>=2.2.1` / `torchvision>=0.21` / `wandb>=0.16.3` 均开口、
  且不钉 numpy），因此与 locate-anything 的 `numpy<2` / `triton>=3.1` **无需任何 override**
  即可共存。已验证解析：numpy 1.26.4、torch 2.6+cu124、triton 3.2、transformers 4.57.1。
- `torch==2.6.0` / `torchvision==0.21.0` 显式钉到已验证可跑 ACT+CUDA 的版本。
- `setuptools` 必须装——triton 运行时 JIT 构建要用，uv 不默认带。
- Python 3.11（别用 3.8：locate-anything 的 numpy>=1.25 在 3.8 上装不上）。

自检（装完确认 import、版本与 CUDA）：
```bash
uv run python -c "import mcap, mcap_protobuf, cv2, numpy, torch, transformers, triton, eaglevl; \
from lerobot.common.policies.act.modeling_act import ACTPolicy; \
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
print('OK numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
已验证：上述全部通过；`eaglevl`(locate-anything)可导入；阶段 A/B 用**合成检测**在真实
episode 上端到端跑通——建库、编码、画点、ZOH 全部正确，cv2 解码 h264 后确认目标点为绿、
框点为蓝，落在预期像素。**两个脚本开箱即用**。

> **视频编码固定 h264**（转换器默认 `--vcodec h264`）。lerobot 默认的 `libsvtav1`(AV1)
> 在本机 FFmpeg 4.4 上**解不了**（cv2 与 torchcodec 均失败）；h264 通用可解，训练读取无碍。
>
> **torchcodec**（lerobot 默认视频后端）在本机报 `Could not load libtorchcodec`（FFmpeg 库
> 加载问题，与 codec 无关）。转换不受影响（编码走 av/ffmpeg）。训练读数据时二选一：修好
> torchcodec 的 FFmpeg，或给 lerobot 配 `video_backend="pyav"`（`av` 已装，能解 h264）。
>
> **只用 `uv sync` / `uv run`，别用 `uv pip install`**——后者绕过锁文件，会把 numpy/opencv
> 顶到 >=2 破坏约束（`uv run` 会自动同步回锁定状态）。

## 1. 脚本

同一个 venv 跑。分两阶段不是为隔离依赖，而是**检测是最慢的 GPU 环节（~28k 帧）**，
存下坐标后调标记样式只需重画、不必重检测。

| 文件 | 作用 |
|---|---|
| `galbot_mcap.py` | mcap 读取 + 对齐 + 8 维 state/action。库文件，不直接跑 |
| `markers.py` | 纯 cv2 画目标/框中心点 + ZOH。无重依赖，训练/部署 import 同一份保证一致 |
| `detect_markers.py`（阶段 A） | 逐帧 locate-anything 检测 → 写中心坐标到 `detections/<ep>.parquet`（漏检记 NaN） |
| `convert_mcap_to_lerobot.py`（阶段 B） | 读 mcap + 坐标 → ZOH → 画点 → 写 LeRobot 数据集 |

## 2. 阶段 A：检测

```bash
cd /home/galbot/moon-galbot
uv run python tools/detect_markers.py \
  --data-dir /home/galbot/1105_1696 \
  --out-dir /home/galbot/moon-galbot/detections \
  --target-label "cola bottle" --bin-label "basket"
```
- 逐帧检测头相机，按 head-frame 时间戳存中心坐标；一次性，约 28k 帧。
- `--target-label` / `--bin-label`：旧 50 条固定 `cola bottle` / `basket`；将来多物体数据
  每条传各自目标词（框标签通常不变）。
- 只读 `*.SYNC.mcap`；`*.FIN.mcap` 不用。

## 3. 阶段 B：转换

```bash
export HF_LEROBOT_HOME=/home/galbot/moon-galbot/lerobot_data

# 先拿 2 条跑通
uv run python tools/convert_mcap_to_lerobot.py \
  --data-dir /home/galbot/1105_1696 \
  --detections-dir /home/galbot/moon-galbot/detections \
  --output-root $HF_LEROBOT_HOME \
  --repo-id galbot_g1_marked \
  --fps 15 --max-episodes 2 --overwrite

# 跑通后去掉 --max-episodes 全量
```
阶段 B 只读阶段 A 产的坐标；漏检帧用 ZOH 沿用上一次中心。调标记样式时只重跑 B。

## 4. 转换后（venv-convert）

```bash
# 计算 norm stats（ACT 训练前必做），按 lerobot 原生流程对 galbot_g1_marked 计算
```

## 5. 验证（照做确认没白跑）

1. **标记对**：抽几条 episode，把 `observation.images.head` dump 成图，看圆点是否稳定
   压在目标物/框上；抓取遮挡段圆点应 ZOH 不乱跳。
2. **schema 对**：
   ```bash
   .venv/bin/python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
   d=LeRobotDataset('galbot_g1_marked', root='$HF_LEROBOT_HOME/galbot_g1_marked'); \
   print(list(d.features))"
   # 应含 observation.images.head/wrist、observation.state、action
   ```
3. **数值合理**：转换打印的每条 `max joint span` / `min grip` 无异常，QA 无超阈值告警。
4. **能训**：拿 2 条 episode 起 lerobot ACTPolicy 训几个 step，不报维度错。
