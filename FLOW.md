# moon-galbot 项目流程文档

> **项目目标**：Galbot G1 机器人通过语音指令，从桌上随机摆放的多个物体中抓取指定目标并放入框中。支持途中挪动干扰和在线修正（"往上抓一点 / 抓紧一点"）。

---

## 0. 核心架构：VLM 解耦 + 视觉标记 + 双速 ACT

```
┌──────────────────────────────────────────────────────────────────┐
│                         语音接口                                  │
│  机器人内置麦克风 → KWS 关键词识别 → ZMQ 6000 发送标签               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ 物体标签 / 修正标签
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    双速运行架构（工作站）                           │
│                                                                  │
│   ┌── VLM 刷新环 (1.5Hz) ──┐     ┌── ACT 控制环 (15Hz) ──┐       │
│   │  locate-anything 3B   │     │  单策略端到端           │       │
│   │  检测目标物 + 框       │     │  pick → move → place    │       │
│   │       │               │     │       ▲                 │       │
│   │       ▼               │     │       │                 │       │
│   │  bbox 中心坐标         │ ──▶ │  头相机画标记           │       │
│   │  (SharedState 共享)   │     │  + ACT 推理 + SDK 执行  │       │
│   └───────────────────────┘     └─────────────────────────┘       │
│                                                                  │
│   关键设计：                                                       │
│   · 头部相机 = 画标记（目标绿点 + 框蓝点）→ 告诉策略"在哪"            │
│   · 腕部相机 = 原始画面不画标记 → 保留抓取时机的近距离视觉线索        │
│   · 在线修正 = 平移标记 + 夹爪 effort + 物体记忆 JSON               │
└──────────────────────────────────────────────────────────────────┘
```

**为什么这样设计？** 此前 pi0.5 端到端方案中，框的位置一变，pick 就受影响——策略在小数据下把场景布局与行为纠缠在一起学了。本架构把 grounding（"认哪个 / 在哪"）剥离给预训练 VLM，让 action 策略坍缩成"朝标记伺服"，从根本上消除此问题。代价是 ACT 无从零训练的行为先验，必须用扰动示教补回。

---

## 1. 项目结构

```
moon-galbot/
├── tools/                          # 离线数据处理管线
│   ├── galbot_mcap.py              # 读 SYNC mcap，时间对齐，组装 state/action（库）
│   ├── markers.py                  # 纯 cv2 画点 + ZOH 零阶保持（训练/部署共用）
│   ├── detect_markers.py           # 阶段A：locate-anything 逐帧检测 → detections/*.parquet
│   ├── convert_mcap_to_lerobot.py  # 阶段B：mcap + detections → lerobot 数据集 (h264)
│   └── preview_markers.py          # 检查阶段A：把中心画回真实帧，打印漏检率
├── training/
│   └── train_act.sh                # lerobot 原生 ACT 训练启动脚本
├── deploy/                         # 真机部署（双速运行时）
│   ├── config.py                   # 部署契约常量（关节组/夹爪/home 姿态/安全阈值/控制频率）
│   ├── shared_state.py             # 线程安全共享状态：VLM 中心 + 目标标签 + 命令计数
│   ├── vlm_worker.py               # 低频 VLM 刷新环（线程）→ 持续更新目标/框中心
│   ├── policy_runtime.py           # 高频 ACT 控制环：抓图 → 画标记 → 推理 → SDK 执行
│   ├── correction.py               # 在线修正三件套：标记平移 / 夹爪 effort / 记忆 JSON
│   ├── voice.py                    # 语音接口：消费 ZMQ → KWS 标签 → 物体/修正路由
│   └── run_g1.py                   # 主入口，串起以上全部
├── docs/adr/
│   └── 0001-vlm-decoupled-act-over-pi05.md  # 架构决策记录
├── README.md                       # 使用说明
├── CONTEXT.md                      # 术语定义
├── PLAN.md                         # 计划文档
└── pyproject.toml                  # 项目配置与依赖
```

---

## 2. 端到端流程概览

```
步骤1            步骤2             步骤2.5          步骤3             步骤4          步骤5
采数据     →     阶段A 检测    →   检查检测     →   阶段B 转换    →   训练 ACT   →   真机部署
遥操作录         VLM 逐帧检        漏检率/位置       mcap+坐标          lerobot       双速运行时
*.SYNC.mcap      测目标+框         可视化验证       → lerobot 数据集   训练脚本       抓取闭环
                 → *.parquet                        (h264 视频)
```

---

## 3. 详细流程

### 步骤 1：采集数据（比赛现场，遥操作）

**输入**：遥操作控制机器人执行抓取-放框任务
**输出**：`*.SYNC.mcap` 文件（每 episode 一个）

**数据配方（关键质量要求）**：

| 要求 | 做法 | 目的 |
|------|------|------|
| 位置泛化 | 每条 episode 随机摆放物体和框位置 | 治"背地图" |
| 抗挪动干扰 | 约 1/4 ~ 1/3 集在接近途中人为挪动物体，重新接近完成 | ACT 无行为先验，恢复行为必须显式示教 |
| 多物体在场 | 桌上摆多个物体，只抓目标物，目标物轮换 | 否则学成"抓最近的"，语音选物失效 |
| 量级 | 每任务 50~100 条 | — |

> **注意**：管线只用 `*.SYNC.mcap`（时间对齐后的数据）；`*.FIN.mcap` 是原始未对齐数据，用不上。

---

### 步骤 2：阶段 A — VLM 离线检测

**脚本**：[tools/detect_markers.py](tools/detect_markers.py)

**输入**：
- `*.SYNC.mcap` 原始数据
- locate-anything 3B 模型权重
- 目标物和框的文字描述（如 `"cola bottle"`, `"basket"`）

**处理流程**：
```
*.SYNC.mcap
    │
    ▼
┌─────────────────────────────┐
│ galbot_mcap.read_episode()  │  ← 读 mcap，提取 head_ts + head_jpeg
│ 得 EpisodeStreams           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ decimate_indices()          │  ← 30fps → 15fps 抽帧
│ 选帧索引 sel                │
└──────────────┬──────────────┘
               │ 对每个抽帧
               ▼
┌─────────────────────────────┐
│ locate-anything             │  ← ground_single(target_label)
│ 检测目标物 bbox             │     取最大框中心 → (tx, ty)
│ 检测框 bbox                 │     取最大框中心 → (bx, by)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 写入 detections/            │  ← 每帧一行 parquet：
│ <episode>.parquet           │     head_ts, target_x/y, bin_x/y
└─────────────────────────────┘
```

**输出**：`detections/<episode>.parquet`（仅坐标，KB 级，可反复复用）

**命令**：
```bash
uv run python tools/detect_markers.py \
  --data-dir <数据目录> --out-dir detections \
  --target-label "cola bottle" --bin-label "basket" \
  --model <LocateAnything-3B路径> --max-episodes 1
```

---

### 步骤 2.5：检查检测效果

**脚本**：[tools/preview_markers.py](tools/preview_markers.py)

**作用**：把检测中心画回真实帧，生成预览图，打印漏检率。

**验收标准**：
- 框（basket）漏检率 ≈ 0%
- 目标物遮挡导致漏检几帧属正常
- 翻预览图确认：**绿点压在目标上，蓝点在框上**

**命令**：
```bash
uv run python tools/preview_markers.py \
  --data-dir <数据目录> --detections-dir detections --out-dir preview --n 12
```

---

### 步骤 3：阶段 B — 转换为 lerobot 数据集

**脚本**：[tools/convert_mcap_to_lerobot.py](tools/convert_mcap_to_lerobot.py)

**输入**：
- `*.SYNC.mcap` 原始数据
- `detections/*.parquet` 检测坐标

**处理流程**：
```
*.SYNC.mcap  +  detections/*.parquet
           │
           ▼
┌──────────────────────────────────────┐
│ 1. galbot_mcap.read_episode()        │  ← 读 mcap 各流
│ 2. decimate_indices() 抽帧 30→15fps │
│ 3. assemble_state_action()           │  ← 8维 state + 8维 action
│    · state = 7 关节角 + 夹爪开度[0,1] │     (sensor 当前值 + target ZOH)
│    · action = 7 目标关节 + 夹爪指令   │
│ 4. _load_centers() + zoh_fill()     │  ← 检测坐标 ZOH 补全
│ 5. markers.draw_markers()           │  ← 头相机画绿/蓝点（原分辨率）
│ 6. cv2.resize() → 640x480           │  ← resize 头相机
│ 7. dataset.add_frame()              │  ← 逐帧写入 lerobot 数据集
└──────────────────┬───────────────────┘
                   │
                   ▼
     lerobot 数据集 (h264 视频编码)
     · observation.images.head   (640x480, 画了标记)
     · observation.images.wrist  (640x360, 无标记)
     · observation.state         (8维 float32)
     · action                    (8维 float32)
     · task                      (文本 prompt)
```

**输出**：`$HF_LEROBOT_HOME/galbot_g1_marked/` 标准 lerobot 数据集

**关键约定**：
- 标记在头相机原分辨率 (1280×960) 上画，再 resize 到 640×480——与部署完全一致
- 漏检用 ZOH（零阶保持）沿用上一次成功检测——复现部署时低频 VLM 刷新之间标记不变的行为
- 视频编码默认 h264（本机 FFmpeg AV1 解码有问题）

**命令**：
```bash
export HF_LEROBOT_HOME=<lerobot_data路径>
uv run python tools/convert_mcap_to_lerobot.py \
  --data-dir <数据目录> --detections-dir detections \
  --output-root $HF_LEROBOT_HOME --repo-id galbot_g1_marked --fps 15 --overwrite
```

---

### 步骤 4：训练 ACT

**脚本**：[training/train_act.sh](training/train_act.sh)

**流程**：
```
lerobot 数据集
      │
      ▼
┌──────────────────────────┐
│ lerobot.scripts.train    │
│ · policy.type = act      │
│ · n_action_steps = 15    │  ← ~1s@15fps 执行一段再重规划
│ · chunk_size = 50        │  ← 预测 50 步，执行前 15 步
│ · video_backend = pyav   │  ← h264 解码
│ · batch_size = 8         │
│ · steps = 100000         │
└──────────┬───────────────┘
           │
           ▼
  outputs/act_galbot_g1_marked/checkpoints/<step>/pretrained_model
```

**命令**：
```bash
export HF_LEROBOT_HOME=<lerobot_data路径>
bash training/train_act.sh
```

**产物**：ACT checkpoint 目录，部署用 `--checkpoint` 指向它。

---

### 步骤 5：真机部署（双速运行时）

**主入口**：[deploy/run_g1.py](deploy/run_g1.py)

#### 5.1 部署架构（线程级）

```
┌─────────────────────────────────────────────────────────┐
│  run_g1.py (主线程)                                      │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  VLMWorker        │  │  PolicyRuntime   │             │
│  │  (daemon 线程)    │  │  (主线程调用)     │             │
│  │                   │  │                  │             │
│  │  循环 1.5Hz:      │  │  run_episode():  │             │
│  │  ① 抓头相机        │  │  ┌────────────┐ │             │
│  │  ② ground 目标    │  │  │ capture()  │ │  ← 15Hz    │
│  │  ③ ground 框      │  │  │ 抓头+腕相机 │ │             │
│  │  ④ set_centers()  │  │  │ 读 Shared   │ │             │
│  │     ↓              │  │  │ State 中心  │ │             │
│  │  SharedState       │  │  │ +修正偏移   │ │             │
│  │  ┌──────────────┐  │  │  │ draw_       │ │             │
│  │  │ target_xy    │──┼──┼─▶│ markers()   │ │             │
│  │  │ bin_xy       │  │  │  │ resize      │ │             │
│  │  │ target_label │  │  │  │ → obs dict  │ │             │
│  │  │ bin_label    │  │  │  └─────┬──────┘ │             │
│  │  │ cmd_id       │  │  │        ▼        │             │
│  │  └──────────────┘  │  │  ┌────────────┐ │             │
│  └──────────────────┘  │  │  │ infer_     │ │             │
│                        │  │  │ chunk()    │ │ ← ACT 推理 │
│  ┌──────────────────┐  │  │  │ → (T,8)    │ │             │
│  │  VoiceMicClient   │  │  │  └─────┬──────┘ │             │
│  │  (daemon 线程)    │  │  │        ▼        │             │
│  │                   │  │  │  ┌────────────┐ │             │
│  │  ZMQ SUB 6000    │  │  │  │ 安全检查    │ │             │
│  │  收 KWS 标签     │  │  │  │ 夹爪滞回    │ │             │
│  │  → set_labels()  │  │  │  │ SDK 执行    │ │             │
│  │  → apply_hits()  │  │  │  └────────────┘ │             │
│  └──────────────────┘  │  │  循环至 done 或  │             │
│                        │  │  max_chunks      │             │
│  ┌──────────────────┐  │  └─────────────────┘             │
│  │  CorrectionMemory │  │                                  │
│  │  (JSON 持久化)    │◀┼── 修正量读写                     │
│  └──────────────────┘  │                                  │
└─────────────────────────────────────────────────────────┘
```

#### 5.2 两种运行模式

**模式一：无语音 bring-up（推荐先用）**

```bash
# Dry-run：VLM 定位 → 画标记 → 推理 → 打印动作，机器人不动
uv run python deploy/run_g1.py \
  --checkpoint <checkpoint路径> \
  --model <LocateAnything-3B路径> \
  --target-label "cola bottle" --bin-label "basket"

# 调好后真动
uv run python deploy/run_g1.py \
  --checkpoint <checkpoint路径> \
  --model <LocateAnything-3B路径> \
  --target-label "cola bottle" --bin-label "basket" \
  --execute --go-home
```

**模式二：语音模式**

```bash
uv run python deploy/run_g1.py \
  --checkpoint <checkpoint路径> \
  --model <LocateAnything-3B路径> \
  --mic-addr tcp://<机器人IP>:6000 --execute --go-home
```

语音流程：
1. 人对机器人说"帮我拿可乐"
2. 机器人内置麦克风 → KWS 关键词识别 → 命中 `cola` 标签
3. ZMQ PUB/SUB 发送 `{"type":"kwd_cmd","data":"cola"}` 到工作站
4. [voice.py](deploy/voice.py) 路由：`cola` → `"cola bottle"` → `SharedState.set_labels()`
5. VLMWorker 读到新标签，开始检测目标位置
6. `wait_for_center()` 等待 VLM 产出有效检测
7. PolicyRuntime 执行抓取 episode
8. VoiceMicClient 说"完成"

#### 5.3 在线修正流程

```
用户说"往上抓一点"
       │
       ▼
┌──────────────────────────┐
│ KWS 识别 → corr_up 标签  │
│ ZMQ → VoiceMicClient     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ correction_labels 路由    │
│ corr_up → ("dy", -1)     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ CorrectionMemory         │
│ .apply_hits("cola",      │
│   [("dy", -1)])          │
│ → dy 累加 -40px          │
│ → 保存 corrections.json  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ PolicyRuntime.capture()  │
│ .apply_offset()          │
│ → 标记中心 (x, y-40)     │
│ → draw_markers() 画偏上  │
│ → ACT 策略自然跟过去      │
└──────────────────────────┘
```

修正三件套：

| 类型 | 关键词示例 | 机制 | 持久化 |
|------|-----------|------|--------|
| 空间类 | "往上/往下/往左/往右" | 平移目标标记（像素偏移） | `corrections.json` 中 dx, dy |
| 力度类 | "抓紧一点/松一点" | 调节夹爪 effort | `corrections.json` 中 grip |
| 记忆 | 自动 | 同物体下次指令自动预加载上次修正量 | `corrections.json` |

---

## 4. 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           离线管线                                       │
│                                                                         │
│  遥操作                  *.SYNC.mcap                                     │
│    │                        │                                           │
│    ▼                        ▼                                           │
│  G1 机器人 ──────────► galbot_mcap.py ──► EpisodeStreams                │
│  (采集时 30fps)           │                                              │
│                           ├──► detect_markers.py ──► detections/        │
│                           │    (阶段A: VLM检测)      *.parquet          │
│                           │                           │                 │
│                           └──► convert_mcap_to_lerobot.py ◄──┘          │
│                                (阶段B: 组装数据集)                        │
│                                    │                                    │
│                                    ▼                                    │
│                              lerobot 数据集                              │
│                              (h264, 15fps)                              │
│                                    │                                    │
│                                    ▼                                    │
│                              train_act.sh                               │
│                              (ACT 训练)                                  │
│                                    │                                    │
│                                    ▼                                    │
│                              ACT checkpoint                             │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ 加载
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           在线部署                                       │
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ 语音输入     │     │ VLM 定位     │     │ ACT 控制     │               │
│  │             │     │             │     │             │               │
│  │ 人声 → KWS │ │     │ locate-     │     │ capture()   │               │
│  │ → ZMQ 标签 │─┼───▶│ anything    │────▶│ 抓头+腕相机  │               │
│  │             │     │ 3B (1.5Hz) │     │ + Shared     │               │
│  │ voice.py   │     │ vlm_worker  │     │   State中心  │               │
│  └─────────────┘     │ .py         │     │ draw_        │               │
│                      └─────────────┘     │ markers()    │               │
│                            │             │ resize       │               │
│                            ▼             │ ACT 推理     │               │
│                      SharedState         │ SDK 执行     │               │
│                      (线程安全)          │ → G1 机器人  │               │
│                            ▲             │             │               │
│                            │             │ policy_      │               │
│                      ┌─────────────┐     │ runtime.py  │               │
│                      │ 在线修正     │     │ (15Hz)      │               │
│                      │ correction │────▶│             │               │
│                      │ memory +    │     └─────────────┘               │
│                      │ offset      │                                   │
│                      └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 关键设计决策

| 决策 | 原因 | 影响 |
|------|------|------|
| VLM 只在头相机检测 | 腕相机近距离视角下物体可能不成完整外观，VLM 检测不可靠 | 腕相机保留原始画面，提供抓取时机线索 |
| 标记用实心圆点而非 bbox | 最小信息量——策略只需知道"在哪" | 降低对画法的过拟合风险 |
| ZOH 填补漏检 | 复现部署时低频刷新之间标记不变的行为 | 训练/部署分布必须严格一致 |
| markers.py 训推共用 | 标记外观（颜色/半径/线宽）不一致是此类方案头号翻车点 | 改标记 = 必须重跑阶段 B + 重训 |
| 帧率 30→15fps | 平衡存储/训练成本与控制精度 | 与 VLM 刷新率 1.5Hz 匹配：VLM 每 10 个控制步刷新一次 |
| ACT n_action_steps=15 | ~1s@15fps 执行一段再重规划 | 抗挪动反应性够；默认 100 太长 |
| 扰动示教占 1/4~1/3 | ACT 无行为先验，恢复行为必须显式在数据里 | 不采则抗挪动失败 |

---

## 6. 依赖关系

```
moon-galbot (本项目)
    │
    ├── locate-anything (../eagle/Embodied)
    │   └── nvidia/LocateAnything-3B 模型
    │
    ├── lerobot (huggingface/lerobot, git)
    │   └── ACT 策略
    │
    ├── galbot_sdk (G1 机器人 SDK)
    │   └── GalbotRobot, 相机/关节/夹爪控制
    │
    └── galbot-mic-service (机器人端语音服务)
        └── VAD + KWS → ZMQ 6000
```

---

## 7. 验证清单

根据 [PLAN.md](PLAN.md)，完整验收包括：

- [ ] **VLM**：比赛桌面照片验证全部物体+框检出率与延迟
- [ ] **语音**：真机喊指令，验证 KWS 标签命中正确率
- [ ] **策略（关键验收）**：
  - [ ] 框换位置 → pick 不受影响
  - [ ] 途中挪物体 → 重新接近成功
  - [ ] 多物体在场 → 只抓语音指定的
- [ ] **修正**：
  - [ ] "往上抓一点" → 抓取点上移
  - [ ] 同一物体二次指令 → 自动带上次修正量
- [ ] **阶段A 漏检率**：框 ≈ 0%，目标物遮挡漏几帧正常
- [ ] **标记一致性**：训练和部署画出的标记像素级一致
