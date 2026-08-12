# AudioRobust-Bench 3.0 新增内容与发布说明

## 版本定位

3.0 将 2.0 的真实模型 smoke 与统一扰动协议整理成正式、可复用、可发布的 benchmark
版本。它没有重新训练 Whisper、WavLM 或 AST，也不凭空增加新的“准确率”；主要升级是
把版本、实验合同、证据、学习材料和 README 入口统一，形成能被他人复跑和审计的交付。

## 3.0 正式化内容

### 1. 统一版本与发布入口

- Python 包版本升级为 `3.0.0`；
- 增加运行时 `__version__`；
- README 增加 3.0 release、GPU、真实模型、测试和许可证徽章；
- 升级说明、学习手册、manifest、计划和 Demo GIF 均可一键进入。

### 2. 保留并固化真实模型协议

- Faster-Whisper tiny.en：有参考文本的 ASR 字符准确度；
- WavLM：相对 clean anchor 的说话人 embedding 一致性；
- AST AudioSet：相对 clean top-5 的事件标签一致性；
- clean、10dB、0dB 使用同一公开样本与稳定 SHA-256；
- 模型、环境、输入、输出与 `measurement_scope` 写入 manifest。

### 3. 统一可靠性合同

- corruption case 使用稳定 ID、强度和随机种子；
- 扰动层、任务 adapter 和聚合器相互解耦；
- 同时报告均值、失败率、强度切片和校准指标；
- 指标缺失、模型下载失败或依赖冲突必须显式失败；
- 不同任务不强行压缩成一个不可解释总分。

### 4. 可复现交付

- README Demo GIF 由真实 smoke manifest 生成；
- 环境 freeze、模型 ID、样本哈希和逐强度结果进入 Git；
- 远端原始日志和产物另存 F 盘，不把大模型或缓存提交到仓库；
- 学习手册记录依赖、下载、指标命名和 claim boundary 的踩坑过程。

## 已验证结果

| 任务 | 结果与含义 |
|---|---|
| ASR | clean/10/0dB：1.0000 / 0.9639 / 0.8554，均值 0.9398 |
| Speaker | 相对 clean embedding consistency 均值 0.8951 |
| Event | 相对 clean top-5 consistency 均值 0.4667 |
| GPU | RTX 4090，PyTorch 2.9.0+cu128 |
| 测试 | 5 passed；Ruff passed |

这些是单个公开样本的工程 smoke，不是 corpus-level benchmark；Speaker 与 Event 也是
一致性指标，不是带人工真值的准确率。

## 主要文件

```text
src/audio_robust_bench/core.py       manifest、slice、ECE 与报告
src/audio_robust_bench/audio.py      确定性音频扰动
src/audio_robust_bench/adapters.py   三任务指标适配
scripts/run_hf_gpu_smoke.py          真实模型 GPU 入口
artifacts/smoke/run_manifest.json    机器可读证据
assets/audio_robust_v2_demo.gif      真实结果动画（历史文件名保留）
```

## 与 2.0 的关系

2.0 完成了核心技术验证，3.0 是正式发布与证据治理版本。保留旧 Demo 文件名和
`V2_UPGRADE_AND_LEARNING_ZH.md`，避免破坏历史链接；所有新对外版本入口统一指向 3.0。

深入原理、完整排障和面试问答见
[3.0 学习与踩坑手册](V3_LEARNING_AND_INTERVIEW_ZH.md) 与
[2.0 原始工程复盘](V2_UPGRADE_AND_LEARNING_ZH.md)。
