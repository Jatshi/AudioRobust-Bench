# AudioRobust-Bench 3.0 深度学习、踩坑与面试手册

> 目标：能从零实现一个跨 ASR、说话人和事件识别的可靠性 benchmark，讲清 manifest、
> corruption、adapter、slice、校准、分布式评测和指标声明边界。

## 1. Benchmark 真正解决什么

干净样本 Demo 无法回答系统在噪声、混响、带宽限制、削波和丢包下何时失效。不同任务
若各自使用不同音频、强度和指标，也无法公平比较。AudioRobust-Bench 用同一组稳定 case
驱动多个任务，并把输入、扰动、模型、输出和指标语义写入 manifest。

3.0 不是再训练一个模型，而是把可靠性测量做成可复现基础设施。

## 2. Manifest 为什么是核心

每个 case 至少包括 source ID、corruption type、strength、seed 和稳定 output ID。ID 应由
这些字段哈希生成，不能使用当前时间。这样不同 worker 和不同机器得到同一 ID，结果可以
合并、去重和审计。

一个完整 manifest 还应记录源文件 SHA-256、模型 revision、采样率、metric scope、环境
版本和失败原因。

## 3. 三层解耦

### Corruption 层

输入波形，输出扰动波形。它不知道 Whisper、WavLM 或 AST，只负责确定性信号处理。

### Adapter 层

把波形送入任务模型，输出统一的 `score/confidence/error`。它不负责生成噪声。

### Aggregator 层

按任务、扰动和强度聚合均值、失败率、slice 与校准。它不重新运行模型。

这种边界使新增第二个 ASR 时不修改扰动层，也使 CI 可以使用小型 fake adapter 验证合同。

## 4. SNR 扰动怎样实现

给定干净信号功率 `P_s` 和目标 SNR `q` dB，需要噪声功率：

`P_n = P_s / 10^(q/10)`。

将固定 seed 生成的噪声归一到 `P_n` 后相加。同一个 case 必须使用同一 seed；否则不同
模型实际上看到了不同噪声 realization，比较不公平。

混响、带宽、削波和丢包也要把所有参数写进 case，不能只写一个模糊标签“noisy”。

## 5. 三类指标语义不同

- ASR：有参考文本，字符准确度/错误率是真值任务质量；
- Speaker：本次 smoke 使用相对 clean embedding cosine consistency，不是克隆 MOS；
- Event：相对 clean top-5 标签重合，不是人工真值分类准确率。

统一的是协议和报告结构，不是把三个分数粗暴平均。README 和 manifest 使用
`measurement_scope`，避免把一致性写成准确率。

## 6. 校准与 ECE

模型可能准确率尚可但过度自信。将置信度分桶，ECE 可写为：

`ECE = Σ_b (n_b/N) * |accuracy_b - confidence_b|`。

ECE 依赖桶数和样本量，小型 smoke 只验证计算链路，不能给稳定校准结论。正式 benchmark
应报告 reliability diagram、各强度 ECE 和置信区间。

## 7. 怎样解释真实结果

公开 JFK 样本在 clean/10/0dB 的 ASR 字符准确度为 1.0000/0.9639/0.8554，说明噪声
增强后有单调退化。Speaker consistency 均值 0.8951，Event top-5 consistency 0.4667。

只能说“真实模型 smoke 揭示了该样本的退化趋势”。单样本不代表总体分布，后两项也没有
人工真值。诚实边界是项目可信度的一部分。

## 8. 分布式评测设计

稳定 case ID 允许将 manifest 按哈希分片：`worker = hash(case_id) mod world_size`。worker
输出 append-only 结果；聚合器按 ID 去重并检查所有预期 case 是否完成。

每个 worker 可以各自加载模型，也可请求集中服务。无论哪种，最终报告必须记录
world size、分片规则、失败重试和模型 revision。单机多进程合同不等于多 GPU scaling。

## 9. 关键踩坑案例

### Python/SciPy ABI 冲突

Python 3.10 无法安装某些新 SciPy 约束。解决是按 ABI 固定 `scipy==1.15.3` 并保存
environment freeze，而不是无差别升级所有包。

### AutoDL 下载公开样本超时

本地下载后先验 SHA-256，再上传远端；下载器增加“已有且哈希正确则复用”和镜像回退。
这样节省付费时间，也避免来源不明文件。

### 指标名字过度声明

WavLM cosine 一度容易被叫“克隆质量”，AST overlap 容易被叫“事件准确率”。修复是
重新命名 consistency，并在 manifest 写清有无人工真值。

### 缺依赖时静默使用假数据

正式路径禁止 silent fallback。模型不可用或推理失败必须记录 error，测试中的 fake
adapter 只能用于合同单测，不能混入公开结果。

## 10. 高频面试问答

**为什么不用一个总分？** 三个任务量纲与风险不同，总分会隐藏局部灾难；应提供任务级
曲线和部署阈值。

**clean anchor 有什么价值？** 当人工标签昂贵时，它能快速测稳定性；但不能替代真值，
所以必须标注 consistency。

**怎样防 benchmark 被刷分？** 保留隐藏扰动组合，固定公开协议，要求提交逐 case manifest、
多强度曲线和校准，而不是只接受最佳均值。

**3.0 为什么没有新模型权重？** 这是评测基础设施，不训练自有模型；重复上传 Whisper、
WavLM、AST 既无必要也可能违反上游分发边界。

## 11. 亲手练习

- 新增带通或削波 corruption 并测试边界；
- 扩展到 20/10/5/0/-5dB 并画曲线；
- 手算一个过度自信预测集的 ECE；
- 增加第二个 ASR adapter，证明 corruption 无需修改；
- 两进程分片后验证结果与单进程完全一致；
- 逐句区分 accuracy、similarity、consistency 和 calibration。

2.0 的真实模型接入和原始过程继续参考
[2.0 升级、学习与工程复盘](V2_UPGRADE_AND_LEARNING_ZH.md)。
