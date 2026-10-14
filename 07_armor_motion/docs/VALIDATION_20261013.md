# 验证记录：2026-10-13

工程运行位置：`/home/rm/robotmaster/07_armor_motion`。本记录对应 ROI 灯带重提取、测量质量权重、干净漏检恢复和标定三维刚体模式的集成版本。

## 输入和计量范围

- 输入：`/home/rm/datasets/outpost/前哨站视频.mp4`，925 帧，668×688，名义 30 fps。
- SHA256：`954081d075c01060df6608001f3991fb81126415daad3bdfe8402121d85809c0`。
- 预测提前量保持 0.1 秒，没有通过缩短预测时间增加覆盖率。
- 实际视频误差比较使用未来接受的检测中心，不是人工真值。未评分预测仍保存在 CSV，不从总数中删除。
- 合成测试的未来真值只供生成测量和事后评分，不输入在线估计器。
- 无新的效果 MP4，无人工标签训练，无 GitHub 提交。

## 自动测试

默认构建与 OpenVINO 2024.6 构建各通过 45/45 项 CTest。测试明细由各构建目录的 `Testing/Temporary/LastTest.log` 保存。

新增覆盖包括：

- ROI 原图坐标还原、弱灯带、截断轮廓、单灯带、共享灯带、重复候选、分类失败候选和倾斜灯带。
- 两轴质量权重、方差上限、非法数值拒绝、外扩不改变测量、降权测量不能绕过快速恢复条件。
- 成熟轨迹短缺测、竞争/拒绝候选、恢复中断、持续歧义及未成熟轨迹不提前放行。
- 三维 IPPE 多解、高低错落、等高相对编号、±π 展开、静止/正反转/平移、同帧多观测、漏检、超时及重新初始化。
- 近似等价联合关联的歧义保持、空帧不能解除歧义、身份分支独立过期，以及单分支存活不能凭历史分数直接冒充身份确认。
- 真实主程序的 56 帧合成无损视频端到端测试：灯带检测、位姿、共同状态、缺测超时、新 epoch、日志和无视频输出。
- 配置解析、运行中热更新与状态重置、缺失标定拒绝、图像缩放与去畸变坐标一致性。
- 原始 925 帧视频回归。

`config/calibrated_video.template.yaml` 直接启动返回“requires measured K/D/resolution, axis rotation, lamp dimensions, radius, pitch and three heights”，未创建运行输出。没有填入演示标定使测试视频冒充真实验证。

## 全视频主程序运行

以下六个运行目录的最后状态均为 `completed`，`frames=925`，`video_enabled=0`：

| 配置 | outputs 下的目录 |
|---|---|
| 默认二维 | `integration_default_20261013` |
| 测量质量 | `integration_quality_20261013` |
| 干净恢复 | `integration_recovery_20261013` |
| 网络辅助 | `integration_neural_20261013` |
| 网络 ROI 重提取 | `integration_roi_20261013` |
| 三项组合二维 | `integration_combined_20261013` |

默认 `tracking.csv` 和 `motion_state.csv` 与修改前 `integration_baseline_20261013` 逐字节一致。默认与干净恢复的实际视频跟踪结果也一致；自然输入没有触发快速恢复条件，不能把受控实验的改善描述为该自然视频已经改善。

## ROI 重提取

网络候选总数为 965。未启用 ROI 时接受 790 个实测双灯带观测；启用后保留这 790 个，并新增 16 个 `accepted_roi_measured_pair`。分类拒绝 95 个，缺少灯带支持从 80 个降为 64 个。

原有 790 个观测的中心和实测端点不变。新增结果均带有全局双灯带索引，没有同帧共享灯带；实测端点在图像范围内。这证明来源和几何约束成立，不等于经过人工确认的识别准确率。

新增观测帧号（零起始）：

`113, 132, 151, 172, 324, 325, 344, 363, 364, 441, 535, 572, 707, 725, 746, 766`

| 指标 | 网络辅助 | ROI 重提取 |
|---|---:|---:|
| 发出预测 | 611 | 621 |
| 可评分预测 | 494 | 504 |
| 视频末端无法评分 | 3 | 3 |
| 历史中断 | 2 | 7 |
| 目标时刻无接受观测 | 112 | 107 |
| 全部可评分均值 / px | 3.138 | 3.213 |
| 全部可评分 RMSE / px | 4.914 | 4.981 |

两组共同的 494 个预测样本误差完全一致，均值 3.138 px、P95 11.196 px、RMSE 4.914 px、最大 20.326 px。新增 10 个可评分样本改变了总体样本构成，因此不能直接用总体均值差称为退步或进步，也不能宣称已有预测变准。

证据：`outputs/integration_prediction_neural_20261013.csv`、`integration_prediction_roi_20261013.csv`，以及对应 `.txt` 和 `.confidence.csv`。

## 测量质量权重

默认发出 678 个预测、评分 557 个；质量权重发出 684 个、评分 565 个。共同 557 个样本比较：

| 指标 / px | 默认 | 质量权重 |
|---|---:|---:|
| 均值 | 3.637 | 3.732 |
| RMSE | 5.477 | 5.560 |
| P95 | 11.954 | 11.866 |
| 最大 | 19.755 | 20.151 |

没有稳定的误差改善，因此默认未开启。残差尺度是有界启发式膨胀，不被描述为已校准协方差或准确率。

证据：`outputs/integration_prediction_default_20261013.csv`、`integration_prediction_quality_20261013.csv` 及各主运行的 `measurement_reliability.csv`。

## 三项组合对照

补充使用 `combined_2d.yaml` 完整评估：发出 627 个预测、评分 510 个，视频末端 3 个、历史中断 7 个、无接受目标 107 个。相对仅 ROI 配置，共同 504 个样本，新增 6 个可评分样本，无基线独有样本。

共同样本均值从 3.213 增至 3.290 px，RMSE 从 4.981 增至 5.046 px，P95 从 11.196 降至 10.595 px，最大误差从 20.326 增至 20.953 px。组合并没有一致精度优势，未设为默认。完整组合评分均值为 3.303 px、RMSE 5.043 px，与共同样本口径不同。

证据：`outputs/integration_prediction_combined_20261013.csv` 及对应 `.confidence.csv`。该对照没有证明哪一项单独导致全部变化，也不将多开开关视为自动优化成功。

## 受控缺测恢复

证据目录：`outputs/controlled_occlusion_verified_20261013`。

缓存不修改的原始检测结果，固定 27 个互不重叠片段，每片分别屏蔽全部候选 1、2、3 帧，共 81 个场景；预热和恢复区间各 5 帧。参考局部 ID 只用于选段和事后计分，不送入被测跟踪器。27 个片段中 26 个在屏蔽前成熟，另 1 个保留在统计中。

快路径与基线在每一步的位置、速度、几何等数值状态相等；区别仅在恢复后的预测就绪条件。所有场景原局部 ID 仍存在，错误参考匹配为 0。缺 2/3 帧各有 1 个场景在恢复区间始终未再接受对应观测，该失败未删除，也不能因为 ID 仍 coast 就算成功恢复。

| 屏蔽帧数 | 基线恢复均值 / 帧 | 快路径恢复均值 / 帧 | 基线可评分观测预测 | 快路径可评分观测预测 |
|---|---:|---:|---:|---:|
| 1 | 5.000 | 2.667 | 27/135 | 90/135 |
| 2 | 5.000 | 2.462 | 26/135 | 92/135 |
| 3 | 5.000 | 5.000 | 26/135 | 26/135 |

恢复均值仅统计实际恢复场景，并与未恢复数量一起解释。新增运动预测误差：缺 1 帧的 30 个样本均值 3.619 px、最大 11.739 px；缺 2 帧的 33 个样本均值 4.230 px、最大 12.277 px。该结果说明特定干净缺测条件下就绪更早，不证明真实遮挡或所有机动场景的预测更准，故保留独立实验配置。

## 复现

在工程目录执行；输出路径使用尚不存在的新名字：

```bash
./build/outpost_video config/video.yaml --headless --no-video
./build-openvino-2024/outpost_video config/roi_refinement.yaml --headless --no-video

./build/prediction_evaluation config/video.yaml outputs/recheck_default.csv
./build/prediction_evaluation config/measurement_quality.yaml outputs/recheck_quality.csv
./build/compare_predictions outputs/recheck_default.csv outputs/recheck_quality.csv

./build-openvino-2024/prediction_evaluation config/neural_supported.yaml outputs/recheck_neural.csv
./build-openvino-2024/prediction_evaluation config/roi_refinement.yaml outputs/recheck_roi.csv
./build/compare_predictions outputs/recheck_neural.csv outputs/recheck_roi.csv

./build-openvino-2024/prediction_evaluation config/combined_2d.yaml outputs/recheck_combined.csv
./build/compare_predictions outputs/recheck_roi.csv outputs/recheck_combined.csv

./build/occlusion_replay config/video.yaml outputs/recheck_occlusion 5 5
```

标定三维模式的真实视频精度、外部物理身份、人工标签检测精度与新模型训练均不在本次已完成验证范围内。
