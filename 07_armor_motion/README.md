# 自瞄作业二：运动解算

## 实验完成内容

我使用 C++17 和 OpenCV 完成了前哨站三块装甲板的灯带检测、连续相对编号、相位与角速度估计、运动状态判断、未来 3 帧位置预测、视频叠加和逐帧 CSV 输出。工程在 Ubuntu 虚拟机中完成编译、测试、模型标定和整段视频处理。

程序分为两个可执行文件：

- `outpost_calibrate` 读取标定视频，生成固定相机下的二维几何和相位四边形模型。
- `outpost_solver` 加载模型后，从第 0 帧开始单遍处理待识别视频。

画面中的绿色实线四边形是当前帧最高置信度的真实双灯带检测框，绿色虚线四边形是另外两个相对槽位的模型位置，红色虚线四边形是未来 3 帧目标。`A1`、`A2`、`A3` 是连续相对编号，`USED` 和 `GATED` 表示当前绿色候选是否通过门控并参与状态更新。

## 工程结构

```text
07_armor_motion/
├── CMakeLists.txt
├── config/
│   ├── outpost_model.yaml
│   └── video.yaml
├── include/
│   ├── armor_detector.hpp
│   ├── config.hpp
│   ├── render_policy.hpp
│   ├── rigid_model.hpp
│   └── types.hpp
├── src/
│   ├── armor_detector.cpp
│   ├── calibrate.cpp
│   ├── config.cpp
│   ├── main.cpp
│   ├── render_policy.cpp
│   └── rigid_model.cpp
└── tests/
    └── test_main.cpp
```

## 环境与运行

我使用 Ubuntu、C++17、CMake 3.16 及以上版本、OpenCV 4 和 GCC。依赖安装命令为：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

我在工程目录中完成 Release 编译和测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

我先生成标定模型，再处理完整视频：

```bash
VIDEO="/absolute/path/to/前哨站视频.mp4"
./build/outpost_calibrate "$VIDEO" config/outpost_model.yaml config/video.yaml
./build/outpost_solver "$VIDEO" output/识别效果.mp4 config/video.yaml \
  --model config/outpost_model.yaml --no-gui
```

`--no-gui` 用于无桌面环境下处理完整视频，`--debug-lights` 用于额外绘制灯带轮廓。求解器同时生成 MP4 和同名 CSV。

## 算法实现

### 1. 灯带检测与真实外缘四边形

我使用归一化 ROI 限定检测区域，结合红通道亮度、`R-G` 和 `R-B` 通道差提取红橙色区域。轮廓经过面积、长度、宽高比和倾角筛选后形成灯带候选，两根灯带再按照中心间距、高度差、长度比和倾角差完成配对。

每根灯带保留长轴、宽度和四个顶点。我使用左右灯带的外侧上下端点构造固定顺序 `{TL, TR, BR, BL}`，并检查凸性、面积、边长比例和灯带包含关系。普通连接不能完整包住灯带时，程序使用两条灯带外侧边与上下支撑线求交修复；仍不合法的配对和单灯带不会生成装甲板框。因此框的梯形、收窄和倾斜直接来自灯带形状。

### 2. 静态二维模型标定

标定程序按 `geometry_calibration_stride` 从视频采样真实检测框，使用 RANSAC 拟合装甲板中心的二维仿射椭圆：

```text
p_i(theta) = center + axis_cos * cos(theta + i*120°)
                      + axis_sin * sin(theta + i*120°)
```

有效观测按相位分入 36 个循环区间。每个区间对中心残差和四个角点相对中心的偏移取鲁棒统计，缺失区间使用循环插值和平滑补齐。`config/outpost_model.yaml` 保存模型版本、图像尺寸、去畸变状态、相机参数指纹、仿射几何和相位四边形模型，不保存角速度或预设轨迹。

本次视频分辨率为 668 × 688，没有对应相机的实测标定板数据，所以 `camera_enabled` 为 `0`。相机内参、畸变参数、装甲板尺寸、旋转半径和俯角仍保存在 `config/video.yaml`，后续取得实测值后可直接修改并重新标定模型。

### 3. 门控关联与连续编号

求解器维护三个固定槽位。候选关联代价由相位 NIS、像素重投影误差、检测得分惩罚和槽位切换惩罚组成。只有同时通过 `phase_nis_gate` 和 `max_observation_distance_px` 的候选才会更新状态；被拒候选不会修改编号、速度或换板历史。

连续跟踪时只允许保留当前槽位或经过图像大跳变检验的相邻槽位。进入 `TEMP_LOST` 后，重新允许三个槽位参与关联，以便遮挡或短时漏检后恢复。槽位更新只在观测被接受后提交，避免高分离群候选造成 ID 交换。

### 4. 相位 Kalman 滤波与跟踪状态机

我使用状态向量 `[phase, angular_speed]` 的二状态 Kalman 滤波器。预测使用常角速度模型，过程噪声由角加速度标准差和实际时间间隔构造；更新前使用 NIS 门控，更新后使用 Joseph 形式计算协方差并重新对称。

跟踪状态为：

```text
LOST -> DETECTING -> TRACKING -> TEMP_LOST -> LOST
```

首次观测进入 `DETECTING`，连续确认后进入 `TRACKING`。短时漏检进入 `TEMP_LOST` 并继续模型预测，超过显示时限后隐藏预测框，超过丢失时限后进入 `LOST`。时间戳倒退、非有限值或间隔过大时立即清空状态重新确认。

### 5. 在线角速度与方向估计

程序没有写死角速度，也没有使用预设轨迹代替估计。大跳变只有在位移范围、运动方向和轨道中心关系同时满足门限时才作为已确认换板事件。

连续三个换板事件对应机构转过 120°，程序使用事件时间间隔计算角速度大小，并对最近 `physical_speed_window` 个结果取中位数。旋转方向由最近 `physical_direction_window` 个已确认槽位变化多数投票得到。当前配置关闭角速度吸附，吸附目标值和容差均为零。

### 6. 未来 3 帧图像预测

我使用最近 5 个通过图像连续性门控的检测中心计算全部点对斜率，再分别取横纵方向的 Theil–Sen 中位数。连续运动时，未来中心为：

```text
future_center = current_center + robust_velocity * prediction_lead_s
```

真实换板发生后，程序保存已经发生的退出框、进入框和槽位步进，形成在线换板模板。速度在预测时域内越过已学习退出位置时，未来目标切换到对应进入框，并保持到真实槽位切换或 `image_prediction_handover_hold_s` 超时，避免预测框在换板边界闪烁。

原始未来框再经过持续的 alpha-beta 中心和速度状态平滑，四个角点相对中心的形状也独立平滑。未来编号改变、时间间隔异常或创新过大时，状态直接重置到新目标，不在两块装甲板之间生成不存在的过渡框。短时漏检期间沿用最后一次可信速度，超过跟踪时限后停止绘制。

### 7. 输出与重叠显示

绿色当前框始终来自本帧真实灯带四边形，不使用模型框替换。红色未来框与绿色框高度重叠时仍保留，只切换为细线半透明样式；距离和 IoU 使用进入、退出两组阈值形成迟滞，避免样式闪烁。

CSV 逐帧记录候选是否存在、观测是否采用、三个槽位来源、相位、角速度、方向、跟踪状态、NIS、协方差、重投影误差、图像速度、换板事件、原始与滤波后的未来目标、未来编号和渲染状态。

## 配置文件

`config/video.yaml` 集中保存以下可修改参数：

- 相机内参、畸变系数和参考分辨率；
- 装甲板尺寸、旋转半径和前哨站俯角；
- ROI、颜色阈值、灯带筛选、双灯带配对和四边形留量；
- 几何采样步长、RANSAC、相位分箱和循环平滑；
- 跟踪确认、短时显示、丢失超时和时间戳上限；
- Kalman 过程噪声、观测噪声、NIS 与像素关联门限；
- 物理角速度统计窗口和方向投票窗口；
- 预测提前时间、图像速度窗口、换板门限和保持时间；
- 未来目标滤波增益、最大速度、重置距离和重叠显示阈值。

## 验证

`tests/test_main.cpp` 覆盖灯带四边形、灯带包含、不同倾角与长度、极窄和退化配对、单灯带拒绝、相位循环插值、三个槽位关系、编号连续性、NIS 门控、离群候选、短时丢失、时间戳复位、未来框滤波、换板重置、重叠显示迟滞、标定模型版本与相机指纹以及 YAML 参数校验。

整段 925 帧视频回归中，923 帧检测到有效双灯带候选，796 帧观测通过门控；跟踪状态为 794 帧 `TRACKING`、129 帧 `TEMP_LOST` 和 2 帧 `DETECTING`。运动段方向保持为 `CCW`，有效角速度约为 62.07–63.16°/s。所有单元测试均通过。
