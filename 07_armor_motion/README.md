# 自瞄作业二：运动解算

## 实验完成内容

我使用 C++17 和 OpenCV 完成了前哨站三块装甲板的灯带检测、连续相对编号、相位与角速度估计、运动状态判断、未来 3 帧位置预测、视频叠加和逐帧 CSV 输出。工程在 Ubuntu 虚拟机中完成编译、测试、模型标定和整段视频处理。

程序分为两个可执行文件：

- `outpost_calibrate` 读取标定视频，生成固定相机下的二维几何和相位四边形模型。
- `outpost_solver` 加载模型后，从第 0 帧开始单遍处理待识别视频。

画面中的绿色实线四边形是通过门控的真实双灯带检测框，绿色虚线四边形是未观测槽位的模型位置，红色虚线四边形是未来 3 帧目标。`A1`、`A2`、`A3` 是连续相对编号，`USED` 和 `GATED` 表示当前候选是否通过门控并参与状态更新。

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

本次视频分辨率为 668 × 688，没有对应相机的实测标定板数据，所以 `camera_enabled` 为 `0`。相机内参、畸变参数、装甲板尺寸、旋转半径和俯角均作为 YAML 配置项保存，本次运行使用固定相机二维模型。

### 3. 门控关联与连续编号

求解器维护三个固定槽位。候选关联代价由相位 NIS、像素重投影误差、检测得分惩罚和槽位切换惩罚组成。只有同时通过 `phase_nis_gate` 和 `max_observation_distance_px` 的候选才会更新状态；被拒候选不会修改编号、速度或换板历史。

连续跟踪时只允许保留当前槽位或经过图像大跳变检验的相邻槽位。进入 `TEMP_LOST` 后先保持原关联约束；超过图像历史时限仍未恢复时，才允许三个槽位参与重捕获。槽位更新只在观测被接受后提交，避免高分离群候选造成 ID 交换。

### 4. 相位 Kalman 滤波与跟踪状态机

我使用状态向量 `[phase, angular_speed]` 的二状态 Kalman 滤波器。预测使用常角速度模型，过程噪声由角加速度标准差和实际时间间隔构造；更新前使用 NIS 门控，更新后使用 Joseph 形式计算协方差并重新对称。

跟踪状态为：

```text
LOST -> DETECTING -> TRACKING -> TEMP_LOST -> LOST
```

首次观测进入 `DETECTING`，连续确认后进入 `TRACKING`。短时漏检进入 `TEMP_LOST` 并继续模型预测，超过显示时限后隐藏预测框，超过丢失时限后进入 `LOST`。时间戳倒退、非有限值或间隔过大时立即清空状态重新确认。

### 5. 在线角速度与方向估计

程序没有写死角速度，也没有使用预设轨迹代替估计。大跳变只有在位移范围、运动方向和轨道中心关系同时满足门限时才作为已确认换板事件。

每次相邻装甲板交接对应 120°。程序统计跨越 `physical_speed_handover_stride` 次交接的时间，用“交接次数 × 120° ÷ 时间间隔”计算角速度大小，并对最近 `physical_speed_window` 个结果取中位数。方向由连续相位滤波速度投票确定，交接槽位只用于计时，不直接作为连续相位的正负号。当前配置关闭角速度吸附，吸附目标值和容差均为零。

### 6. 未来 3 帧相位预测

当前框关联、漏检续跟和未来框共用同一个相位状态。交接间隔得到稳定角速度后使用物理角速度；尚未形成有效交接统计时使用 Kalman 角速度。未来相位为：

```text
future_phase = phase + angular_speed * prediction_lead_s
               + 0.5 * angular_acceleration * prediction_lead_s^2
```

程序按 `future_phase` 查询当前固定 ID 槽位的相位四边形，因此红色未来框的位置、透视收窄和倾斜来自同一相位模型。图像中心的 Theil–Sen 速度仅用于判断检测器的大跳变是否为真实装甲板交接，不再生成另一套未来框，也不使用退出/进入模板或长时间锁存。短时漏检期间继续推进相位；超过显示时限后停止绘制，超过丢失时限后重置。

### 7. 输出与重叠显示

观测通过门控时，对应绿色实线框使用本帧真实灯带四边形；另外两个绿色虚线框使用相位模型。观测被拒或短时漏检时，三个槽位都使用同一状态预测。红色未来框与绿色框高度重叠时仍保留，只切换为细线半透明样式；距离和 IoU 使用进入、退出两组阈值形成迟滞，避免样式闪烁。

CSV 逐帧记录候选是否存在、观测是否采用、三个槽位来源、相位、角速度、方向、跟踪状态、NIS、协方差、重投影误差、图像连续性、换板事件、未来目标、未来编号和渲染状态。

## 配置文件

`config/video.yaml` 集中保存以下可修改参数：

- 相机内参、畸变系数和参考分辨率；
- 装甲板尺寸、旋转半径和前哨站俯角；
- ROI、颜色阈值、灯带筛选、双灯带配对和四边形留量；
- 几何采样步长、RANSAC、相位分箱和循环平滑；
- 跟踪确认、短时显示、丢失超时和时间戳上限；
- Kalman 过程噪声、观测噪声、NIS 与像素关联门限；
- 物理角速度统计窗口和方向投票窗口；
- 预测提前时间、图像速度窗口和换板门限；
- 未来框与当前框的重叠显示阈值。

## 验证

`tests/test_main.cpp` 覆盖灯带四边形、灯带包含、不同倾角与长度、极窄和退化配对、单灯带拒绝、相位循环插值、三个槽位关系、编号连续性、NIS 门控、离群候选、短时丢失、时间戳复位、换板编号、重叠显示迟滞、标定模型版本与相机指纹以及 YAML 参数校验。

整段 925 帧视频回归中，923 帧检测到有效双灯带候选，595 帧观测通过状态门控；跟踪状态为 593 帧 `TRACKING`、330 帧 `TEMP_LOST` 和 2 帧 `DETECTING`。初始化后的 923 帧均连续绘制未来框，运动段方向为 `CW`，有效角速度中位数为 138.46°/s，范围为 135.00–150.00°/s。同一编号未来框的相邻帧位移中位数为 5.46 px，99% 不超过 19.78 px。所有单元测试均通过。
