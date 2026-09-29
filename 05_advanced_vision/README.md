# 进阶题 A：多线程视觉检测与目标跟踪

程序 `rm_vision_sim` 使用 C++17 和 OpenCV，从视频中检测蓝色目标，并用常速度模型进行平滑跟踪和短时遮挡预测。

## 一、流水线结构

```text
FrameSource
    │ BoundedQueue<ImageFrame>
    ▼
ColorDetector
    │ BoundedQueue<DetectionPacket>
    ▼
TargetTracker
    │ BoundedQueue<OutputPacket>
    ▼
ResultWriter
```

程序共启动 4 个工作线程。所有队列都有容量上限：队列满时生产者等待，队列空时消费者等待。视频结束或发生错误时，队列的 `close()` 会唤醒所有等待线程，最终由主线程正常 `join()`。

## 二、模块说明

- `FrameSource`：读取视频，为每帧生成连续帧号和单调递增时间戳。
- `ColorDetector`：BGR 转 HSV，按配置阈值分割蓝色区域，先开运算去除小噪点，再闭运算填补小孔；按面积和圆形度选择候选目标。
- `TargetTracker`：使用常速度 alpha-beta 滤波器，状态包含 `x、y、vx、vy`。检测有效时更新状态；短时漏检时输出预测位置；超出最大漏检帧数后使轨迹失效；距离门限会拒绝明显错误的检测。
- `ResultWriter`：输出逐帧 CSV 和带标注的视频。黄色圆表示检测，绿色十字表示滤波结果，红色十字表示预测结果。

## 三、安装依赖

Ubuntu 22.04：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

## 四、编译和运行

```bash
cd ~/rm_vision_sim
cmake -S . -B build
cmake --build build -j2
./build/generate_test_video data/test_video.mp4
./build/rm_vision_sim config/vision.conf
```

成功后会生成：

- `output/result.csv`：逐帧检测与跟踪结果；
- `output/result.mp4`：可视化视频。

可用以下命令检查：

```bash
head output/result.csv
xdg-open output/result.mp4
```

如果老师提供测试视频，将其复制为 `data/test_video.mp4`，或修改 `config/vision.conf` 的 `input_video` 即可。

## 五、配置参数

| 参数 | 含义 |
|---|---|
| `input_video` | 输入视频路径 |
| `output_video` | 可视化视频路径 |
| `output_csv` | CSV 结果路径 |
| `queue_capacity` | 每个有界队列的最大元素数 |
| `h/s/v_min/max` | HSV 蓝色阈值 |
| `min_area/max_area` | 合法轮廓面积范围 |
| `morphology_kernel` | 形态学核尺寸，偶数会自动调整为下一个奇数 |
| `max_missed_frames` | 最多连续预测帧数 |
| `max_match_distance` | 检测与预测的最大匹配距离（像素） |
| `tracker_alpha` | 位置残差修正比例 |
| `tracker_beta` | 速度残差修正比例 |

## 六、CSV 格式

```text
frame_id,timestamp_us,detected,predicted,measure_x,measure_y,track_x,track_y,vx,vy
```

没有检测或有效轨迹时，相应坐标字段为空。

## 七、算法分析题

### 1. 为什么 HSV 阈值在光照剧烈变化时仍可能失效？

HSV 将颜色与亮度部分分离，但并没有完全消除光照影响。强光会造成过曝，使饱和度下降；弱光会降低亮度并增加噪声；彩色光源还会改变色相。因此固定阈值仍可能漏检或误检。可通过自适应阈值、白平衡、颜色校正或学习型检测器改善。

### 2. 形态学开运算和闭运算分别解决什么问题？

开运算是先腐蚀再膨胀，用来删除孤立的小噪点；闭运算是先膨胀再腐蚀，用来填补目标内部的小孔并连接相邻的小裂缝。

### 3. 多个蓝色轮廓中如何选择真实目标？

先按面积范围排除过小噪声和过大背景，再结合轮廓圆形度打分，选择得分最高者。实际系统还可以加入与预测位置的距离、长宽比、颜色置信度和历史轨迹一致性。

### 4. 为什么直接使用检测中心会抖动？

图像噪声、阈值变化、轮廓边缘变化和像素量化都会使每帧中心产生小偏差。滤波器融合当前测量与历史运动状态，可以减少高频抖动。

### 5. 滤波器的预测和更新各做什么？

预测阶段根据上一时刻的位置、速度和时间间隔估计当前状态；更新阶段用新检测产生的残差修正预测位置和速度。alpha-beta 滤波器是常速度模型的一种简化状态估计方法。

### 6. 目标短暂消失时为什么还能输出预测位置？

跟踪器保留了目标最后的位置和速度。短暂没有测量时，可以用常速度模型外推位置。预测时间越长，误差通常越大，所以程序用 `max_missed_frames` 限制连续预测次数，超过后将轨迹设为无效。

## 八、异常处理

- 配置文件或输入视频不存在：打印 `Error:` 并返回非零退出码；
- 输入视频没有可读帧：报错并正常结束所有线程；
- 视频正常结束：上游关闭队列，关闭信号沿流水线传递，全部线程正常退出。

