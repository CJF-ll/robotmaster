# 作业三：相机标定

Python + OpenCV，使用真实摄像头采集棋盘格图片，输出标定参数、RMS 重投影误差，并显示原始与去畸变画面。

## 标定板

使用 https://www.cnblogs.com/gshang/p/18819764/calib_borad 生成 10 列 × 7 行棋盘格，每格 25 mm。OpenCV 检测参数是 9 列 × 6 行内角点。

按实际大小（100%）打印，不要选择适应纸张。板子含白边需要能放入纸张；可以使用 A4 横向。用尺量确认相邻角点间距，若实际格子边长不是 25 mm，请通过 `--square-mm` 填写实际值。纸张平整贴到硬板上，避免翘曲。

## 安装

在带桌面界面的虚拟机里运行，先将摄像头连接/转接给虚拟机。进入本目录后：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Linux 系统若提示缺少 venv，安装对应系统的 python3-venv。必须安装 `opencv-python`，不能使用不含窗口支持的 `opencv-python-headless`。

### 已有手机摄像头方案

之前的作业使用荣耀 200 Pro 上的 IP Webcam，Ubuntu 通过局域网 MJPEG 读取手机画面。该方式无需 USB 共享。本次地址是 `http://192.168.0.159:8080/video`，每次启动后以手机显示的当前 IP 为准。

手机和 Mac 连接同一 Wi-Fi，启动 IP Webcam 服务器后，在 Ubuntu 检查：

```bash
curl --max-time 5 -I http://192.168.0.159:8080
```

使用手机视频流采集和演示：

```bash
python camera_calibration.py collect --camera http://192.168.0.159:8080/video
python camera_calibration.py calibrate
python camera_calibration.py demo --camera http://192.168.0.159:8080/video --record output/demo.mp4
```

自动采集：加上 `--auto --target 25`。检测到全部角点、姿态稳定至少 0.8 秒且与已保存图片有足够位移后自动保存，两次保存至少间隔 2 秒。屏幕会提示保持稳定或改变位置，Q 可随时退出。仍需主动改变倾斜、位置和距离；自动去重不代表拍摄覆盖已经充分。

若换标定屏幕或改变显示缩放，建议用新的图片目录，例如 `--images images_screen2`，并在标定时使用同一目录。电子标定板需要保持显示比例一致，用尺量实际格子边长并通过 `--square-mm` 填入。

网络流分辨率在 IP Webcam 中设置，程序不会调整网络流尺寸。标定和演示保持同一镜头、分辨率、方向、缩放及焦距，避免自动切换镜头；能设置固定对焦时，选择合适的对焦距离后固定。移动标定板后等待画面稳定，再保存图片。

## 采集

```bash
python camera_calibration.py collect
```

窗口检测到棋盘格时显示角点。空格保存原始图片，Q 退出。建议采集 20–30 张，移动到中央、四角、边缘，改变距离和倾斜角度，保持完整棋盘格在画面内，静止清晰后按空格。默认 640 × 480；可用 `--width 1280 --height 720` 指定采集分辨率。实际摄像头支持的尺寸会在保存时输出。摄像头索引可用 `--camera 1` 等切换。

## 标定

```bash
python camera_calibration.py calibrate
```

结果为 `output/calibration.json`，包含内参、畸变系数、尺寸、每张图片误差及总体 RMS。RMS 定义为所有角点二维残差平方和除以角点总数，再开平方，单位为像素；要求严格小于 0.5。默认使用全部图片。可选 --robust 进行一次异常帧筛选：界限为每帧 RMS 中位数加 3 倍稳健标准差（1.4826 × MAD），与作业阈值无关。只筛选一次，原图全部保留，初始误差及排除记录写入 JSON。若不达标，检查高误差图片是否模糊、反光或板子变形，补拍并重新标定。低误差还需要结合姿态覆盖和实际去畸变效果判断。

## 演示及录像

```bash
python camera_calibration.py demo --record output/demo.mp4
```

左右显示原图与去畸变图，并标注真实 RMS。Q 结束保存视频。此 MP4 按固定帧率编码，播放速度取决于实际采集速度；如需准确保留运行时序，可用虚拟机桌面录屏。演示自动请求标定时的分辨率；不匹配则停止。相机、焦距、数字缩放和图像裁剪应与标定时保持一致。

去畸变保留完整视野，边缘可能出现黑边。可以把直边物体放到画面边缘演示效果；摄像头畸变较小时，两幅图变化可能不明显。

## 提交

GitHub 上传本程序、requirements.txt、README.md 和真实生成的 calibration.json；飞书提交运行效果视频。images 和 output 默认被忽略，若需提交参数，可显式运行 `git add -f output/calibration.json`。提交前确认 RMS 达标，视频能看到两个画面及误差。此仓库未预置或伪造标定结果。

## 本次实测

Ubuntu 22.04 ARM64，Python 3.10、OpenCV 4.5.4。手机视频 1920 × 1080；另一台屏幕显示棋盘，内角点 9 × 6。连续四格实测约 10.4 cm，单格采用 26 mm。

35 张全部拟合 RMS 为 0.504868 px。一次 MAD 筛选界限为 0.621923 px，第 25 张初始误差 0.734925 px，被标记为异常帧。原图保留，其余 34 张重新拟合 RMS 为 0.486025 px。该值是拟合数据的重投影误差，不是独立测试精度；屏幕条纹、反光及对焦变化会影响标定质量。

复现命令：

```bash
python3 camera_calibration.py calibrate --images images_screen2 --square-mm 26 --robust --params output/calibration.json
python3 camera_calibration.py demo --camera http://192.168.0.159:8080/video --params output/calibration.json --record output/demo.mp4 --duration 20
```

--duration 20 运行约 20 秒后保存退出；不加时按 Q 退出。演示使用后台读取最新帧，断流尝试重连。
