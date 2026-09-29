# 数据包与配置格式

## 文本数据包

`camera.txt` 和 `imu.txt` 每行表示一个逻辑数据包：

```text
sequence timestamp_us
```

- `sequence`：非负 32 位包序号。
- `timestamp_us`：非负 64 位单调时间戳，单位为微秒。
- 单个文件内按 `timestamp_us` 非递减排列。
- 生产者读取哪个文件，就为该数据包赋予对应的 `CAMERA` 或 `IMU` 类型。
- 空行、格式错误、负数、字段缺失或多余字段均应报错，不得静默忽略。

公开样例：

```text
0 1000
1 2000
```

## 配置格式

配置文件使用每行一个 `key=value` 的格式，`#` 开头的行是注释：

```ini
queue_capacity=8
max_delta_us=300
camera_file=../data/camera.txt
imu_file=../data/imu.txt
output_file=../data/result.txt
```

所有键都必须存在，未知键可选择拒绝或忽略，但行为必须在提交说明中写清楚。

## 关于真实通信包

基础题使用文本包，目的是把评分集中在线程同步和时间戳算法上，因此不提供战队通信板二进制协议。

若出题人启用附加题，可另行提供固定二进制布局：

```text
magic:u16 | version:u8 | type:u8 | sequence:u32 | timestamp_us:u64 | crc16:u16
```

启用时必须同时给出字节序、CRC 参数、粘包/拆包规则、错误处理规则和至少一组编码解码测试向量。
