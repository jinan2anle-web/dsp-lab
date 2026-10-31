# 实验一第三周：成员1工作区

本目录在第二周基准上补做三类实验：预处理链消融、时域特征组合消融、DTW约束带调参与重复分层划分稳健性验证。

最小验证：

```powershell
C:\ProgramData\anaconda3\python.exe run_week3_experiments.py --smoke
```

正式复现：

```powershell
C:\ProgramData\anaconda3\python.exe run_week3_experiments.py --full
```

固定基准为去直流、双门限端点检测、25 ms帧长、10 ms帧移、Hamming窗。所有模型参数只在训练数据内选择；固定测试集结果仅用于描述性比较。

完整交接内容见 `第一部分交接说明.md`。
