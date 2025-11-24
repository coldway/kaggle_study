# Road Accident Risk Prediction 运行命令 (Windows 10)

## 系统配置

- **操作系统**: Windows 10
- **显卡**: RTX 5070 12GB
- **CPU**: Intel Core i5-12600KF
- **Python**: 建议使用 Python 3.9 或 3.10

## 文件说明

已创建专门针对 Windows 10 优化的 Python 脚本：`road_accident_risk_win.py`

该脚本已针对 Windows 10 和 GPU 环境进行优化，支持自动检测和使用 RTX 5070 显卡。
**注意**：这是 Windows 专用版本，原版 `road_accident_risk_complete.py` 保持不变。

## 环境准备

### 1. 安装 Python

确保已安装 Python 3.9 或 3.10（推荐使用 Anaconda 或 Miniconda）

```powershell
# 检查 Python 版本
python --version
```

### 2. 安装 CUDA 和 PyTorch（GPU 支持）

**RTX 5070 需要 CUDA 11.8 或更高版本**

```powershell
# 安装 PyTorch with CUDA 11.8
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者使用 conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**验证 GPU 是否可用**：
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. 安装其他依赖

```powershell
# 安装基础依赖
python -m pip install pandas numpy scikit-learn

# 安装 pytabkit（TabM 模型库）
python -m pip install pytabkit

# 安装其他可能需要的依赖
python -m pip install tqdm
```

### 4. 数据目录

确保 `playground-series-s5e10` 目录存在并包含以下文件：
- `train.csv` - 训练数据（必需）
- `test.csv` - 测试数据（必需）
- `synthetic_road_accidents_100k.csv` - 原始数据（可选，用于特征工程）

**Windows 路径示例**：
```
C:\Users\YourName\kaggle-study\Predicting Road Accident Risk\
├── road_accident_risk_complete.py
├── playground-series-s5e10\
│   ├── train.csv
│   ├── test.csv
│   └── synthetic_road_accidents_100k.csv (可选)
└── docs\
```

## 运行命令

### 1. 基本运行（使用默认路径）

```powershell
python road_accident_risk_win.py
```

### 2. 指定数据目录

```powershell
python road_accident_risk_win.py --data-dir .\playground-series-s5e10
```

### 3. 指定输出目录

```powershell
python road_accident_risk_win.py --output-dir .\output
```

### 4. 完整参数示例

```powershell
python road_accident_risk_win.py `
    --data-dir .\playground-series-s5e10 `
    --output-dir .\output `
    --skip-install
```

**注意**：Windows PowerShell 使用反引号 `` ` `` 作为行继续符

### 5. 在 CMD 中运行

```cmd
python road_accident_risk_win.py --data-dir .\playground-series-s5e10 --output-dir .\output
```

## GPU 配置

### RTX 5070 12GB 优化建议

1. **自动检测**：脚本会自动检测 CUDA 并启用 GPU
2. **内存管理**：RTX 5070 有 12GB 显存，足够运行 TabM 模型
3. **批次大小**：脚本使用 `batch_size='auto'`，会自动优化

### 验证 GPU 使用

运行脚本后，查看输出中的环境信息：

```
============================================================
Environment Information:
============================================================
python_version: 3.10.0
platform: Windows
pytorch_version: 2.0.0+cu118
cuda_available: True
cuda_device_count: 1
gpu_0_name: NVIDIA GeForce RTX 5070
gpu_0_memory: 12.00 GB
============================================================
```

训练过程中会显示：
```
  → 使用 GPU: NVIDIA GeForce RTX 5070
```

## 输出文件

运行完成后，会在输出目录（默认 `.\output`）生成：

- `oof_tabm_plus_origcol_tuned.csv` - Out-of-Fold 预测结果（用于验证）
- `test_tabm_plus_origcol_tuned.csv` - 测试集预测结果（用于提交）

## Windows 特定注意事项

### 1. 路径分隔符

脚本已自动处理 Windows 路径（使用 `os.path.join`），但手动指定路径时注意：
- 使用反斜杠：`.\playground-series-s5e10`
- 或使用正斜杠：`./playground-series-s5e10`（Python 支持）

### 2. 编码问题

如果遇到编码错误，设置环境变量：

```powershell
$env:PYTHONIOENCODING="utf-8"
python road_accident_risk_complete.py
```

### 3. 长路径问题

如果路径太长，可能需要启用 Windows 长路径支持：
1. 打开组策略编辑器（gpedit.msc）
2. 导航到：计算机配置 > 管理模板 > 系统 > 文件系统
3. 启用"启用 Win32 长路径"

### 4. 权限问题

如果遇到权限错误，以管理员身份运行 PowerShell 或 CMD。

## 性能优化

### RTX 5070 12GB 配置建议

1. **批次大小**：脚本会自动优化，通常可以使用较大的批次
2. **混合精度**：当前配置 `allow_amp=False`，可以尝试启用以加速训练
3. **多进程**：Windows 上支持多进程，但 TabM 默认使用单进程

### 预期性能

- **训练时间**：5 折交叉验证，每折约 15-30 分钟（取决于数据大小）
- **GPU 利用率**：应该能看到较高的 GPU 利用率（使用 `nvidia-smi` 监控）
- **内存使用**：显存使用约 2-4GB（12GB 足够）

## 故障排除

### CUDA 不可用

**问题**：`cuda_available: False`

**解决方案**：
1. 检查 NVIDIA 驱动是否安装
   ```powershell
   nvidia-smi
   ```
2. 重新安装 PyTorch with CUDA
   ```powershell
   python -m pip uninstall torch torchvision torchaudio
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 依赖安装失败

**问题**：`pytabkit` 安装失败

**解决方案**：
```powershell
# 使用国内镜像源
python -m pip install pytabkit -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 路径错误

**问题**：找不到数据文件

**解决方案**：
- 使用绝对路径
- 检查路径中的空格和特殊字符
- 使用引号包裹路径

### 内存不足

**问题**：GPU 内存不足（OOM）

**解决方案**：
- 减小批次大小（修改 `params['batch_size']`）
- 使用 CPU 模式（移除 `device='cuda'` 参数）

## 监控 GPU 使用

在另一个 PowerShell 窗口中运行：

```powershell
# 实时监控 GPU
nvidia-smi -l 1
```

## 示例输出

```
============================================================
Environment Information:
============================================================
python_version: 3.10.0
platform: Windows
pytorch_version: 2.0.0+cu118
cuda_available: True
cuda_device_count: 1
gpu_0_name: NVIDIA GeForce RTX 5070
gpu_0_memory: 12.00 GB
============================================================

*** Installing/checking dependencies...
  ✓ pytabkit already installed

*** 开始执行训练和预测...
*** 数据目录: .\playground-series-s5e10
*** 输出目录: .\output

Train Shape: (517754, 14)
Test Shape: (172585, 13)
...

--- Fold 1/5 ---
  → 使用 GPU: NVIDIA GeForce RTX 5070
Fold 1 RMSE: 0.05606
...
Overall OOF RMSE: 0.05590

*** 预测结果已保存到 .\output\ 目录
```

## 快速开始

```powershell
# 1. 进入项目目录
cd "C:\Users\YourName\kaggle-study\Predicting Road Accident Risk"

# 2. 检查环境
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. 运行脚本（Windows 优化版本）
python road_accident_risk_win.py
```

## 性能对比

### CPU vs GPU（预期）

- **CPU (12600KF)**：每折约 60-90 分钟
- **GPU (RTX 5070)**：每折约 15-30 分钟
- **加速比**：约 3-4 倍

## 注意事项

1. **首次运行**：首次运行会下载模型权重，可能需要一些时间
2. **数据大小**：确保有足够的磁盘空间（至少 1GB 可用空间）
3. **电源管理**：确保电源设置为"高性能"模式以获得最佳性能
4. **温度监控**：长时间训练时注意 GPU 温度，确保散热良好

