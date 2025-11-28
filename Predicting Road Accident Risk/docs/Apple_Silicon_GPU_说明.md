# Apple Silicon (M4) GPU 使用说明

## 问题原因

Apple M4 芯片使用的是 **Metal Performance Shaders (MPS)** 而不是 CUDA。之前的代码只检测了 CUDA，没有检测 MPS，导致 Apple Silicon 的 GPU 无法被使用。

## 已修复的问题

✅ **已更新代码以支持 MPS 检测**

代码现在会：
1. 优先检测 CUDA (NVIDIA GPU)
2. 其次检测 MPS (Apple Silicon)
3. 最后使用 CPU

## 重要注意事项

### 1. pytabkit 可能不支持 MPS

**关键问题**：`pytabkit` 库可能不完全支持 Apple Silicon 的 MPS backend。即使代码检测到了 MPS，`TabM_D_Regressor` 可能仍然会使用 CPU。

### 2. 如何检查是否使用了 GPU

运行训练时，查看输出信息：

```bash
python road_accident_risk_docker.py
```

如果看到：
```
*** GPU 检测成功 (Apple Silicon MPS): Apple M4
*** 将使用 Apple Silicon GPU 加速训练
```

说明代码检测到了 MPS，但实际是否使用 GPU 取决于 `pytabkit` 的支持。

### 3. 检查 GPU 使用情况

在 macOS 上，可以使用 Activity Monitor 或命令行工具检查 GPU 使用：

```bash
# 使用 Activity Monitor 查看 GPU 使用情况
# 或者使用命令行
sudo powermetrics --samplers gpu_power -i 1000
```

### 4. 验证 pytabkit 是否支持 MPS

可以创建一个简单的测试脚本：

```python
import torch
from pytabkit import TabM_D_Regressor
import pandas as pd
import numpy as np

# 检查 MPS 是否可用
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

# 创建测试数据
X_train = pd.DataFrame(np.random.randn(100, 10))
y_train = pd.Series(np.random.randn(100))

# 尝试使用 MPS
try:
    model = TabM_D_Regressor(device='mps')
    model.fit(X_train, y_train, X_train, y_train)
    print("✓ pytabkit 支持 MPS")
except Exception as e:
    print(f"✗ pytabkit 不支持 MPS: {e}")
    print("将使用 CPU 模式")
```

## 解决方案

### 方案 1: 检查 pytabkit 版本和 MPS 支持

```bash
# 检查 pytabkit 版本
pip show pytabkit

# 查看是否有更新版本支持 MPS
pip search pytabkit  # 或访问 PyPI
```

### 方案 2: 使用 CPU 模式（如果 MPS 不支持）

如果 `pytabkit` 不支持 MPS，可以强制使用 CPU：

```python
# 在代码中强制使用 CPU
device = 'cpu'
```

### 方案 3: 等待 pytabkit 更新

`pytabkit` 可能在未来版本中添加 MPS 支持。可以：
- 关注 `pytabkit` 的 GitHub 仓库
- 查看是否有 MPS 支持的计划
- 考虑提交 feature request

## 代码更新说明

### 更新的功能

1. **MPS 检测**
   - 自动检测 Apple Silicon MPS 是否可用
   - 在环境信息中显示 MPS 状态

2. **设备选择逻辑**
   - 优先使用 CUDA (NVIDIA GPU)
   - 其次使用 MPS (Apple Silicon)
   - 最后使用 CPU

3. **信息显示**
   - 显示检测到的设备类型
   - 显示设备名称和相关信息

### 代码示例

```python
# 检测设备
if torch.cuda.is_available():
    device = 'cuda'  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = 'mps'   # Apple Silicon
else:
    device = 'cpu'   # CPU
```

## 性能建议

### 如果 MPS 不可用或不被支持

1. **使用 CPU 模式**
   - Apple M4 的 CPU 性能也很强
   - 对于表格数据，CPU 性能可能足够

2. **优化训练参数**
   - 减小 batch_size
   - 减少模型复杂度
   - 使用更少的折数进行交叉验证

3. **考虑使用其他库**
   - 如果 `pytabkit` 不支持 MPS，可以考虑其他支持 MPS 的表格数据模型库

## 验证步骤

1. **运行训练脚本**
   ```bash
   python road_accident_risk_docker.py
   ```

2. **查看输出信息**
   - 检查是否检测到 MPS
   - 查看设备使用情况

3. **监控 GPU 使用**
   - 使用 Activity Monitor
   - 或使用命令行工具

4. **检查训练速度**
   - 如果使用 GPU，训练速度应该明显加快
   - 如果速度没有提升，可能仍在使用 CPU

## 常见问题

### Q: 为什么检测到了 MPS 但 GPU 没有被使用？

A: 这可能是因为 `pytabkit` 库不支持 MPS backend。即使 PyTorch 支持 MPS，具体的模型库可能不支持。

### Q: 如何确认是否使用了 GPU？

A: 
1. 查看 Activity Monitor 中的 GPU 使用情况
2. 检查训练速度（GPU 应该明显更快）
3. 查看代码输出中的设备信息

### Q: 可以强制使用 CPU 吗？

A: 可以，在代码中设置 `device = 'cpu'` 或修改参数。

### Q: MPS 和 CUDA 有什么区别？

A:
- **CUDA**: NVIDIA GPU 使用的并行计算平台
- **MPS**: Apple Silicon 使用的 Metal Performance Shaders
- 两者都是 GPU 加速，但使用不同的 API

## 总结

✅ **代码已更新**：现在支持检测和使用 Apple Silicon MPS

⚠️ **限制**：`pytabkit` 可能不完全支持 MPS，可能需要使用 CPU 模式

💡 **建议**：
- 先运行代码查看是否检测到 MPS
- 检查 GPU 使用情况
- 如果 MPS 不被支持，使用 CPU 模式（M4 的 CPU 性能也很强）

