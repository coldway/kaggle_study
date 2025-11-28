# Docker 与 Apple Silicon (M4) 使用说明

## 重要提示

**Apple Silicon MPS 在 Docker 容器中的限制**：

⚠️ **Apple Silicon 的 MPS (Metal Performance Shaders) 通常在 macOS 主机上直接运行，而不是在 Docker 容器中。**

这是因为：
1. Docker 容器无法直接访问 macOS 的 Metal API
2. MPS 需要 macOS 系统级别的支持
3. 在容器中，MPS 可能不可用，代码会自动回退到 CPU

## 推荐方案

### 方案 1: 直接在 macOS 上运行（推荐）⭐

对于 Apple Silicon (M4/M1/M2/M3)，**建议直接在 macOS 上运行 Python 脚本**，而不是使用 Docker：

```bash
# 安装依赖
pip install torch pandas numpy scikit-learn pytabkit

# 直接运行
python road_accident_risk_docker.py
```

**优势**：
- ✅ 可以使用 MPS GPU 加速
- ✅ 性能最佳
- ✅ 无需 Docker 开销

### 方案 2: 使用 Docker（CPU 模式）

如果必须使用 Docker，代码会自动检测 MPS，如果不可用会使用 CPU：

```bash
# 使用自动检测脚本
./docker-run.sh

# 或手动使用 CPU 模式
docker-compose -f docker-compose.cpu.yml up --build
```

**注意**：在 Docker 容器中，MPS 可能不可用，将使用 CPU。

### 方案 3: 使用 Docker MPS 模式（实验性）

我们提供了 `docker-compose.mps.yml` 和 `Dockerfile.mps`，但请注意：

```bash
# 使用 MPS 模式（可能仍会回退到 CPU）
docker-compose -f docker-compose.mps.yml up --build
```

## Docker 文件说明

### 1. Dockerfile.mps

专门为 Apple Silicon 创建的 Dockerfile：
- 安装支持 MPS 的 PyTorch
- 代码会自动检测 MPS 是否可用
- 如果 MPS 不可用，自动使用 CPU

### 2. docker-compose.mps.yml

Apple Silicon MPS 模式的 Docker Compose 配置：
- 使用 `Dockerfile.mps`
- 配置了 `platform: linux/arm64`
- 不包含 GPU 配置（MPS 不需要）

### 3. docker-run.sh

自动检测脚本已更新：
- 自动检测 Apple Silicon
- 如果检测到 Apple Silicon，会提示建议直接运行
- 支持 `--mps` 参数强制使用 MPS 模式

## 使用示例

### 自动检测（推荐）

```bash
./docker-run.sh
```

脚本会：
1. 检测是否是 Apple Silicon
2. 如果是，提示建议直接运行
3. 如果仍要使用 Docker，使用 MPS 配置

### 强制使用 MPS 模式

```bash
./docker-run.sh --mps --build
```

### 强制使用 CPU 模式

```bash
./docker-run.sh --cpu --build
```

## 验证 MPS 是否可用

在容器中运行后，查看输出：

```
*** GPU 检测成功 (Apple Silicon MPS): Apple M4
*** 将使用 Apple Silicon GPU 加速训练
```

如果看到：
```
*** 未检测到 GPU，将使用 CPU 训练
```

说明 MPS 在容器中不可用，已自动使用 CPU。

## 性能对比

### 直接在 macOS 上运行（使用 MPS）
- ✅ GPU 加速可用
- ✅ 性能最佳
- ✅ 训练速度快

### 在 Docker 容器中运行
- ⚠️ MPS 可能不可用
- ⚠️ 使用 CPU（但 M4 的 CPU 性能也很强）
- ⚠️ 有 Docker 开销

## 常见问题

### Q: 为什么 Docker 容器中 MPS 不可用？

A: Docker 容器无法直接访问 macOS 的 Metal API。MPS 需要系统级别的支持，这在容器中通常不可用。

### Q: 如何在 Docker 中使用 GPU？

A: 对于 Apple Silicon，建议直接在 macOS 上运行。如果必须使用 Docker，代码会自动使用 CPU。

### Q: MPS 和 CUDA 有什么区别？

A:
- **CUDA**: NVIDIA GPU 使用的并行计算平台（可在 Docker 中使用）
- **MPS**: Apple Silicon 使用的 Metal Performance Shaders（通常在主机上运行）

### Q: 如何确认是否使用了 GPU？

A:
1. 查看代码输出中的设备信息
2. 使用 Activity Monitor 查看 GPU 使用情况
3. 观察训练速度（GPU 应该明显更快）

## 总结

✅ **推荐**：对于 Apple Silicon，直接在 macOS 上运行 Python 脚本

⚠️ **限制**：Docker 容器中 MPS 可能不可用，会使用 CPU

💡 **建议**：
- 如果追求最佳性能，直接在 macOS 上运行
- 如果需要环境隔离，可以使用 Docker（但会使用 CPU）
- M4 的 CPU 性能也很强，CPU 模式也能提供不错的性能

