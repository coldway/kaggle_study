# MPS 在 Docker 容器中不可用的技术原因

## 问题现象

从终端输出可以看到：
```
pytorch_version: 2.8.0+cpu
mps_available: False
*** 未检测到 GPU，将使用 CPU 训练
```

即使检测到 Apple Silicon，在 Docker 容器中 MPS 仍然不可用。

## 根本原因

### 技术限制

**MPS (Metal Performance Shaders) 在 Docker 容器中永远不可用**，这是技术限制，不是配置问题。

原因：

1. **MPS 需要 macOS 系统级别的 Metal API**
   - MPS 依赖于 macOS 的 Metal 框架
   - Metal 是 macOS/iOS 专有的图形和计算 API

2. **Docker 容器运行在 Linux 环境中**
   - 即使在 macOS 上运行 Docker，容器内部仍然是 Linux 环境
   - Docker Desktop for Mac 使用 Linux 虚拟机运行容器

3. **Metal API 无法在 Linux 中访问**
   - Metal API 是 macOS 系统级别的 API
   - Linux 容器无法访问 macOS 的系统 API
   - 这是架构层面的限制

### 架构对比

```
macOS 主机直接运行 Python:
  Python → PyTorch → MPS → Metal API → Apple Silicon GPU ✅

Docker 容器中运行:
  Python → PyTorch → MPS → ❌ (无法访问 Metal API) → 回退到 CPU
```

## 解决方案

### 方案 1: 直接在 macOS 上运行（强烈推荐）⭐

```bash
# 安装依赖
pip install torch pandas numpy scikit-learn pytabkit

# 直接运行（可以使用 MPS GPU）
python road_accident_risk_docker.py
```

**优势**：
- ✅ 可以使用 MPS GPU 加速
- ✅ 性能最佳
- ✅ 无需 Docker 开销
- ✅ 充分利用 Apple Silicon 的性能

### 方案 2: 使用 Docker（仅 CPU）

如果必须使用 Docker（例如需要环境隔离），代码会自动使用 CPU：

```bash
./docker-run.sh --mps
# 或
docker-compose -f docker-compose.mps.yml up
```

**注意**：
- ⚠️ MPS 在容器中不可用
- ⚠️ 将使用 CPU（但 M4 的 CPU 性能也很强）
- ⚠️ 有 Docker 开销

## 验证方法

### 在 macOS 上直接运行

```bash
python road_accident_risk_docker.py
```

应该看到：
```
mps_available: True
*** GPU 检测成功 (Apple Silicon MPS): Apple M4
*** 将使用 Apple Silicon GPU 加速训练
```

### 在 Docker 容器中运行

```bash
docker-compose -f docker-compose.mps.yml up
```

会看到：
```
mps_available: False
*** 未检测到 GPU，将使用 CPU 训练
```

## 为什么 Dockerfile.mps 存在？

`Dockerfile.mps` 和 `docker-compose.mps.yml` 是为了：
1. **完整性**：提供所有可能的配置选项
2. **一致性**：保持配置文件的统一结构
3. **未来兼容**：如果未来 Docker 支持 Metal API（可能性很小）

但实际上，在 Docker 容器中 MPS 永远不可用。

## 性能对比

### macOS 直接运行（使用 MPS）
- GPU 加速：✅ 可用
- 训练速度：🚀 最快
- 资源利用：✅ 充分利用 GPU

### Docker 容器运行（CPU）
- GPU 加速：❌ 不可用
- 训练速度：🐢 较慢（但仍可接受，M4 CPU 性能强）
- 资源利用：⚠️ 仅使用 CPU

## 总结

✅ **推荐**：对于 Apple Silicon，直接在 macOS 上运行 Python 脚本

❌ **限制**：Docker 容器中 MPS 永远不可用（技术限制）

💡 **建议**：
- 如果追求最佳性能 → 直接在 macOS 上运行
- 如果需要环境隔离 → 使用 Docker（但会使用 CPU）
- M4 的 CPU 性能也很强，CPU 模式也能提供不错的性能

## 常见问题

### Q: 为什么检测到 Apple Silicon 但 MPS 不可用？

A: 检测到的是**主机系统**的 Apple Silicon，但 Docker 容器运行在 Linux 环境中，无法访问 macOS 的 Metal API。

### Q: 可以修改 Docker 配置让 MPS 可用吗？

A: 不可以。这是架构层面的限制，不是配置问题。MPS 需要 macOS 系统级别的支持，而 Docker 容器无法访问。

### Q: 未来 Docker 会支持 MPS 吗？

A: 可能性很小。这需要 Docker 能够访问 macOS 的系统 API，这在架构上很困难。

### Q: 如何在 Docker 中使用 GPU？

A: 对于 Apple Silicon，无法在 Docker 中使用 GPU。建议直接在 macOS 上运行。

