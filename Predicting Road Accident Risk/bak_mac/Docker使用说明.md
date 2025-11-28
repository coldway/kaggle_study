# Docker 使用说明

## 文件说明

本项目包含以下 Docker 配置文件：

- `Dockerfile` - CPU 版本的 Dockerfile（默认安装 CPU 版本的 PyTorch）
- `Dockerfile.gpu` - GPU 版本的 Dockerfile（安装支持 CUDA 的 PyTorch，用于 NVIDIA GPU）
- `Dockerfile.mps` - Apple Silicon MPS 版本的 Dockerfile（支持 MPS，用于 Apple M4/M1/M2/M3）
- `docker-compose.yml` - GPU 模式的 Docker Compose 配置（NVIDIA CUDA）
- `docker-compose.cpu.yml` - CPU 模式的 Docker Compose 配置
- `docker-compose.mps.yml` - Apple Silicon MPS 模式的 Docker Compose 配置
- `.dockerignore` - Docker 构建时忽略的文件
- `docker-run.sh` - **自动检测脚本（Linux/macOS）** - 自动检测 GPU/CPU/MPS 并运行
- `docker-run.bat` - **自动检测脚本（Windows）** - 自动检测 GPU/CPU 并运行

**重要提示**：
- 对于 Apple Silicon (M4/M1/M2/M3)，脚本会自动检测并切换到 Python 模式
- 直接运行 `road_accident_risk_mac.py` 进行训练，可以使用 MPS 加速
- 训练结果和模型保存方式与 Docker 模式完全一致
- 详见 `Docker_Apple_Silicon_说明.md`

## 🚀 快速开始（推荐：自动检测）

### 方式 1: 自动检测 GPU/CPU/MPS（最简单，推荐）

**Linux/macOS:**
```bash
# 自动检测并运行（推荐）
./docker-run.sh

# 构建并运行
./docker-run.sh --build

# 后台运行
./docker-run.sh --detached

# 强制使用 CPU 模式
./docker-run.sh --cpu

# 强制使用 NVIDIA GPU 模式（需要系统支持）
./docker-run.sh --gpu

# 强制使用 Apple Silicon MPS 模式（需要 Apple Silicon）
./docker-run.sh --mps
```

**注意**：对于 Apple Silicon，脚本会自动检测并提示建议直接运行 Python 脚本。

**Windows:**
```cmd
REM 自动检测并运行（推荐）
docker-run.bat

REM 构建并运行
docker-run.bat --build

REM 后台运行
docker-run.bat --detached

REM 强制使用 CPU 模式
docker-run.bat --cpu
```

**自动检测脚本的优势：**
- ✅ 自动检测系统是否支持 GPU（NVIDIA CUDA）
- ✅ **自动检测 Apple Silicon，并切换到 Python 模式（不使用 Docker）**
- ✅ 自动选择正确的配置文件或执行方式
- ✅ 无需手动判断和选择
- ✅ 显示 GPU 信息（如果可用）
- ✅ 支持强制指定模式（GPU/CPU/MPS）

**重要：Apple Silicon 自动切换**
- 当检测到 Apple Silicon (M4/M1/M2/M3) 时，脚本会自动切换到 Python 模式
- 直接运行 `road_accident_risk_mac.py` 进行训练，可以使用 MPS 加速
- 不会使用 Docker（因为 Docker 容器中 MPS 不可用）
- 训练结果和模型保存方式与 Docker 模式完全一致

### 方式 2: 手动选择（传统方式）

#### 使用 GPU（如果系统支持）

```bash
# 构建并运行（GPU 模式）
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

#### 使用 CPU

```bash
# 构建并运行（CPU 模式）
docker-compose -f docker-compose.cpu.yml up --build

# 后台运行
docker-compose -f docker-compose.cpu.yml up -d --build

# 查看日志
docker-compose -f docker-compose.cpu.yml logs -f

# 停止
docker-compose -f docker-compose.cpu.yml down
```

## 详细说明

### 1. 自动检测脚本工作原理

`docker-run.sh` (Linux/macOS) 和 `docker-run.bat` (Windows) 脚本会：

1. **自动检测 GPU 支持**
   - 检查 `nvidia-smi` 命令是否可用
   - 检查 Docker GPU 运行时是否可用
   - 显示 GPU 信息（如果可用）

2. **自动检测 Apple Silicon**
   - 检查系统架构（ARM64）和操作系统（macOS）
   - **如果检测到 Apple Silicon：自动切换到 Python 模式**
   - 直接运行 `road_accident_risk_mac.py` 进行训练
   - 可以使用 MPS 加速，性能优于 Docker CPU 模式

3. **自动选择配置或执行方式**
   - 如果检测到 NVIDIA GPU：使用 `docker-compose.yml` 和 `Dockerfile.gpu`
   - 如果检测到 Apple Silicon：**切换到 Python 模式，不使用 Docker**
   - 如果未检测到 GPU：使用 `docker-compose.cpu.yml` 和 `Dockerfile`

4. **支持手动覆盖**
   - `--cpu` 参数：强制使用 CPU 模式（仅 Docker）
   - `--gpu` 参数：强制使用 GPU 模式（需要系统支持）
   - `--mps` 参数：强制使用 Apple Silicon MPS 模式（自动切换到 Python）

### 2. 检查 GPU 支持（手动检查）

如果你想手动检查系统是否支持 GPU：

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

如果上述命令失败，请使用 CPU 模式或使用自动检测脚本。

### 2. 构建镜像

#### GPU 模式构建

```bash
# 使用 docker-compose 构建
docker-compose build

# 或直接使用 docker 构建
docker build -f Dockerfile.gpu -t road-accident-risk:latest .
```

#### CPU 模式构建

```bash
# 使用 docker-compose 构建
docker-compose -f docker-compose.cpu.yml build

# 或直接使用 docker 构建
docker build -f Dockerfile -t road-accident-risk:latest .
```

### 3. 运行容器

#### GPU 模式运行

```bash
# 前台运行（可以看到实时输出）
docker-compose up

# 后台运行
docker-compose up -d

# 运行并重新构建
docker-compose up --build
```

#### CPU 模式运行

```bash
# 前台运行
docker-compose -f docker-compose.cpu.yml up

# 后台运行
docker-compose -f docker-compose.cpu.yml up -d

# 运行并重新构建
docker-compose -f docker-compose.cpu.yml up --build
```

### 4. 查看日志

```bash
# GPU 模式日志
docker-compose logs -f

# CPU 模式日志
docker-compose -f docker-compose.cpu.yml logs -f

# 查看最后 100 行日志
docker-compose logs --tail=100
```

### 5. 进入容器

```bash
# GPU 模式容器
docker-compose exec training bash

# CPU 模式容器
docker-compose -f docker-compose.cpu.yml exec training bash
```

### 6. 清理

```bash
# 停止并删除容器
docker-compose down

# 停止并删除容器和镜像
docker-compose down --rmi all

# 清理所有未使用的资源
docker system prune -a
```

## 配置说明

### docker-compose.yml (GPU 模式)

- 使用 `Dockerfile.gpu` 构建镜像
- 配置了 GPU 支持（`deploy.resources.reservations.devices`）
- 挂载数据目录和输出目录

### docker-compose.cpu.yml (CPU 模式)

- 使用 `Dockerfile` 构建镜像（CPU 版本）
- 不包含 GPU 配置
- 设置 `CUDA_VISIBLE_DEVICES=` 强制使用 CPU

### 数据目录挂载

- `./playground-series-s5e10:/app/playground-series-s5e10:ro` - 数据目录（只读）
- `./output:/app/output` - 输出目录（可写）

### 环境变量

- `PYTHONUNBUFFERED=1` - 确保 Python 输出实时显示
- `CUDA_VISIBLE_DEVICES=` - 在 CPU 模式中强制禁用 GPU

## 常见问题

### 1. GPU 不可用

**问题**：运行 GPU 模式时出现错误

**解决方案**：
- 使用 CPU 模式：`docker-compose -f docker-compose.cpu.yml up`
- 检查 NVIDIA Docker 运行时是否安装：`docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi`

### 2. 权限问题

**问题**：无法写入输出目录

**解决方案**：
```bash
# 修复输出目录权限
chmod -R 755 output/
sudo chown -R $USER:$USER output/
```

### 3. 数据文件找不到

**问题**：容器内找不到数据文件

**解决方案**：
- 确保数据目录存在：`ls -la playground-series-s5e10/`
- 检查挂载路径是否正确
- 确保文件权限正确：`chmod -R 755 playground-series-s5e10/`

### 4. 构建失败

**问题**：Docker 构建时出错

**解决方案**：
```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建（不使用缓存）
docker-compose build --no-cache
```

### 5. PyTorch CUDA 版本不匹配

**问题**：GPU 模式下 PyTorch 无法使用 CUDA

**解决方案**：
- 检查 CUDA 版本：`nvidia-smi`
- 修改 `Dockerfile.gpu` 中的 PyTorch 安装命令，选择匹配的 CUDA 版本
- 常见版本：
  - CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
  - CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

## 性能优化建议

### GPU 模式

- 确保有足够的 GPU 显存（建议至少 8GB）
- 如果显存不足，可以在代码中减小 batch_size

### CPU 模式

- 使用多核 CPU 可以加快训练速度
- 考虑减少模型复杂度（n_blocks, d_block）以加快训练

## 示例命令

### 完整流程（自动检测，推荐）

```bash
# Linux/macOS
# 1. 自动检测并运行（最简单）
./docker-run.sh --build

# 2. 查看结果
ls -la output/
```

**Apple Silicon 用户（自动切换）**：
```bash
# 在 Apple Silicon (M1/M2/M3/M4) 上运行
./docker-run.sh

# 脚本会自动检测到 Apple Silicon，并切换到 Python 模式
# 直接运行 road_accident_risk_mac.py，可以使用 MPS 加速
# 训练结果保存在 output/ 目录，与 Docker 模式完全一致
```

```cmd
REM Windows
REM 1. 自动检测并运行（最简单）
docker-run.bat --build

REM 2. 查看结果
dir output
```

### 完整流程（GPU 模式，手动）

```bash
# 1. 检查 GPU
nvidia-smi

# 2. 构建镜像
docker-compose build

# 3. 运行训练
docker-compose up

# 4. 查看结果
ls -la output/
```

### 完整流程（CPU 模式，手动）

```bash
# 1. 构建镜像
docker-compose -f docker-compose.cpu.yml build

# 2. 运行训练
docker-compose -f docker-compose.cpu.yml up

# 3. 查看结果
ls -la output/
```

### 后台运行并查看日志

```bash
# GPU 模式
docker-compose up -d
docker-compose logs -f

# CPU 模式
docker-compose -f docker-compose.cpu.yml up -d
docker-compose -f docker-compose.cpu.yml logs -f
```

## 注意事项

1. **数据目录**：确保 `playground-series-s5e10` 目录存在并包含必要的数据文件
2. **输出目录**：`output` 目录会自动创建，但确保有写入权限
3. **GPU 驱动**：使用 GPU 模式需要安装 NVIDIA 驱动和 nvidia-docker2
4. **资源使用**：训练过程可能需要较长时间和大量资源，请确保系统有足够的资源

