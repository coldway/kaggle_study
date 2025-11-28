# Docker 自动检测使用指南

## 为什么需要自动检测？

Docker Compose 本身不支持条件判断，无法自动检测系统是否支持 GPU。因此，我们提供了自动检测脚本，让使用更加便捷。

## 解决方案

### 方案 1: 使用自动检测脚本（推荐）⭐

我们提供了智能脚本，自动检测系统是否支持 GPU，然后选择相应的配置：

**Linux/macOS:**
```bash
./docker-run.sh
```

**Windows:**
```cmd
docker-run.bat
```

### 方案 2: 手动选择配置

如果你知道系统配置，可以手动选择：

- **GPU 模式**: `docker-compose up`
- **CPU 模式**: `docker-compose -f docker-compose.cpu.yml up`

## 自动检测脚本功能

### 自动检测
- ✅ 检测 `nvidia-smi` 是否可用
- ✅ 检测 Docker GPU 运行时是否可用
- ✅ 显示 GPU 信息（如果可用）

### 自动选择
- ✅ 检测到 GPU → 使用 `docker-compose.yml` + `Dockerfile.gpu`
- ✅ 未检测到 GPU → 使用 `docker-compose.cpu.yml` + `Dockerfile`

### 手动覆盖
- `--cpu`: 强制使用 CPU 模式
- `--gpu`: 强制使用 GPU 模式（需要系统支持）
- `--build`: 构建镜像后再运行
- `--detached`: 后台运行

## 使用示例

### 最简单的使用方式

```bash
# Linux/macOS
./docker-run.sh

# Windows
docker-run.bat
```

脚本会自动：
1. 检测系统是否支持 GPU
2. 选择正确的配置文件
3. 构建并运行容器

### 带参数的使用

```bash
# 构建并运行
./docker-run.sh --build

# 后台运行
./docker-run.sh --detached

# 强制使用 CPU
./docker-run.sh --cpu

# 强制使用 GPU（需要系统支持）
./docker-run.sh --gpu
```

## 为什么 Docker Compose 不能自动检测？

Docker Compose 的限制：
1. **不支持条件逻辑**: Docker Compose YAML 文件不支持 if/else 等条件判断
2. **构建时确定**: Dockerfile 在构建时就需要确定，无法在运行时动态选择
3. **GPU 检测需要运行时**: GPU 检测需要在运行时进行，而 Docker Compose 配置是静态的

## 我们的解决方案

通过创建智能启动脚本，我们实现了：
- ✅ **自动检测**: 在运行前检测系统配置
- ✅ **自动选择**: 根据检测结果选择正确的配置
- ✅ **用户友好**: 无需手动判断，一键运行
- ✅ **灵活覆盖**: 支持手动指定模式

## 技术实现

### 检测逻辑

```bash
# 检测 nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        # GPU 可用
    fi
fi

# 检测 Docker GPU 支持
if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    # GPU 可用
fi
```

### 配置选择

```bash
if GPU_AVAILABLE; then
    COMPOSE_FILE="docker-compose.yml"
    DOCKERFILE="Dockerfile.gpu"
else
    COMPOSE_FILE="docker-compose.cpu.yml"
    DOCKERFILE="Dockerfile"
fi
```

## 总结

虽然 Docker Compose 本身不支持自动检测，但通过智能脚本，我们实现了：
- 🎯 **自动化**: 无需手动判断
- 🚀 **便捷性**: 一键运行
- 🔧 **灵活性**: 支持手动覆盖
- 📊 **信息透明**: 显示检测结果和 GPU 信息

**推荐使用自动检测脚本，让 Docker 使用更加简单！**

