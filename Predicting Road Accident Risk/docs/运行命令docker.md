# Road Accident Risk Prediction - Docker 运行命令

## 文件说明

已创建完整的 Python 脚本：`road_accident_risk_docker.py`

该脚本从 notebook `s5e10-tabm-tuned-further.ipynb` 改写而来，支持在 Docker 容器中运行，自动检测 GPU/CPU。

## 环境要求

### 1. Docker 环境

确保已安装 Docker 和 Docker Compose（如需要）。

### 2. 数据目录结构

确保项目目录结构如下：

```
.
├── road_accident_risk_docker.py
├── playground-series-s5e10/
│   ├── train.csv
│   ├── test.csv
│   └── synthetic_road_accidents_100k.csv (可选)
└── output/ (自动创建)
```

## Docker 运行方式

### 方式 1: 使用 GPU（推荐）

如果系统支持 GPU（NVIDIA GPU + CUDA）：

```bash
# 构建 Docker 镜像（包含所有依赖）
docker build -t road-accident-risk:latest -f Dockerfile .

# 运行容器（GPU 支持）
docker run --gpus all \
    -v $(pwd)/playground-series-s5e10:/app/playground-series-s5e10 \
    -v $(pwd)/output:/app/output \
    road-accident-risk:latest \
    python road_accident_risk_docker.py
```

### 方式 2: 仅使用 CPU

如果系统不支持 GPU 或想强制使用 CPU：

```bash
# 构建 Docker 镜像
docker build -t road-accident-risk:latest -f Dockerfile .

# 运行容器（CPU 模式）
docker run \
    -v $(pwd)/playground-series-s5e10:/app/playground-series-s5e10 \
    -v $(pwd)/output:/app/output \
    road-accident-risk:latest \
    python road_accident_risk_docker.py
```

### 方式 3: 使用 Docker Compose（推荐用于开发）

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  training:
    build: .
    image: road-accident-risk:latest
    volumes:
      - ./playground-series-s5e10:/app/playground-series-s5e10
      - ./output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python road_accident_risk_docker.py
```

运行：

```bash
# GPU 模式
docker-compose up

# CPU 模式（注释掉 deploy 部分）
docker-compose up
```

## Dockerfile 示例

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制代码文件
COPY road_accident_risk_docker.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    pytabkit \
    torch

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["python", "road_accident_risk_docker.py"]
```

## 本地运行（非 Docker）

如果不想使用 Docker，也可以直接在本地运行：

```bash
# 安装依赖
pip install pandas numpy scikit-learn pytabkit torch

# 运行脚本
python road_accident_risk_docker.py
```

## 命令行参数

### 基本参数

```bash
# 使用默认路径
python road_accident_risk_docker.py

# 指定数据目录
python road_accident_risk_docker.py --data-dir ./playground-series-s5e10

# 指定输出目录
python road_accident_risk_docker.py --output-dir ./output

# 跳过依赖安装（Docker 中通常不需要）
python road_accident_risk_docker.py --skip-install
```

### 完整参数示例

```bash
python road_accident_risk_docker.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output \
    --skip-install
```

## GPU/CPU 自动检测

脚本会自动检测可用的计算资源：

- **GPU 模式**：如果检测到 CUDA，自动使用 GPU 加速
- **CPU 模式**：如果没有 GPU 或 CUDA 不可用，自动切换到 CPU

检测信息会在运行时显示：

```
======================================================================
Environment Information (Docker)
======================================================================
python_version        : 3.9.18
pytorch_version       : 2.0.0
cuda_available        : True
cuda_device_count     : 1
----------------------------------------------------------------------
GPU Information:
  gpu_0_name          : NVIDIA GeForce RTX 3090
  gpu_0_memory        : 24.00 GB
======================================================================

*** GPU 检测成功: NVIDIA GeForce RTX 3090 (24.00 GB)
*** 将使用 GPU 加速训练
```

## 输出文件

运行完成后，会在输出目录（默认 `./output`）生成：

- `oof_tabm_plus_origcol_tuned.csv` - Out-of-Fold 预测结果（用于验证）
- `test_tabm_plus_origcol_tuned.csv` - 测试集预测结果（用于提交）

## 注意事项

### 1. 数据挂载

确保 Docker 容器可以访问数据目录：

```bash
# 检查数据目录是否存在
ls -la playground-series-s5e10/

# 确保文件权限正确
chmod -R 755 playground-series-s5e10/
```

### 2. GPU 支持

如果使用 GPU，需要：

- 安装 NVIDIA Docker 运行时（nvidia-docker2）
- 确保 NVIDIA 驱动已安装
- 确保 CUDA 版本兼容

检查 GPU 支持：

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 3. 内存和存储

- **内存**：建议至少 8GB RAM（GPU 模式可能需要更多）
- **存储**：确保有足够的磁盘空间存储输出文件

### 4. 原始数据文件

如果 `synthetic_road_accidents_100k.csv` 不存在，脚本会自动跳过基于原始数据的特征工程，但仍会使用其他特征。

## 故障排除

### Docker 构建失败

```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建
docker build --no-cache -t road-accident-risk:latest -f Dockerfile .
```

### GPU 不可用

```bash
# 检查 NVIDIA Docker 运行时
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 如果失败，安装 nvidia-docker2
# Ubuntu/Debian:
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### 数据文件找不到

```bash
# 检查挂载路径
docker run --rm -v $(pwd)/playground-series-s5e10:/app/playground-series-s5e10 \
    road-accident-risk:latest ls -la /app/playground-series-s5e10
```

### 权限问题

```bash
# 修复输出目录权限
sudo chown -R $USER:$USER output/
chmod -R 755 output/
```

## 性能优化建议

### GPU 模式

- 使用较大的 batch_size（如果 GPU 内存允许）
- 启用混合精度训练（allow_amp=True，如果支持）

### CPU 模式

- 减少 n_blocks 和 d_block 参数以加快训练
- 使用较小的 batch_size

## 示例输出

```
======================================================================
Environment Information (Docker)
======================================================================
python_version        : 3.9.18
pytorch_version       : 2.0.0
cuda_available        : True
cuda_device_count     : 1
----------------------------------------------------------------------
GPU Information:
  gpu_0_name          : NVIDIA GeForce RTX 3090
  gpu_0_memory        : 24.00 GB
======================================================================

*** Installing/checking dependencies...
  ✓ pytabkit already installed
  ✓ PyTorch with CUDA available

*** 开始执行训练和预测...
*** 数据目录: ./playground-series-s5e10
*** 输出目录: ./output

Train Shape: (517754, 14)
Test Shape: (172585, 13)
Orig Shape: (100000, 13)

12 Base Features:['road_type', 'num_lanes', ...]
12 Orig Features Created!!
25 Features.

*** GPU 检测成功: NVIDIA GeForce RTX 3090 (24.00 GB)
*** 将使用 GPU 加速训练

--- Fold 1/5 ---
  → 使用 GPU: NVIDIA GeForce RTX 3090
Fold 1 RMSE: 0.05606
--- Fold 2/5 ---
  → 使用 GPU: NVIDIA GeForce RTX 3090
Fold 2 RMSE: 0.05589
...
Overall OOF RMSE: 0.05590

*** 预测结果已保存到 ./output 目录
*** 训练和预测完成!
```

## 快速开始

```bash
# 1. 确保数据目录存在
ls playground-series-s5e10/train.csv

# 2. 构建 Docker 镜像
docker build -t road-accident-risk:latest -f Dockerfile .

# 3. 运行训练（GPU）
docker run --gpus all \
    -v $(pwd)/playground-series-s5e10:/app/playground-series-s5e10 \
    -v $(pwd)/output:/app/output \
    road-accident-risk:latest

# 4. 查看结果
ls output/
```

