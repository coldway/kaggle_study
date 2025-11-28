FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制代码文件
COPY road_accident_risk_docker.py .

# 安装 PyTorch
# ⚠️ 重要：MPS 在 Docker 容器中不可用！
# MPS 需要 macOS 系统级别的 Metal API 支持
# Docker 容器运行在 Linux 环境中，无法访问 macOS 的 Metal API
# 因此，即使安装了支持 MPS 的 PyTorch，在容器中 MPS 也永远不可用
# 代码会自动检测并回退到 CPU 模式
# 
# 对于 Apple Silicon，强烈建议直接在 macOS 上运行 Python 脚本，而不是使用 Docker
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# 安装其他依赖
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    pytabkit

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ⚠️ 重要说明：Apple Silicon MPS 在 Docker 容器中不可用！
# 
# 技术原因：
# 1. MPS 需要 macOS 系统级别的 Metal API 支持
# 2. Docker 容器运行在 Linux 环境中（即使是在 macOS 上）
# 3. Metal API 是 macOS 专有的，无法在 Linux 容器中访问
# 4. 因此，在容器中 MPS 永远不可用，代码会自动使用 CPU
#
# 推荐方案：
# 对于 Apple Silicon，强烈建议直接在 macOS 上运行：
#   python road_accident_risk_docker.py
# 
# 这样可以：
# - ✅ 使用 MPS GPU 加速
# - ✅ 获得最佳性能
# - ✅ 无需 Docker 开销

# 默认命令
CMD ["python", "road_accident_risk_docker.py"]

