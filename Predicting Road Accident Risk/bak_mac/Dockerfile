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

# 安装 Python 依赖
# 先安装 PyTorch（根据是否有 GPU 选择版本）
# 这里安装 CPU 版本，如果需要 GPU 版本，可以在运行时使用不同的 Dockerfile
# 注意：此版本不支持 CUDA 或 MPS，仅支持 CPU
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    pytabkit

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 默认命令
CMD ["python", "road_accident_risk_docker.py"]

