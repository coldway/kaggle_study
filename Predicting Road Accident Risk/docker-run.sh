#!/bin/bash
# Docker 自动检测 GPU/CPU 并运行训练脚本
# 注意: 在 Apple Silicon 上会自动切换到 Python 模式（不使用 Docker）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Road Accident Risk Prediction"
echo "自动检测 GPU/CPU/MPS 模式"
echo "=========================================="
echo ""

# 检测 GPU 支持
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
        return 0
    fi
    return 1
}

# 检测 Apple Silicon
check_apple_silicon() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            return 0
        fi
    fi
    return 1
}

# 保存原始参数
ORIGINAL_ARGS=("$@")

echo "正在检测系统 GPU 支持..."
USE_GPU=false

if check_gpu; then
    echo -e "${GREEN}✓ 检测到 NVIDIA GPU 支持${NC}"
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
        echo ""
    fi
    USE_GPU=true
    COMPOSE_FILE="docker-compose.yml"
    DOCKERFILE="Dockerfile.gpu"
elif check_apple_silicon; then
    echo -e "${GREEN}✓ 检测到 Apple Silicon (M4/M1/M2/M3)${NC}"
    echo ""
    echo -e "${YELLOW}自动切换到 Python 模式（不使用 Docker）${NC}"
    echo ""
    echo -e "${YELLOW}原因:${NC}"
    echo "  - MPS 需要 macOS 系统级别的 Metal API"
    echo "  - Docker 容器运行在 Linux 环境中，无法访问 macOS 的 Metal API"
    echo "  - 直接在 macOS 上运行 Python 脚本可以使用 MPS 加速"
    echo ""
    
    PYTHON_SCRIPT="road_accident_risk.py"
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo -e "${RED}错误: 找不到 Python 脚本 $PYTHON_SCRIPT${NC}"
        exit 1
    fi
    
    DATA_DIR="playground-series-s5e10"
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${YELLOW}警告: 数据目录 $DATA_DIR 不存在${NC}"
        echo "请确保数据目录存在并包含 train.csv 和 test.csv"
    fi
    
    mkdir -p output
    
    SKIP_INSTALL=false
    for arg in "${ORIGINAL_ARGS[@]}"; do
        case $arg in
            --skip-install)
                SKIP_INSTALL=true
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --build, -b      构建镜像后再运行（仅Docker模式）"
                echo "  --detached, -d   后台运行（仅Docker模式）"
                echo "  --cpu            强制使用 CPU 模式（仅Docker模式）"
                echo "  --gpu            强制使用 NVIDIA GPU 模式（仅Docker模式）"
                echo "  --skip-install   跳过依赖安装（Python模式）"
                echo "  --help, -h       显示此帮助信息"
                echo ""
                echo "注意: 在 Apple Silicon 上，将自动使用 Python 模式"
                exit 0
                ;;
        esac
    done
    
    PYTHON_CMD="python3 $PYTHON_SCRIPT --mode train --data-dir $DATA_DIR --output-dir output"
    if [ "$SKIP_INSTALL" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --skip-install"
    fi
    
    echo "=========================================="
    echo "使用模式: Python (Apple Silicon MPS)"
    echo "脚本: $PYTHON_SCRIPT"
    echo "数据目录: $DATA_DIR"
    echo "输出目录: output"
    echo "=========================================="
    echo ""
    echo "启动训练..."
    echo ""
    
    $PYTHON_CMD
    exit $?
else
    echo -e "${YELLOW}⚠ 未检测到 GPU 支持，将使用 CPU 模式${NC}"
    echo ""
    USE_GPU=false
    COMPOSE_FILE="docker-compose.cpu.yml"
    DOCKERFILE="Dockerfile"
fi

# 解析命令行参数
BUILD=false
DETACHED=false
FORCE_CPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b)
            BUILD=true
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --cpu)
            FORCE_CPU=true
            USE_GPU=false
            COMPOSE_FILE="docker-compose.cpu.yml"
            DOCKERFILE="Dockerfile"
            echo -e "${YELLOW}强制使用 CPU 模式${NC}"
            shift
            ;;
        --gpu)
            if check_gpu; then
                USE_GPU=true
                COMPOSE_FILE="docker-compose.yml"
                DOCKERFILE="Dockerfile.gpu"
                echo -e "${GREEN}强制使用 NVIDIA GPU 模式${NC}"
            else
                echo -e "${RED}错误: 系统不支持 NVIDIA GPU${NC}"
                exit 1
            fi
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --build, -b      构建镜像后再运行"
            echo "  --detached, -d   后台运行"
            echo "  --cpu            强制使用 CPU 模式"
            echo "  --gpu            强制使用 NVIDIA GPU 模式（需要系统支持）"
            echo "  --help, -h       显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必要的文件
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}错误: 找不到配置文件 $COMPOSE_FILE${NC}"
    exit 1
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}错误: 找不到 Dockerfile $DOCKERFILE${NC}"
    exit 1
fi

# 检查数据目录
if [ ! -d "playground-series-s5e10" ]; then
    echo -e "${YELLOW}警告: 数据目录 playground-series-s5e10 不存在${NC}"
    echo "请确保数据目录存在并包含 train.csv 和 test.csv"
fi

# 创建输出目录
mkdir -p output

# 构建命令
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"

if [ "$BUILD" = true ]; then
    echo ""
    echo "=========================================="
    echo "构建 Docker 镜像..."
    echo "=========================================="
    $COMPOSE_CMD build --no-cache
    echo ""
fi

# 运行命令
echo "=========================================="
if [ "$USE_GPU" = true ]; then
    echo -e "使用模式: ${GREEN}NVIDIA GPU (CUDA)${NC}"
else
    echo -e "使用模式: ${YELLOW}CPU${NC}"
fi
echo "配置文件: $COMPOSE_FILE"
echo "Dockerfile: $DOCKERFILE"
echo "=========================================="
echo ""

if [ "$DETACHED" = true ]; then
    echo "后台运行容器..."
    $COMPOSE_CMD up -d
    echo ""
    echo "容器已在后台运行"
    echo "使用以下命令查看日志:"
    echo "  $COMPOSE_CMD logs -f"
    echo ""
    echo "使用以下命令停止容器:"
    echo "  $COMPOSE_CMD down"
else
    echo "启动容器..."
    echo ""
    $COMPOSE_CMD up
fi

