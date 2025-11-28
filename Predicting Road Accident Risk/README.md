# Road Accident Risk Prediction

道路事故风险预测项目，基于 TabM 模型的回归任务。

## 功能特性

- ✅ 支持 GPU/MPS/CPU 自动检测和训练
- ✅ 支持 Windows/macOS/Linux 本地运行
- ✅ 支持 Docker 运行（GPU/CPU 模式）
- ✅ 模型保存和加载
- ✅ 单次和批量预测
- ✅ HTTP 服务接口
- ✅ Apple Silicon 自动切换到 Python 模式（不使用 Docker）

## 简单可快速运行版本：
本代码可以在所有平台上进行训练模型

### 文件说明：
road_accident_risk.py      模型训练文件
api_server.py              启动http服务文件
convert_model_to_cpu.py    将mps模型文件转为cpu模型文件
check_model_device_type.py 检查模型文件类型
docker-run.sh              在docker内训练模型，mac下运行
docker-run.bat             在docker内训练模型，win下运行（自动选择gpu还是cpu训练）

### 训练模型（MPS）
```bash
python3 road_accident_risk.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output
#使用docker进行运行在mac上会转为cpu训练，mac上只需要运行 ./docker-run.sh即可
```
### 运行http服务

```bash
docker：docker-compose -f docker-compose.api.yml up --build
python：python3 api_server.py \
    --model-dir ./output/models \
    --host 0.0.0.0 \
    --port 6000 \
    --orig-data-path ./playground-series-s5e10/synthetic_road_accidents_100k.csv
#文档内可能有说需要转成cpu模型文件才能进行docker预测，其实不用，已经修复    
```

### 测试接口:
```bash
# 健康检查
curl http://localhost:5000/health

####  单次预测
curl -X POST http://localhost:6000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "road_type":"urban",
    "num_lanes":1,
    "curvature": 0.5,
    "lighting": "day",
    "weather": "clear",
    "road_signs_present": "yes",
    "public_road": "yes",
    "time_of_day": "morning",
    "holiday": "no",
    "school_season": "yes",
    "speed_limit": 50,
    "num_reported_accidents": 2
  }'

####  批量预测
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

### 模型文件类型检查
```bash
python3 check_model_device_type.py --model-dir ./output/models
```
### mps模型文件转换为cpu文件
```bash
python convert_model_to_cpu.py --model-dir ./output/models --script-dir ./output/models_cpu
```


## 快速开始

### 1. 本地运行（推荐：Apple Silicon）

#### macOS (Apple Silicon)

```bash
# 直接运行 Python 脚本（自动使用 MPS）
python3 road_accident_risk.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output
```

#### Linux/Windows

```bash
# 训练模型
python road_accident_risk.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output

# 如果支持 GPU，脚本会自动检测并使用
```

### 2. Docker 运行

#### 自动检测（推荐）

**Linux/macOS:**
```bash
# 自动检测 GPU/CPU 并运行
./docker-run.sh

# 构建并运行
./docker-run.sh --build

# 后台运行
./docker-run.sh --detached

# 强制使用 CPU 模式
./docker-run.sh --cpu
```

**Windows:**
```cmd
REM 自动检测并运行
docker-run.bat

REM 构建并运行
docker-run.bat --build

REM 后台运行
docker-run.bat --detached
```

**注意**: 在 Apple Silicon Mac 上，`docker-run.sh` 会自动切换到 Python 模式，不使用 Docker。

#### 手动运行

**GPU 模式:**
```bash
docker-compose up --build
```

**CPU 模式:**
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

## 使用说明

### 训练模型

```bash
python road_accident_risk.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output \
    --skip-install
```

训练完成后，模型会保存在 `./output/models/` 目录：
- `model_fold_1.pkl` 到 `model_fold_5.pkl`: 每个 fold 的模型
- `ensemble_model.pkl`: 集成模型包装器（推荐使用）
- `metadata.json`: 模型元数据

### 单次预测

创建输入文件 `input.json`:
```json
{
  "curvature": 0.5,
  "lighting": "day",
  "weather": "clear",
  "road_signs_present": "yes",
  "public_road": "yes",
  "time_of_day": "morning",
  "holiday": "no",
  "school_season": "yes",
  "speed_limit": 50,
  "num_reported_accidents": 2
}
```

运行预测:
```bash
python road_accident_risk.py \
    --mode predict \
    --model-dir ./output/models \
    --input input.json \
    --output prediction.json
```

### 批量预测

使用 CSV 文件:
```bash
python road_accident_risk.py \
    --mode predict \
    --model-dir ./output/models \
    --input input.csv \
    --output predictions.csv \
    --orig-data-path ./playground-series-s5e10/synthetic_road_accidents_100k.csv
```

### HTTP 服务

启动服务:
```bash
python3 api_server.py \
    --model-dir ./output/models \
    --host 0.0.0.0 \
    --port 6000 \
    --orig-data-path ./playground-series-s5e10/synthetic_road_accidents_100k.csv
```

**注意**: HTTP服务使用独立的 `api_server.py` 脚本，专门用于生产环境部署。

测试接口:
```bash
# 健康检查
curl http://localhost:5000/health

# 单次预测
curl -X POST http://localhost:6000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "road_type":"urban",
    "num_lanes":1,
    "curvature": 0.5,
    "lighting": "day",
    "weather": "clear",
    "road_signs_present": "yes",
    "public_road": "yes",
    "time_of_day": "morning",
    "holiday": "no",
    "school_season": "yes",
    "speed_limit": 50,
    "num_reported_accidents": 2
  }'

# 批量预测
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

### Docker HTTP 服务

#### 步骤 1: 转换 MPS 模型为 CPU 格式

```bash
# 将 Mac MPS 训练的模型转换为 CPU 格式（用于 Docker/Linux 部署）
python convert_model_to_cpu.py --model-dir ./output/models

# 不创建备份（如果已经备份过）
python convert_model_to_cpu.py --model-dir ./output/models --no-backup

python convert_model_to_cpu.py --model-dir ./output/models --script-dir ./output/models_cpu
```

#### 步骤 2: 启动 Docker 服务

```bash
# 启动服务
docker-compose -f docker-compose.api.yml up --build

# 查看日志
docker-compose -f docker-compose.api.yml logs -f

# 停止服务
docker-compose -f docker-compose.api.yml down
```

### Linux 原生部署（不使用 Docker）

如果 Docker 无法使用，可以在 Linux 系统上直接部署：

1. **在 Mac 上转换模型**：
   ```bash
   python convert_model_to_cpu.py --model-dir ./output/models
   ```

2. **传输文件到 Linux**：
   ```bash
   # 传输模型和代码
   scp -r ./output/models user@linux-server:/path/to/deployment/models
   scp road_accident_risk.py api_server.py user@linux-server:/path/to/deployment/
   ```

3. **在 Linux 上安装依赖**：
   ```bash
   # 安装 PyTorch CPU 版本
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # 安装其他依赖
   pip install pandas numpy scikit-learn pytabkit flask
   ```

4. **启动服务**：
   ```bash
   python api_server.py --model-dir ./models --host 0.0.0.0 --port 5000
   ```

详细部署指南请参考：[Mac_to_Linux_部署指南.md](docs/Mac_to_Linux_部署指南.md)

```bash
# 启动服务
docker-compose -f docker-compose.api.yml up --build

# 查看日志
docker-compose -f docker-compose.api.yml logs -f

# 停止服务
docker-compose -f docker-compose.api.yml down
```

## API 接口

### GET /health
健康检查

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
预测接口（支持单次和批量）

**请求体（单次）:**
```json
{
  "curvature": 0.5,
  "lighting": "day",
  ...
}
```

**响应（单次）:**
```json
{
  "prediction": 0.234
}
```

**请求体（批量）:**
```json
[{...}, {...}]
```

**响应（批量）:**
```json
{
  "predictions": [0.234, 0.567]
}
```

### POST /predict/batch
批量预测接口

### GET /model/info
获取模型信息

## 命令行参数

### 训练模式
- `--data-dir`: 数据目录路径（默认: `./playground-series-s5e10`）
- `--output-dir`: 输出目录（默认: `./output`）
- `--skip-install`: 跳过依赖安装

### 预测模式
- `--mode predict`: 预测模式
- `--model-dir`: 模型目录（默认: `./output/models`）
- `--input`: 输入文件路径（CSV 或 JSON）
- `--output`: 输出文件路径
- `--orig-data-path`: 原始数据文件路径（用于特征工程）

### 服务模式
HTTP服务请使用 `api_server.py` 脚本
- `--model-dir`: 模型目录
- `--host`: 服务主机地址（默认: `0.0.0.0`）
- `--port`: 服务端口（默认: `5000`）
- `--orig-data-path`: 原始数据文件路径

## 环境要求

### Python 依赖
- Python 3.9+
- pytabkit
- pandas
- numpy
- scikit-learn
- torch
- flask (HTTP 服务需要)

### Docker 要求
- Docker
- Docker Compose
- NVIDIA Docker (GPU 模式需要)

## 文件结构

```
.
├── road_accident_risk.py          # 主 Python 脚本
├── Dockerfile                      # CPU 版本 Dockerfile
├── Dockerfile.gpu                 # GPU 版本 Dockerfile
├── Dockerfile.api                 # HTTP 服务 Dockerfile
├── docker-compose.yml             # GPU 模式配置
├── docker-compose.cpu.yml         # CPU 模式配置
├── docker-compose.api.yml         # HTTP 服务配置
├── docker-run.sh                  # 自动检测脚本 (Linux/macOS)
├── docker-run.bat                 # 自动检测脚本 (Windows)
├── playground-series-s5e10/       # 数据目录
│   ├── train.csv
│   ├── test.csv
│   └── synthetic_road_accidents_100k.csv (可选)
└── output/                        # 输出目录
    ├── models/                    # 模型目录
    │   ├── model_fold_1.pkl
    │   ├── ...
    │   ├── ensemble_model.pkl
    │   └── metadata.json
    ├── oof_tabm_plus_origcol_tuned.csv
    └── test_tabm_plus_origcol_tuned.csv
```

## 注意事项

1. **Apple Silicon**: 在 Apple Silicon Mac 上，脚本会自动使用 Python 模式（MPS 加速），不使用 Docker
2. **原始数据**: 如果训练时使用了原始数据文件，预测时也需要提供该文件路径
3. **模型保存**: 训练时会自动保存所有 fold 的模型和集成模型包装器
4. **GPU 支持**: 脚本会自动检测 CUDA (NVIDIA) 和 MPS (Apple Silicon) 支持

## 故障排除

### 问题: 找不到数据文件
**解决**: 确保 `playground-series-s5e10` 目录存在并包含 `train.csv` 和 `test.csv`

### 问题: GPU 不可用
**解决**: 使用 CPU 模式或检查 GPU 驱动

### 问题: 端口被占用
**解决**: 使用 `--port` 参数指定其他端口

### 问题: Docker 构建失败
**解决**: 检查 Docker 版本和网络连接

### 问题: MPS 训练的模型无法在 Docker 中加载
**原因**: MPS (Metal Performance Shaders) 是 Apple Silicon 特有的，Docker 容器（Linux）不支持 MPS 设备。

**解决方案**:

1. **重新训练并保存为 CPU 模型**（推荐）:
   - 训练代码已更新，会自动将模型保存为 CPU 格式
   - 重新运行训练即可

2. **转换现有模型**:
   ```bash
   # 使用转换脚本将 MPS 模型转换为 CPU 模型
   python convert_model_to_cpu.py --model-dir ./output/models
   ```
   脚本会自动创建备份，然后转换所有模型文件。

3. **手动转换**:
   - 在 macOS 上加载模型
   - 将模型移动到 CPU: `model.to('cpu')`
   - 重新保存模型

**注意**: 从 v1.1 开始，训练代码会自动将模型保存为 CPU 格式，确保跨平台兼容性。

## 许可证

本项目基于 Kaggle 竞赛代码改写。


简单版：

本地训练（推荐）
```bash
python3 road_accident_risk.py \
    --data-dir ./playground-series-s5e10 \
    --output-dir ./output
```
Docker 训练
```bash
./docker-run.sh --build
```
预测
```bash
python3 road_accident_risk.py \
    --mode predict \
    --model-dir ./output/models \
    --input data.json
```
HTTP 服务
```bash
python3 api_server.py \
    --model-dir ./output/models \
    --port 5000
```