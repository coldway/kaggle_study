# Mac MPS 模型转 Linux 部署指南

本指南说明如何将在 Mac 上使用 MPS 加速训练的模型转换为 CPU 格式，并在 Linux 系统上部署 HTTP 服务。

## 流程概述

```
Mac (MPS训练) → 模型转换 (CPU格式) → Linux (CPU部署) → HTTP服务
```

## 步骤 1: 在 Mac 上训练模型（使用 MPS）

在 Mac 上正常训练模型，模型会使用 MPS 加速：

```bash
cd "/Users/yuanrui/go/src/new/kaggle_study/Predicting Road Accident Risk"
python road_accident_risk.py --data-dir ./playground-series-s5e10 --output-dir ./output
```

训练完成后，模型文件会保存在 `./output/models/` 目录中。

## 步骤 2: 在 Mac 上转换模型为 CPU 格式

使用 `convert_model_to_cpu.py` 脚本将 MPS 模型转换为 CPU 格式：

```bash
# 转换模型（会自动创建备份）
python convert_model_to_cpu.py --model-dir ./output/models

# 或者不创建备份
python convert_model_to_cpu.py --model-dir ./output/models --no-backup
```

转换脚本会：
- 自动创建模型备份（除非使用 `--no-backup`）
- 将所有模型文件中的 MPS 设备引用转换为 CPU
- 递归处理模型内部的所有 PyTorch 张量
- 使用 `torch.save` 保存转换后的模型，确保设备信息被正确转换

转换完成后，模型文件已经可以在 Linux 上使用了。

## 步骤 3: 将模型文件传输到 Linux 系统

将转换后的模型目录传输到 Linux 系统：

```bash
# 方法1: 使用 scp
scp -r ./output/models user@linux-server:/path/to/deployment/models

# 方法2: 使用 rsync（推荐，支持断点续传）
rsync -avz --progress ./output/models/ user@linux-server:/path/to/deployment/models/

# 方法3: 打包后传输
tar -czf models_cpu.tar.gz ./output/models
scp models_cpu.tar.gz user@linux-server:/path/to/deployment/
# 在 Linux 上解压
ssh user@linux-server "cd /path/to/deployment && tar -xzf models_cpu.tar.gz"
```

**需要传输的文件：**
- `model_fold_1.pkl` 到 `model_fold_5.pkl`（所有 fold 的模型）
- `ensemble_model.pkl`（集成模型）
- `metadata.json`（元数据文件）

## 步骤 4: 在 Linux 系统上准备环境

### 4.1 安装 Python 和依赖

```bash
# 在 Linux 系统上
# 安装 Python 3.9+（如果还没有）
sudo apt-get update
sudo apt-get install python3 python3-pip

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install pandas numpy scikit-learn pytabkit flask
```

### 4.2 准备代码文件

将以下文件复制到 Linux 系统：
- `road_accident_risk.py`
- `api_server.py`

```bash
# 在 Mac 上
scp road_accident_risk.py api_server.py user@linux-server:/path/to/deployment/
```

## 步骤 5: 在 Linux 上启动 HTTP 服务

### 5.1 使用原生 Python 启动

```bash
# 在 Linux 系统上
cd /path/to/deployment
source venv/bin/activate  # 如果使用虚拟环境

# 启动 HTTP 服务
python api_server.py \
  --model-dir ./models \
  --host 0.0.0.0 \
  --port 5000 \
  --orig-data-path /path/to/synthetic_road_accidents_100k.csv
```

### 5.2 使用 systemd 管理服务（生产环境）

创建 systemd 服务文件 `/etc/systemd/system/road-accident-api.service`：

```ini
[Unit]
Description=Road Accident Risk Prediction API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/deployment
Environment="PATH=/path/to/deployment/venv/bin"
ExecStart=/path/to/deployment/venv/bin/python api_server.py \
  --model-dir /path/to/deployment/models \
  --host 0.0.0.0 \
  --port 5000 \
  --orig-data-path /path/to/synthetic_road_accidents_100k.csv
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable road-accident-api
sudo systemctl start road-accident-api
sudo systemctl status road-accident-api
```

### 5.3 使用 Gunicorn（生产环境推荐）

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务（多进程）
gunicorn -w 4 -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  api_server:app
```

注意：使用 Gunicorn 需要修改 `api_server.py`，将 `create_app` 函数暴露为 `app`：

```python
# 在 api_server.py 末尾添加
app = create_app('/path/to/models', '/path/to/data.csv')
```

## 步骤 6: 验证服务

```bash
# 健康检查
curl http://localhost:5000/health

# 测试预测
curl -X POST http://localhost:5000/predict \
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
```

## 常见问题

### Q1: 转换后的模型在 Linux 上仍然报 MPS 错误？

**A:** 确保：
1. 转换脚本成功运行，没有错误
2. 在 Linux 上使用的是 PyTorch CPU 版本
3. `api_server.py` 中的 MPS 禁用代码已生效

### Q2: 如何验证模型已正确转换？

**A:** 在 Mac 上转换后，可以尝试在 Mac 上加载模型（禁用 MPS）：

```python
import torch
torch.backends.mps.is_available = lambda: False
from road_accident_risk import load_model
model, metadata = load_model('./output/models')
# 如果成功加载且没有 MPS 错误，说明转换成功
```

### Q3: 模型文件太大，传输慢怎么办？

**A:** 
- 使用 `rsync` 支持断点续传
- 使用压缩：`tar -czf models.tar.gz models/`
- 考虑使用对象存储服务（如 S3、OSS）作为中转

### Q4: 如何在 Linux 上使用 Docker？

**A:** 参考 `Dockerfile.api` 和 `docker-compose.api.yml`，但确保：
1. 模型文件已转换为 CPU 格式
2. Dockerfile 中使用 PyTorch CPU 版本
3. 环境变量 `CUDA_VISIBLE_DEVICES=""` 已设置

## 完整示例脚本

### Mac 端：转换和打包

```bash
#!/bin/bash
# convert_and_package.sh

MODEL_DIR="./output/models"
OUTPUT_DIR="./deployment_package"

# 1. 转换模型
echo "转换模型到 CPU 格式..."
python convert_model_to_cpu.py --model-dir $MODEL_DIR --no-backup

# 2. 创建部署包目录
mkdir -p $OUTPUT_DIR
cp -r $MODEL_DIR $OUTPUT_DIR/models
cp road_accident_risk.py $OUTPUT_DIR/
cp api_server.py $OUTPUT_DIR/

# 3. 创建 requirements.txt
cat > $OUTPUT_DIR/requirements.txt << EOF
torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
torchaudio --index-url https://download.pytorch.org/whl/cpu
pandas
numpy
scikit-learn
pytabkit
flask
gunicorn
EOF

# 4. 创建启动脚本
cat > $OUTPUT_DIR/start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python api_server.py \
  --model-dir ./models \
  --host 0.0.0.0 \
  --port 5000
EOF
chmod +x $OUTPUT_DIR/start.sh

# 5. 打包
tar -czf deployment_package.tar.gz $OUTPUT_DIR
echo "部署包已创建: deployment_package.tar.gz"
```

### Linux 端：部署脚本

```bash
#!/bin/bash
# deploy.sh

DEPLOY_DIR="/opt/road-accident-api"
PACKAGE_FILE="deployment_package.tar.gz"

# 1. 解压
mkdir -p $DEPLOY_DIR
tar -xzf $PACKAGE_FILE -C $DEPLOY_DIR --strip-components=1

# 2. 创建虚拟环境
cd $DEPLOY_DIR
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务（或配置 systemd）
./start.sh
```

## 总结

1. **在 Mac 上训练**：使用 MPS 加速训练模型
2. **在 Mac 上转换**：使用 `convert_model_to_cpu.py` 转换为 CPU 格式
3. **传输到 Linux**：使用 scp/rsync 传输模型和代码
4. **在 Linux 上部署**：安装依赖，启动 HTTP 服务
5. **验证服务**：使用 curl 测试 API 接口

这样就可以在 Linux 系统上使用 Mac MPS 训练的模型了！

