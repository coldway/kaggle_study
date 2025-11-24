# 快速修复：unsloth 安装问题

## 错误信息
```
ModuleNotFoundError: No module named 'unsloth'
```

## 快速解决方案

### 方法 1: 在终端中安装（推荐）

打开终端，运行以下命令：

```bash
cd "/Users/yuanrui/go/src/new/kaggle study/ARC Prize 2024"

# 1. 升级 pip
python3 -m pip install --upgrade pip

# 2. 安装基础依赖
python3 -m pip install numpy tqdm tokenizers

# 3. 安装 PyTorch (macOS CPU 版本)
python3 -m pip install torch torchvision torchaudio

# 4. 安装 Transformers 生态
python3 -m pip install transformers datasets accelerate peft trl

# 5. 安装 Unsloth
python3 -m pip install unsloth
```

### 方法 2: 在 Notebook 中安装

在运行 Cell 7 之前，先创建一个新单元格并运行：

```python
# 安装所有依赖
!python3 -m pip install --upgrade pip
!python3 -m pip install numpy tqdm tokenizers torch torchvision torchaudio transformers datasets accelerate peft trl unsloth
```

### 方法 3: 验证安装

运行以下代码验证：

```python
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth installed successfully")
except ImportError as e:
    print(f"✗ Unsloth not installed: {e}")
```

## 如果安装失败

### 问题 1: 网络超时
使用国内镜像：
```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unsloth
```

### 问题 2: 编译错误
尝试从源码安装：
```bash
python3 -m pip install git+https://github.com/unslothai/unsloth.git
```

### 问题 3: 依赖冲突
使用虚拟环境：
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # 如果有的话
```

## 安装完成后

重新运行 Cell 7 和后续单元格。

