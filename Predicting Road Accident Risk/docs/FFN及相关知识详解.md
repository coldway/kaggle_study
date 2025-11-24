# FFN (Feed-Forward Network) 及相关知识详解

## 📋 目录

1. [FFN 基础概念](#ffn-基础概念)
2. [FFN 在 Transformer 中的作用](#ffn-在-transformer-中的作用)
3. [FFN 实现详解](#ffn-实现详解)
4. [激活函数](#激活函数)
5. [正则化技术](#正则化技术)
6. [其他相关概念](#其他相关概念)
7. [完整代码示例](#完整代码示例)

---

## FFN 基础概念

### 什么是 FFN？

**FFN (Feed-Forward Network)**，也称为**前馈神经网络**或**多层感知机 (MLP)**，是深度学习中最基础的网络结构。

### 基本结构

```
输入层 (Input Layer)
    ↓
隐藏层 1 (Hidden Layer 1)
    ↓
隐藏层 2 (Hidden Layer 2)
    ↓
...
    ↓
输出层 (Output Layer)
```

### 数学表示

对于一个简单的 FFN：

```
y = f(W₂ · f(W₁ · x + b₁) + b₂)
```

其中：
- `x`: 输入向量
- `W₁, W₂`: 权重矩阵
- `b₁, b₂`: 偏置向量
- `f`: 激活函数

### FFN 的特点

1. **单向传播**：信息从输入到输出单向流动
2. **全连接**：每一层的每个神经元都与下一层的所有神经元连接
3. **非线性变换**：通过激活函数引入非线性

---

## FFN 在 Transformer 中的作用

### Transformer 架构中的 FFN

在 Transformer 架构中（包括 TabM），FFN 是每个 Block 的重要组成部分：

```
Transformer Block:
    ├─ Multi-Head Self-Attention
    ├─ Add & Norm (残差连接 + 层归一化)
    ├─ FFN (Feed-Forward Network)  ← 这里
    └─ Add & Norm (残差连接 + 层归一化)
```

### FFN 的作用

1. **非线性变换**：注意力机制是线性变换，FFN 提供非线性
2. **特征增强**：将注意力后的特征进一步处理
3. **维度扩展**：通常先扩展到更大维度，再压缩回原维度

### 典型的 FFN 结构

```python
FFN(x) = ReLU(W₂ · ReLU(W₁ · x + b₁) + b₂)
```

或者使用 GELU：

```python
FFN(x) = GELU(W₂ · GELU(W₁ · x + b₁) + b₂)
```

**维度变化**：
- 输入：`d_model` (例如 432)
- 中间层：`d_ff` (通常是 `d_model * 4`，例如 1728)
- 输出：`d_model` (例如 432)

---

## FFN 实现详解

### 1. 基础 FFN 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicFFN(nn.Module):
    """基础的前馈神经网络"""
    def __init__(self, d_model, d_ff, activation='relu', dropout=0.0):
        """
        Args:
            d_model: 模型维度（输入和输出维度）
            d_ff: 前馈网络中间层维度（通常是 d_model * 4）
            activation: 激活函数类型
            dropout: Dropout 概率
        """
        super().__init__()
        
        # 第一层：扩展到 d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # 第二层：压缩回 d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish = SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) 或 (batch_size, d_model)
        Returns:
            out: 相同形状的输出
        """
        # 第一层：扩展维度
        x = self.linear1(x)  # (..., d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二层：压缩回原维度
        x = self.linear2(x)  # (..., d_model)
        x = self.dropout(x)
        
        return x
```

### 2. 带残差连接的 FFN

```python
class FFNWithResidual(nn.Module):
    """带残差连接的 FFN"""
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.0):
        super().__init__()
        self.ffn = BasicFFN(d_model, d_ff, activation, dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # 残差连接
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = x + residual  # 残差连接
        return x
```

### 3. TabM 中的 FFN 实现

```python
class TabMFFN(nn.Module):
    """TabM 中使用的前馈网络"""
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        """
        Args:
            d_model: 模型维度（对应 d_block，例如 432）
            d_ff: 前馈网络维度（通常为 d_model * 4）
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (d_model * 4)  # 默认扩展 4 倍
        
        # 第一层：扩展到 d_ff
        self.linear1 = nn.Linear(d_model, self.d_ff)
        
        # 第二层：压缩回 d_model
        self.linear2 = nn.Linear(self.d_ff, d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数（TabM 通常使用 GELU）
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features, d_model)
        Returns:
            out: (batch_size, num_features, d_model)
        """
        # 扩展维度
        x = self.linear1(x)  # (batch_size, num_features, d_ff)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # 压缩回原维度
        x = self.linear2(x)  # (batch_size, num_features, d_model)
        x = self.dropout2(x)
        
        return x
```

### 4. 高级 FFN 变体

#### 4.1 门控 FFN (Gated FFN)

```python
class GatedFFN(nn.Module):
    """门控前馈网络（类似 GLU）"""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # 输出两倍维度
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # 第一层输出分成两部分
        gate_output = self.linear1(x)  # (..., d_ff * 2)
        gate, value = gate_output.chunk(2, dim=-1)  # 分成两部分
        
        # 门控机制
        gated = self.activation(gate) * value  # 元素级乘法
        
        # 第二层
        output = self.linear2(gated)
        output = self.dropout(output)
        
        return output
```

#### 4.2 深度 FFN (Deep FFN)

```python
class DeepFFN(nn.Module):
    """多层前馈网络"""
    def __init__(self, d_model, d_ff, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        
        # 第一层：扩展到 d_ff
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # 中间层：保持 d_ff 维度
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # 最后一层：压缩回 d_model
        layers.append(nn.Linear(d_ff, d_model))
        layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
```

---

## 激活函数

### 1. ReLU (Rectified Linear Unit)

```python
ReLU(x) = max(0, x)
```

**特点**：
- ✅ 计算简单快速
- ✅ 解决梯度消失问题（正区间）
- ❌ 死亡 ReLU 问题（负区间梯度为 0）

**实现**：
```python
class ReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))
```

### 2. GELU (Gaussian Error Linear Unit)

```python
GELU(x) = x * Φ(x)
```

其中 `Φ(x)` 是标准正态分布的累积分布函数。

**近似公式**：
```python
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**特点**：
- ✅ 平滑的激活函数
- ✅ 在 Transformer 中表现优秀
- ✅ 避免死亡神经元问题

**实现**：
```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / 3.14159)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 3. Swish / SiLU

```python
Swish(x) = x * sigmoid(x)
```

**特点**：
- ✅ 平滑且可微
- ✅ 在某些任务上比 ReLU 更好
- ✅ 自门控机制

**实现**：
```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

### 4. 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 | 使用场景 |
|---------|------|------|------|---------|
| **ReLU** | max(0, x) | 简单快速 | 死亡神经元 | 通用 |
| **GELU** | x * Φ(x) | 平滑，性能好 | 计算稍慢 | Transformer |
| **Swish** | x * σ(x) | 平滑，自门控 | 计算稍慢 | 某些任务 |
| **Tanh** | tanh(x) | 输出范围 [-1, 1] | 梯度消失 | 较少使用 |
| **Sigmoid** | 1/(1+e⁻ˣ) | 输出范围 [0, 1] | 梯度消失 | 输出层 |

### 5. 激活函数可视化

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)

# ReLU
relu = np.maximum(0, x)

# GELU (近似)
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Swish
swish = x * (1 / (1 + np.exp(-x)))

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(x, relu, label='ReLU')
plt.plot(x, gelu, label='GELU')
plt.plot(x, swish, label='Swish')
plt.legend()
plt.grid(True)
plt.title('Activation Functions Comparison')
plt.show()
```

---

## 正则化技术

### 1. Dropout

**原理**：训练时随机将部分神经元输出置为 0，防止过拟合。

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # Dropout 概率
    
    def forward(self, x):
        if self.training:
            # 训练时：随机置零
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)  # 缩放以保持期望值
        else:
            # 推理时：不做任何操作
            return x
```

**在 FFN 中的使用**：
```python
class FFNWithDropout(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # 第一层后 Dropout
        x = self.linear2(x)
        x = self.dropout2(x)  # 第二层后 Dropout
        return x
```

### 2. Layer Normalization

**原理**：对每个样本的特征维度进行归一化。

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 可学习缩放
        self.beta = nn.Parameter(torch.zeros(d_model))   # 可学习偏移
        self.eps = eps
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta
```

**在 Transformer Block 中的使用**：
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.ffn = BasicFFN(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # 自注意力 + 残差 + 归一化
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)[0]
        x = x + residual
        
        # FFN + 残差 + 归一化
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x
```

### 3. Batch Normalization vs Layer Normalization

| 特性 | Batch Normalization | Layer Normalization |
|------|---------------------|---------------------|
| **归一化维度** | 批次维度 | 特征维度 |
| **适用场景** | CNN, 大批次 | Transformer, RNN |
| **训练/推理** | 需要区分 | 一致 |
| **位置** | 通常在激活前 | 通常在激活后 |

**Batch Normalization**：
```python
# 对批次维度归一化
# x: (batch_size, features)
mean = x.mean(dim=0)  # 对批次维度求均值
std = x.std(dim=0)
normalized = (x - mean) / (std + eps)
```

**Layer Normalization**：
```python
# 对特征维度归一化
# x: (batch_size, features)
mean = x.mean(dim=-1, keepdim=True)  # 对特征维度求均值
std = x.std(dim=-1, keepdim=True)
normalized = (x - mean) / (std + eps)
```

### 4. 权重衰减 (Weight Decay)

**原理**：L2 正则化，在损失函数中添加权重的平方和。

```python
# 在优化器中设置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 正则化系数
)
```

**数学表示**：
```
Loss = Original_Loss + λ * Σ(w²)
```

其中 `λ` 是 `weight_decay` 参数。

---

## 其他相关概念

### 1. 残差连接 (Residual Connection)

**原理**：将输入直接加到输出上，缓解梯度消失问题。

```python
class ResidualFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.ffn = BasicFFN(d_model, d_ff, dropout=dropout)
    
    def forward(self, x):
        residual = x
        x = self.ffn(x)
        return x + residual  # 残差连接
```

**为什么有效**：
1. **梯度流动**：梯度可以直接通过残差连接传播
2. **身份映射**：如果 FFN 学习到恒等映射，残差连接保证至少是恒等
3. **深层网络**：使训练更深的网络成为可能

### 2. 注意力机制与 FFN 的配合

在 Transformer 中，注意力机制和 FFN 配合工作：

```
输入 x
  ↓
Self-Attention: 学习特征间的关系
  ↓
Add & Norm: 残差连接 + 归一化
  ↓
FFN: 非线性变换和特征增强
  ↓
Add & Norm: 残差连接 + 归一化
  ↓
输出
```

**分工**：
- **Self-Attention**：学习"哪些特征重要"（特征选择）
- **FFN**：学习"如何变换特征"（特征变换）

### 3. 位置编码 (Positional Encoding)

虽然 FFN 本身不涉及位置编码，但在 Transformer 中，位置信息很重要：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 4. 梯度裁剪 (Gradient Clipping)

**原理**：限制梯度的大小，防止梯度爆炸。

```python
# 方法 1: 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法 2: 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**在训练循环中使用**：
```python
optimizer.zero_grad()
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### 5. 学习率调度

**ReduceLROnPlateau**：当验证损失不再下降时降低学习率。

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 最小化指标
    factor=0.5,        # 学习率衰减因子
    patience=10,       # 等待轮数
    verbose=True
)

# 在训练循环中
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)  # 根据验证损失调整
```

**CosineAnnealingLR**：余弦退火调度。

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # 最大周期数
    eta_min=1e-6  # 最小学习率
)
```

---

## 完整代码示例

### TabM Block 完整实现（包含 FFN）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TabMBlock(nn.Module):
    """完整的 TabM Block，包含注意力机制和 FFN"""
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.0, tabm_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or (d_model * 4)
        self.tabm_k = tabm_k
        
        # 1. 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 2. FFN 层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features, d_model)
        Returns:
            out: (batch_size, num_features, d_model)
        """
        # 第一部分：自注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)
        
        # 第二部分：FFN + 残差连接
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x


class TabMNet(nn.Module):
    """完整的 TabM 网络"""
    def __init__(
        self,
        num_numeric,
        categorical_cardinalities,
        d_embedding=24,
        n_blocks=5,
        d_block=432,
        n_heads=8,
        dropout=0.0,
        tabm_k=32,
    ):
        super().__init__()
        
        # 特征嵌入（简化版，实际需要 PWL 嵌入）
        self.numeric_embedding = nn.Linear(num_numeric, d_embedding)
        if len(categorical_cardinalities) > 0:
            self.categorical_embedding = nn.ModuleList([
                nn.Embedding(card, d_embedding)
                for card in categorical_cardinalities
            ])
        else:
            self.categorical_embedding = None
        
        # 输入投影
        num_features = num_numeric + len(categorical_cardinalities)
        self.input_projection = nn.Linear(d_embedding, d_block)
        
        # TabM Blocks
        self.blocks = nn.ModuleList([
            TabMBlock(d_block, n_heads, dropout=dropout, tabm_k=tabm_k)
            for _ in range(n_blocks)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_block, d_block // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_block // 2, 1)
        )
    
    def forward(self, numeric_features, categorical_features=None):
        # 特征嵌入
        embedded = []
        
        # 数值特征
        if numeric_features is not None:
            numeric_emb = self.numeric_embedding(numeric_features)
            embedded.append(numeric_emb)
        
        # 分类特征
        if self.categorical_embedding is not None and categorical_features is not None:
            cat_embs = []
            for i, emb in enumerate(self.categorical_embedding):
                cat_embs.append(emb(categorical_features[:, i]))
            cat_emb = torch.stack(cat_embs, dim=1)
            embedded.append(cat_emb)
        
        # 拼接
        if len(embedded) == 2:
            x = torch.cat(embedded, dim=1)
        elif len(embedded) == 1:
            x = embedded[0]
        else:
            raise ValueError("需要至少一种特征类型")
        
        # 投影
        x = self.input_projection(x)  # (batch_size, num_features, d_block)
        
        # TabM Blocks
        for block in self.blocks:
            x = block(x)
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch_size, d_block, num_features)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_block)
        
        # 输出
        output = self.output_layer(x)
        return output.squeeze(-1)


# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = TabMNet(
        num_numeric=8,
        categorical_cardinalities=[3, 4, 5],  # 3 个分类特征
        d_embedding=24,
        n_blocks=5,
        d_block=432,
        n_heads=8,
        dropout=0.0,
        tabm_k=32,
    )
    
    # 创建示例数据
    batch_size = 32
    numeric = torch.randn(batch_size, 8)
    categorical = torch.randint(0, 3, (batch_size, 3))
    
    # 前向传播
    output = model(numeric, categorical)
    print(f"Input shape: numeric={numeric.shape}, categorical={categorical.shape}")
    print(f"Output shape: {output.shape}")
```

---

## FFN 设计原则

### 1. 维度扩展原则

**为什么先扩展再压缩？**

```
d_model → d_ff (扩展) → d_model (压缩)
```

**原因**：
1. **表达能力**：更大的中间维度提供更强的表达能力
2. **非线性变换**：在更大的空间中进行非线性变换
3. **信息流动**：扩展-压缩的过程类似于"瓶颈"结构

### 2. 激活函数选择

**在 FFN 中通常使用**：
- **GELU**：Transformer 中的标准选择
- **ReLU**：简单快速，但可能不如 GELU
- **Swish**：在某些任务上表现更好

### 3. Dropout 位置

**通常的位置**：
1. 激活函数之后
2. 线性层之后
3. 残差连接之前（可选）

### 4. 残差连接的重要性

**为什么需要残差连接？**

1. **梯度流动**：使梯度可以直接传播
2. **身份映射**：如果 FFN 学习到恒等映射，至少保持原值
3. **深层网络**：使训练深层网络成为可能

---

## 性能优化

### 1. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 优化器选择

**AdamW**（推荐）：
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

**Adam**：
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)
```

---

## 总结

### FFN 的关键点

1. **结构**：扩展 → 激活 → 压缩
2. **作用**：提供非线性变换和特征增强
3. **配合**：与注意力机制配合工作
4. **正则化**：使用 Dropout 和 LayerNorm

### 在 TabM 中的位置

```
输入特征
  ↓
特征嵌入（PWL + Categorical）
  ↓
输入投影
  ↓
TabM Block 1
  ├─ Self-Attention
  ├─ Add & Norm
  ├─ FFN ← 这里
  └─ Add & Norm
  ↓
TabM Block 2-5
  ...
  ↓
全局池化
  ↓
输出层
```

### 相关技术栈

- **激活函数**：GELU, ReLU, Swish
- **正则化**：Dropout, LayerNorm, Weight Decay
- **优化技巧**：残差连接, 梯度裁剪, 学习率调度
- **性能优化**：混合精度, 优化器选择

---

## 参考资料

- **Transformer 论文**：Attention Is All You Need
- **GELU 论文**：Gaussian Error Linear Units
- **Layer Normalization 论文**：Layer Normalization
- **PyTorch 文档**：https://pytorch.org/docs/stable/index.html

