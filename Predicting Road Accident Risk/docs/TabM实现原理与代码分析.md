# TabM_D_Regressor 实现原理与代码分析

## 📋 目录

1. [TabM 概述](#tabm-概述)
2. [核心架构](#核心架构)
3. [关键组件实现](#关键组件实现)
4. [完整实现代码](#完整实现代码)
5. [训练流程分析](#训练流程分析)
6. [关键技术详解](#关键技术详解)

---

## TabM 概述

**TabM (Tabular Model)** 是一个专门为表格数据设计的深度学习模型，结合了 Transformer 架构和表格数据的特殊需求。

### 核心特点

1. **混合特征处理**：同时处理数值特征和分类特征
2. **数值嵌入（PWL）**：使用分段线性嵌入处理连续数值
3. **Transformer 架构**：使用注意力机制学习特征交互
4. **端到端训练**：从原始特征到预测的完整流程

---

## 核心架构

### 整体架构图

```
输入数据
  │
  ├─ 数值特征 ──> PWL 数值嵌入 ──┐
  │                              │
  └─ 分类特征 ──> 分类嵌入 ──────┤
                                 │
                                 ▼
                          特征拼接/融合
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │  TabM Block 1        │
                    │  - Self-Attention    │
                    │  - FFN               │
                    │  - Residual          │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  TabM Block 2       │
                    │  ...                │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  TabM Block N       │
                    └─────────────────────┘
                                 │
                                 ▼
                           全局池化/聚合
                                 │
                                 ▼
                            输出层
                                 │
                                 ▼
                           回归预测值
```

### 架构层次

1. **输入层**：特征嵌入
2. **编码层**：多个 TabM Block（Transformer 风格）
3. **聚合层**：特征聚合
4. **输出层**：回归预测

---

## 关键组件实现

### 1. 数值特征嵌入（PWL - Piecewise Linear）

**原理**：将连续数值特征分成多个区间（bins），每个区间学习一个嵌入向量，通过线性插值计算最终嵌入。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PWLNumericEmbedding(nn.Module):
    """
    分段线性数值嵌入（Piecewise Linear Embedding）
    类似 TabM 中的数值特征处理方式
    """
    def __init__(self, num_features, n_bins=112, embed_dim=24):
        """
        Args:
            num_features: 数值特征数量
            n_bins: 分箱数量（对应 num_emb_n_bins）
            embed_dim: 嵌入维度（对应 d_embedding）
        """
        super().__init__()
        self.num_features = num_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        
        # 为每个数值特征创建嵌入表
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_bins, embed_dim) for _ in range(num_features)
        ])
        
        # 可学习的边界点（bin edges）
        # 初始化为均匀分布
        self.bin_edges = nn.ParameterList([
            nn.Parameter(torch.linspace(0, 1, n_bins - 1)) 
            for _ in range(num_features)
        ])
        
        # 可学习的权重（用于插值）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(n_bins)) 
            for _ in range(num_features)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features) 数值特征
        Returns:
            embedded: (batch_size, num_features, embed_dim) 嵌入向量
        """
        batch_size, num_features = x.shape
        embedded_features = []
        
        for i in range(num_features):
            # 1. 归一化到 [0, 1]
            x_i = x[:, i]
            x_min = x_i.min()
            x_max = x_i.max()
            if x_max > x_min:
                x_norm = (x_i - x_min) / (x_max - x_min + 1e-8)
            else:
                x_norm = torch.zeros_like(x_i)
            
            # 2. 找到对应的 bin 索引
            # 使用 bucketize 找到每个值属于哪个区间
            bin_indices = torch.bucketize(x_norm, self.bin_edges[i], right=True)
            bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
            
            # 3. 获取基础嵌入
            base_embed = self.embeddings[i](bin_indices)  # (batch_size, embed_dim)
            
            # 4. 线性插值（简化版本）
            # 计算在 bin 内的位置
            bin_width = 1.0 / self.n_bins
            bin_pos = (x_norm - bin_indices.float() * bin_width) / bin_width
            bin_pos = torch.clamp(bin_pos, 0, 1)
            
            # 5. 应用权重
            weight = self.weights[i][bin_indices].unsqueeze(-1)  # (batch_size, 1)
            embedded = base_embed * weight * (1 + bin_pos.unsqueeze(-1))
            
            embedded_features.append(embedded)
        
        # 堆叠所有特征
        return torch.stack(embedded_features, dim=1)  # (batch_size, num_features, embed_dim)
```

### 2. 分类特征嵌入

```python
class CategoricalEmbedding(nn.Module):
    """
    分类特征嵌入
    """
    def __init__(self, categorical_cardinalities, embed_dim=24):
        """
        Args:
            categorical_cardinalities: 每个分类特征的基数（类别数量）列表
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in categorical_cardinalities
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_categorical) 分类特征索引
        Returns:
            embedded: (batch_size, num_categorical, embed_dim) 嵌入向量
        """
        embedded_features = []
        for i, emb in enumerate(self.embeddings):
            embedded_features.append(emb(x[:, i]))
        return torch.stack(embedded_features, dim=1)
```

### 3. TabM Block（Transformer 风格）

```python
class TabMBlock(nn.Module):
    """
    TabM 的核心块，基于 Transformer 架构
    对应参数：n_blocks, d_block
    """
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.0, tabm_k=32):
        """
        Args:
            d_model: 模型维度（对应 d_block）
            n_heads: 注意力头数
            d_ff: 前馈网络维度（通常为 d_model * 4）
            dropout: Dropout 概率
            tabm_k: TabM 的 k 参数（控制注意力范围）
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or d_model * 4
        self.tabm_k = tabm_k
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),  # 或 ReLU
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # TabM 特定的特征选择机制（简化版）
        self.feature_selector = nn.Linear(d_model, tabm_k)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features, d_model) 特征嵌入
        Returns:
            out: (batch_size, num_features, d_model) 输出
        """
        # 1. 自注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        
        # TabM 特定的注意力机制
        # 可以选择 top-k 特征进行注意力计算
        attn_weights = self.feature_selector(x)  # (batch_size, num_features, k)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 简化的注意力计算
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + attn_out
        
        # 2. 前馈网络 + 残差连接
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x
```

### 4. 特征融合层

```python
class FeatureFusion(nn.Module):
    """
    融合数值和分类特征
    """
    def __init__(self, num_numeric, num_categorical, embed_dim=24):
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.embed_dim = embed_dim
        
        # 特征类型嵌入（可选）
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0: numeric, 1: categorical
    
    def forward(self, numeric_emb, categorical_emb):
        """
        Args:
            numeric_emb: (batch_size, num_numeric, embed_dim)
            categorical_emb: (batch_size, num_categorical, embed_dim)
        Returns:
            fused: (batch_size, num_features, embed_dim)
        """
        # 添加类型嵌入
        numeric_type = self.type_embedding(torch.zeros(self.num_numeric, dtype=torch.long))
        categorical_type = self.type_embedding(torch.ones(self.num_categorical, dtype=torch.long))
        
        numeric_emb = numeric_emb + numeric_type.unsqueeze(0)
        categorical_emb = categorical_emb + categorical_type.unsqueeze(0)
        
        # 拼接
        fused = torch.cat([numeric_emb, categorical_emb], dim=1)
        return fused
```

---

## 完整实现代码

### TabM_D_Regressor 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional

class TabM_D_Regressor:
    """
    TabM (Tabular Model) 回归器
    基于 Transformer 架构的表格数据深度学习模型
    """
    
    def __init__(
        self,
        batch_size='auto',
        patience=16,
        allow_amp=False,
        arch_type='tabm-mini',
        tabm_k=32,
        gradient_clipping_norm=1.0,
        share_training_batches=False,
        lr=0.000624068703424289,
        weight_decay=0.0019090968357478807,
        n_blocks=5,
        d_block=432,
        dropout=0.0,
        num_emb_type='pwl',
        d_embedding=24,
        num_emb_n_bins=112,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        初始化 TabM 模型
        
        Args:
            batch_size: 批次大小，'auto' 表示自动计算
            patience: 早停耐心值
            allow_amp: 是否使用混合精度训练
            arch_type: 架构类型（'tabm-mini', 'tabm-base', 'tabm-large'）
            tabm_k: TabM 的 k 参数
            gradient_clipping_norm: 梯度裁剪范数
            lr: 学习率
            weight_decay: 权重衰减
            n_blocks: TabM Block 数量
            d_block: 每个 Block 的维度
            dropout: Dropout 概率
            num_emb_type: 数值嵌入类型（'pwl' 表示分段线性）
            d_embedding: 嵌入维度
            num_emb_n_bins: 数值嵌入分箱数
            device: 计算设备
        """
        self.patience = patience
        self.allow_amp = allow_amp
        self.arch_type = arch_type
        self.tabm_k = tabm_k
        self.gradient_clipping_norm = gradient_clipping_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.num_emb_type = num_emb_type
        self.d_embedding = d_embedding
        self.num_emb_n_bins = num_emb_n_bins
        self.device = device
        
        # 自动计算批次大小
        if batch_size == 'auto':
            self.batch_size = 256 if device == 'cuda' else 32
        else:
            self.batch_size = batch_size
        
        # 模型组件（将在 fit 时初始化）
        self.model = None
        self.numeric_scaler = StandardScaler()
        self.categorical_encoders = []
        self.num_numeric_features = 0
        self.num_categorical_features = 0
        self.categorical_cardinalities = []
    
    def _build_model(self, num_numeric, categorical_cardinalities):
        """构建模型架构"""
        return TabMNet(
            num_numeric=num_numeric,
            categorical_cardinalities=categorical_cardinalities,
            d_embedding=self.d_embedding,
            num_emb_n_bins=self.num_emb_n_bins,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            tabm_k=self.tabm_k,
        ).to(self.device)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, cat_col_names=None):
        """
        训练模型
        
        Args:
            X_train: 训练特征（DataFrame）
            y_train: 训练目标（Series）
            X_val: 验证特征（DataFrame，可选）
            y_val: 验证目标（Series，可选）
            cat_col_names: 分类特征列名列表
        """
        # 1. 数据预处理
        X_train_processed, X_val_processed = self._preprocess_data(
            X_train, X_val, cat_col_names, fit=True
        )
        
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(self.device) if y_val is not None else None
        
        # 2. 构建模型
        self.model = self._build_model(
            self.num_numeric_features,
            self.categorical_cardinalities
        )
        
        # 3. 训练
        self._train(
            X_train_processed, y_train_tensor,
            X_val_processed, y_val_tensor
        )
    
    def _preprocess_data(self, X_train, X_val=None, cat_col_names=None, fit=False):
        """数据预处理"""
        if cat_col_names is None:
            cat_col_names = []
        
        # 分离数值和分类特征
        numeric_cols = [col for col in X_train.columns if col not in cat_col_names]
        categorical_cols = cat_col_names
        
        # 处理数值特征
        X_train_numeric = X_train[numeric_cols].values
        if fit:
            X_train_numeric = self.numeric_scaler.fit_transform(X_train_numeric)
            self.num_numeric_features = len(numeric_cols)
        else:
            X_train_numeric = self.numeric_scaler.transform(X_train_numeric)
        
        # 处理分类特征
        X_train_categorical = []
        if fit:
            self.categorical_encoders = []
            self.categorical_cardinalities = []
        
        for i, col in enumerate(categorical_cols):
            if fit:
                le = LabelEncoder()
                X_train_cat = le.fit_transform(X_train[col].astype(str).fillna('unknown'))
                self.categorical_encoders.append(le)
                self.categorical_cardinalities.append(len(le.classes_))
            else:
                le = self.categorical_encoders[i]
                X_train_cat = le.transform(X_train[col].astype(str).fillna('unknown'))
            X_train_categorical.append(X_train_cat)
        
        X_train_categorical = np.column_stack(X_train_categorical) if X_train_categorical else np.array([]).reshape(len(X_train), 0)
        
        # 处理验证集
        if X_val is not None:
            X_val_numeric = self.numeric_scaler.transform(X_val[numeric_cols].values)
            X_val_categorical = []
            for i, col in enumerate(categorical_cols):
                le = self.categorical_encoders[i]
                X_val_cat = le.transform(X_val[col].astype(str).fillna('unknown'))
                X_val_categorical.append(X_val_cat)
            X_val_categorical = np.column_stack(X_val_categorical) if X_val_categorical else np.array([]).reshape(len(X_val), 0)
            return (X_train_numeric, X_train_categorical), (X_val_numeric, X_val_categorical)
        
        return (X_train_numeric, X_train_categorical), None
    
    def _train(self, X_train, y_train, X_val=None, y_val=None):
        """训练循环"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.patience // 2, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 创建数据加载器
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        if X_val is not None:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        for epoch in range(1000):  # 最大 epoch 数
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                numeric, categorical, target = batch
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                
                if self.allow_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(numeric, categorical)
                        loss = criterion(pred, target)
                    torch.cuda.amp.scale_loss(loss, optimizer).backward()
                else:
                    pred = self.model(numeric, categorical)
                    loss = criterion(pred, target)
                    loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping_norm
                )
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        numeric, categorical, target = batch
                        numeric = numeric.to(self.device)
                        categorical = categorical.to(self.device)
                        target = target.to(self.device)
                        
                        pred = self.model(numeric, categorical)
                        loss = criterion(pred, target)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    self.model.load_state_dict(self.best_model_state)
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}')
    
    def predict(self, X):
        """预测"""
        X_processed, _ = self._preprocess_data(X, fit=False)
        
        self.model.eval()
        predictions = []
        
        dataset = TabularDataset(X_processed, None)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        
        with torch.no_grad():
            for batch in loader:
                numeric, categorical = batch
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                
                pred = self.model(numeric, categorical)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions)


class TabMNet(nn.Module):
    """
    TabM 神经网络架构
    """
    def __init__(
        self,
        num_numeric,
        categorical_cardinalities,
        d_embedding=24,
        num_emb_n_bins=112,
        n_blocks=5,
        d_block=432,
        dropout=0.0,
        tabm_k=32,
    ):
        super().__init__()
        
        # 1. 特征嵌入层
        if num_numeric > 0:
            self.numeric_embedding = PWLNumericEmbedding(
                num_numeric, num_emb_n_bins, d_embedding
            )
        else:
            self.numeric_embedding = None
        
        if len(categorical_cardinalities) > 0:
            self.categorical_embedding = CategoricalEmbedding(
                categorical_cardinalities, d_embedding
            )
        else:
            self.categorical_embedding = None
        
        # 2. 特征融合
        self.feature_fusion = FeatureFusion(
            num_numeric, len(categorical_cardinalities), d_embedding
        )
        
        # 3. 投影到模型维度
        num_features = num_numeric + len(categorical_cardinalities)
        self.input_projection = nn.Linear(d_embedding, d_block)
        
        # 4. TabM Blocks
        self.blocks = nn.ModuleList([
            TabMBlock(d_block, dropout=dropout, tabm_k=tabm_k)
            for _ in range(n_blocks)
        ])
        
        # 5. 全局聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 6. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_block, d_block // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_block // 2, 1)
        )
    
    def forward(self, numeric_features, categorical_features):
        """
        Args:
            numeric_features: (batch_size, num_numeric) 或 None
            categorical_features: (batch_size, num_categorical) 或 None
        Returns:
            output: (batch_size, 1) 预测值
        """
        # 1. 特征嵌入
        embedded_features = []
        
        if self.numeric_embedding is not None:
            numeric_emb = self.numeric_embedding(numeric_features)
            embedded_features.append(numeric_emb)
        
        if self.categorical_embedding is not None:
            categorical_emb = self.categorical_embedding(categorical_features)
            embedded_features.append(categorical_emb)
        
        # 2. 特征融合
        if len(embedded_features) == 2:
            x = self.feature_fusion(embedded_features[0], embedded_features[1])
        elif len(embedded_features) == 1:
            x = embedded_features[0]
        else:
            raise ValueError("至少需要一种特征类型")
        
        # 3. 投影到模型维度
        x = self.input_projection(x)  # (batch_size, num_features, d_block)
        
        # 4. TabM Blocks
        for block in self.blocks:
            x = block(x)
        
        # 5. 全局聚合（平均池化）
        x = x.transpose(1, 2)  # (batch_size, d_block, num_features)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_block)
        
        # 6. 输出
        output = self.output_layer(x)  # (batch_size, 1)
        
        return output.squeeze(-1)


class TabularDataset(torch.utils.data.Dataset):
    """表格数据数据集"""
    def __init__(self, features, targets=None):
        self.numeric = torch.FloatTensor(features[0])
        self.categorical = torch.LongTensor(features[1]) if features[1].size > 0 else None
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.numeric)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            if self.categorical is not None:
                return self.numeric[idx], self.categorical[idx], self.targets[idx]
            else:
                return self.numeric[idx], torch.tensor([]), self.targets[idx]
        else:
            if self.categorical is not None:
                return self.numeric[idx], self.categorical[idx]
            else:
                return self.numeric[idx], torch.tensor([])
```

---

## 训练流程分析

### 完整训练流程

```python
# 1. 初始化模型
model = TabM_D_Regressor(
    arch_type='tabm-mini',
    tabm_k=32,
    n_blocks=5,
    d_block=432,
    lr=0.000624,
    weight_decay=0.001909,
    d_embedding=24,
    num_emb_n_bins=112,
)

# 2. 训练
model.fit(
    X_train, y_train,
    X_val, y_val,
    cat_col_names=['road_type', 'lighting', 'weather', ...]
)

# 3. 预测
predictions = model.predict(X_test)
```

### 数据流

```
原始数据 (DataFrame)
  │
  ├─ 数值特征 ──> StandardScaler ──> PWL 嵌入 ──┐
  │                                              │
  └─ 分类特征 ──> LabelEncoder ──> 分类嵌入 ─────┤
                                                 │
                                                 ▼
                                          特征融合
                                                 │
                                                 ▼
                                         输入投影 (d_embedding -> d_block)
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │   TabM Block 1          │
                                    │   - Self-Attention      │
                                    │   - FFN                 │
                                    │   - Residual            │
                                    └────────────────────────┘
                                                 │
                                    ┌────────────────────────┐
                                    │   TabM Block 2-5        │
                                    │   ...                   │
                                    └────────────────────────┘
                                                 │
                                                 ▼
                                          全局平均池化
                                                 │
                                                 ▼
                                            输出层
                                                 │
                                                 ▼
                                           预测值
```

---

## 关键技术详解

### 1. PWL 数值嵌入详解

**为什么使用 PWL？**

1. **非线性映射**：将连续数值映射到嵌入空间，捕捉非线性关系
2. **分箱策略**：将连续值分成多个区间，每个区间学习不同的表示
3. **可学习边界**：bin edges 是可学习的参数，可以自适应调整

**实现细节**：

```python
# 示例：处理一个数值特征
x = 0.75  # 归一化后的值
n_bins = 112

# 1. 找到对应的 bin
bin_idx = int(x * n_bins)  # 84

# 2. 获取该 bin 的嵌入向量
embedding = embedding_table[bin_idx]  # (embed_dim,)

# 3. 线性插值（可选）
bin_pos = (x - bin_idx / n_bins) * n_bins  # 在 bin 内的位置
# 可以插值相邻 bin 的嵌入
```

### 2. TabM Block 详解

**核心组件**：

1. **Self-Attention**：学习特征间的交互关系
2. **FFN (Feed-Forward Network)**：非线性变换
3. **Residual Connection**：缓解梯度消失
4. **Layer Normalization**：稳定训练

**注意力机制**：

```python
# 自注意力计算
Q = Linear(x)  # Query
K = Linear(x)  # Key
V = Linear(x)  # Value

# 注意力分数
scores = Q @ K.T / sqrt(d_k)
attn_weights = softmax(scores)

# 加权求和
output = attn_weights @ V
```

### 3. 特征交互学习

**TabM 如何学习特征交互？**

1. **注意力权重**：显示哪些特征对预测最重要
2. **多层堆叠**：每层学习不同层次的交互
3. **全局聚合**：将所有特征信息融合

**示例**：

```
Block 1: 学习基础特征表示
Block 2: 学习两两特征交互
Block 3: 学习高阶特征交互
Block 4-5: 进一步精炼表示
```

### 4. 训练技巧

#### 4.1 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**作用**：防止梯度爆炸，稳定训练

#### 4.2 学习率调度

```python
scheduler = ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
```

**作用**：验证损失不下降时降低学习率

#### 4.3 早停机制

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    save_best_model()
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop_training()
```

**作用**：防止过拟合，节省训练时间

---

## 参数说明

### 关键超参数

| 参数 | 说明 | 典型值 | 影响 |
|------|------|--------|------|
| `n_blocks` | TabM Block 数量 | 5 | 模型深度，越多越复杂 |
| `d_block` | Block 维度 | 432 | 模型容量，越大表达能力越强 |
| `d_embedding` | 嵌入维度 | 24 | 特征表示维度 |
| `num_emb_n_bins` | 数值嵌入分箱数 | 112 | 数值特征离散化粒度 |
| `tabm_k` | TabM k 参数 | 32 | 注意力范围控制 |
| `lr` | 学习率 | 0.000624 | 训练速度，太大可能不稳定 |
| `weight_decay` | 权重衰减 | 0.001909 | 正则化强度 |
| `dropout` | Dropout 概率 | 0.0 | 防止过拟合 |
| `patience` | 早停耐心值 | 16 | 早停等待轮数 |

### 参数调优建议

1. **从小开始**：先使用较小的 `n_blocks` 和 `d_block`
2. **逐步增加**：如果欠拟合，增加模型容量
3. **学习率**：通常从 1e-4 到 1e-3 之间
4. **批次大小**：GPU 内存允许的情况下，越大越好
5. **早停**：根据验证集表现调整 `patience`

---

## 性能优化

### 1. 混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    pred = model(x)
    loss = criterion(pred, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优势**：减少显存占用，加速训练

### 2. 数据并行

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**优势**：多 GPU 加速训练

### 3. 批次大小优化

```python
# 自动调整批次大小
if device == 'cuda':
    batch_size = 256  # GPU 可以更大
else:
    batch_size = 32   # CPU 较小
```

---

## 总结

### TabM 的核心优势

1. **专门设计**：针对表格数据优化
2. **混合特征**：原生支持数值和分类特征
3. **特征交互**：自动学习复杂交互关系
4. **可扩展**：可以通过增加 blocks 提高性能

### 实现要点

1. **PWL 嵌入**：处理数值特征的关键
2. **Transformer 架构**：学习特征交互
3. **端到端训练**：从原始特征到预测
4. **训练技巧**：梯度裁剪、学习率调度、早停

### 适用场景

- ✅ 表格数据回归/分类任务
- ✅ 特征交互复杂
- ✅ 有 GPU 资源
- ✅ 追求高性能

### 局限性

- ❌ 训练时间长
- ❌ 需要 GPU（CPU 训练很慢）
- ❌ 需要调参
- ❌ 内存占用较大

---

## 参考资料

- **pytabkit**: https://github.com/georgian-io/pytabkit
- **Transformer 论文**: "Attention Is All You Need"
- **TabM 相关研究**: 表格数据深度学习方法

