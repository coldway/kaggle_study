#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Accident Risk Prediction - Complete Script
从 notebook s5e10-tabm-tuned-further.ipynb 改写而来
支持 GPU/MPS/CPU 自动检测，支持训练、预测、HTTP服务
支持 Windows/macOS/Linux 本地运行和 Docker 运行
"""

import os
import sys
import platform
import argparse
import warnings
import pickle
import json
warnings.simplefilter('ignore')

# ============================================================================
# 环境检测和配置
# ============================================================================

def detect_environment():
    """检测本地环境配置"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        # 检测 Apple Silicon MPS 支持
        mps_available = False
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except:
            pass
        
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'pytorch_version': torch.__version__,
            'cuda_available': cuda_available,
            'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
            'mps_available': mps_available,
        }
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                env_info[f'gpu_{i}_name'] = gpu_props.name
                env_info[f'gpu_{i}_memory'] = f"{gpu_props.total_memory / 1024**3:.2f} GB"
        elif mps_available:
            env_info['mps_device'] = 'Apple Silicon (MPS)'
            env_info['mps_device_name'] = platform.processor() if hasattr(platform, 'processor') else 'Apple Silicon'
    except ImportError:
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'pytorch_version': 'Not installed',
            'cuda_available': False,
            'cuda_device_count': 0,
            'mps_available': False,
        }
    return env_info

def print_environment_info(env_info):
    """打印环境信息"""
    print('=' * 70)
    print('Environment Information:')
    print('=' * 70)
    for key, value in env_info.items():
        if not key.startswith('gpu_') and not key.startswith('mps_'):
            print(f'{key:25s}: {value}')
    gpu_keys = [k for k in env_info.keys() if k.startswith('gpu_')]
    if gpu_keys:
        print('-' * 70)
        print('GPU Information (CUDA):')
        for key in sorted(gpu_keys):
            print(f'  {key:23s}: {env_info[key]}')
    if env_info.get('mps_available', False):
        print('-' * 70)
        print('GPU Information (Apple Silicon MPS):')
        if 'mps_device' in env_info:
            print(f'  {"device":23s}: {env_info["mps_device"]}')
        if 'mps_device_name' in env_info:
            print(f'  {"device_name":23s}: {env_info["mps_device_name"]}')
    print('=' * 70)

def install_dependencies():
    """安装必要的依赖"""
    print('*** Installing/checking dependencies...')
    import subprocess
    try:
        import pytabkit
        print('  ✓ pytabkit already installed')
    except ImportError:
        print('  → Installing pytabkit...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pytabkit'])

# ============================================================================
# 主代码（从 notebook 提取并修改路径）
# ============================================================================

def run_training_pipeline(data_dir='./playground-series-s5e10', output_dir='./output'):
    """执行完整的训练和预测流程"""
    
    import warnings
    warnings.simplefilter('ignore')

    import pandas as pd
    import numpy as np

    # 数据路径配置（本地路径）
    data_dir = os.path.normpath(data_dir)
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # 原始数据文件（可能不存在）
    orig_file = os.path.join(data_dir, 'synthetic_road_accidents_100k.csv')
    if os.path.exists(orig_file):
        orig = pd.read_csv(orig_file)
        print('Train Shape:', train.shape)
        print('Test Shape:', test.shape)
        print('Orig Shape:', orig.shape)
        USE_ORIG = True
    else:
        print('Train Shape:', train.shape)
        print('Test Shape:', test.shape)
        print(f'警告: 原始数据文件不存在 ({orig_file})，将跳过基于原始数据的特征工程')
        orig = None
        USE_ORIG = False

    print('\n训练数据预览:')
    print(train.head(3))

    # 特征定义
    TARGET = 'accident_risk'
    BASE = [col for col in train.columns if col not in ['id', TARGET]]
    CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 
            'time_of_day', 'holiday', 'school_season']

    print(f'{len(BASE)} Base Features:{BASE}')

    # 基于原始数据的特征（Orig Features）
    ORIG = []

    if USE_ORIG and orig is not None:
        for col in BASE:
            tmp = orig.groupby(col)[TARGET].mean()
            new_col_name = f"orig_{col}"
            tmp.name = new_col_name
            train = train.merge(tmp, on=col, how='left')
            test = test.merge(tmp, on=col, how='left')
            ORIG.append(new_col_name)
        print(len(ORIG), 'Orig Features Created!!')
    else:
        print('跳过 Orig Features 创建（原始数据不可用）')

    # 元特征（Meta Features）
    META = []

    dataframes = [train, test]
    if USE_ORIG and orig is not None:
        dataframes.append(orig)

    for df in dataframes:
        base_risk = (
            0.3 * df["curvature"] + 
            0.2 * (df["lighting"] == "night").astype(int) + 
            0.1 * (df["weather"] != "clear").astype(int) + 
            0.2 * (df["speed_limit"] >= 60).astype(int) + 
            0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
        )
        df['Meta'] = base_risk

    META.append('Meta')

    # 缺失值处理
    if USE_ORIG and orig is not None and 'orig_curvature' in train.columns:
        train['orig_curvature'] = train['orig_curvature'].fillna(orig[TARGET].mean())
        test['orig_curvature'] = test['orig_curvature'].fillna(orig[TARGET].mean())

    # 特征整合
    FEATURES = BASE + ORIG + META
    print(len(FEATURES), 'Features.')

    # 准备训练数据
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]

    # 交叉验证设置
    from sklearn.model_selection import KFold

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 抑制输出
    from contextlib import contextmanager

    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w", encoding='utf-8') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    # 模型参数配置
    params = {
        'batch_size': 'auto',
        'patience': 16,
        'allow_amp': False,
        'arch_type': 'tabm-mini',
        'tabm_k': 32,
        'gradient_clipping_norm': 1.0, 
        'share_training_batches': False,
        'lr': 0.000624068703424289,
        'weight_decay': 0.0019090968357478807,
        'n_blocks': 5,
        'd_block': 432, 
        'dropout': 0.0, 
        'num_emb_type': 'pwl',
        'd_embedding': 24,
        'num_emb_n_bins': 112,
    }

    # 导入模型和评估指标
    from pytabkit import TabM_D_Regressor
    from sklearn.metrics import root_mean_squared_error

    # 检测 GPU 并配置
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = False
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except:
            pass
        
        if cuda_available:
            use_gpu = True
            device = 'cuda'
            print(f'\n*** GPU 检测成功 (CUDA): {torch.cuda.get_device_name(0)}')
            print(f'*** 将使用 GPU 加速训练\n')
        elif mps_available:
            use_gpu = True
            device = 'mps'
            print(f'\n*** GPU 检测成功 (Apple Silicon MPS)')
            print(f'*** 将使用 Apple Silicon GPU 加速训练\n')
        else:
            use_gpu = False
            device = 'cpu'
            print('\n*** 未检测到 GPU，将使用 CPU 训练\n')
    except Exception as e:
        use_gpu = False
        device = 'cpu'
        print(f'\n*** GPU 检测失败: {e}')
        print('*** 将使用 CPU 训练\n')

    # 交叉验证训练
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    models = []  # 保存所有fold的模型

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'--- Fold {fold+1}/{N_SPLITS} ---')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 配置模型参数
        model_params = params.copy()
        if use_gpu:
            model_params['device'] = device
            if device == 'cuda':
                print(f'  → 使用 GPU (CUDA): {torch.cuda.get_device_name(0)}')
            elif device == 'mps':
                print(f'  → 使用 GPU (Apple Silicon MPS)')
        
        with suppress_stdout():
            model = TabM_D_Regressor(**model_params)
            model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test)
        models.append(model)  # 保存模型

        print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")
        
        # 清理 GPU 缓存
        if use_gpu:
            try:
                import torch
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                    except:
                        pass
            except:
                pass

    test_preds /= N_SPLITS

    print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")

    # 保存结果
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv(
        os.path.join(output_dir, 'oof_tabm_plus_origcol_tuned.csv'), 
        index=False
    )
    pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(
        os.path.join(output_dir, 'test_tabm_plus_origcol_tuned.csv'), 
        index=False
    )
    print(f'\n*** 预测结果已保存到 {output_dir} 目录')
    
    # 保存模型和元数据
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存每个fold的模型
    for fold, model in enumerate(models):
        model_path = os.path.join(model_dir, f'model_fold_{fold+1}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'  ✓ 保存模型: {model_path}')
    
    # 保存元数据
    metadata = {
        'FEATURES': FEATURES,
        'CATS': CATS,
        'BASE': BASE,
        'ORIG': ORIG,
        'META': META,
        'TARGET': TARGET,
        'params': params,
        'N_SPLITS': N_SPLITS,
        'USE_ORIG': USE_ORIG,
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'  ✓ 保存元数据: {metadata_path}')
    
    # 创建并保存集成模型包装器
    ensemble_model = EnsembleModel(models, metadata)
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    print(f'  ✓ 保存集成模型包装器: {ensemble_path}')
    print(f'  → 集成模型包含 {len(ensemble_model)} 个fold，预测时会自动平均结果')
    
    return models, metadata

# ============================================================================
# 模型加载和预测功能
# ============================================================================

class EnsembleModel:
    """集成模型包装器"""
    def __init__(self, models, metadata):
        self.models = models
        self.metadata = metadata
        self.n_folds = len(models)
    
    def predict(self, X):
        import numpy as np
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred
    
    def __len__(self):
        return self.n_folds

def load_model(model_dir, use_ensemble=True):
    """加载保存的模型和元数据"""
    model_dir = os.path.normpath(model_dir)
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    if use_ensemble and os.path.exists(ensemble_path):
        # 尝试使用 torch.load 加载模型
        try:
            import torch
            ensemble_model = torch.load(ensemble_path, weights_only=False)
        except Exception as e:
            # 如果 torch.load 失败（可能文件不是 PyTorch 格式），尝试使用自定义 Unpickler
            # 使用自定义 Unpickler 来正确加载 EnsembleModel 类
            # 这解决了从其他模块导入时 pickle 找不到类的问题
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 如果查找的是 EnsembleModel，确保从正确的模块加载
                    if name == 'EnsembleModel':
                        # 尝试从当前模块加载
                        if module == '__main__' or module == 'road_accident_risk':
                            return EnsembleModel
                        # 如果模块名不匹配，也返回 EnsembleModel（向后兼容）
                        return EnsembleModel
                    # 其他类使用默认查找
                    return super().find_class(module, name)
            
            with open(ensemble_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                ensemble_model = unpickler.load()
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f'✓ 成功加载集成模型（包含 {ensemble_model.n_folds} 个fold）')
        return ensemble_model, metadata
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'元数据文件不存在: {metadata_path}')
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    models = []
    for fold in range(1, metadata['N_SPLITS'] + 1):
        model_path = os.path.join(model_dir, f'model_fold_{fold}.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        # 使用 torch.load 来加载模型
        try:
            import torch
            model = torch.load(model_path, weights_only=False)
            models.append(model)
        except:
            # 如果 torch.load 失败，使用标准的 pickle.load
            with open(model_path, 'rb') as f:
                models.append(pickle.load(f))
    
    print(f'✓ 成功加载 {len(models)} 个模型')
    
    if use_ensemble:
        ensemble_model = EnsembleModel(models, metadata)
        return ensemble_model, metadata
    else:
        return models, metadata

def prepare_features(data, metadata, orig_data=None):
    """准备特征数据（包括特征工程）"""
    import pandas as pd
    import numpy as np
    
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
    
    if metadata['USE_ORIG'] and orig_data is not None:
        for col in metadata['BASE']:
            if f'orig_{col}' in metadata['ORIG']:
                tmp = orig_data.groupby(col)[metadata['TARGET']].mean()
                new_col_name = f"orig_{col}"
                tmp.name = new_col_name
                df = df.merge(tmp, on=col, how='left')
                if 'orig_curvature' in df.columns:
                    df['orig_curvature'] = df['orig_curvature'].fillna(orig_data[metadata['TARGET']].mean())
    elif metadata['USE_ORIG'] and metadata.get('ORIG'):
        for orig_col in metadata['ORIG']:
            if orig_col not in df.columns:
                df[orig_col] = 0.0
    
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk
    
    missing_features = set(metadata['FEATURES']) - set(df.columns)
    if missing_features:
        raise ValueError(f'缺少必需的特征: {missing_features}')
    
    return df[metadata['FEATURES']]

def predict_single(model_or_models, metadata, data, orig_data=None):
    """单次预测"""
    import numpy as np
    X = prepare_features(data, metadata, orig_data)
    
    if isinstance(model_or_models, EnsembleModel):
        predictions = model_or_models.predict(X)
    else:
        predictions = []
        for model in model_or_models:
            pred = model.predict(X)
            predictions.append(pred)
        predictions = np.mean(predictions, axis=0)
    
    return predictions[0] if len(predictions) == 1 else predictions

def predict_batch(model_or_models, metadata, data, orig_data=None):
    """批量预测"""
    import numpy as np
    X = prepare_features(data, metadata, orig_data)
    
    if isinstance(model_or_models, EnsembleModel):
        predictions = model_or_models.predict(X)
    else:
        predictions = []
        for model in model_or_models:
            pred = model.predict(X)
            predictions.append(pred)
        predictions = np.mean(predictions, axis=0)
    
    return predictions


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Road Accident Risk Prediction - Training and Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 训练模型
  python road_accident_risk.py --data-dir ./playground-series-s5e10 --output-dir ./output

  # 单次预测
  python road_accident_risk.py --mode predict --model-dir ./output/models --input data.json

  # 批量预测
  python road_accident_risk.py --mode predict --model-dir ./output/models --input data.csv

注意: HTTP服务请使用 api_server.py
        '''
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict'],
                       help='运行模式: train(训练), predict(预测)')
    parser.add_argument('--data-dir', type=str, default='./playground-series-s5e10',
                       help='数据目录路径（训练模式）')
    parser.add_argument('--skip-install', action='store_true',
                       help='跳过依赖安装')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录（训练模式）')
    parser.add_argument('--model-dir', type=str, default='./output/models',
                       help='模型目录（预测模式）')
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件路径（预测模式，支持JSON或CSV）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（预测模式）')
    parser.add_argument('--orig-data-path', type=str, default=None,
                       help='原始数据文件路径（用于特征工程，预测模式）')
    args = parser.parse_args()

    # 检测环境
    env_info = detect_environment()
    print_environment_info(env_info)

    # 安装依赖
    if not args.skip_install:
        install_dependencies()

    if args.mode == 'train':
        if not os.path.exists(args.data_dir):
            print(f'错误: 数据目录不存在: {args.data_dir}')
            return 1

        os.makedirs(args.output_dir, exist_ok=True)

        print('\n*** 开始执行训练和预测...')
        print(f'*** 数据目录: {args.data_dir}')
        print(f'*** 输出目录: {args.output_dir}')
        
        try:
            run_training_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)
            print('\n*** 训练和预测完成!')
            return 0
        except Exception as e:
            print(f'\n*** 错误: {e}')
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.mode == 'predict':
        if not args.input:
            print('错误: 预测模式需要指定 --input 参数')
            return 1
        
        if not os.path.exists(args.model_dir):
            print(f'错误: 模型目录不存在: {args.model_dir}')
            return 1
        
        if not os.path.exists(args.input):
            print(f'错误: 输入文件不存在: {args.input}')
            return 1
        
        print(f'\n*** 加载模型: {args.model_dir}')
        model_or_models, metadata = load_model(args.model_dir)
        
        orig_data = None
        if args.orig_data_path and os.path.exists(args.orig_data_path):
            import pandas as pd
            orig_data = pd.read_csv(args.orig_data_path)
            print(f'✓ 加载原始数据: {args.orig_data_path}')
        
        import pandas as pd
        if args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        elif args.input.endswith('.json'):
            import json
            with open(args.input, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if isinstance(json_data, list):
                data = pd.DataFrame(json_data)
            else:
                data = pd.DataFrame([json_data])
        else:
            print('错误: 不支持的文件格式，请使用CSV或JSON')
            return 1
        
        print(f'✓ 加载输入数据: {len(data)} 条记录')
        
        predictions = predict_batch(model_or_models, metadata, data, orig_data)
        data['prediction'] = predictions
        
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = f'{base_name}_predictions.csv'
        
        data.to_csv(output_path, index=False)
        print(f'\n✓ 预测完成，结果已保存到: {output_path}')
        print(f'\n预测结果预览:')
        print(data[['prediction']].head(10))
        
        return 0
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

