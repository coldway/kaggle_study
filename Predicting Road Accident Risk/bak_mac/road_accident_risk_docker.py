#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Accident Risk Prediction - Docker Version
从 notebook s5e10-tabm-tuned-further.ipynb 改写而来
支持 GPU/CPU 自动检测，可在 Docker 容器中运行
"""

import os
import sys
import warnings
warnings.simplefilter('ignore')

# ============================================================================
# 环境检测和配置（Docker 优化）
# ============================================================================

def detect_environment():
    """检测本地环境配置（Docker 优化）"""
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
            'pytorch_version': torch.__version__,
            'cuda_available': cuda_available,
            'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
            'mps_available': mps_available,
        }
        if cuda_available:
            # 获取 CUDA GPU 详细信息
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                env_info[f'gpu_{i}_name'] = gpu_props.name
                env_info[f'gpu_{i}_memory'] = f"{gpu_props.total_memory / 1024**3:.2f} GB"
                env_info[f'gpu_{i}_compute_capability'] = f"{gpu_props.major}.{gpu_props.minor}"
        elif mps_available:
            # Apple Silicon MPS 信息
            import platform
            env_info['mps_device'] = 'Apple Silicon (MPS)'
            env_info['mps_device_name'] = platform.processor() if hasattr(platform, 'processor') else 'Apple Silicon'
    except ImportError:
        env_info = {
            'python_version': sys.version.split()[0],
            'pytorch_version': 'Not installed',
            'cuda_available': False,
            'cuda_device_count': 0,
            'mps_available': False,
        }
    return env_info

def print_environment_info(env_info):
    """打印环境信息（Docker 优化格式）"""
    print('=' * 70)
    print('Environment Information (Docker)')
    print('=' * 70)
    basic_keys = ['python_version', 'pytorch_version', 'cuda_available', 'cuda_device_count', 'mps_available']
    for key in basic_keys:
        if key in env_info:
            print(f'{key:25s}: {env_info[key]}')
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
    
    # 检查 pytabkit
    try:
        import pytabkit
        print('  ✓ pytabkit already installed')
    except ImportError:
        print('  → Installing pytabkit...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pytabkit'])
    
    # 检查 PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print('  ✓ PyTorch with CUDA available')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print('  ✓ PyTorch with Apple Silicon MPS available')
        else:
            print('  ⚠ PyTorch installed but no GPU available (will use CPU)')
    except ImportError:
        print('  ⚠ PyTorch not installed')
        print('  → Installing PyTorch...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])

# ============================================================================
# 主代码（从 notebook 提取并修改路径）
# ============================================================================

def run_training_pipeline(data_dir='./playground-series-s5e10', output_dir='./output'):
    """执行完整的训练和预测流程"""
    
    # Cell 2: 忽略警告
    import warnings
    warnings.simplefilter('ignore')

    # Cell 3: 数据加载
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

    train.head(3)

    # Cell 4: 特征定义
    TARGET = 'accident_risk'
    BASE = [col for col in train.columns if col not in ['id', TARGET]]
    CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 
            'time_of_day', 'holiday', 'school_season']

    print(f'{len(BASE)} Base Features:{BASE}')

    # Cell 5: 基于原始数据的特征（Orig Features）
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

    # Cell 6: 元特征（Meta Features）
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

    # Cell 7: 缺失值处理
    if USE_ORIG and orig is not None and 'orig_curvature' in train.columns:
        train['orig_curvature'] = train['orig_curvature'].fillna(orig[TARGET].mean())
        test['orig_curvature'] = test['orig_curvature'].fillna(orig[TARGET].mean())

    # Cell 8: 特征整合
    FEATURES = BASE + ORIG + META
    print(len(FEATURES), 'Features.')

    # Cell 9: 准备训练数据
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]

    # Cell 10: 交叉验证设置
    from sklearn.model_selection import KFold

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Cell 12: 抑制输出
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

    # Cell 13: 模型参数配置
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

    # Cell 14: 导入模型和评估指标
    from pytabkit import TabM_D_Regressor
    from sklearn.metrics import root_mean_squared_error

    # Cell 15: 交叉验证训练
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    # 检测 GPU 并配置（支持 CUDA 和 Apple Silicon MPS）
    try:
        import torch
        import platform
        
        # 检测 CUDA (NVIDIA GPU)
        cuda_available = torch.cuda.is_available()
        
        # 检测 MPS (Apple Silicon)
        mps_available = False
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except:
            pass
        
        # 选择设备：优先 CUDA，其次 MPS，最后 CPU
        if cuda_available:
            use_gpu = True
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'\n*** GPU 检测成功 (CUDA): {gpu_name} ({gpu_memory:.2f} GB)')
            print(f'*** 将使用 GPU 加速训练\n')
        elif mps_available:
            use_gpu = True
            device = 'mps'
            processor = platform.processor() if hasattr(platform, 'processor') else 'Apple Silicon'
            print(f'\n*** GPU 检测成功 (Apple Silicon MPS): {processor}')
            print(f'*** 将使用 Apple Silicon GPU 加速训练\n')
            print(f'*** 注意: 某些操作可能回退到 CPU，这是正常的\n')
        else:
            use_gpu = False
            device = 'cpu'
            print('\n*** 未检测到 GPU，将使用 CPU 训练\n')
    except Exception as e:
        use_gpu = False
        device = 'cpu'
        print(f'\n*** GPU 检测失败: {e}')
        print('*** 将使用 CPU 训练\n')

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
            else:
                print(f'  → 使用设备: {device}')
        
        with suppress_stdout():
            model = TabM_D_Regressor(**model_params)
            model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test)

        print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")
        
        # 清理 GPU 缓存
        if use_gpu:
            try:
                import torch
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    # MPS 不需要手动清理缓存，但可以同步
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

    # Cell 16: 保存结果
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

# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Road Accident Risk Prediction - Docker Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python road_accident_risk_docker.py
  python road_accident_risk_docker.py --data-dir ./playground-series-s5e10 --output-dir ./output
  python road_accident_risk_docker.py --skip-install
        '''
    )
    parser.add_argument('--data-dir', type=str, default='./playground-series-s5e10',
                       help='数据目录路径')
    parser.add_argument('--skip-install', action='store_true',
                       help='跳过依赖安装')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    args = parser.parse_args()

    # 检测环境
    env_info = detect_environment()
    print_environment_info(env_info)

    # 安装依赖
    if not args.skip_install:
        install_dependencies()

    # 检查数据目录
    data_dir = os.path.normpath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f'错误: 数据目录不存在: {data_dir}')
        print(f'请确保数据目录存在并包含 train.csv 和 test.csv 文件')
        return 1

    # 创建输出目录
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('\n*** 开始执行训练和预测...')
    print(f'*** 数据目录: {data_dir}')
    print(f'*** 输出目录: {output_dir}\n')
    
    # 执行训练和预测流程
    try:
        run_training_pipeline(data_dir=data_dir, output_dir=output_dir)
        print('\n*** 训练和预测完成!')
        return 0
    except KeyboardInterrupt:
        print('\n*** 用户中断训练')
        return 130
    except Exception as e:
        print(f'\n*** 错误: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

