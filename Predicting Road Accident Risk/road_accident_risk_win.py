#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Accident Risk Prediction - Windows 10 Optimized Version
专门针对 Windows 10 + RTX 5070 12GB + Intel i5-12600KF 优化
从 notebook 改写而来，支持本地运行和 GPU 加速
"""

import os
import sys
import platform
import argparse
import warnings
warnings.simplefilter('ignore')

# ============================================================================
# 环境检测和配置（Windows 优化）
# ============================================================================

def detect_environment():
    """检测本地环境配置（Windows 优化）"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'platform_version': platform.version(),
            'pytorch_version': torch.__version__,
            'cuda_available': cuda_available,
            'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
        }
        if cuda_available:
            # 获取 GPU 详细信息
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                env_info[f'gpu_{i}_name'] = gpu_props.name
                env_info[f'gpu_{i}_memory'] = f"{gpu_props.total_memory / 1024**3:.2f} GB"
                env_info[f'gpu_{i}_compute_capability'] = f"{gpu_props.major}.{gpu_props.minor}"
                # 获取当前 GPU 使用情况
                try:
                    import torch.cuda
                    env_info[f'gpu_{i}_allocated'] = f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
                    env_info[f'gpu_{i}_reserved'] = f"{torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
                except:
                    pass
    except ImportError:
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'platform_version': platform.version(),
            'pytorch_version': 'Not installed',
            'cuda_available': False,
            'cuda_device_count': 0,
        }
    return env_info

def print_environment_info(env_info):
    """打印环境信息（Windows 优化格式）"""
    print('=' * 70)
    print('Environment Information (Windows 10 Optimized)')
    print('=' * 70)
    # 先打印基本信息
    basic_keys = ['python_version', 'platform', 'platform_version', 'pytorch_version', 
                  'cuda_available', 'cuda_device_count']
    for key in basic_keys:
        if key in env_info:
            print(f'{key:25s}: {env_info[key]}')
    # 再打印 GPU 详细信息
    gpu_keys = [k for k in env_info.keys() if k.startswith('gpu_')]
    if gpu_keys:
        print('-' * 70)
        print('GPU Information:')
        for key in sorted(gpu_keys):
            print(f'  {key:23s}: {env_info[key]}')
    print('=' * 70)

def install_dependencies():
    """安装必要的依赖（Windows 优化）"""
    print('*** Installing/checking dependencies...')
    import subprocess
    
    # 检查 pytabkit
    try:
        import pytabkit
        print('  ✓ pytabkit already installed')
    except ImportError:
        print('  → Installing pytabkit...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pytabkit'])
    
    # 检查 PyTorch with CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print('  ⚠ PyTorch installed but CUDA not available')
            print('  → To enable GPU support, install PyTorch with CUDA:')
            print('     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
        else:
            print('  ✓ PyTorch with CUDA available')
    except ImportError:
        print('  ⚠ PyTorch not installed')
        print('  → To install PyTorch with CUDA support:')
        print('     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')

# ============================================================================
# 主代码（从 notebook 提取并修改路径，Windows 优化）
# ============================================================================

def run_training_pipeline(data_dir='./playground-series-s5e10', output_dir='./output'):
    """执行完整的训练和预测流程（Windows + GPU 优化）"""
    
    # Cell 2 code
    import warnings
    warnings.simplefilter('ignore')

    # Cell 3 code
    import pandas as pd, numpy as np

    # 数据路径配置（Windows 路径兼容）
    data_dir = os.path.normpath(data_dir)  # 规范化路径
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # 原始数据文件（可能不存在，需要处理）
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

    # Cell 4 code
    TARGET = 'accident_risk'
    BASE = [col for col in train.columns if col not in ['id', TARGET]]
    CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

    print(f'{len(BASE)} Base Features:{BASE}')

    # Cell 5 code
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

    # Cell 6 code
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

    # Cell 7 code
    if USE_ORIG and orig is not None and 'orig_curvature' in train.columns:
        train['orig_curvature'] = train['orig_curvature'].fillna(orig[TARGET].mean())
        test['orig_curvature'] = test['orig_curvature'].fillna(orig[TARGET].mean())

    # Cell 8 code
    FEATURES = BASE + ORIG + META
    print(len(FEATURES), 'Features.')

    # Cell 9 code
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]

    # Cell 10 code
    from sklearn.model_selection import KFold

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Cell 12 code
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

    # Cell 13 code - Windows + RTX 5070 优化参数
    params = {
        'batch_size': 'auto',  # 自动优化，RTX 5070 12GB 可以支持较大批次
        'patience': 16,
        'allow_amp': False,  # 可以尝试 True 以加速训练（RTX 5070 支持）
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

    # Cell 14 code
    from pytabkit import TabM_D_Regressor
    from sklearn.metrics import root_mean_squared_error

    # Cell 15 code
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    # 检测 GPU 并配置（Windows + RTX 5070 优化）
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        if use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'\n*** GPU 检测成功: {gpu_name} ({gpu_memory:.2f} GB)')
            print(f'*** 将使用 GPU 加速训练\n')
        else:
            print('\n*** 未检测到 GPU，将使用 CPU 训练（速度较慢）\n')
    except:
        use_gpu = False
        device = 'cpu'
        print('\n*** GPU 检测失败，将使用 CPU 训练\n')

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'--- Fold {fold+1}/{N_SPLITS} ---')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 配置模型参数（GPU 优化）
        model_params = params.copy()
        if use_gpu:
            model_params['device'] = device
            print(f'  → 使用 GPU: {torch.cuda.get_device_name(0)}')
            # 显示 GPU 内存使用情况
            if fold == 0:  # 只在第一个 fold 显示
                print(f'  → GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        
        with suppress_stdout():
            model = TabM_D_Regressor(**model_params)
            model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test)

        print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")
        
        # 清理 GPU 缓存（Windows 优化）
        if use_gpu:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    test_preds /= N_SPLITS

    print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")

    # Cell 16 code
    output_dir = os.path.normpath(output_dir)  # 规范化路径
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
# 主函数（Windows 优化）
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Road Accident Risk Prediction - Windows 10 Optimized (RTX 5070)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python road_accident_risk_win.py
  python road_accident_risk_win.py --data-dir .\\playground-series-s5e10 --output-dir .\\output
  python road_accident_risk_win.py --skip-install
        '''
    )
    parser.add_argument('--data-dir', type=str, default='./playground-series-s5e10',
                       help='数据目录路径（Windows 路径格式）')
    parser.add_argument('--orig-file', type=str, default=None,
                       help='原始数据文件路径（如果不存在则跳过）')
    parser.add_argument('--skip-install', action='store_true',
                       help='跳过依赖安装')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录（Windows 路径格式）')
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
    print(f'*** 输出目录: {output_dir}')
    print(f'*** 系统: Windows 10')
    print(f'*** 优化: RTX 5070 12GB + Intel i5-12600KF\n')
    
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

