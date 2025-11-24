#!/usr/bin/env python3
"""
Road Accident Risk Prediction - Complete Script
从 notebook 改写而来，支持本地运行
"""

import os
import sys
import platform
import argparse
import warnings
warnings.simplefilter('ignore')

# ============================================================================
# 环境检测和配置
# ============================================================================

def detect_environment():
    """检测本地环境配置"""
    try:
        import torch
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError:
        env_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.system(),
            'pytorch_version': 'Not installed',
            'cuda_available': False,
            'cuda_device_count': 0,
        }
    return env_info

def print_environment_info(env_info):
    """打印环境信息"""
    print('=' * 60)
    print('Environment Information:')
    print('=' * 60)
    for key, value in env_info.items():
        print(f'{key}: {value}')
    print('=' * 60)

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
    
    # Cell 2 code
    import warnings
    warnings.simplefilter('ignore')

    # Cell 3 code
    import pandas as pd, numpy as np

    # 数据路径配置
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')

    # 原始数据文件（可能不存在，需要处理）
    orig_file = f'{data_dir}/synthetic_road_accidents_100k.csv'
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
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    # Cell 13 code
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

    # You may want to try these two sets of parameters as well
#params = {'batch_size': 'auto',
#          'patience': 16,
#          'allow_amp': False,
#          'arch_type': 'tabm-mini',
#          'tabm_k': 32,
#          'gradient_clipping_norm': 1.0, 
#          'share_training_batches': False,
#          'lr': 0.0017539221864098504,
#          'weight_decay': 0.0006814972152714441,
#          'n_blocks': 4,
#          'd_block': 128, 
#          'dropout': 0.0, 
#          'num_emb_type': 'pwl',
#          'd_embedding': 24,
#          'num_emb_n_bins': 59,
#         }
#
#params = {'batch_size': 'auto',
#          'patience': 16,
#          'allow_amp': False,
#          'arch_type': 'tabm-mini',
#          'tabm_k': 32,
#          'gradient_clipping_norm': 1.0, 
#          'share_training_batches': False,
#          'lr': 0.00024387748784930943,
#          'weight_decay': 0.0,
#          'n_blocks': 5,
#          'd_block': 512, 
#          'dropout': 0.0, 
#          'num_emb_type': 'pwl',
#          'd_embedding': 32,
#          'num_emb_n_bins': 54,
#         }

    # Cell 14 code
    from pytabkit import TabM_D_Regressor
    from sklearn.metrics import root_mean_squared_error

    # Cell 15 code
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'--- Fold {fold+1}/{N_SPLITS} ---')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        with suppress_stdout():
            model = TabM_D_Regressor(**params)
            model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test)

        print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")

    test_preds /= N_SPLITS

    print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")

    # Cell 16 code
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv(f'{output_dir}/oof_tabm_plus_origcol_tuned.csv', index=False)
    pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(f'{output_dir}/test_tabm_plus_origcol_tuned.csv', index=False)
    print(f'\n*** 预测结果已保存到 {output_dir}/ 目录')

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Road Accident Risk Prediction - Local Runner')
    parser.add_argument('--data-dir', type=str, default='./playground-series-s5e10',
                       help='数据目录路径')
    parser.add_argument('--orig-file', type=str, default=None,
                       help='原始数据文件路径（如果不存在则跳过）')
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
    if not os.path.exists(args.data_dir):
        print(f'错误: 数据目录不存在: {args.data_dir}')
        return 1

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置全局变量
    global DATA_DIR
    DATA_DIR = args.data_dir
    global output_dir
    output_dir = args.output_dir

    print('\n*** 开始执行训练和预测...')
    print(f'*** 数据目录: {args.data_dir}')
    print(f'*** 输出目录: {args.output_dir}')
    
    # 执行训练和预测流程
    try:
        run_training_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)
        print('\n*** 训练和预测完成!')
        return 0
    except Exception as e:
        print(f'\n*** 错误: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
