#!/usr/bin/env python3
"""
Road Accident Risk Prediction - Complete Script
从 notebook 改写而来，支持本地运行
支持模型保存、预测和HTTP服务
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
    models = []  # 保存所有fold的模型

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'--- Fold {fold+1}/{N_SPLITS} ---')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        with suppress_stdout():
            model = TabM_D_Regressor(**params)
            model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test)
        models.append(model)  # 保存模型

        print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val, oof_preds[val_idx]):.5f}")

    test_preds /= N_SPLITS

    print(f"Overall OOF RMSE: {root_mean_squared_error(y, oof_preds):.5f}")

    # Cell 16 code
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv(f'{output_dir}/oof_tabm_plus_origcol_tuned.csv', index=False)
    pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv(f'{output_dir}/test_tabm_plus_origcol_tuned.csv', index=False)
    print(f'\n*** 预测结果已保存到 {output_dir}/ 目录')
    
    # 保存模型和元数据
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存每个fold的模型（保留原始模型文件，便于调试和单独使用）
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
    
    # 创建并保存集成模型包装器（方便使用，看起来像单个模型）
    # 注意：这不是平均模型权重，而是包装器，预测时会自动平均所有fold的预测结果
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
    """
    集成模型包装器
    将多个fold的模型包装成一个统一的模型接口
    预测时自动对所有fold的预测结果进行平均
    """
    def __init__(self, models, metadata):
        """
        初始化集成模型
        
        Args:
            models: 模型列表（每个fold的模型）
            metadata: 模型元数据
        """
        self.models = models
        self.metadata = metadata
        self.n_folds = len(models)
    
    def predict(self, X):
        """
        预测方法（与单个模型接口一致）
        
        Args:
            X: 输入特征（DataFrame或numpy数组）
        
        Returns:
            预测结果（numpy数组）
        """
        import numpy as np
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        # 平均所有fold的预测
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred
    
    def __len__(self):
        """返回fold数量"""
        return self.n_folds

def load_model(model_dir, use_ensemble=True):
    """
    加载保存的模型和元数据
    
    Args:
        model_dir: 模型目录路径
        use_ensemble: 是否返回集成模型包装器（默认True）
                    如果False，返回原始模型列表
    
    Returns:
        如果use_ensemble=True: (EnsembleModel, metadata)
        如果use_ensemble=False: (models_list, metadata)
    """
    model_dir = os.path.normpath(model_dir)
    
    # 首先尝试加载集成模型（如果存在）
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    if use_ensemble and os.path.exists(ensemble_path):
        # 加载集成模型
        with open(ensemble_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f'✓ 成功加载集成模型（包含 {ensemble_model.n_folds} 个fold）')
        return ensemble_model, metadata
    
    # 加载元数据
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'元数据文件不存在: {metadata_path}')
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 加载所有fold的模型
    models = []
    for fold in range(1, metadata['N_SPLITS'] + 1):
        model_path = os.path.join(model_dir, f'model_fold_{fold}.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))
    
    print(f'✓ 成功加载 {len(models)} 个模型')
    
    if use_ensemble:
        # 创建并返回集成模型
        ensemble_model = EnsembleModel(models, metadata)
        return ensemble_model, metadata
    else:
        # 返回原始模型列表
        return models, metadata

def prepare_features(data, metadata, orig_data=None):
    """准备特征数据（包括特征工程）"""
    import pandas as pd
    import numpy as np
    
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
    
    # 添加Orig特征（如果可用）
    if metadata['USE_ORIG'] and orig_data is not None:
        for col in metadata['BASE']:
            if f'orig_{col}' in metadata['ORIG']:
                tmp = orig_data.groupby(col)[metadata['TARGET']].mean()
                new_col_name = f"orig_{col}"
                tmp.name = new_col_name
                df = df.merge(tmp, on=col, how='left')
                # 填充缺失值
                if 'orig_curvature' in df.columns:
                    df['orig_curvature'] = df['orig_curvature'].fillna(orig_data[metadata['TARGET']].mean())
    elif metadata['USE_ORIG'] and metadata.get('ORIG'):
        # 如果没有原始数据但需要ORIG特征，使用默认值（训练时的平均值）
        # 注意：这只是一个fallback，最好提供原始数据
        for orig_col in metadata['ORIG']:
            if orig_col not in df.columns:
                # 使用0作为默认值（实际应用中应该使用训练时的平均值）
                df[orig_col] = 0.0
    
    # 添加Meta特征
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk
    
    # 确保所有必需的特征都存在
    missing_features = set(metadata['FEATURES']) - set(df.columns)
    if missing_features:
        raise ValueError(f'缺少必需的特征: {missing_features}')
    
    return df[metadata['FEATURES']]

def predict_single(model_or_models, metadata, data, orig_data=None):
    """
    单次预测
    
    Args:
        model_or_models: 可以是EnsembleModel对象或模型列表
        metadata: 模型元数据
        data: 输入数据（字典或DataFrame）
        orig_data: 原始数据（用于特征工程，可选）
    
    Returns:
        预测结果（单个值或数组）
    """
    import numpy as np
    X = prepare_features(data, metadata, orig_data)
    
    # 如果传入的是EnsembleModel，直接使用其predict方法
    if isinstance(model_or_models, EnsembleModel):
        predictions = model_or_models.predict(X)
    else:
        # 如果是模型列表，手动平均
        predictions = []
        for model in model_or_models:
            pred = model.predict(X)
            predictions.append(pred)
        predictions = np.mean(predictions, axis=0)
    
    return predictions[0] if len(predictions) == 1 else predictions

def predict_batch(model_or_models, metadata, data, orig_data=None):
    """
    批量预测
    
    Args:
        model_or_models: 可以是EnsembleModel对象或模型列表
        metadata: 模型元数据
        data: 输入数据（DataFrame）
        orig_data: 原始数据（用于特征工程，可选）
    
    Returns:
        预测结果（numpy数组）
    """
    import numpy as np
    X = prepare_features(data, metadata, orig_data)
    
    # 如果传入的是EnsembleModel，直接使用其predict方法
    if isinstance(model_or_models, EnsembleModel):
        predictions = model_or_models.predict(X)
    else:
        # 如果是模型列表，手动平均
        predictions = []
        for model in model_or_models:
            pred = model.predict(X)
            predictions.append(pred)
        predictions = np.mean(predictions, axis=0)
    
    return predictions

# ============================================================================
# HTTP服务功能
# ============================================================================

def create_app(model_dir, orig_data_path=None):
    """创建Flask应用"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print('错误: Flask未安装，请运行: pip install flask')
        sys.exit(1)
    
    app = Flask(__name__)
    
    # 加载模型
    print(f'正在加载模型: {model_dir}')
    model_or_models, metadata = load_model(model_dir, use_ensemble=True)
    
    # 加载原始数据（如果可用）
    orig_data = None
    if orig_data_path and os.path.exists(orig_data_path):
        import pandas as pd
        orig_data = pd.read_csv(orig_data_path)
        print(f'✓ 加载原始数据: {orig_data_path}')
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({'status': 'healthy', 'model_loaded': True})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """预测接口（支持单次和批量）"""
        try:
            data = request.json
            if not data:
                return jsonify({'error': '请求体为空'}), 400
            
            # 支持单条或多条数据
            if isinstance(data, dict):
                # 单条数据
                import pandas as pd
                df = pd.DataFrame([data])
                predictions = predict_batch(model_or_models, metadata, df, orig_data)
                result = {'prediction': float(predictions[0])}
            elif isinstance(data, list):
                # 批量数据
                import pandas as pd
                df = pd.DataFrame(data)
                predictions = predict_batch(model_or_models, metadata, df, orig_data)
                result = {'predictions': [float(p) for p in predictions]}
            else:
                return jsonify({'error': '无效的数据格式'}), 400
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch_endpoint():
        """批量预测接口"""
        try:
            data = request.json
            if not data or not isinstance(data, list):
                return jsonify({'error': '请求体必须是数组格式'}), 400
            
            import pandas as pd
            df = pd.DataFrame(data)
            predictions = predict_batch(model_or_models, metadata, df, orig_data)
            result = {'predictions': [float(p) for p in predictions]}
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model/info', methods=['GET'])
    def model_info():
        """获取模型信息"""
        return jsonify({
            'n_folds': metadata['N_SPLITS'],
            'features': metadata['FEATURES'],
            'n_features': len(metadata['FEATURES']),
            'target': metadata['TARGET'],
        })
    
    return app

def run_server(model_dir, host='0.0.0.0', port=5000, orig_data_path=None, debug=False):
    """运行HTTP服务"""
    app = create_app(model_dir, orig_data_path)
    print(f'\n*** HTTP服务启动')
    print(f'*** 地址: http://{host}:{port}')
    print(f'*** 健康检查: http://{host}:{port}/health')
    print(f'*** 预测接口: http://{host}:{port}/predict')
    print(f'*** 批量预测: http://{host}:{port}/predict/batch')
    print(f'*** 模型信息: http://{host}:{port}/model/info')
    print(f'\n按 Ctrl+C 停止服务\n')
    app.run(host=host, port=port, debug=debug)

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Road Accident Risk Prediction - Local Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 训练模型
  python road_accident_risk_mac.py --data-dir ./playground-series-s5e10 --output-dir ./output

  # 单次预测
  python road_accident_risk_mac.py --mode predict --model-dir ./output/models --input data.json

  # 批量预测
  python road_accident_risk_mac.py --mode predict --model-dir ./output/models --input data.csv

  # 启动HTTP服务
  python road_accident_risk_mac.py --mode serve --model-dir ./output/models --port 5000
        '''
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict', 'serve'],
                       help='运行模式: train(训练), predict(预测), serve(HTTP服务)')
    parser.add_argument('--data-dir', type=str, default='./playground-series-s5e10',
                       help='数据目录路径（训练模式）')
    parser.add_argument('--orig-file', type=str, default=None,
                       help='原始数据文件路径（如果不存在则跳过）')
    parser.add_argument('--skip-install', action='store_true',
                       help='跳过依赖安装')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录（训练模式）')
    parser.add_argument('--model-dir', type=str, default='./output/models',
                       help='模型目录（预测/服务模式）')
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件路径（预测模式，支持JSON或CSV）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（预测模式）')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='HTTP服务主机地址（服务模式）')
    parser.add_argument('--port', type=int, default=5000,
                       help='HTTP服务端口（服务模式）')
    parser.add_argument('--orig-data-path', type=str, default=None,
                       help='原始数据文件路径（用于特征工程，预测/服务模式）')
    args = parser.parse_args()

    # 检测环境
    env_info = detect_environment()
    print_environment_info(env_info)

    # 安装依赖
    if not args.skip_install:
        install_dependencies()

    if args.mode == 'train':
        # 训练模式
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
        # 预测模式
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
        model_or_models, metadata = load_model(args.model_dir, use_ensemble=True)
        
        # 加载原始数据（如果可用）
        orig_data = None
        if args.orig_data_path and os.path.exists(args.orig_data_path):
            import pandas as pd
            orig_data = pd.read_csv(args.orig_data_path)
            print(f'✓ 加载原始数据: {args.orig_data_path}')
        
        # 读取输入数据
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
        
        # 执行预测
        predictions = predict_batch(model_or_models, metadata, data, orig_data)
        data['prediction'] = predictions
        
        # 保存结果
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
    
    elif args.mode == 'serve':
        # HTTP服务模式
        if not os.path.exists(args.model_dir):
            print(f'错误: 模型目录不存在: {args.model_dir}')
            return 1
        
        try:
            run_server(
                model_dir=args.model_dir,
                host=args.host,
                port=args.port,
                orig_data_path=args.orig_data_path,
                debug=False
            )
        except KeyboardInterrupt:
            print('\n*** 服务已停止')
            return 0
        except Exception as e:
            print(f'\n*** 错误: {e}')
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
