#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Accident Risk Prediction - API Server
生产环境HTTP服务，专门用于模型预测服务
"""

import os
import sys
import argparse
import warnings
warnings.simplefilter('ignore')

# 检测是否在 Docker 环境中
# 在 Docker 中，禁用 MPS 设备（因为 Docker 不支持 MPS）
# 在本地 Python3 环境中，允许使用 MPS
IS_DOCKER = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'

# 在导入任何 PyTorch 相关模块之前，彻底禁用 MPS（如果在 Docker 中）
# 这必须在导入 road_accident_risk 之前完成，因为 road_accident_risk 会导入 pytabkit
# 而 pytabkit 可能在导入时检查 MPS 可用性
if IS_DOCKER:
    # Docker 环境：彻底禁用 MPS，强制使用 CPU
    # 必须在导入任何可能使用 MPS 的模块之前完成
    try:
        import torch
        
        # 创建一个始终返回 False 的类（比 lambda 更可靠）
        class MPSDisabled:
            def __call__(self, *args, **kwargs):
                return False
            def __bool__(self):
                return False
            def __repr__(self):
                return 'False'
        
        mps_disabled = MPSDisabled()
        
        # 禁用所有 MPS 相关的检查
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = mps_disabled
            if hasattr(torch.backends.mps, 'is_built'):
                torch.backends.mps.is_built = mps_disabled
        
        if hasattr(torch, 'mps'):
            if hasattr(torch.mps, 'is_available'):
                torch.mps.is_available = mps_disabled
            if hasattr(torch.mps, 'is_built'):
                torch.mps.is_built = mps_disabled
        
        # 替换设备恢复函数，确保加载时不会使用 MPS
        import torch.serialization
        _original_restore_location = torch.serialization.default_restore_location
        
        def _cpu_restore_location(storage, location):
            """将所有设备映射到 CPU"""
            if isinstance(location, str):
                if 'mps' in location.lower() or 'cuda' in location.lower():
                    location = 'cpu'
            elif isinstance(location, dict):
                device_type = location.get('device_type', '')
                if device_type in ['mps', 'cuda'] or 'mps' in str(location).lower():
                    location = 'cpu'
            return _original_restore_location(storage, location)
        
        torch.serialization.default_restore_location = _cpu_restore_location
        
        # 替换 _mps_deserialize（如果存在）
        if hasattr(torch.serialization, '_mps_deserialize'):
            def _mps_deserialize_cpu(obj, location):
                return obj.cpu()
            torch.serialization._mps_deserialize = _mps_deserialize_cpu
        
        # 拦截可能的 MPS 设备创建
        # 包装 torch.device 函数（如果存在）
        if hasattr(torch, 'device'):
            _original_device = torch.device
            def _device_wrapper(device):
                if isinstance(device, str) and 'mps' in device.lower():
                    return _original_device('cpu')
                return _original_device(device)
            torch.device = _device_wrapper
        
        print('✓ Docker 环境：MPS 设备已彻底禁用，将使用 CPU')
    except ImportError:
        pass
else:
    # 本地环境：允许使用 MPS（如果可用）
    print('✓ 本地环境：支持 MPS 设备（如果可用）')

# 导入模型加载和预测功能
# 注意：此时如果是在 Docker 中，MPS 已经被禁用
from road_accident_risk import load_model, predict_batch, EnsembleModel

# 导入后再次确保 MPS 被禁用（防止导入过程中被重新启用）
if IS_DOCKER:
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False
            if hasattr(torch.backends.mps, 'is_built'):
                torch.backends.mps.is_built = lambda: False
        if hasattr(torch, 'mps'):
            if hasattr(torch.mps, 'is_available'):
                torch.mps.is_available = lambda: False
            if hasattr(torch.mps, 'is_built'):
                torch.mps.is_built = lambda: False
    except:
        pass

def create_app(model_dir, orig_data_path=None):
    """创建Flask应用"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print('错误: Flask未安装，请运行: pip install flask')
        sys.exit(1)
    
    app = Flask(__name__)
    
    print(f'正在加载模型: {model_dir}')
    model_or_models, metadata = load_model(model_dir)
    
    # 验证加载的模型类型
    # 如果 load_model 返回的不是 EnsembleModel 对象，尝试修复
    if not isinstance(model_or_models, EnsembleModel):
        # 如果返回的是整数或其他类型，说明加载失败，尝试重新加载
        if isinstance(model_or_models, (int, float, str)):
            print(f'警告: 模型加载返回了意外类型: {type(model_or_models)}')
            print(f'尝试重新加载模型...')
            # 强制使用 pickle 加载
            import pickle
            ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    model_or_models = pickle.load(f)
                if not isinstance(model_or_models, EnsembleModel):
                    raise ValueError(f'加载的模型不是 EnsembleModel 类型: {type(model_or_models)}')
    
    print('✓ 模型加载完成')
    
    orig_data = None
    if orig_data_path and os.path.exists(orig_data_path):
        import pandas as pd
        orig_data = pd.read_csv(orig_data_path)
        print(f'✓ 加载原始数据: {orig_data_path}')
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查接口"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'service': 'road-accident-risk-api'
        })
    
    @app.route('/ready', methods=['GET'])
    def ready():
        """就绪检查接口（用于Kubernetes）"""
        if model_or_models is None:
            return jsonify({'status': 'not ready'}), 503
        return jsonify({'status': 'ready'}), 200
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """预测接口（支持单次和批量）"""
        try:
            # 在预测前再次确保 MPS 被禁用（如果在 Docker 中）
            if IS_DOCKER:
                try:
                    import torch
                    if hasattr(torch.backends, 'mps'):
                        torch.backends.mps.is_available = lambda: False
                        if hasattr(torch.backends.mps, 'is_built'):
                            torch.backends.mps.is_built = lambda: False
                    if hasattr(torch, 'mps'):
                        if hasattr(torch.mps, 'is_available'):
                            torch.mps.is_available = lambda: False
                        if hasattr(torch.mps, 'is_built'):
                            torch.mps.is_built = lambda: False
                except:
                    pass
            
            data = request.json
            if not data:
                return jsonify({'error': '请求体为空'}), 400
            
            import pandas as pd
            if isinstance(data, dict):
                # 单次预测
                df = pd.DataFrame([data])
                predictions = predict_batch(model_or_models, metadata, df, orig_data)
                result = {'prediction': float(predictions[0])}
            elif isinstance(data, list):
                # 批量预测
                df = pd.DataFrame(data)
                predictions = predict_batch(model_or_models, metadata, df, orig_data)
                result = {'predictions': [float(p) for p in predictions]}
            else:
                return jsonify({'error': '无效的数据格式'}), 400
            
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': f'输入验证失败: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch_endpoint():
        """批量预测接口"""
        try:
            # 在预测前再次确保 MPS 被禁用（如果在 Docker 中）
            if IS_DOCKER:
                try:
                    import torch
                    if hasattr(torch.backends, 'mps'):
                        torch.backends.mps.is_available = lambda: False
                        if hasattr(torch.backends.mps, 'is_built'):
                            torch.backends.mps.is_built = lambda: False
                    if hasattr(torch, 'mps'):
                        if hasattr(torch.mps, 'is_available'):
                            torch.mps.is_available = lambda: False
                        if hasattr(torch.mps, 'is_built'):
                            torch.mps.is_built = lambda: False
                except:
                    pass
            
            data = request.json
            if not data or not isinstance(data, list):
                return jsonify({'error': '请求体必须是数组格式'}), 400
            
            import pandas as pd
            df = pd.DataFrame(data)
            predictions = predict_batch(model_or_models, metadata, df, orig_data)
            result = {'predictions': [float(p) for p in predictions]}
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': f'输入验证失败: {str(e)}'}), 400
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

def main():
    parser = argparse.ArgumentParser(
        description='Road Accident Risk Prediction - API Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 启动HTTP服务
  python api_server.py --model-dir ./output/models --port 5000

  # 使用原始数据进行特征工程
  python api_server.py --model-dir ./output/models \\
      --orig-data-path ./playground-series-s5e10/synthetic_road_accidents_100k.csv
        '''
    )
    parser.add_argument('--model-dir', type=str, required=True,
                       help='模型目录路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务主机地址（默认: 0.0.0.0）')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务端口（默认: 5000）')
    parser.add_argument('--orig-data-path', type=str, default=None,
                       help='原始数据文件路径（用于特征工程）')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式（仅开发环境）')
    args = parser.parse_args()
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f'错误: 模型目录不存在: {args.model_dir}')
        return 1
    
    try:
        app = create_app(args.model_dir, args.orig_data_path)
        
        print(f'\n*** HTTP服务启动')
        print(f'*** 地址: http://{args.host}:{args.port}')
        print(f'*** 健康检查: http://{args.host}:{args.port}/health')
        print(f'*** 就绪检查: http://{args.host}:{args.port}/ready')
        print(f'*** 预测接口: http://{args.host}:{args.port}/predict')
        print(f'*** 批量预测: http://{args.host}:{args.port}/predict/batch')
        print(f'*** 模型信息: http://{args.host}:{args.port}/model/info')
        print(f'\n按 Ctrl+C 停止服务\n')
        
        # 生产环境建议使用 gunicorn 或 uwsgi
        # gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print('\n*** 服务已停止')
        return 0
    except Exception as e:
        print(f'\n*** 错误: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
