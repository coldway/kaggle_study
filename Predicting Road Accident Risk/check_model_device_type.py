#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型文件的设备类型（MPS/CPU/CUDA）
"""

import os
import sys
import pickle
import argparse

def check_tensor_device(obj, path="", max_depth=5, devices_found=None):
    """递归检查对象中的 PyTorch 张量设备"""
    if devices_found is None:
        devices_found = set()
    
    if max_depth <= 0:
        return devices_found
    
    try:
        import torch
        
        # 如果是 PyTorch 张量
        if isinstance(obj, torch.Tensor):
            device_str = str(obj.device)
            devices_found.add(device_str)
            if 'mps' in device_str.lower():
                devices_found.add('MPS')
            elif 'cuda' in device_str.lower():
                devices_found.add('CUDA')
            elif 'cpu' in device_str.lower():
                devices_found.add('CPU')
            return devices_found
        
        # 如果是 PyTorch 模型
        if hasattr(obj, 'parameters'):
            try:
                for param in obj.parameters():
                    device_str = str(param.device)
                    devices_found.add(device_str)
                    if 'mps' in device_str.lower():
                        devices_found.add('MPS')
                    elif 'cuda' in device_str.lower():
                        devices_found.add('CUDA')
                    elif 'cpu' in device_str.lower():
                        devices_found.add('CPU')
            except:
                pass
        
        # 检查对象的设备属性（TabM 模型可能有 device 属性）
        for attr_name in ['device', '_device', 'training_device', '_training_device']:
            if hasattr(obj, attr_name):
                try:
                    device_attr = getattr(obj, attr_name)
                    if isinstance(device_attr, str):
                        devices_found.add(device_attr)
                        if 'mps' in device_attr.lower():
                            devices_found.add('MPS')
                        elif 'cuda' in device_attr.lower():
                            devices_found.add('CUDA')
                        elif 'cpu' in device_attr.lower():
                            devices_found.add('CPU')
                except:
                    pass
        
        # 如果是字典
        if isinstance(obj, dict):
            for key, value in obj.items():
                check_tensor_device(value, f"{path}.{key}", max_depth-1, devices_found)
        
        # 如果是列表或元组
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                check_tensor_device(item, f"{path}[{i}]", max_depth-1, devices_found)
        
        # 如果是对象
        elif hasattr(obj, '__dict__'):
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):
                            check_tensor_device(attr_value, f"{path}.{attr_name}", max_depth-1, devices_found)
                    except:
                        pass
    except Exception as e:
        pass
    
    return devices_found

def check_model_device(model_path):
    """检查单个模型文件的设备类型"""
    print(f'\n检查模型: {os.path.basename(model_path)}')
    
    try:
        import torch
        
        # 方法1: 尝试使用 torch.load（不使用 map_location，保持原始设备）
        try:
            # 不使用 map_location，让模型按原始设备加载
            model = torch.load(model_path, weights_only=False)
            
            print(f'  → 使用 torch.load 加载成功')
            devices = check_tensor_device(model)
            print(f'  → 检测到的设备: {devices}')
            
            if 'MPS' in devices or any('mps' in str(d).lower() for d in devices):
                print(f'  ⚠️  包含 MPS 设备（需要转换）')
                return 'mps'
            elif 'CUDA' in devices or any('cuda' in str(d).lower() for d in devices):
                print(f'  ⚠️  包含 CUDA 设备（需要转换）')
                return 'cuda'
            elif 'CPU' in devices or all('cpu' in str(d).lower() for d in devices if d):
                print(f'  ✓ 仅包含 CPU 设备')
                return 'cpu'
            else:
                print(f'  ? 未检测到明确的设备信息')
                return 'unknown'
        except Exception as e:
            print(f'  → torch.load 失败: {str(e)[:100]}')
            
            # 方法2: 尝试使用 pickle.load
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                print(f'  → 使用 pickle.load 加载成功')
                devices = check_tensor_device(model)
                print(f'  → 检测到的设备: {devices}')
                
                if 'MPS' in devices or any('mps' in str(d).lower() for d in devices):
                    print(f'  ⚠️  包含 MPS 设备（需要转换）')
                    return 'mps'
                elif 'CUDA' in devices or any('cuda' in str(d).lower() for d in devices):
                    print(f'  ⚠️  包含 CUDA 设备（需要转换）')
                    return 'cuda'
                elif 'CPU' in devices or all('cpu' in str(d).lower() for d in devices if d):
                    print(f'  ✓ 仅包含 CPU 设备')
                    return 'cpu'
                else:
                    print(f'  ? 未检测到明确的设备信息')
                    return 'unknown'
            except Exception as e2:
                print(f'  ✗ pickle.load 也失败: {str(e2)[:100]}')
                # 最后尝试：检查文件内容
                with open(model_path, 'rb') as f:
                    data = f.read(10240)
                    if b'mps' in data.lower():
                        print(f'  ⚠️  文件内容包含 "mps" 字符串（可能是 MPS 设备）')
                        return 'mps'
                    elif b'cuda' in data.lower():
                        print(f'  ⚠️  文件内容包含 "cuda" 字符串（可能是 CUDA 设备）')
                        return 'cuda'
                    else:
                        print(f'  ? 无法确定设备类型')
                        return 'unknown'
    except ImportError:
        print(f'  → PyTorch 未安装，无法检查设备')
        return 'error'

def check_ensemble_model(ensemble_path):
    """检查集成模型"""
    print(f'\n检查集成模型: {os.path.basename(ensemble_path)}')
    
    try:
        # 首先尝试 pickle.load（因为 EnsembleModel 是 Python 对象）
        try:
            # 添加当前目录到路径，以便找到 EnsembleModel
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            from road_accident_risk import EnsembleModel
            
            # 使用自定义 Unpickler 来正确加载 EnsembleModel 类
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 如果查找的是 EnsembleModel，确保从正确的模块加载
                    if name == 'EnsembleModel':
                        if module == '__main__' or module == 'road_accident_risk':
                            return EnsembleModel
                        return EnsembleModel
                    return super().find_class(module, name)
            
            with open(ensemble_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                ensemble_model = unpickler.load()
            
            print(f'  → 使用 pickle.load 加载成功')
            print(f'  → 类型: {type(ensemble_model)}')
            
            if isinstance(ensemble_model, EnsembleModel):
                print(f'  → 包含 {len(ensemble_model)} 个模型')
                
                # 检查每个模型的设备
                all_devices = set()
                for i, model in enumerate(ensemble_model.models):
                    devices = check_tensor_device(model, f"models[{i}]", max_depth=3)
                    all_devices.update(devices)
                
                print(f'  → 检测到的设备: {all_devices}')
                
                if 'MPS' in all_devices or any('mps' in str(d).lower() for d in all_devices):
                    print(f'  ⚠️  包含 MPS 设备（需要转换）')
                    return 'mps'
                elif 'CUDA' in all_devices or any('cuda' in str(d).lower() for d in all_devices):
                    print(f'  ⚠️  包含 CUDA 设备（需要转换）')
                    return 'cuda'
                elif 'CPU' in all_devices or all('cpu' in str(d).lower() for d in all_devices if d):
                    print(f'  ✓ 仅包含 CPU 设备')
                    return 'cpu'
                else:
                    print(f'  ? 未检测到明确的设备信息')
                    return 'unknown'
            else:
                print(f'  ⚠️  不是 EnsembleModel 类型: {type(ensemble_model)}')
                return 'unknown'
        except Exception as e:
            print(f'  → pickle.load 失败: {str(e)[:100]}')
            # 尝试 torch.load
            try:
                import torch
                if hasattr(torch.backends, 'mps'):
                    original_mps = torch.backends.mps.is_available
                    torch.backends.mps.is_available = lambda: False
                
                model = torch.load(ensemble_path, map_location='cpu', weights_only=False)
                
                if hasattr(torch.backends, 'mps'):
                    torch.backends.mps.is_available = original_mps
                
                print(f'  → 使用 torch.load 加载成功')
                devices = check_tensor_device(model)
                print(f'  → 检测到的设备: {devices}')
                
                if 'MPS' in devices or any('mps' in str(d).lower() for d in devices):
                    return 'mps'
                elif 'CUDA' in devices or any('cuda' in str(d).lower() for d in devices):
                    return 'cuda'
                elif 'CPU' in devices:
                    return 'cpu'
                else:
                    return 'unknown'
            except Exception as e2:
                print(f'  ✗ torch.load 也失败: {str(e2)[:100]}')
                return 'error'
    except ImportError:
        print(f'  → 无法导入必要的模块')
        return 'error'

def main():
    parser = argparse.ArgumentParser(description='检查模型文件的设备类型')
    parser.add_argument('--model-dir', type=str, default='./output/models',
                       help='模型目录路径')
    args = parser.parse_args()
    
    model_dir = os.path.normpath(args.model_dir)
    
    if not os.path.exists(model_dir):
        print(f'错误: 模型目录不存在: {model_dir}')
        return 1
    
    print('=' * 70)
    print('检查模型文件的设备类型')
    print('=' * 70)
    
    # 检查各个 fold 的模型
    fold_results = []
    for fold in range(1, 6):
        model_path = os.path.join(model_dir, f'model_fold_{fold}.pkl')
        if os.path.exists(model_path):
            result = check_model_device(model_path)
            fold_results.append(result)
        else:
            print(f'\n模型文件不存在: model_fold_{fold}.pkl')
    
    # 检查集成模型
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    if os.path.exists(ensemble_path):
        ensemble_result = check_ensemble_model(ensemble_path)
    else:
        print(f'\n集成模型文件不存在: ensemble_model.pkl')
        ensemble_result = None
    
    # 总结
    print('\n' + '=' * 70)
    print('检查结果总结')
    print('=' * 70)
    
    mps_count = fold_results.count('mps')
    cuda_count = fold_results.count('cuda')
    cpu_count = fold_results.count('cpu')
    unknown_count = fold_results.count('unknown')
    
    print(f'\nFold 模型统计:')
    print(f'  MPS:   {mps_count}/5')
    print(f'  CUDA:  {cuda_count}/5')
    print(f'  CPU:   {cpu_count}/5')
    print(f'  未知:  {unknown_count}/5')
    
    if ensemble_result:
        print(f'\n集成模型: {ensemble_result.upper()}')
    
    print('\n' + '=' * 70)
    
    if mps_count > 0 or ensemble_result == 'mps':
        print('⚠️  警告: 检测到 MPS 设备信息，建议运行转换脚本:')
        print(f'   python convert_model_to_cpu.py --model-dir {model_dir}')
    elif cuda_count > 0 or ensemble_result == 'cuda':
        print('⚠️  警告: 检测到 CUDA 设备信息，建议运行转换脚本:')
        print(f'   python convert_model_to_cpu.py --model-dir {model_dir}')
    elif cpu_count == 5 and ensemble_result == 'cpu':
        print('✓ 所有模型都是 CPU 格式，可以直接在 Docker 中使用')
    else:
        print('? 无法完全确定模型设备类型，建议运行转换脚本以确保兼容性')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
