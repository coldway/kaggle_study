#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换脚本：将 MPS/CUDA 训练的模型转换为 CPU 模型
用于在 Docker 等不支持 MPS 的环境中部署模型
"""

import os
import sys
import pickle
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path

def convert_model_to_cpu(model_dir, backup=True, script_dir=None, output_dir=None):
    """
    将模型目录中的所有模型转换为 CPU 模型
    
    Args:
        model_dir: 模型目录路径（输入目录）
        backup: 是否创建备份（仅在未指定 output_dir 时有效）
        script_dir: 脚本所在目录（用于导入模块），如果为 None 则自动检测
        output_dir: 输出目录路径（如果指定，转换后的模型将保存到此目录，不覆盖原文件）
    """
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    model_dir = os.path.normpath(model_dir)
    
    if not os.path.exists(model_dir):
        print(f'错误: 模型目录不存在: {model_dir}')
        return False
    
    # 确定输出目录
    if output_dir is not None:
        output_dir = os.path.normpath(output_dir)
        # 如果指定了输出目录，不创建备份（因为原文件不会被覆盖）
        backup = False
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f'\n*** 输出目录: {output_dir}')
        print(f'✓ 转换后的模型将保存到此目录，原文件不会被修改')
    else:
        # 如果没有指定输出目录，使用原目录（会覆盖原文件）
        output_dir = model_dir
        # 创建备份（如果需要）
        if backup:
            backup_dir = f"{model_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f'\n*** 创建备份: {backup_dir}')
            shutil.copytree(model_dir, backup_dir)
            print(f'✓ 备份完成')
    
    # 加载元数据
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f'错误: 元数据文件不存在: {metadata_path}')
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f'\n*** 开始转换模型到 CPU...')
    print(f'输入目录: {model_dir}')
    if output_dir != model_dir:
        print(f'输出目录: {output_dir}')
    print(f'Fold 数量: {metadata.get("N_SPLITS", "未知")}')
    
    # 导入必要的库
    try:
        import torch
        # 禁用 MPS 设备
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False
    except ImportError:
        print('警告: PyTorch 未安装，将尝试直接转换')
    
    # 复制元数据到输出目录（如果输出目录不同）
    if output_dir != model_dir:
        output_metadata_path = os.path.join(output_dir, 'metadata.json')
        shutil.copy2(metadata_path, output_metadata_path)
        print(f'✓ 已复制元数据到输出目录')
    
    # 转换每个 fold 的模型
    models = []
    for fold in range(1, metadata['N_SPLITS'] + 1):
        model_path = os.path.join(model_dir, f'model_fold_{fold}.pkl')
        if not os.path.exists(model_path):
            print(f'警告: 模型文件不存在: {model_path}')
            continue
        
        # 确定输出路径
        output_model_path = os.path.join(output_dir, f'model_fold_{fold}.pkl')
        
        print(f'\n处理 Fold {fold}...')
        try:
            # 加载模型
            try:
                import torch
                # 使用 torch.load 并映射到 CPU
                def map_location_func(storage, location):
                    if isinstance(location, str) and ('mps' in location.lower() or 'cuda' in location.lower()):
                        return storage.cpu()
                    elif isinstance(location, dict):
                        device_type = location.get('device_type', '')
                        if device_type in ['mps', 'cuda']:
                            return storage.cpu()
                    return storage.cpu()
                
                model = torch.load(model_path, map_location=map_location_func, weights_only=False)
            except:
                # 如果 torch.load 失败，使用 pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # 将模型移动到 CPU（更彻底的方式）
            try:
                import torch
                # 禁用 MPS 设备
                if hasattr(torch.backends, 'mps'):
                    torch.backends.mps.is_available = lambda: False
                
                # 递归函数：将对象中的所有 PyTorch 张量移动到 CPU
                def move_to_cpu_recursive(obj):
                    """递归地将对象中的所有 PyTorch 张量移动到 CPU"""
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    elif isinstance(obj, dict):
                        return {k: move_to_cpu_recursive(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_cpu_recursive(item) for item in obj)
                    elif hasattr(obj, '__dict__'):
                        # 对于对象，尝试移动其属性
                        for attr_name in dir(obj):
                            if not attr_name.startswith('_'):
                                try:
                                    attr_value = getattr(obj, attr_name)
                                    if isinstance(attr_value, torch.Tensor):
                                        setattr(obj, attr_name, attr_value.cpu())
                                except:
                                    pass
                    return obj
                
                # 设置设备属性
                if hasattr(model, 'device'):
                    model.device = 'cpu'
                elif hasattr(model, '_device'):
                    model._device = 'cpu'
                
                # 如果模型有内部 PyTorch 模型，也移动到 CPU
                if hasattr(model, 'model'):
                    if hasattr(model.model, 'device'):
                        model.model.device = 'cpu'
                    elif hasattr(model.model, '_device'):
                        model.model._device = 'cpu'
                    # 尝试使用 to() 方法
                    if hasattr(model.model, 'to'):
                        try:
                            model.model.to('cpu')
                        except:
                            pass
                    # 递归移动所有张量
                    try:
                        model.model = move_to_cpu_recursive(model.model)
                    except:
                        pass
                
                # TabM 模型可能有额外的设备相关属性
                # 检查并设置所有可能的设备属性
                for attr_name in ['device', '_device', 'training_device', '_training_device']:
                    if hasattr(model, attr_name):
                        try:
                            setattr(model, attr_name, 'cpu')
                        except:
                            pass
                
                # 如果模型有 _model 属性（TabM 可能使用）
                if hasattr(model, '_model'):
                    try:
                        if hasattr(model._model, 'to'):
                            model._model.to('cpu')
                        if hasattr(model._model, 'device'):
                            model._model.device = 'cpu'
                        elif hasattr(model._model, '_device'):
                            model._model._device = 'cpu'
                    except:
                        pass
                
                # 尝试使用 to() 方法
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except:
                        pass
                
                # 递归移动模型中的所有张量
                try:
                    model = move_to_cpu_recursive(model)
                except:
                    pass
                
                # 如果模型有参数，也尝试移动到 CPU
                if hasattr(model, 'parameters'):
                    try:
                        for param in model.parameters():
                            if hasattr(param, 'data'):
                                param.data = param.data.cpu() if hasattr(param.data, 'cpu') else param.data
                    except:
                        pass
            except Exception as e:
                print(f'  警告: 移动模型到 CPU 时出错: {e}')
                import traceback
                traceback.print_exc()
            
            # 保存转换后的模型
            # 重要：使用 torch.save 而不是 pickle.dump，确保设备信息被正确转换
            # torch.save 会正确处理设备映射，确保保存的模型不包含 MPS 设备信息
            try:
                import torch
                # 确保在保存时也禁用 MPS
                if hasattr(torch.backends, 'mps'):
                    torch.backends.mps.is_available = lambda: False
                
                # 使用 torch.save 保存，它会自动处理设备映射
                # _use_new_zipfile_serialization=False 确保兼容性
                torch.save(model, output_model_path, _use_new_zipfile_serialization=False)
                print(f'  → 使用 torch.save 保存模型到: {output_model_path}')
            except Exception as e:
                # 如果 torch.save 失败，回退到 pickle.dump
                print(f'  → 使用 pickle.dump 保存模型（torch.save 失败: {e}）')
                with open(output_model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            models.append(model)
            print(f'  ✓ Fold {fold} 转换完成')
            
        except Exception as e:
            print(f'  ✗ Fold {fold} 转换失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    # 转换集成模型
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    output_ensemble_path = os.path.join(output_dir, 'ensemble_model.pkl')
    if os.path.exists(ensemble_path):
        print(f'\n处理集成模型...')
        try:
            # 导入 EnsembleModel
            # 如果指定了 script_dir，使用它；否则自动检测脚本目录
            if script_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 临时添加脚本目录到 Python 路径（不改变当前工作目录）
            script_dir_abs = os.path.abspath(script_dir)
            if script_dir_abs not in sys.path:
                sys.path.insert(0, script_dir_abs)
            
            try:
                from road_accident_risk import EnsembleModel
            except ImportError:
                # 如果导入失败，尝试从当前目录导入
                print('  尝试从当前目录导入 road_accident_risk...')
                from road_accident_risk import EnsembleModel
            
            # 创建新的集成模型（会自动确保所有模型在 CPU 上）
            ensemble_model = EnsembleModel(models, metadata)
            
            # 保存转换后的集成模型
            # 注意：EnsembleModel 是 Python 对象，不是 PyTorch 模型，应该使用 pickle.dump
            # 虽然它包含 PyTorch 模型，但 pickle 可以正确处理嵌套的 PyTorch 对象
            # 使用 pickle.dump 确保 EnsembleModel 对象结构完整
            with open(output_ensemble_path, 'wb') as f:
                pickle.dump(ensemble_model, f)
            print(f'  → 使用 pickle.dump 保存集成模型到: {output_ensemble_path}')
            
            print(f'  ✓ 集成模型转换完成')
            
        except Exception as e:
            print(f'  ✗ 集成模型转换失败: {e}')
            import traceback
            traceback.print_exc()
            # 恢复原始工作目录
            os.chdir(original_cwd)
            return False
    
    # 更新元数据中的设备信息
    output_metadata_path = os.path.join(output_dir, 'metadata.json')
    if 'params' in metadata and 'device' in metadata['params']:
        metadata['params']['device'] = 'cpu'
        with open(output_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'\n✓ 元数据已更新（设备设置为 CPU）')
    
    print(f'\n*** 模型转换完成！')
    print(f'所有模型已转换为 CPU 格式，可以在 Docker 等环境中使用')
    
    # 恢复原始工作目录
    os.chdir(original_cwd)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='将 MPS/CUDA 训练的模型转换为 CPU 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 转换模型（自动创建备份）
  python convert_model_to_cpu.py --model-dir ./output/models

  # 转换模型（不创建备份）
  python convert_model_to_cpu.py --model-dir ./output/models --no-backup

  # 指定输出目录（转换后的模型保存到新目录，不覆盖原文件，不创建备份）
  python convert_model_to_cpu.py --model-dir ./output/models --output-dir ./output/models_cpu

  # 指定脚本目录（用于导入模块）
  python convert_model_to_cpu.py --model-dir ./output/models --script-dir ./
        '''
    )
    parser.add_argument('--model-dir', type=str, required=True,
                       help='模型目录路径（输入目录）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录路径（如果指定，转换后的模型将保存到此目录，不覆盖原文件，且不创建备份）')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份（默认会创建备份，仅在未指定 --output-dir 时有效）')
    parser.add_argument('--script-dir', type=str, default=None,
                       help='脚本所在目录（用于导入 road_accident_risk 模块），默认自动检测')
    args = parser.parse_args()
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        success = convert_model_to_cpu(
            args.model_dir, 
            backup=not args.no_backup,
            script_dir=args.script_dir,
            output_dir=args.output_dir
        )
        
        if success:
            print('\n✓ 转换成功！')
            return 0
        else:
            print('\n✗ 转换失败！')
            return 1
    finally:
        # 确保恢复原始工作目录
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)

if __name__ == '__main__':
    sys.exit(main())

