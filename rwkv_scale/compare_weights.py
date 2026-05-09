"""
权重参数对比脚本
对比训练后的模型和初始化时的模型权重参数，特别关注scale维度（4096-6144）
"""

import torch
import numpy as np
from collections import OrderedDict
import re
import json
from pathlib import Path

# ============================================================================
# 配置参数
# ============================================================================

# 初始化模型路径
INIT_MODEL_PATH = "/mnt/data/train/models/rwkv_state_dict_expanded_L64_C6144_bo_init.pth"

# 训练后的模型路径
TRAINED_MODEL_PATH = "/mnt/data/Outs/rwkv7-megatron/ckpt/scale/bo_init/ckpt/iter_0000500/rwkv_state_dict.pth"

# 原始维度和新维度
C_ORIG = 4096  # 原始embedding维度
C_NEW = 6144   # 扩展后的embedding维度
SCALE_START = 4096  # scale维度的起始位置
SCALE_END = 6144    # scale维度的结束位置

# 输出结果保存路径
OUTPUT_DIR = "/mnt/data/Codes/RWKV/RWKV-Scale/weight_comparison_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 辅助函数
# ============================================================================

def get_scale_slice(tensor, axis_with_c_dim):
    """
    从张量中提取scale维度（4096-6144）的切片
    
    Args:
        tensor: 输入张量
        axis_with_c_dim: 包含C维度的轴索引（可能是一个列表，因为可能有多个轴包含C维度）
    
    Returns:
        提取的scale维度切片，以及原始维度的切片（0-4096）
    """
    if isinstance(axis_with_c_dim, int):
        axis_with_c_dim = [axis_with_c_dim]
    
    # 构建切片
    slices_scale = [slice(None)] * tensor.dim()
    slices_orig = [slice(None)] * tensor.dim()
    
    for axis in axis_with_c_dim:
        slices_scale[axis] = slice(SCALE_START, SCALE_END)
        slices_orig[axis] = slice(0, SCALE_START)
    
    scale_part = tensor[tuple(slices_scale)]
    orig_part = tensor[tuple(slices_orig)]
    
    return scale_part, orig_part, slices_scale, slices_orig


def analyze_tensor_diff(init_tensor, trained_tensor, param_name):
    """
    分析两个张量之间的差异
    
    Returns:
        dict: 包含各种统计信息的字典
    """
    diff = trained_tensor.float() - init_tensor.float()
    
    stats = {
        'param_name': param_name,
        'shape': list(trained_tensor.shape),
        'dtype': str(trained_tensor.dtype),
        
        # 整体统计
        'init_mean': float(init_tensor.float().mean().item()),
        'init_std': float(init_tensor.float().std().item()),
        'init_min': float(init_tensor.float().min().item()),
        'init_max': float(init_tensor.float().max().item()),
        
        'trained_mean': float(trained_tensor.float().mean().item()),
        'trained_std': float(trained_tensor.float().std().item()),
        'trained_min': float(trained_tensor.float().min().item()),
        'trained_max': float(trained_tensor.float().max().item()),
        
        # 差异统计
        'diff_mean': float(diff.mean().item()),
        'diff_std': float(diff.std().item()),
        'diff_abs_mean': float(diff.abs().mean().item()),
        'diff_max': float(diff.max().item()),
        'diff_min': float(diff.min().item()),
        
        # 相对变化
        'relative_change_mean': float((diff.abs() / (init_tensor.abs() + 1e-8)).mean().item()),
    }
    
    return stats


def analyze_scale_dimension(init_tensor, trained_tensor, param_name, axis_with_c_dim):
    """
    专门分析scale维度（4096-6144）的变化
    
    Args:
        init_tensor: 初始化时的张量
        trained_tensor: 训练后的张量
        param_name: 参数名称
        axis_with_c_dim: 包含C维度的轴索引列表
    
    Returns:
        dict: scale维度的详细分析结果
    """
    if isinstance(axis_with_c_dim, int):
        axis_with_c_dim = [axis_with_c_dim]
    
    # 提取scale维度和原始维度
    init_scale, init_orig, _, _ = get_scale_slice(init_tensor, axis_with_c_dim)
    trained_scale, trained_orig, _, _ = get_scale_slice(trained_tensor, axis_with_c_dim)
    
    # 计算scale维度的差异
    scale_diff = trained_scale.float() - init_scale.float()
    
    # 计算原始维度的差异（作为对比）
    orig_diff = trained_orig.float() - init_orig.float()
    
    stats = {
        'param_name': param_name,
        'shape': list(trained_tensor.shape),
        
        # Scale维度（4096-6144）的统计
        'scale_init_mean': float(init_scale.float().mean().item()),
        'scale_init_std': float(init_scale.float().std().item()),
        'scale_init_min': float(init_scale.float().min().item()),
        'scale_init_max': float(init_scale.float().max().item()),
        
        'scale_trained_mean': float(trained_scale.float().mean().item()),
        'scale_trained_std': float(trained_scale.float().std().item()),
        'scale_trained_min': float(trained_scale.float().min().item()),
        'scale_trained_max': float(trained_scale.float().max().item()),
        
        'scale_diff_mean': float(scale_diff.mean().item()),
        'scale_diff_std': float(scale_diff.std().item()),
        'scale_diff_abs_mean': float(scale_diff.abs().mean().item()),
        'scale_diff_max': float(scale_diff.max().item()),
        'scale_diff_min': float(scale_diff.min().item()),
        
        # 原始维度（0-4096）的统计（作为对比）
        'orig_init_mean': float(init_orig.float().mean().item()),
        'orig_init_std': float(init_orig.float().std().item()),
        'orig_trained_mean': float(trained_orig.float().mean().item()),
        'orig_trained_std': float(trained_orig.float().std().item()),
        'orig_diff_abs_mean': float(orig_diff.abs().mean().item()),
        'scale_diff_abs_mean': float(scale_diff.abs().mean().item()),
        
        # 对比：scale维度的变化是否比原始维度更大
        'scale_vs_orig_change_ratio': float(scale_diff.abs().mean().item() / (orig_diff.abs().mean().item() + 1e-8)),
    }
    
    return stats


def identify_c_dimension_axis(param_name, tensor_shape):
    """
    识别参数中哪个轴包含C维度（6144）
    
    Returns:
        list: 包含C维度的轴索引列表
    """
    axes = []
    
    # 1D向量: (C,)
    if len(tensor_shape) == 1 and tensor_shape[0] == C_NEW:
        axes = [0]
    
    # 2D矩阵: 可能的情况
    elif len(tensor_shape) == 2:
        if tensor_shape[0] == C_NEW:  # (C, D) 或 (C, C)
            axes.append(0)
        if tensor_shape[1] == C_NEW:  # (D, C) 或 (C, C)
            axes.append(1)
    
    # 3D张量: (1, 1, C)
    elif len(tensor_shape) == 3 and tensor_shape[2] == C_NEW:
        axes = [2]
    
    # Embedding和Head: (Vocab, C)
    elif 'emb.weight' in param_name or 'head.weight' in param_name:
        if tensor_shape[1] == C_NEW:
            axes = [1]
    
    return axes


# ============================================================================
# 主对比函数
# ============================================================================

def compare_models(init_path, trained_path):
    """
    对比两个模型的权重参数
    """
    print("=" * 80)
    print("开始权重参数对比")
    print("=" * 80)
    
    # 加载模型
    print(f"\n[1/4] 加载初始化模型: {init_path}")
    init_state_dict = torch.load(init_path, map_location="cpu")
    print(f"  初始化模型包含 {len(init_state_dict)} 个参数")
    
    print(f"\n[2/4] 加载训练后模型: {trained_path}")
    trained_state_dict = torch.load(trained_path, map_location="cpu")
    print(f"  训练后模型包含 {len(trained_state_dict)} 个参数")
    
    # 检查参数键是否匹配
    init_keys = set(init_state_dict.keys())
    trained_keys = set(trained_state_dict.keys())
    
    if init_keys != trained_keys:
        print(f"\n[警告] 参数键不完全匹配！")
        only_init = init_keys - trained_keys
        only_trained = trained_keys - init_keys
        if only_init:
            print(f"  仅在初始化模型中: {only_init}")
        if only_trained:
            print(f"  仅在训练后模型中: {only_trained}")
    
    common_keys = init_keys & trained_keys
    print(f"\n  共同参数数量: {len(common_keys)}")
    
    # 分类参数
    print(f"\n[3/4] 分析参数...")
    
    all_stats = []
    scale_dim_stats = []
    
    # 需要特别关注的参数类型
    scale_related_params = []
    
    for key in sorted(common_keys):
        init_tensor = init_state_dict[key]
        trained_tensor = trained_state_dict[key]
        
        # 检查形状是否匹配
        if init_tensor.shape != trained_tensor.shape:
            print(f"  [跳过] {key}: 形状不匹配 {init_tensor.shape} vs {trained_tensor.shape}")
            continue
        
        # 识别是否包含C维度
        c_axes = identify_c_dimension_axis(key, trained_tensor.shape)
        
        # 整体统计
        stats = analyze_tensor_diff(init_tensor, trained_tensor, key)
        all_stats.append(stats)
        
        # 如果包含C维度，进行scale维度分析
        if c_axes:
            scale_stats = analyze_scale_dimension(init_tensor, trained_tensor, key, c_axes)
            scale_dim_stats.append(scale_stats)
            scale_related_params.append(key)
            
            # 打印关键信息
            print(f"\n  [Scale维度参数] {key}")
            print(f"    形状: {trained_tensor.shape}")
            print(f"    Scale维度(4096-6144):")
            print(f"      初始化: mean={scale_stats['scale_init_mean']:.6f}, std={scale_stats['scale_init_std']:.6f}")
            print(f"      训练后: mean={scale_stats['scale_trained_mean']:.6f}, std={scale_stats['scale_trained_std']:.6f}")
            print(f"      变化: diff_mean={scale_stats['scale_diff_mean']:.6f}, diff_abs_mean={scale_stats['scale_diff_abs_mean']:.6f}")
            print(f"    Scale vs Orig变化比: {scale_stats['scale_vs_orig_change_ratio']:.4f}")
    
    print(f"\n[4/4] 生成报告...")
    
    # 保存详细统计结果
    output_file = Path(OUTPUT_DIR) / "weight_comparison_detailed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'all_params': all_stats,
            'scale_dim_params': scale_dim_stats
        }, f, indent=2, ensure_ascii=False)
    print(f"  详细统计已保存到: {output_file}")
    
    # 生成摘要报告
    generate_summary_report(all_stats, scale_dim_stats, scale_related_params)
    
    # 生成scale维度专门报告
    generate_scale_dimension_report(scale_dim_stats)
    
    print("\n" + "=" * 80)
    print("权重对比完成！")
    print("=" * 80)


def generate_summary_report(all_stats, scale_dim_stats, scale_related_params):
    """
    生成摘要报告
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("权重参数对比摘要报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append(f"总参数数量: {len(all_stats)}")
    report_lines.append(f"包含Scale维度的参数数量: {len(scale_dim_stats)}")
    report_lines.append("")
    
    # 整体统计
    report_lines.append("-" * 80)
    report_lines.append("整体统计（所有参数）")
    report_lines.append("-" * 80)
    
    avg_diff = np.mean([s['diff_abs_mean'] for s in all_stats])
    max_diff = max([s['diff_max'] for s in all_stats])
    min_diff = min([s['diff_min'] for s in all_stats])
    
    report_lines.append(f"平均绝对差异: {avg_diff:.6f}")
    report_lines.append(f"最大差异: {max_diff:.6f}")
    report_lines.append(f"最小差异: {min_diff:.6f}")
    report_lines.append("")
    
    # Scale维度参数统计
    if scale_dim_stats:
        report_lines.append("-" * 80)
        report_lines.append("Scale维度（4096-6144）参数统计")
        report_lines.append("-" * 80)
        
        # 按变化大小排序
        sorted_scale_stats = sorted(scale_dim_stats, 
                                   key=lambda x: x['scale_diff_abs_mean'], 
                                   reverse=True)
        
        report_lines.append("\n变化最大的Scale维度参数（Top 10）:")
        for i, stats in enumerate(sorted_scale_stats[:10], 1):
            report_lines.append(f"  {i}. {stats['param_name']}")
            report_lines.append(f"     形状: {stats['shape']}")
            report_lines.append(f"     Scale维度变化(绝对平均): {stats['scale_diff_abs_mean']:.6f}")
            report_lines.append(f"     Scale vs Orig变化比: {stats['scale_vs_orig_change_ratio']:.4f}")
            report_lines.append("")
        
        # 平均统计
        avg_scale_diff = np.mean([s['scale_diff_abs_mean'] for s in scale_dim_stats])
        avg_orig_diff = np.mean([s['orig_diff_abs_mean'] for s in scale_dim_stats])
        avg_ratio = np.mean([s['scale_vs_orig_change_ratio'] for s in scale_dim_stats])
        
        report_lines.append(f"\nScale维度平均变化: {avg_scale_diff:.6f}")
        report_lines.append(f"原始维度平均变化: {avg_orig_diff:.6f}")
        report_lines.append(f"Scale/Orig变化比（平均）: {avg_ratio:.4f}")
        report_lines.append("")
    
    # 保存报告
    report_file = Path(OUTPUT_DIR) / "summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  摘要报告已保存到: {report_file}")
    
    # 同时打印到控制台
    print('\n'.join(report_lines))


def generate_scale_dimension_report(scale_dim_stats):
    """
    生成Scale维度的专门报告
    """
    if not scale_dim_stats:
        print("  没有包含Scale维度的参数，跳过专门报告")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Scale维度（4096-6144）详细分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 按参数类型分类
    param_categories = {
        'LayerNorm': [],
        'Attention': [],
        'FFN': [],
        'Embedding': [],
        'Head': [],
        'LoRA': [],
        'Other': []
    }
    
    for stats in scale_dim_stats:
        name = stats['param_name']
        if 'ln' in name or 'LayerNorm' in name:
            param_categories['LayerNorm'].append(stats)
        elif 'att' in name:
            if 'w1' in name or 'w2' in name or 'a1' in name or 'a2' in name or 'v1' in name or 'v2' in name or 'g1' in name or 'g2' in name:
                param_categories['LoRA'].append(stats)
            else:
                param_categories['Attention'].append(stats)
        elif 'ffn' in name:
            param_categories['FFN'].append(stats)
        elif 'emb' in name:
            param_categories['Embedding'].append(stats)
        elif 'head' in name:
            param_categories['Head'].append(stats)
        else:
            param_categories['Other'].append(stats)
    
    # 按类别输出
    for category, params in param_categories.items():
        if not params:
            continue
        
        report_lines.append("-" * 80)
        report_lines.append(f"{category} 参数 ({len(params)} 个)")
        report_lines.append("-" * 80)
        
        # 按变化大小排序
        sorted_params = sorted(params, key=lambda x: x['scale_diff_abs_mean'], reverse=True)
        
        for stats in sorted_params:
            report_lines.append(f"\n参数: {stats['param_name']}")
            report_lines.append(f"  形状: {stats['shape']}")
            report_lines.append(f"  Scale维度(4096-6144)初始化:")
            report_lines.append(f"    mean={stats['scale_init_mean']:.6f}, std={stats['scale_init_std']:.6f}")
            report_lines.append(f"    min={stats['scale_init_min']:.6f}, max={stats['scale_init_max']:.6f}")
            report_lines.append(f"  Scale维度训练后:")
            report_lines.append(f"    mean={stats['scale_trained_mean']:.6f}, std={stats['scale_trained_std']:.6f}")
            report_lines.append(f"    min={stats['scale_trained_min']:.6f}, max={stats['scale_trained_max']:.6f}")
            report_lines.append(f"  Scale维度变化:")
            report_lines.append(f"    diff_mean={stats['scale_diff_mean']:.6f}")
            report_lines.append(f"    diff_abs_mean={stats['scale_diff_abs_mean']:.6f}")
            report_lines.append(f"    diff_std={stats['scale_diff_std']:.6f}")
            report_lines.append(f"    diff_range=[{stats['scale_diff_min']:.6f}, {stats['scale_diff_max']:.6f}]")
            report_lines.append(f"  Scale vs Orig变化比: {stats['scale_vs_orig_change_ratio']:.4f}")
            report_lines.append("")
    
    # 保存报告
    report_file = Path(OUTPUT_DIR) / "scale_dimension_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Scale维度专门报告已保存到: {report_file}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    compare_models(INIT_MODEL_PATH, TRAINED_MODEL_PATH)

