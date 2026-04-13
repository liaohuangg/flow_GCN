#!/usr/bin/env python3
"""
快速可视化布局结果
"""

import sys
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def find_latest_results():
    """查找最新的推理结果"""
    logs_dir = Path("logs")

    if not logs_dir.exists():
        return None

    # 查找所有输出目录
    output_dirs = list(logs_dir.glob("*/outputs"))

    if not output_dirs:
        return None

    # 返回最新的
    return max(output_dirs, key=lambda p: p.stat().st_mtime)

def visualize_placement(pkl_file, output_file="placement.png"):
    """可视化单个布局"""
    # 加载数据
    with open(pkl_file, 'rb') as f:
        placement = pickle.load(f)

    print(f"\n布局信息:")
    print(f"  文件: {pkl_file}")
    print(f"  单元数量: {placement.shape[0]}")
    print(f"  坐标范围:")
    print(f"    X: [{placement[:,0].min():.3f}, {placement[:,0].max():.3f}]")
    print(f"    Y: [{placement[:,1].min():.3f}, {placement[:,1].max():.3f}]")

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：散点图
    ax1 = axes[0]
    scatter = ax1.scatter(placement[:, 0], placement[:, 1],
                         s=2, alpha=0.6, c=range(len(placement)),
                         cmap='viridis')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'Chip Placement - {pkl_file.stem}', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cell Index')

    # 右图：密度热图
    ax2 = axes[1]
    # 创建2D直方图
    h, xedges, yedges = np.histogram2d(
        placement[:, 0], placement[:, 1],
        bins=50
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax2.imshow(h.T, extent=extent, origin='lower',
                    cmap='hot', aspect='auto', interpolation='bilinear')
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_title('Density Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Cell Density')

    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化保存到: {output_file}")

    # 尝试显示（如果在交互环境）
    try:
        plt.show()
    except:
        pass

def visualize_all_placements(output_dir, output_prefix="comparison"):
    """可视化并比较多个布局"""
    pkl_files = sorted(list(output_dir.glob("*.pkl")))[:6]  # 最多6个

    if not pkl_files:
        print("未找到 .pkl 文件")
        return

    n = len(pkl_files)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"\n可视化 {n} 个布局...")

    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as f:
            placement = pickle.load(f)

        ax = axes[idx]
        ax.scatter(placement[:, 0], placement[:, 1],
                  s=1, alpha=0.6, c=range(len(placement)),
                  cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(pkl_file.stem, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        print(f"  {idx+1}. {pkl_file.name}: {placement.shape[0]} cells")

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_file = f"{output_prefix}_all.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图保存到: {output_file}")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  ChipDiffusion 布局可视化工具")
    print("="*60)

    if len(sys.argv) > 1:
        # 指定文件
        pkl_file = Path(sys.argv[1])
        if not pkl_file.exists():
            print(f"❌ 文件不存在: {pkl_file}")
            sys.exit(1)

        output_file = sys.argv[2] if len(sys.argv) > 2 else "placement.png"
        visualize_placement(pkl_file, output_file)
    else:
        # 自动查找最新结果
        output_dir = find_latest_results()

        if output_dir is None:
            print("\n❌ 未找到推理结果")
            print("\n使用方法:")
            print("  1. 自动查找: python visualize_placement.py")
            print("  2. 指定文件: python visualize_placement.py path/to/sample.pkl")
            print("\n请先运行推理: python quick_inference.py")
            sys.exit(1)

        print(f"\n找到推理结果: {output_dir}")

        # 询问用户
        print("\n选择可视化模式:")
        print("  1. 详细可视化第一个结果 (推荐)")
        print("  2. 对比所有结果")

        choice = input("\n请选择 (1/2) [1]: ").strip() or "1"

        pkl_files = sorted(list(output_dir.glob("*.pkl")))

        if choice == "1":
            if pkl_files:
                visualize_placement(pkl_files[0])
        elif choice == "2":
            visualize_all_placements(output_dir)
        else:
            print("无效选择")
            sys.exit(1)

    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
