#!/usr/bin/env python3
"""
Mac 环境一键配置脚本
自动创建环境、安装依赖并验证
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        return False
    print(f"\n✅ 完成: {description}")
    return True

def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   ChipDiffusion Mac 环境配置向导                  ║
    ║   Apple Silicon (M1/M2/M3/M4/M5) 专用            ║
    ╚══════════════════════════════════════════════════╝
    """)

    # 检查是否在项目根目录
    if not os.path.exists('environment_mac.yaml'):
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)

    # 检查架构
    arch = subprocess.check_output(['uname', '-m']).decode().strip()
    if arch != 'arm64':
        print(f"⚠️  警告: 检测到架构 {arch}，此脚本专为 Apple Silicon (arm64) 设计")
        response = input("是否继续? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("\n开始配置环境...\n")

    steps = [
        (
            "conda env create -f environment_mac.yaml",
            "创建 Conda 环境"
        ),
    ]

    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print("\n❌ 配置失败，请检查错误信息")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("🎉 环境配置完成！")
    print(f"{'='*60}")

    print("\n下一步:")
    print("  1. 激活环境:")
    print("     conda activate chipdiffusion")
    print("\n  2. 验证安装:")
    print("     python test_mac_setup.py")
    print("\n  3. 快速测试:")
    print("     PYTHONPATH=. python data-gen/generate.py versions@_global_=v1 num_train_samples=5 num_val_samples=2")
    print("\n  4. 阅读文档:")
    print("     cat INSTALL_MAC.md")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()