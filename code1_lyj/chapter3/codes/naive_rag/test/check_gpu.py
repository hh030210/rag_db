"""检查GPU可用性"""
import torch

print("=" * 60)
print("GPU 检查")
print("=" * 60)

print(f"\nPyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("\n⚠ CUDA 不可用，请检查:")
    print("  1. 是否安装了CUDA版本的PyTorch")
    print("  2. GPU驱动是否正常")
    print("  3. 运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
