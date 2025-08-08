# Save this as check_gpu.py and run it
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")