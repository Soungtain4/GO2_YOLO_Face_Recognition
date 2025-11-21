import sys
import torch
import torchvision

print("="*40)
print("Environment Check")
print("="*40)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

print("-" * 40)

try:
    import facenet_pytorch
    print(f"facenet_pytorch: {facenet_pytorch.__version__}") # Note: facenet_pytorch might not have __version__ attribute in some versions
except ImportError:
    print("facenet_pytorch: Not installed")
except AttributeError:
    print("facenet_pytorch: Installed (version unknown)")

print("-" * 40)
print("Testing InceptionResnetV1 loading...")

try:
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("[SUCCESS] InceptionResnetV1 loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load InceptionResnetV1: {e}")
    print("\nPossible causes:")
    print("1. Torch/Torchvision version mismatch")
    print("2. Network issue (if downloading weights fails)")
    print("3. Corrupted installation")

print("="*40)
