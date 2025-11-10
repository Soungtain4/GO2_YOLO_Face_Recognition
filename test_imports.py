"""Test if all dependencies are working"""
import sys
print("Python version:", sys.version)
print()

print("Testing imports...")

try:
    import cv2
    print("[OK] cv2:", cv2.__version__)
except Exception as e:
    print("[FAIL] cv2:", e)

try:
    import torch
    print("[OK] torch:", torch.__version__)
    print("  - CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("[FAIL] torch:", e)

try:
    import numpy as np
    print("[OK] numpy:", np.__version__)
except Exception as e:
    print("[FAIL] numpy:", e)

try:
    from ultralytics import YOLO
    print("[OK] ultralytics (YOLO)")
except Exception as e:
    print("[FAIL] ultralytics:", e)

try:
    from facenet_pytorch import InceptionResnetV1
    print("[OK] facenet_pytorch (InceptionResnetV1)")
except Exception as e:
    print("[FAIL] facenet_pytorch:", e)

try:
    from PIL import Image
    print("[OK] PIL (Pillow)")
except Exception as e:
    print("[FAIL] PIL:", e)

print()
print("All dependency tests completed!")
