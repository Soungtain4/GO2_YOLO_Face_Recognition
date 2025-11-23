import numpy as np
import cv2
import time

try:
    import maccel
except ImportError:
    print("Error: 'maccel' library not found. This script must be run on the Unitree Go2 robot.")
    exit(1)

def main():
    model_path = "../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq"
    input_size = (512, 640) # H, W
    
    print(f"Loading model: {model_path}")
    try:
        acc = maccel.Accelerator()
        model = maccel.Model(model_path)
        model.launch(acc)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded successfully.")
    
    # Try to inspect model inputs if available
    try:
        if hasattr(model, 'inputs'):
            print(f"Model inputs: {model.inputs}")
        if hasattr(model, 'input_shapes'):
            print(f"Model input shapes: {model.input_shapes}")
        if hasattr(model, 'get_input_info'):
            print(f"Model input info: {model.get_input_info()}")
    except:
        pass

    # Create dummy input base (H, W, 3)
    dummy_base = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:input_size[0], :input_size[1]] = dummy_base
    
    # Normalize
    normalized = padded.astype(np.float32) / 255.0
    
    # Test shapes
    test_shapes = [
        ("HWC (512, 640, 3)", normalized),
        ("NHWC (1, 512, 640, 3)", np.expand_dims(normalized, axis=0)),
        ("NCHW (1, 3, 512, 640)", np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]),
        ("CHW (3, 512, 640)", np.transpose(normalized, (2, 0, 1)))
    ]
    
    for name, data in test_shapes:
        print(f"\nTesting input shape: {name} - {data.shape}")
        try:
            start_time = time.time()
            outputs = model.infer([data])
            end_time = time.time()
            print(f"  [SUCCESS] Inference time: {(end_time - start_time)*1000:.2f} ms")
            
            print("\n" + "="*40)
            print(f"OUTPUT INSPECTION ({name})")
            print("="*40)
            
            print(f"Type of outputs: {type(outputs)}")
            
            if isinstance(outputs, (list, tuple)):
                print(f"Number of output tensors: {len(outputs)}")
                for i, out in enumerate(outputs):
                    print(f"  Output {i}: type={type(out)}, shape={out.shape}, dtype={out.dtype}")
                    flat = out.flatten()
                    print(f"    First 10 values: {flat[:10]}")
            else:
                print(f"Output shape: {outputs.shape}")
            print("="*40)
            break # Stop after first success
            
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            
    model.dispose()

if __name__ == "__main__":
    main()
