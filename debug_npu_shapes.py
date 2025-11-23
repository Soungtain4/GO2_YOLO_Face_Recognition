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
    
    # Create dummy input
    print(f"Creating dummy input: {input_size}")
    dummy_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    
    # Preprocess (mimic NPUYoloDetector.preprocess)
    # Letterbox resize (already correct size, just fill)
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:input_size[0], :input_size[1]] = dummy_img
    
    # Normalize
    normalized = padded.astype(np.float32) / 255.0
    
    print("Running inference...")
    try:
        start_time = time.time()
        outputs = model.infer([normalized])
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
        
        print("\n" + "="*40)
        print("OUTPUT INSPECTION")
        print("="*40)
        
        print(f"Type of outputs: {type(outputs)}")
        
        if isinstance(outputs, (list, tuple)):
            print(f"Number of output tensors: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"  Output {i}: type={type(out)}, shape={out.shape}, dtype={out.dtype}")
                # Print first few values to see if it looks like boxes or scores
                flat = out.flatten()
                print(f"    First 10 values: {flat[:10]}")
        else:
            print(f"Output shape: {outputs.shape}")
            
        print("="*40)
        
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        model.dispose()

if __name__ == "__main__":
    main()
