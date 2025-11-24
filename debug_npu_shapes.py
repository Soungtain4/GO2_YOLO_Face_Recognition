import numpy as np
import cv2
import time
import os

try:
    import maccel
except ImportError:
    print("Error: 'maccel' library not found. This script must be run on the Unitree Go2 robot.")
    exit(1)

def main():
    # Load custom model
    # model_path = "yolov8n-face.mxq"
    model_path = "inception_resnet_v1_vggface2.mxq"
    
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        # Fallback to check parent dir if not found
        if os.path.exists("../" + model_path):
            model_path = "../" + model_path
            print(f"Found at: {model_path}")
        else:
            return

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

    # Test shapes and types
    resolutions = [
        ("160x160", (160, 160)),
        ("640x640", (640, 640))
    ]
    
    for res_name, (h, w) in resolutions:
        # Create dummy inputs for this resolution
        dummy_base = np.zeros((h, w, 3), dtype=np.uint8)
        
        # float32 normalized
        norm_hwc = dummy_base.astype(np.float32) / 255.0
        norm_nhwc = np.expand_dims(norm_hwc, axis=0)
        norm_nchw = np.transpose(norm_hwc, (2, 0, 1))[np.newaxis, ...]
        
        # uint8 raw
        uint8_hwc = dummy_base
        uint8_nhwc = np.expand_dims(uint8_hwc, axis=0)
        
        tests = [
            (f"{res_name} HWC float32", norm_hwc),
            (f"{res_name} NHWC float32", norm_nhwc),
            (f"{res_name} NCHW float32", norm_nchw),
            (f"{res_name} HWC uint8", uint8_hwc),
            (f"{res_name} NHWC uint8", uint8_nhwc),
        ]
        
        for name, data in tests:
            print(f"\nTesting: {name} - Shape: {data.shape}, Dtype: {data.dtype}")
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
                    flat = outputs.flatten()
                    print(f"    First 10 values: {flat[:10]}")
                print("="*40)
                
                # If successful, we can stop
                model.dispose()
                return
                
            except Exception as e:
                print(f"  [FAIL] Error: {e}")
    
    model.dispose()

if __name__ == "__main__":
    main()
