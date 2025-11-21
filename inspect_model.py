import numpy as np
import os
import sys

try:
    import maccel
except ImportError:
    print("Error: maccel not found")
    sys.exit(1)

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        # Try relative path
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq"))
        if os.path.exists(alt_path):
            print(f"Found at: {alt_path}")
            model_path = alt_path
        else:
            return

    print(f"Loading model: {model_path}")
    acc = maccel.Accelerator()
    model = maccel.Model(model_path)
    model.launch(acc)

    # Dummy input (1, 512, 640, 3) - based on filename 640_512 but usually it's (H, W)
    # Filename says 640_512. Usually WxH or HxW.
    # Code used input_size=(512, 640) (W, H) -> 640x512 image?
    # Let's try 640x512x3 (H, W, C)
    # The NPU usually expects NHWC or NCHW. maccel usually takes list of numpy arrays.
    
    # Let's check the code again.
    # self.input_w, self.input_h = input_size (512, 640)
    # padded = np.full((self.input_h, self.input_w, 3)...
    # So H=640, W=512?
    # Wait, code said: input_size=(512, 640) # (W, H)
    # self.input_w = 512, self.input_h = 640.
    # So image is 640(H) x 512(W).
    
    dummy_input = np.zeros((640, 512, 3), dtype=np.float32)
    
    print("Running inference...")
    outputs = model.infer([dummy_input])
    
    print(f"Number of output tensors: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        
    model.dispose()

if __name__ == "__main__":
    inspect_model("face_yolov8n_640_512.mxq")
