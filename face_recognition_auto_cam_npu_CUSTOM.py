"""
YOLO + InceptionResnetV1 Face Recognition System (Full NPU Version)
- Auto-detects first available camera
- Face Detection: YOLOv8-Face (NPU: yolov8n-face.mxq)
- Face Recognition: InceptionResnetV1 (NPU: inception_resnet_v1_vggface2.mxq)
- Output: Save annotated images + JSON logs
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime
from PIL import Image

try:
    import maccel
except ImportError:
    print("Error: 'maccel' library not found. Please run on the robot.")
    exit(1)

def find_available_camera(max_checks=10):
    """
    Find the first available camera index
    """
    print(f"Scanning for cameras (indices 0 to {max_checks-1})...")
    for index in range(max_checks):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCCESS] Camera found at index: {index}")
                cap.release()
                return index
            cap.release()
    return None

class NPUYoloDetector:
    """
    YOLO Face Detector using NPU (maccel)
    """
    def __init__(self, model_path, input_size=(512, 640), conf_thres=0.45, iou_thres=0.5):
        print(f"Loading YOLO model from NPU: {model_path}")
        self.acc = maccel.Accelerator()
        self.model = maccel.Model(model_path)
        self.model.launch(self.acc)

        self.input_w, self.input_h = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.strides = [8, 16, 32]
        self.grids = self._generate_grids()

        print(f"NPU YOLO initialized (input: {self.input_w}x{self.input_h})")

    def _generate_grids(self):
        grids = []
        for stride in self.strides:
            grid_h = self.input_h // stride
            grid_w = self.input_w // stride
            grid = []
            for i in range(grid_h):
                for j in range(grid_w):
                    grid.append([j, i])
            grids.append(np.array(grid, dtype=np.float32))
        return grids

    def _letterbox_resize(self, img, fill_value=114):
        h, w = img.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_h, self.input_w, 3), fill_value, dtype=img.dtype)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        metadata = {'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y, 'orig_shape': (h, w)}
        return padded, metadata

    def preprocess(self, img_rgb):
        padded, metadata = self._letterbox_resize(img_rgb, fill_value=114)
        normalized = padded.astype(np.float32) / 255.0
        return normalized, metadata

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _decode_boxes_dfl(self, box_output, class_output, grid, stride):
        # box_output: (H, W, 64) -> DFL distribution
        # class_output: (H, W, 1) -> Class score
        
        # Flatten if needed
        if len(box_output.shape) == 3:
            box_output = box_output.reshape(-1, 64)
        if len(class_output.shape) == 3:
            class_output = class_output.reshape(-1, 1)
            
        num_anchors = box_output.shape[0]
        boxes = []
        scores = []
        
        # Check if output is already sigmoid-activated (probabilities)
        # If all values are between 0 and 1, assume probabilities.
        is_probability = np.all((class_output >= 0) & (class_output <= 1))
        
        if is_probability:
            # Values are already probabilities
            # Filter directly
            mask = class_output[:, 0] > self.conf_thres
            indices = np.where(mask)[0]
        else:
            # Values are logits
            inverse_conf = -np.log(1 / self.conf_thres - 1)
            mask = class_output[:, 0] > inverse_conf
            indices = np.where(mask)[0]

        for i in indices:
            class_score = class_output[i, 0]
            
            if is_probability:
                conf = class_score
            else:
                conf = 1 / (1 + np.exp(-class_score))
            
            # Decode box (DFL)
            box = []
            for j in range(4):
                start_idx = j * 16
                end_idx = start_idx + 16
                dist = box_output[i, start_idx:end_idx]
                prob = self._softmax(dist)
                value = np.sum(prob * np.arange(16))
                box.append(value)
            
            grid_x, grid_y = grid[i]
            x1 = (grid_x - box[0] + 0.5) * stride
            y1 = (grid_y - box[1] + 0.5) * stride
            x2 = (grid_x + box[2] + 0.5) * stride
            y2 = (grid_y + box[3] + 0.5) * stride
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            
        return np.array(boxes), np.array(scores)

    @staticmethod
    def _compute_iou(box, boxes):
        """Compute IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _nms(self, boxes, scores):
        """
        Apply Non-Maximum Suppression

        Args:
            boxes: (N, 4) array [x1, y1, x2, y2]
            scores: (N,) array of confidence scores

        Returns:
            Filtered boxes and scores
        """
        if len(boxes) == 0:
            return boxes, scores

        # Sort by score
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            # Keep highest scoring box
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]

            ious = self._compute_iou(current_box, other_boxes)

            # Keep boxes with IoU below threshold
            indices = indices[1:][ious < self.iou_thres]

        return boxes[keep], scores[keep]

    def _scale_boxes_to_original(self, boxes, metadata):
        """
        Scale boxes back to original image coordinates

        Args:
            boxes: (N, 4) array [x1, y1, x2, y2] in model input coordinates
            metadata: Dict with 'scale', 'pad_x', 'pad_y', 'orig_shape'

        Returns:
            Scaled boxes in original image coordinates
        """
        if len(boxes) == 0:
            return boxes

        scale = metadata['scale']
        pad_x = metadata['pad_x']
        pad_y = metadata['pad_y']
        orig_h, orig_w = metadata['orig_shape']

        # Remove padding and scale
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes_scaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        # Clip to image boundaries
        boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, orig_w)
        boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, orig_h)

        return boxes_scaled

    def postprocess(self, outputs, metadata):
        # Outputs: 6 tensors
        # 0: (20, 20, 1)  -> Class (Stride 32)
        # 1: (20, 20, 64) -> Box (Stride 32)
        # 2: (40, 40, 1)  -> Class (Stride 16)
        # 3: (40, 40, 64) -> Box (Stride 16)
        # 4: (80, 80, 1)  -> Class (Stride 8)
        # 5: (80, 80, 64) -> Box (Stride 8)
        
        # Note: The order might be reversed (Stride 8 first) or mixed.
        # Based on shapes from debug log:
        # Out 0: (20, 20, 1) -> Stride 32 Class
        # Out 1: (20, 20, 64) -> Stride 32 Box
        # Out 2: (40, 40, 1) -> Stride 16 Class
        # Out 3: (40, 40, 64) -> Stride 16 Box
        # Out 4: (80, 80, 1) -> Stride 8 Class
        # Out 5: (80, 80, 64) -> Stride 8 Box
        
        # self.strides = [8, 16, 32] -> We need to match them.
        # So we should process in reverse order of outputs or map them correctly.
        
        # Mapping:
        # Stride 8 (80x80): Index 4 (Class), Index 5 (Box)
        # Stride 16 (40x40): Index 2 (Class), Index 3 (Box)
        # Stride 32 (20x20): Index 0 (Class), Index 1 (Box)
        
        # Let's organize them
        layer_indices = [
            (4, 5), # Stride 8
            (2, 3), # Stride 16
            (0, 1)  # Stride 32
        ]
        
        all_boxes = []
        all_scores = []
        
        for i, (class_idx, box_idx) in enumerate(layer_indices):
            grid = self.grids[i]
            stride = self.strides[i]
            
            class_out = outputs[class_idx]
            box_out = outputs[box_idx]
            
            boxes, scores = self._decode_boxes_dfl(box_out, class_out, grid, stride)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
        
        if len(all_boxes) > 0:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
        else:
            return np.array([]), np.array([])
            
        boxes, scores = self._nms(boxes, scores)
        boxes = self._scale_boxes_to_original(boxes, metadata)
        return boxes, scores

    def detect(self, img_rgb):
        preprocessed, metadata = self.preprocess(img_rgb)
        outputs = self.model.infer([preprocessed])
        boxes, scores = self.postprocess(outputs, metadata)
        return boxes, scores

class NPUFaceRecognizer:
    """
    InceptionResnetV1 Face Recognizer using NPU (maccel)
    """
    def __init__(self, model_path):
        print(f"Loading FaceNet model from NPU: {model_path}")
        self.acc = maccel.Accelerator()
        self.model = maccel.Model(model_path)
        self.model.launch(self.acc)
        print("NPU FaceNet initialized.")

    def preprocess(self, face_img):
        # Resize to 160x160
        resized = cv2.resize(face_img, (160, 160))
        # Normalize (standard FaceNet: (x - 127.5) / 128.0)
        normalized = (resized.astype(np.float32) - 127.5) / 128.0
        # Expand dims (1, 160, 160, 3) - Assuming NHWC for NPU
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor

    def get_embedding(self, face_img):
        input_tensor = self.preprocess(face_img)
        outputs = self.model.infer([input_tensor])
        
        # Output shape from debug: (1, 1, 512)
        # We need a flat 512 vector
        embedding = outputs[0].flatten()
        
        # L2 Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

class YOLOFaceRecognitionSystemNPU:
    def __init__(self,
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 output_dir="recognition_output",
                 yolo_model_path="../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq", # Using existing one for safety or custom? User said custom.
                 facenet_model_path="inception_resnet_v1_vggface2.mxq",
                 threshold=0.6):
        
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.output_dir = output_dir
        self.threshold = threshold
        Path(self.output_dir).mkdir(exist_ok=True)

        # Initialize NPU models
        # User confirmed models are in the same directory as the script
        
        yolo_path = "yolov8n-face.mxq"
        facenet_path = "inception_resnet_v1_vggface2.mxq"
        
        # Check if files exist, if not try parent directory (fallback)
        if not os.path.exists(yolo_path):
             if os.path.exists("../" + yolo_path): 
                 yolo_path = "../" + yolo_path
                 print(f"Found YOLO model in parent directory: {yolo_path}")
        
        if not os.path.exists(facenet_path):
             if os.path.exists("../" + facenet_path): 
                 facenet_path = "../" + facenet_path
                 print(f"Found FaceNet model in parent directory: {facenet_path}")

        self.yolo_detector = NPUYoloDetector(model_path=yolo_path, input_size=(640, 640))
        self.face_recognizer = NPUFaceRecognizer(model_path=facenet_path)

        self.visitors_info = self._load_visitors_info()
        self.known_embeddings = {}
        self.known_names = {}
        self._precompute_embeddings()
        
        print(f"System initialized. {len(self.known_embeddings)} faces registered.")

    def _load_visitors_info(self):
        if not os.path.exists(self.visitors_json):
            return {}
        with open(self.visitors_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        visitors_map = {}
        for visitor in data.get('visitors', []):
            if 'image_filename' in visitor:
                visitors_map[visitor['image_filename']] = visitor
        return visitors_map

    def _precompute_embeddings(self):
        if not os.path.exists(self.registered_faces_dir):
            print(f"Warning: {self.registered_faces_dir} not found.")
            return
        face_files = list(Path(self.registered_faces_dir).glob("*.jpg")) + list(Path(self.registered_faces_dir).glob("*.png"))
        print(f"Computing embeddings for {len(face_files)} registered faces...")
        
        for face_file in face_files:
            try:
                img = cv2.imread(str(face_file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes, scores = self.yolo_detector.detect(img_rgb)
                
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = img_rgb[y1:y2, x1:x2]
                    embedding = self.face_recognizer.get_embedding(face)
                    
                    filename = face_file.name
                    self.known_embeddings[filename] = embedding
                    
                    if filename in self.visitors_info:
                        visitor = self.visitors_info[filename]
                        self.known_names[filename] = {
                            'name': visitor.get('name', 'Unknown'),
                            'destination': visitor.get('destination', 'Unknown')
                        }
                    else:
                        self.known_names[filename] = {
                            'name': filename.split('.')[0],
                            'destination': 'Unknown'
                        }
                    print(f"[OK] Registered: {self.known_names[filename]['name']}")
                else:
                    print(f"[FAIL] No face detected in {face_file.name}")
            except Exception as e:
                print(f"Error processing {face_file.name}: {e}")

    def recognize_face(self, face_img):
        if len(self.known_embeddings) == 0: return None
        embedding = self.face_recognizer.get_embedding(face_img)
        min_distance = float('inf')
        best_match = None
        
        for filename, known_embedding in self.known_embeddings.items():
            distance = np.linalg.norm(embedding - known_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = filename
                
        if min_distance < self.threshold:
            result = self.known_names[best_match].copy()
            result['distance'] = float(min_distance)
            result['confidence'] = max(0, (1 - min_distance / self.threshold) * 100)
            return result
        else:
            if best_match:
                print(f"  [Debug] Best match: {best_match} (Dist: {min_distance:.4f} > Threshold: {self.threshold})")
            else:
                print(f"  [Debug] No match found (Dist: {min_distance})")
                
        return None

    def run_webcam_batch(self, camera_index=0, duration_seconds=10, capture_interval=2):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("\n" + "="*60)
        print("YOLO Face Recognition Started (Full NPU Version)")
        print("="*60)
        
        start_time = time.time()
        last_capture_time = 0
        frame_count = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= duration_seconds:
                    print(f"\nSession completed ({duration_seconds}s)")
                    break
                
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if current_time - last_capture_time >= capture_interval:
                    print(f"\n[{elapsed_time:.1f}s] Capturing frame {frame_count}...")
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, scores = self.yolo_detector.detect(frame_rgb)
                    
                    if len(boxes) > 0:
                        print(f"Detected {len(boxes)} face(s)")
                        for idx, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            try:
                                face = frame_rgb[y1:y2, x1:x2]
                                
                                # Check if face image is valid (not empty)
                                if face.size == 0:
                                    print(f"  Face {idx + 1}: Invalid face crop ({x1}, {y1}, {x2}, {y2})")
                                    continue
                                    
                                result = self.recognize_face(face)
                                if result:
                                    name = result['name']
                                    conf = result['confidence']
                                    label = f"{name} ({conf:.1f}%)"
                                    cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    print(f"  Face {idx + 1}: {name} ({conf:.1f}%)")
                                else:
                                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    print(f"  Face {idx + 1}: Unknown")
                            except Exception as e:
                                print(f"Error: {e}")
                                
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = Path(self.output_dir) / f"capture_npu_{timestamp}.jpg"
                        cv2.imwrite(str(output_path), frame)
                        print(f"Saved: {output_path}")
                    else:
                        print("No faces detected")
                    
                    last_capture_time = current_time
                time.sleep(0.01)
                
                
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()

    def run_burst_capture(self, camera_index=0, num_frames=10):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("\n" + "="*60)
        print(f"Burst Capture Started (Saving {num_frames} frames)")
        print("="*60)
        
        count = 0
        try:
            while count < num_frames:
                ret, frame = cap.read()
                if not ret: break
                
                count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = Path(self.output_dir) / f"burst_{timestamp}.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"[{count}/{num_frames}] Saved: {output_path}")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()

def main():
    print("\n" + "="*60)
    print("YOLO + FaceNet Recognition System (Full NPU Version)")
    print("="*60)
    
    try:
        system = YOLOFaceRecognitionSystemNPU()
        
        while True:
            print("\nOptions:")
            print("1. Burst Capture (Save only)")
            print("2. Webcam Batch (Save + Recognize)")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                camera_index = find_available_camera()
                if camera_index is None:
                    print("Error: No cameras found!")
                    continue
                system.run_burst_capture(camera_index=camera_index, num_frames=10)
                
            elif choice == '2':
                camera_index = find_available_camera()
                if camera_index is None:
                    print("Error: No cameras found!")
                    continue
                    
                duration = input("Enter duration in seconds (default 10): ").strip()
                duration = int(duration) if duration else 10
                
                interval = input("Enter capture interval in seconds (default 2): ").strip()
                interval = float(interval) if interval else 2.0
                
                system.run_webcam_batch(camera_index=camera_index, duration_seconds=duration, capture_interval=interval)
                
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid option")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
