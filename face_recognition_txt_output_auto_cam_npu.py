"""
YOLO + InceptionResnetV1 Face Recognition System (NPU Version + TXT Output)
- Auto-detects first available camera
- Face Detection: YOLOv8-Face (NPU: yolov8n-face.mxq)
- Face Recognition: InceptionResnetV1 (NPU: inception_resnet_v1_vggface2.mxq)
- Output: Saves destination to TXT file
- Logic: "Fire and Forget" (Immediate save and exit)
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

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
        if len(box_output.shape) == 3:
            box_output = box_output.reshape(-1, 64)
        if len(class_output.shape) == 3:
            class_output = class_output.reshape(-1, 1)

        num_anchors = box_output.shape[0]
        boxes = []
        scores = []

        is_probability = np.all((class_output >= 0) & (class_output <= 1))

        if is_probability:
            mask = class_output[:, 0] > self.conf_thres
            indices = np.where(mask)[0]
        else:
            inverse_conf = -np.log(1 / self.conf_thres - 1)
            mask = class_output[:, 0] > inverse_conf
            indices = np.where(mask)[0]

        for i in indices:
            class_score = class_output[i, 0]
            if is_probability:
                conf = class_score
            else:
                conf = 1 / (1 + np.exp(-class_score))

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
        if len(boxes) == 0:
            return boxes, scores

        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]

            ious = self._compute_iou(current_box, other_boxes)

            indices = indices[1:][ious < self.iou_thres]

        return boxes[keep], scores[keep]

    def _scale_boxes_to_original(self, boxes, metadata):
        if len(boxes) == 0:
            return boxes

        scale = metadata['scale']
        pad_x = metadata['pad_x']
        pad_y = metadata['pad_y']
        orig_h, orig_w = metadata['orig_shape']

        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes_scaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, orig_w)
        boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, orig_h)

        return boxes_scaled

    def postprocess(self, outputs, metadata):
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
        resized = cv2.resize(face_img, (160, 160))
        normalized = (resized.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor

    def get_embedding(self, face_img):
        input_tensor = self.preprocess(face_img)
        outputs = self.model.infer([input_tensor])
        embedding = outputs[0].flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

class YOLOFaceRecognitionSystemTXT:
    def __init__(self,
                 output_file="destination_output.txt",
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 output_dir="recognition_output",
                 yolo_model_path="../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq",
                 facenet_model_path="inception_resnet_v1_vggface2.mxq",
                 threshold=0.6):

        self.output_file = output_file
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.output_dir = output_dir
        self.threshold = threshold
        Path(self.output_dir).mkdir(exist_ok=True)

        # Initialize NPU models
        yolo_path = "yolov8n-face.mxq"
        facenet_path = "inception_resnet_v1_vggface2.mxq"

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

    def save_destination(self, destination):
        """Save destination to TXT file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"{destination}\n")
                f.write(f"# Timestamp: {timestamp}\n")

            print(f"[TXT] Destination saved to {self.output_file}")
            print(f"[TXT] Content: {destination}")
            return True
        except Exception as e:
            print(f"[TXT] Error saving file: {e}")
            return False

    def run_recognition(self, camera_index=0):
        """
        Main loop for TXT Output Mode (NPU)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("\n" + "="*60)
        print("Face Recognition Started (TXT Output Mode - NPU)")
        print("="*60)
        print(" - Detects faces continuously")
        print(f" - Saves destination to {self.output_file}")
        print(" - Exits after successful save")
        print("="*60 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, scores = self.yolo_detector.detect(frame_rgb)

                if len(boxes) > 0:
                    for idx, (box, score) in enumerate(zip(boxes, scores)):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        try:
                            face = frame_rgb[y1:y2, x1:x2]
                            if face.size == 0: continue

                            result = self.recognize_face(face)
                            if result:
                                name = result['name']
                                destination = result.get('destination', 'Unknown')
                                conf = result['confidence']

                                label = f"{name} ({conf:.1f}%)"
                                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame, f"Dest: {destination}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Logic: Save to TXT and Exit
                                print(f"\n[MATCH] Recognized: {name}")
                                print(f" -> Destination: {destination}")

                                # Save to TXT
                                if self.save_destination(destination):
                                    print(" -> Saved to file! Exiting...")
                                    time.sleep(1.0)
                                    return  # Exit function
                                else:
                                    print(" -> Save failed. Retrying...")
                                    time.sleep(2)

                            else:
                                cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error: {e}")

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()

def main():
    # Output file name
    OUTPUT_FILE = "destination_output.txt"

    # Initialize System
    system = YOLOFaceRecognitionSystemTXT(output_file=OUTPUT_FILE)

    # Auto-detect camera
    camera_index = find_available_camera()
    if camera_index is None:
        print("Error: No cameras found!")
        return

    # Run Recognition
    system.run_recognition(camera_index=camera_index)

if __name__ == "__main__":
    main()
