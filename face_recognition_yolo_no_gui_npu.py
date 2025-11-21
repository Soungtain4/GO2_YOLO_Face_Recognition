"""
YOLO + InceptionResnetV1 Face Recognition System (NPU Version)
- Face Detection: YOLOv8-Face (NPU with maccel)
- Face Recognition: InceptionResnetV1 (facenet-pytorch with PyTorch)
- Output: Save annotated images + JSON logs

Key Differences from PyTorch version:
- YOLO detection uses NPU (maccel) for hardware acceleration
- Manual preprocessing and postprocessing for YOLO
- InceptionResnetV1 recognition still uses PyTorch
"""

import cv2
import torch
import numpy as np
import numpy as np
try:
    import maccel
except ImportError:
    maccel = None
    print("Warning: 'maccel' library not found. NPU features will not work.")
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import json
import os
from pathlib import Path
from datetime import datetime
import time


class NPUYoloDetector:
    """
    YOLO Face Detector using NPU (maccel)

    This class handles:
    - NPU initialization
    - YOLO preprocessing (letterbox resize, normalization)
    - NPU inference
    - YOLO postprocessing (DFL decoding, NMS, coordinate scaling)
    """

    def __init__(self, model_path, input_size=(512, 640), conf_thres=0.15, iou_thres=0.5):
        """
        Initialize NPU YOLO detector

        Args:
            model_path: Path to .mxq model file
            input_size: (width, height) tuple for model input
            conf_thres: Confidence threshold for detection
            iou_thres: IoU threshold for NMS
        """
        print(f"Loading YOLO model from NPU: {model_path}")

        # Initialize NPU
        # Initialize NPU
        if maccel is None:
            raise ImportError("Cannot initialize NPUYoloDetector: 'maccel' library is not available.")
            
        self.acc = maccel.Accelerator()
        self.model = maccel.Model(model_path)
        self.model.launch(self.acc)

        # Model parameters
        self.input_w, self.input_h = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_layers = 3
        self.strides = [8, 16, 32]

        # Pre-generate grids for postprocessing
        self.grids = self._generate_grids()

        print(f"NPU YOLO initialized (input: {self.input_w}x{self.input_h})")

    def _generate_grids(self):
        """Generate anchor grids for each YOLO detection layer"""
        grids = []
        for stride in self.strides:
            grid_h = self.input_h // stride
            grid_w = self.input_w // stride

            grid = []
            for i in range(grid_h):
                for j in range(grid_w):
                    grid.append([j, i])  # [x, y]

            grids.append(np.array(grid, dtype=np.float32))

        return grids

    def _letterbox_resize(self, img, fill_value=114):
        """
        Resize image with aspect ratio preservation and padding

        Args:
            img: Input image (H, W, 3) BGR or RGB
            fill_value: Padding color (gray=114)

        Returns:
            padded: Resized and padded image
            metadata: Dict with scale, pad_x, pad_y, orig_shape
        """
        h, w = img.shape[:2]

        # Calculate scale to fit inside target while maintaining aspect ratio
        scale = min(self.input_w / w, self.input_h / h)

        # New dimensions after scaling
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        padded = np.full((self.input_h, self.input_w, 3), fill_value, dtype=img.dtype)

        # Center the resized image
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        metadata = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_shape': (h, w)
        }

        return padded, metadata

    def preprocess(self, img_rgb):
        """
        Preprocess image for YOLO NPU inference

        Args:
            img_rgb: Input image (H, W, 3) RGB format, uint8

        Returns:
            preprocessed: (H, W, 3) float32 array, range [0, 1]
            metadata: Preprocessing metadata for coordinate conversion
        """
        # Letterbox resize
        padded, metadata = self._letterbox_resize(img_rgb, fill_value=114)

        # Normalize to [0, 1]
        normalized = padded.astype(np.float32) / 255.0

        return normalized, metadata

    @staticmethod
    def _softmax(x):
        """Softmax along last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _decode_boxes_dfl(self, output, grid, stride):
        """
        Decode YOLOv8 boxes with Distribution Focal Loss format

        Args:
            output: NPU output for one layer (num_anchors, 80)
                   Format: [16*4 box coords, 1 class score, 15 landmarks]
            grid: Grid coordinates for this layer (num_anchors, 2)
            stride: Stride value for this layer

        Returns:
            boxes: (N, 4) array [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
        """
        num_anchors = output.shape[0]
        boxes = []
        scores = []

        # Pre-calculate inverse confidence threshold (optimization)
        inverse_conf = -np.log(1 / self.conf_thres - 1)

        for i in range(num_anchors):
            # Get class score (index 64)
            class_score = output[i, 64]

            # Apply confidence threshold (before sigmoid for efficiency)
            if class_score < inverse_conf:
                continue

            conf = 1 / (1 + np.exp(-class_score))  # sigmoid

            # Decode box coordinates (indices 0-63, 16 values per coordinate)
            box = []
            for j in range(4):
                start_idx = j * 16
                end_idx = start_idx + 16
                dist = output[i, start_idx:end_idx]

                # Apply softmax to distribution
                prob = self._softmax(dist)

                # Calculate expected value
                value = np.sum(prob * np.arange(16))
                box.append(value)

            # Convert to corner format
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
        """
        Complete postprocessing pipeline for YOLO face detection

        Args:
            outputs: List of NPU output arrays (one per detection layer)
            metadata: Metadata from preprocessing

        Returns:
            boxes: (N, 4) array [x1, y1, x2, y2] in original image coordinates
            scores: (N,) array of confidence scores
        """
        # Decode each layer
        all_boxes = []
        all_scores = []

        for i, (output, grid, stride) in enumerate(zip(outputs, self.grids, self.strides)):
            boxes, scores = self._decode_boxes_dfl(output, grid, stride)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)

        # Concatenate all layers
        if len(all_boxes) > 0:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
        else:
            return np.array([]), np.array([])

        # Apply NMS
        boxes, scores = self._nms(boxes, scores)

        # Scale back to original image
        boxes = self._scale_boxes_to_original(boxes, metadata)

        return boxes, scores

    def detect(self, img_rgb):
        """
        Run face detection on RGB image

        Args:
            img_rgb: Input image (H, W, 3) RGB format, uint8

        Returns:
            boxes: (N, 4) array [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
        """
        # Preprocess
        preprocessed, metadata = self.preprocess(img_rgb)

        # NPU inference
        outputs = self.model.infer([preprocessed])

        # Postprocess
        boxes, scores = self.postprocess(outputs, metadata)

        return boxes, scores

    def __del__(self):
        """Cleanup NPU resources"""
        if hasattr(self, 'model'):
            self.model.dispose()


class YOLOFaceRecognitionSystemNoGUI:
    def __init__(self,
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 output_dir="recognition_output",
                 yolo_model_path="../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq",
                 threshold=0.6,
                 device=None):
        """
        Initialize YOLO + FaceNet face recognition system (NPU Version)

        Args:
            registered_faces_dir: Directory containing registered face images
            visitors_json: JSON file with visitor information
            output_dir: Directory to save output images and logs
            yolo_model_path: Path to .mxq YOLO model for NPU
            threshold: Distance threshold for face matching
            device: 'cuda' or 'cpu' for PyTorch (InceptionResnetV1)
        """
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.output_dir = output_dir
        self.threshold = threshold

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

        # Set device for PyTorch (InceptionResnetV1)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using PyTorch device for face recognition: {self.device}")

        # Initialize NPU YOLO for face detection
        print("Initializing NPU YOLO face detector...")
        self.yolo_detector = NPUYoloDetector(
            model_path=yolo_model_path,
            input_size=(512, 640),  # (W, H)
            conf_thres=0.15,
            iou_thres=0.5
        )

        # Initialize InceptionResnetV1 for face recognition (PyTorch)
        print("Loading InceptionResnetV1 (FaceNet) model...")
        self.face_model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)

        # Load visitor information
        self.visitors_info = self._load_visitors_info()

        # Pre-compute embeddings for registered faces
        self.known_embeddings = {}
        self.known_names = {}
        self._precompute_embeddings()

        print(f"System initialized. {len(self.known_embeddings)} faces registered.")

        # Recognition log
        self.recognition_log = []

    def _load_visitors_info(self):
        """Load visitor information from JSON file"""
        if not os.path.exists(self.visitors_json):
            print(f"Warning: {self.visitors_json} not found.")
            return {}

        with open(self.visitors_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create mapping: image_filename -> visitor_info
        visitors_map = {}
        for visitor in data.get('visitors', []):
            if 'image_filename' in visitor:
                visitors_map[visitor['image_filename']] = visitor

        return visitors_map

    def _precompute_embeddings(self):
        """Pre-compute face embeddings for all registered faces"""
        if not os.path.exists(self.registered_faces_dir):
            print(f"Warning: {self.registered_faces_dir} directory not found.")
            return

        face_files = list(Path(self.registered_faces_dir).glob("*.jpg")) + \
                     list(Path(self.registered_faces_dir).glob("*.png"))

        print(f"Computing embeddings for {len(face_files)} registered faces...")

        for face_file in face_files:
            try:
                # Load image
                img = cv2.imread(str(face_file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # NPU YOLO face detection
                boxes, scores = self.yolo_detector.detect(img_rgb)

                # Get the first detected face
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])

                    # Crop face
                    face = img_rgb[y1:y2, x1:x2]

                    # Get embedding
                    embedding = self._get_embedding(face)

                    # Store
                    filename = face_file.name
                    self.known_embeddings[filename] = embedding

                    # Store visitor info
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

    def _get_embedding(self, face_img):
        """
        Get face embedding using InceptionResnetV1 (PyTorch)

        Args:
            face_img: Face image (RGB numpy array)

        Returns:
            Embedding vector (512-dim)
        """
        # Resize to 160x160 (required for InceptionResnetV1)
        face_pil = Image.fromarray(face_img)
        face_resized = face_pil.resize((160, 160))

        # Convert to tensor
        face_tensor = torch.tensor(np.array(face_resized)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.face_model(face_tensor)

        return embedding.cpu().numpy().flatten()

    def recognize_face(self, face_img):
        """
        Recognize a face by comparing with registered faces

        Args:
            face_img: Face image (RGB numpy array)

        Returns:
            dict with 'name', 'destination', 'distance', or None if no match
        """
        if len(self.known_embeddings) == 0:
            return None

        # Get embedding for input face
        embedding = self._get_embedding(face_img)

        # Compare with all known faces
        min_distance = float('inf')
        best_match = None

        for filename, known_embedding in self.known_embeddings.items():
            # Euclidean distance
            distance = np.linalg.norm(embedding - known_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match = filename

        # Check threshold
        if min_distance < self.threshold:
            result = self.known_names[best_match].copy()
            result['distance'] = float(min_distance)
            result['confidence'] = max(0, (1 - min_distance / self.threshold) * 100)
            return result

        return None

    def process_image(self, image_path, save_output=True):
        """
        Process a single image for face recognition

        Args:
            image_path: Path to input image
            save_output: Whether to save annotated output image

        Returns:
            Recognition results
        """
        print(f"\nProcessing: {image_path}")

        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Cannot read image {image_path}")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # NPU YOLO face detection
        boxes, scores = self.yolo_detector.detect(frame_rgb)

        recognition_results = []

        # Process detected faces
        if len(boxes) > 0:
            print(f"Detected {len(boxes)} face(s)")

            for idx, (box, score) in enumerate(zip(boxes, scores)):
                # Get bounding box
                x1, y1, x2, y2 = map(int, box)
                confidence = float(score)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                try:
                    # Crop face
                    face = frame_rgb[y1:y2, x1:x2]

                    # Recognize
                    result = self.recognize_face(face)

                    face_result = {
                        'face_id': idx + 1,
                        'bbox': [x1, y1, x2, y2],
                        'detection_confidence': confidence
                    }

                    if result:
                        # Known face
                        name = result['name']
                        destination = result['destination']
                        dist = result['distance']
                        conf = result['confidence']

                        face_result.update({
                            'recognized': True,
                            'name': name,
                            'destination': destination,
                            'distance': dist,
                            'confidence': conf
                        })

                        # Display info on image
                        label = f"{name} ({conf:.1f}%)"
                        cv2.putText(frame, label, (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Dest: {destination}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        print(f"  Face {idx + 1}: {name} -> {destination} (conf: {conf:.1f}%)")
                    else:
                        # Unknown face
                        face_result['recognized'] = False
                        cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"  Face {idx + 1}: Unknown")

                    recognition_results.append(face_result)

                except Exception as e:
                    print(f"Error recognizing face {idx + 1}: {e}")

            # Save annotated image
            if save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"output_npu_{timestamp}.jpg"
                output_path = Path(self.output_dir) / output_filename
                cv2.imwrite(str(output_path), frame)
                print(f"\nAnnotated image saved: {output_path}")

        else:
            print("No faces detected")

        return recognition_results

    def run_webcam_batch(self, camera_index=0, duration_seconds=10, capture_interval=2):
        """
        Run face recognition on webcam for a specified duration (No GUI)
        Captures and processes frames at intervals

        Args:
            camera_index: Camera index (default 0)
            duration_seconds: How long to run recognition
            capture_interval: Seconds between captures
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("\n" + "="*60)
        print("YOLO Face Recognition Started (NPU Version - No GUI)")
        print("="*60)
        print(f"Duration: {duration_seconds} seconds")
        print(f"Capture interval: {capture_interval} seconds")
        print(f"Output directory: {self.output_dir}")
        print("="*60 + "\n")

        start_time = time.time()
        last_capture_time = 0
        frame_count = 0

        session_log = {
            'start_time': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'version': 'NPU',
            'captures': []
        }

        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check if duration exceeded
                if elapsed_time >= duration_seconds:
                    print(f"\nSession completed ({duration_seconds}s)")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                frame_count += 1

                # Capture and process at intervals
                if current_time - last_capture_time >= capture_interval:
                    print(f"\n[{elapsed_time:.1f}s] Capturing frame {frame_count}...")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # NPU YOLO face detection
                    boxes, scores = self.yolo_detector.detect(frame_rgb)

                    capture_result = {
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed_time,
                        'frame_number': frame_count,
                        'faces': []
                    }

                    # Process detected faces
                    if len(boxes) > 0:
                        print(f"Detected {len(boxes)} face(s)")

                        for idx, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = map(int, box)
                            confidence = float(score)

                            # Draw on frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            try:
                                face = frame_rgb[y1:y2, x1:x2]
                                result = self.recognize_face(face)

                                face_info = {
                                    'face_id': idx + 1,
                                    'bbox': [x1, y1, x2, y2],
                                    'detection_confidence': confidence
                                }

                                if result:
                                    name = result['name']
                                    destination = result['destination']
                                    conf = result['confidence']

                                    face_info.update({
                                        'recognized': True,
                                        'name': name,
                                        'destination': destination,
                                        'confidence': conf
                                    })

                                    # Annotate frame
                                    label = f"{name} ({conf:.1f}%)"
                                    cv2.putText(frame, label, (x1, y1 - 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(frame, f"Dest: {destination}", (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                    print(f"  Face {idx + 1}: {name} -> {destination} ({conf:.1f}%)")
                                else:
                                    face_info['recognized'] = False
                                    cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    print(f"  Face {idx + 1}: Unknown")

                                capture_result['faces'].append(face_info)

                            except Exception as e:
                                print(f"Error recognizing face: {e}")

                        # Save annotated frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"capture_npu_{timestamp}.jpg"
                        output_path = Path(self.output_dir) / output_filename
                        cv2.imwrite(str(output_path), frame)
                        print(f"Saved: {output_filename}")

                        capture_result['output_image'] = output_filename

                    else:
                        print("No faces detected")

                    session_log['captures'].append(capture_result)
                    last_capture_time = current_time

                # Small delay to reduce CPU usage
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nRecognition interrupted by user")

        finally:
            cap.release()

            # Save session log
            session_log['end_time'] = datetime.now().isoformat()
            log_filename = f"session_log_npu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = Path(self.output_dir) / log_filename

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(session_log, f, indent=2, ensure_ascii=False)

            print(f"\n" + "="*60)
            print("Recognition Session Summary (NPU Version)")
            print("="*60)
            print(f"Total captures: {len(session_log['captures'])}")
            print(f"Session log saved: {log_path}")
            print("="*60)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("YOLO + FaceNet Recognition System (NPU Version)")
    print("="*60)

    # Initialize system
    system = YOLOFaceRecognitionSystemNoGUI(
        registered_faces_dir="registered_faces",
        visitors_json="visitors_info.json",
        output_dir="recognition_output",
        yolo_model_path="../regulus-npu-demo/face-detection-yolov8n/face_yolov8n_640_512.mxq",
        threshold=0.6,
        device=None
    )

    # Check if NPU initialization failed (though it would raise error in __init__)
    if not hasattr(system, 'yolo_detector') or system.yolo_detector is None:
        print("System initialization failed.")
        return

    while True:
        print("\nOptions:")
        print("  1. Process single image")
        print("  2. Run webcam recognition (timed batch mode)")
        print("  3. Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                system.process_image(image_path, save_output=True)
            else:
                print(f"Error: Image not found - {image_path}")

        elif choice == '2':
            try:
                duration = input("Enter duration in seconds (default 10): ").strip()
                duration = int(duration) if duration else 10

                interval = input("Enter capture interval in seconds (default 2): ").strip()
                interval = float(interval) if interval else 2

                system.run_webcam_batch(
                    camera_index=0,
                    duration_seconds=duration,
                    capture_interval=interval
                )
            except ValueError:
                print("Invalid input. Using default values.")
                system.run_webcam_batch(camera_index=0, duration_seconds=10, capture_interval=2)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3':
            print("\nExiting...")
            break

        else:
            print("\nInvalid option. Please select 1-3.")


if __name__ == "__main__":
    main()
