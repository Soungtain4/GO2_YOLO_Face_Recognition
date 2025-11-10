"""
YOLO + InceptionResnetV1 Face Recognition System
- Face Detection: YOLOv8-Face (ultralytics)
- Face Recognition: InceptionResnetV1 (facenet-pytorch)
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download


class YOLOFaceRecognitionSystem:
    def __init__(self,
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 yolo_model_name="yolov8n-face.pt",  # YOLO nano for face detection
                 threshold=0.6,
                 device=None):
        """
        Initialize YOLO + FaceNet face recognition system

        Args:
            registered_faces_dir: Directory containing registered face images
            visitors_json: JSON file with visitor information
            yolo_model_name: YOLO model for face detection
            threshold: Distance threshold for face matching
            device: 'cuda' or 'cpu'
        """
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize YOLO for face detection
        print("Loading YOLOv8-Face model...")

        # Download model from HuggingFace if not using local path
        if yolo_model_name == "yolov8n-face.pt":
            try:
                print("Downloading YOLOv8-Face from HuggingFace...")
                yolo_model_name = hf_hub_download(
                    repo_id="arnabdhar/YOLOv8-Face-Detection",
                    filename="model.pt"
                )
                print(f"Model downloaded to: {yolo_model_name}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        self.yolo_model = YOLO(yolo_model_name)

        # Initialize InceptionResnetV1 for face recognition
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

                # YOLO face detection
                results = self.yolo_model(img_rgb, verbose=False)

                # Get the first detected face
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)

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
        Get face embedding using InceptionResnetV1

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

    def run_webcam(self, camera_index=0):
        """
        Run face recognition on webcam feed

        Args:
            camera_index: Camera index (default 0)
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("\n=== YOLO Face Recognition Started ===")
        print("Press 'q' to quit\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO face detection (every frame for smooth visualization)
            results = self.yolo_model(frame_rgb, verbose=False)

            # Process detected faces
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Face recognition (every 5 frames to save computation)
                    if frame_count % 5 == 0:
                        try:
                            # Crop face
                            face = frame_rgb[y1:y2, x1:x2]

                            # Recognize
                            result = self.recognize_face(face)

                            if result:
                                # Known face
                                name = result['name']
                                destination = result['destination']
                                dist = result['distance']
                                conf = result['confidence']

                                # Display info
                                label = f"{name} ({conf:.1f}%)"
                                cv2.putText(frame, label, (x1, y1 - 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame, f"Dest: {destination}", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                print(f"[Frame {frame_count}] Recognized: {name} -> {destination} (dist: {dist:.3f})")
                            else:
                                # Unknown face
                                cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        except Exception as e:
                            print(f"Error recognizing face: {e}")

                    # Display detection confidence
                    cv2.putText(frame, f"Det: {confidence:.2f}", (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Display frame
            cv2.imshow('YOLO + FaceNet Recognition', frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n=== Recognition Stopped ===")


def main():
    """Main function"""
    # Initialize system
    system = YOLOFaceRecognitionSystem(
        registered_faces_dir="registered_faces",
        visitors_json="visitors_info.json",
        yolo_model_name="yolov8n-face.pt",  # Will auto-download if not exists
        threshold=0.6,
        device=None  # Auto-detect CUDA
    )

    # Run webcam recognition
    system.run_webcam(camera_index=0)


if __name__ == "__main__":
    main()
