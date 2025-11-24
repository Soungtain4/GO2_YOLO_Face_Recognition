"""
YOLO + InceptionResnetV1 Face Recognition System (Auto-Cam Version)
- Auto-detects first available camera
- Face Detection: YOLOv8-Face (ultralytics)
- Face Recognition: InceptionResnetV1 (facenet-pytorch)
- Output: Save annotated images + JSON logs
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
from datetime import datetime
from huggingface_hub import hf_hub_download
import time


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


class YOLOFaceRecognitionSystemNoGUI:
    def __init__(self,
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 output_dir="recognition_output",
                 yolo_model_name="yolov8n-face.pt",
                 threshold=0.6,
                 device=None):
        """
        Initialize YOLO + FaceNet face recognition system (No GUI)

        Args:
            registered_faces_dir: Directory containing registered face images
            visitors_json: JSON file with visitor information
            output_dir: Directory to save output images and logs
            yolo_model_name: YOLO model for face detection
            threshold: Distance threshold for face matching
            device: 'cuda' or 'cpu'
        """
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.output_dir = output_dir
        self.threshold = threshold

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

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

        # YOLO face detection
        results = self.yolo_model(frame_rgb, verbose=False)

        recognition_results = []

        # Process detected faces
        if len(results[0].boxes) > 0:
            print(f"Detected {len(results[0].boxes)} face(s)")

            for idx, box in enumerate(results[0].boxes):
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

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
                output_filename = f"output_{timestamp}.jpg"
                output_path = Path(self.output_dir) / output_filename
                cv2.imwrite(str(output_path), frame)
                print(f"\nAnnotated image saved: {output_path}")

        else:
            print("No faces detected")

        return recognition_results

    def run_burst_capture(self, camera_index=0, num_frames=10):
        """
        Capture multiple frames in quick succession and recognize faces
        
        Args:
            camera_index: Camera index
            num_frames: Number of frames to capture
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print(f"\nStarting Burst Capture ({num_frames} frames)...")
        
        results_summary = {}
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i+1}")
                continue
                
            print(f"\nProcessing frame {i+1}/{num_frames}...")
            
            # Save temp file to reuse process_image logic (or refactor, but this is safer for now)
            # Actually, let's just use the logic inline to avoid IO overhead, 
            # but for consistency with existing code structure, let's call a helper or just do it.
            # Re-using internal logic:
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(frame_rgb, verbose=False)
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    try:
                        face = frame_rgb[y1:y2, x1:x2]
                        result = self.recognize_face(face)
                        
                        if result:
                            name = result['name']
                            conf = result['confidence']
                            print(f"  -> Recognized: {name} ({conf:.1f}%)")
                            
                            if name not in results_summary:
                                results_summary[name] = []
                            results_summary[name].append(conf)
                        else:
                            print("  -> Unknown face")
                            
                    except Exception as e:
                        print(f"  -> Error: {e}")
            else:
                print("  -> No faces detected")
                
            time.sleep(0.1) # Small delay
            
        cap.release()
        
        print("\n" + "="*30)
        print("BURST RESULTS SUMMARY")
        print("="*30)
        if results_summary:
            for name, confs in results_summary.items():
                avg_conf = sum(confs) / len(confs)
                print(f"Name: {name}")
                print(f"  - Detection Count: {len(confs)}/{num_frames}")
                print(f"  - Avg Confidence:  {avg_conf:.1f}%")
        else:
            print("No known faces recognized in any frame.")
        print("="*30)

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
        print("YOLO Face Recognition Started (Auto-Cam Version)")
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

                    # YOLO face detection
                    results = self.yolo_model(frame_rgb, verbose=False)

                    capture_result = {
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed_time,
                        'frame_number': frame_count,
                        'faces': []
                    }

                    # Process detected faces
                    if len(results[0].boxes) > 0:
                        print(f"Detected {len(results[0].boxes)} face(s)")

                        for idx, box in enumerate(results[0].boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            confidence = float(box.conf[0].cpu().numpy())

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
                        output_filename = f"capture_{timestamp}.jpg"
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
            log_filename = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = Path(self.output_dir) / log_filename

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(session_log, f, indent=2, ensure_ascii=False)

            print(f"\n" + "="*60)
            print("Recognition Session Summary")
            print("="*60)
            print(f"Total captures: {len(session_log['captures'])}")
            print(f"Session log saved: {log_path}")
            print("="*60)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("YOLO + FaceNet Recognition System (Auto-Cam Version)")
    print("="*60)

    # Initialize system
    system = YOLOFaceRecognitionSystemNoGUI(
        registered_faces_dir="registered_faces",
        visitors_json="visitors_info.json",
        output_dir="recognition_output",
        yolo_model_name="yolov8n-face.pt",
        threshold=0.6,
        device=None
    )

    while True:
        print("\nOptions:")
        print("  1. Burst Capture Verification (10 frames)")
        print("  2. Run webcam recognition (Auto-Cam)")
        print("  3. Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            try:
                # Auto-detect camera
                camera_index = find_available_camera()
                if camera_index is None:
                    print("Error: No cameras found!")
                    continue

                system.run_burst_capture(camera_index=camera_index, num_frames=10)
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '2':
            try:
                # Auto-detect camera
                camera_index = find_available_camera()
                if camera_index is None:
                    print("Error: No cameras found!")
                    continue

                print(f"\n[INFO] Using camera index: {camera_index}")

                duration = input("Enter duration in seconds (default 10): ").strip()
                duration = int(duration) if duration else 10

                interval = input("Enter capture interval in seconds (default 2): ").strip()
                interval = float(interval) if interval else 2

                system.run_webcam_batch(
                    camera_index=camera_index,
                    duration_seconds=duration,
                    capture_interval=interval
                )
            except ValueError:
                print("Invalid input. Using default values.")
                if camera_index is not None:
                    system.run_webcam_batch(camera_index=camera_index, duration_seconds=10, capture_interval=2)
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
