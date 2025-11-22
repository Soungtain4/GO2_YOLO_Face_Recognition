"""
Fast Face Recognition System (CPU Optimized)
- Face Detection: YOLOv8-Face (ultralytics)
- Face Recognition: MobileFaceNet (OpenCV DNN via ONNX)
- Performance: Optimized for CPU usage (MobileFaceNet is ~20x lighter than InceptionResnetV1)

This script automatically downloads the MobileFaceNet ONNX model if missing.
"""

import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class FastFaceRecognitionSystem:
    def __init__(self, 
                 registered_faces_dir="registered_faces",
                 visitors_json="visitors_info.json",
                 output_dir="recognition_output",
                 threshold=0.5): # MobileFaceNet threshold is usually lower (0.4-0.6)
        
        self.registered_faces_dir = registered_faces_dir
        self.visitors_json = visitors_json
        self.output_dir = output_dir
        self.threshold = threshold
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # 1. Initialize YOLO (Detection)
        print("Loading YOLOv8-Face model...")
        try:
            model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
            self.detector = YOLO(model_path)
        except:
            print("Downloading from HF failed, trying local 'yolov8n-face.pt' or standard 'yolov8n.pt'")
            self.detector = YOLO("yolov8n-face.pt")

        # 2. Initialize MobileFaceNet (Recognition) - Optimized for CPU
        print("Loading MobileFaceNet model (CPU Optimized)...")
        self.recog_model_path = "mobilefacenet.onnx"
        if not os.path.exists(self.recog_model_path):
            self._download_mobilefacenet()
            
        self.recognizer = cv2.dnn.readNetFromONNX(self.recog_model_path)
        
        # Load registered faces
        self.known_embeddings = {}
        self.known_names = {}
        self.visitors_info = self._load_visitors_info()
        self._precompute_embeddings()
        
        print(f"System initialized. {len(self.known_embeddings)} faces registered.")

    def _download_mobilefacenet(self):
        """
        Download MobileFaceNet model from Hugging Face (buffalo_s.zip).
        Using 'vladmandic/insightface-faceanalysis' which mirrors the official models.
        """
        print("MobileFaceNet 모델 다운로드 중 (Hugging Face)...")
        print("저장소: vladmandic/insightface-faceanalysis")
        
        repo_id = "vladmandic/insightface-faceanalysis"
        filename = "models/buffalo_s.zip"
        
        import zipfile
        import shutil
        
        try:
            # 1. Download zip using hf_hub_download
            print(f"다운로드 시작: {repo_id}/{filename}")
            zip_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"다운로드 완료: {zip_path}")
            
            # 2. Extract specific file (w600k_mbf.onnx is MobileFaceNet)
            print("압축 해제 및 모델 추출 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # The file inside might be in a subfolder or root
                # We look for 'w600k_mbf.onnx'
                found = False
                for file in zip_ref.namelist():
                    if file.endswith("w600k_mbf.onnx"):
                        with zip_ref.open(file) as source, open(self.recog_model_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        found = True
                        break
                
                if not found:
                    raise Exception("zip 파일 내에 모델(w600k_mbf.onnx)이 없습니다.")
            
            print(f"설치 완료! 모델 저장됨: {self.recog_model_path}")
                
        except Exception as e:
            print(f"다운로드 오류 발생: {e}")
            print("인터넷 연결을 확인하거나, 수동으로 'mobilefacenet.onnx' 파일을 구해서 이 폴더에 넣어주세요.")
            raise

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

    def _preprocess_face(self, face_img):
        """
        Preprocess face for MobileFaceNet
        Expected input: RGB, 112x112
        """
        # Resize to 112x112 (Standard for MobileFaceNet)
        face = cv2.resize(face_img, (112, 112))
        
        # Convert to float32 and normalize to [-1, 1]
        face = (face.astype(np.float32) - 127.5) / 128.0
        
        # HWC to CHW
        face = face.transpose(2, 0, 1)
        
        # Add batch dimension: (1, 3, 112, 112)
        face = np.expand_dims(face, axis=0)
        return face

    def _get_embedding(self, face_img):
        """Get 128-dim embedding using MobileFaceNet"""
        blob = self._preprocess_face(face_img)
        self.recognizer.setInput(blob)
        embedding = self.recognizer.forward()
        
        # Flatten and normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    def _precompute_embeddings(self):
        if not os.path.exists(self.registered_faces_dir):
            return

        face_files = list(Path(self.registered_faces_dir).glob("*.jpg")) + \
                     list(Path(self.registered_faces_dir).glob("*.png"))

        print(f"Computing embeddings for {len(face_files)} registered faces...")

        for face_file in face_files:
            try:
                img = cv2.imread(str(face_file))
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect face
                results = self.detector(img_rgb, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Padding to ensure full face
                    h, w = img_rgb.shape[:2]
                    pad_x = int((x2-x1) * 0.1)
                    pad_y = int((y2-y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    face = img_rgb[y1:y2, x1:x2]
                    embedding = self._get_embedding(face)
                    
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
            except Exception as e:
                print(f"Error processing {face_file.name}: {e}")

    def recognize_face(self, face_img):
        embedding = self._get_embedding(face_img)
        
        max_score = -1.0
        best_match = None
        
        for filename, known_embedding in self.known_embeddings.items():
            # Cosine Similarity
            score = np.dot(embedding, known_embedding)
            
            if score > max_score:
                max_score = score
                best_match = filename
        
        # Convert cosine similarity to distance-like metric for compatibility
        # Cosine score: 1.0 (same), -1.0 (opposite)
        # Threshold: usually 0.5 ~ 0.6 for MobileFaceNet
        
        if max_score > self.threshold:
            result = self.known_names[best_match].copy()
            result['confidence'] = float(max_score * 100)
            result['distance'] = 1.0 - max_score # Just for logging
            return result
            
        return None

    def process_image(self, image_path, save_output=True):
        """Process a single image"""
        print(f"\nProcessing: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found - {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = self.detector(frame_rgb, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            print(f"Detected {len(results[0].boxes)} face(s)")
            
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                conf = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Recognize
                h, w = frame.shape[:2]
                pad_x = int((x2-x1) * 0.1)
                pad_y = int((y2-y1) * 0.1)
                fx1 = max(0, x1 - pad_x)
                fy1 = max(0, y1 - pad_y)
                fx2 = min(w, x2 + pad_x)
                fy2 = min(h, y2 + pad_y)
                
                face_img = frame_rgb[fy1:fy2, fx1:fx2]
                if face_img.size == 0: continue
                
                result = self.recognize_face(face_img)
                
                if result:
                    name = result['name']
                    dest = result['destination']
                    conf_val = result['confidence']
                    text = f"{name} ({conf_val:.1f}%)"
                    cv2.putText(frame, text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Dest: {dest}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"  Face: {name} -> {dest} ({conf_val:.1f}%)")
                else:
                    cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"  Face: Unknown")
            
            if save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"output_fast_{timestamp}.jpg"
                output_path = Path(self.output_dir) / output_filename
                cv2.imwrite(str(output_path), frame)
                print(f"Saved annotated image: {output_path}")
        else:
            print("No faces detected")

    def run_webcam(self):
        # Auto-detect camera
        camera_index = 0
        available_cameras = []
        print("Scanning for cameras...")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"  [FOUND] Camera index {i}")
                cap.release()
        
        if not available_cameras:
            print("Error: No cameras found.")
            return
            
        camera_index = available_cameras[0]
        print(f"Auto-selected camera index: {camera_index}")
        
        cap = cv2.VideoCapture(camera_index)
        
        print("\n" + "="*60)
        print("Fast Face Recognition Started (CPU Optimized)")
        print("="*60)
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect
                results = self.detector(frame_rgb, verbose=False)
                
                if len(results) > 0:
                    for box in results[0].boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Recognize
                        h, w = frame.shape[:2]
                        pad_x = int((x2-x1) * 0.1)
                        pad_y = int((y2-y1) * 0.1)
                        fx1 = max(0, x1 - pad_x)
                        fy1 = max(0, y1 - pad_y)
                        fx2 = min(w, x2 + pad_x)
                        fy2 = min(h, y2 + pad_y)
                        
                        face_img = frame_rgb[fy1:fy2, fx1:fx2]
                        if face_img.size == 0: continue
                        
                        result = self.recognize_face(face_img)
                        
                        if result:
                            text = f"{result['name']} ({result['confidence']:.1f}%)"
                            color = (0, 255, 0)
                            cv2.putText(frame, f"Dest: {result['destination']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            print(f"\rRecognized: {result['name']} ({result['confidence']:.1f}%)", end="")
                        else:
                            text = "Unknown"
                            color = (0, 0, 255)
                            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show fps
                cv2.imshow('Fast Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_batch_voting(self, num_frames=10, interval=0.5):
        """
        Capture multiple frames and vote for the most frequent face.
        Ignores 'Unknown' unless it's the only result.
        """
        # Auto-detect camera
        camera_index = 0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    camera_index = i
                    cap.release()
                    break
                cap.release()
        
        print(f"\nOpening camera {camera_index} for Batch Voting...")
        cap = cv2.VideoCapture(camera_index)
        
        print(f"Capturing {num_frames} frames (Interval: {interval}s)...")
        
        detected_names = []
        
        try:
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error reading frame {i+1}")
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect
                results = self.detector(frame_rgb, verbose=False)
                
                frame_names = []
                if len(results) > 0:
                    for box in results[0].boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        # Recognize
                        h, w = frame.shape[:2]
                        pad_x = int((x2-x1) * 0.1)
                        pad_y = int((y2-y1) * 0.1)
                        fx1 = max(0, x1 - pad_x)
                        fy1 = max(0, y1 - pad_y)
                        fx2 = min(w, x2 + pad_x)
                        fy2 = min(h, y2 + pad_y)
                        
                        face_img = frame_rgb[fy1:fy2, fx1:fx2]
                        if face_img.size == 0: continue
                        
                        result = self.recognize_face(face_img)
                        
                        if result:
                            frame_names.append(result['name'])
                            print(f"  Frame {i+1}: {result['name']} ({result['confidence']:.1f}%)")
                        else:
                            print(f"  Frame {i+1}: Unknown")
                            
                if not frame_names:
                    print(f"  Frame {i+1}: No known face detected")
                
                detected_names.extend(frame_names)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            cap.release()
            
        print("\n" + "="*40)
        print("Voting Results")
        print("="*40)
        
        if not detected_names:
            print("Result: No known faces identified.")
            return
            
        from collections import Counter
        counts = Counter(detected_names)
        
        print("Counts:")
        for name, count in counts.items():
            print(f"  {name}: {count}/{num_frames}")
            
        most_common = counts.most_common(1)[0]
        winner_name = most_common[0]
        winner_count = most_common[1]
        
        print("-" * 40)
        print(f"FINAL DECISION: {winner_name} (Detected in {winner_count} frames)")
        print("=" * 40)

def main():
    print("\n" + "="*60)
    print("Fast Face Recognition System (CPU Optimized)")
    print("="*60)
    
    system = FastFaceRecognitionSystem()
    
    while True:
        print("\nOptions:")
        print("  1. Run webcam recognition (Real-time)")
        print("  2. Run batch voting (10 frames -> Best Match)")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            system.run_webcam()
        elif choice == '2':
            system.run_batch_voting()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
