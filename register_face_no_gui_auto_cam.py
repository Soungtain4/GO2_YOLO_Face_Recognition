"""
Face Registration Tool (Auto-Cam Version)
- Auto-detects first available camera
- Capture face from webcam without GUI
- Auto-capture after countdown
- Save to registered_faces/
- Update visitors_info.json
"""

import cv2
import json
import os
import time
from datetime import datetime
from pathlib import Path


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


def load_visitors_info(json_file="visitors_info.json"):
    """Load existing visitors information"""
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"visitors": []}


def save_visitors_info(data, json_file="visitors_info.json"):
    """Save visitors information to JSON"""
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def capture_face_auto(cap, countdown_seconds=3, num_captures=3):
    """
    Auto-capture face after countdown without GUI

    Args:
        cap: OpenCV VideoCapture object
        countdown_seconds: Seconds to wait before capture
        num_captures: Number of photos to take

    Returns:
        Best quality frame from captures
    """
    print(f"\nPreparing to capture in {countdown_seconds} seconds...")
    print("Please position your face in front of the camera.")

    # Warmup frames
    for _ in range(10):
        cap.read()

    # Countdown
    for i in range(countdown_seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\nCapturing photos...")
    captured_frames = []

    for i in range(num_captures):
        ret, frame = cap.read()
        if ret:
            captured_frames.append(frame)
            print(f"  Photo {i+1}/{num_captures} captured")
            time.sleep(0.5)  # Small delay between captures
        else:
            print(f"  Failed to capture photo {i+1}")

    if not captured_frames:
        return None

    # Return the middle frame (usually best quality)
    best_frame_idx = len(captured_frames) // 2
    print(f"\nUsing photo {best_frame_idx + 1} as final image")

    return captured_frames[best_frame_idx]


def register_new_face(auto_mode=True):
    """
    Register a new face using webcam (No GUI)

    Args:
        auto_mode: If True, auto-capture. If False, manual capture on Enter
    """
    # Create registered_faces directory if not exists
    faces_dir = Path("registered_faces")
    faces_dir.mkdir(exist_ok=True)

    # Load existing visitors info
    visitors_data = load_visitors_info()

    # Get person information
    print("\n" + "="*60)
    print("FACE REGISTRATION (Auto-Cam Version)")
    print("="*60)
    print()

    name = input("Enter name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return False

    destination = input("Enter destination: ").strip()
    if not destination:
        print("Error: Destination cannot be empty")
        return False

    # Generate unique ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_id = f"person_{timestamp}"
    image_filename = f"{person_id}.jpg"

    # Auto-detect camera
    camera_index = find_available_camera()
    if camera_index is None:
        print("Error: No cameras found!")
        return False

    # Open webcam
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return False

    print("Camera opened successfully!")

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured_frame = None

    if auto_mode:
        # Auto-capture mode
        captured_frame = capture_face_auto(cap, countdown_seconds=3, num_captures=3)
    else:
        # Manual capture mode
        print("\n" + "="*60)
        print("Manual capture mode:")
        print("  - Press ENTER when ready to capture")
        print("  - Type 'q' to quit")
        print("="*60)

        user_input = input("\nPress ENTER to capture (or 'q' to quit): ").strip().lower()

        if user_input == 'q':
            print("Registration cancelled")
            cap.release()
            return False

        # Capture frame
        ret, frame = cap.read()
        if ret:
            captured_frame = frame
            print("Photo captured!")
        else:
            print("Error: Failed to capture frame")

    cap.release()
    print("Camera released")

    # Save captured image
    if captured_frame is not None:
        image_path = faces_dir / image_filename
        success = cv2.imwrite(str(image_path), captured_frame)

        if success:
            print(f"\nImage saved: {image_path}")

            # Verify image size
            file_size = os.path.getsize(image_path)
            print(f"File size: {file_size / 1024:.2f} KB")

            # Add to visitors info
            new_visitor = {
                "id": person_id,
                "name": name,
                "destination": destination,
                "image_filename": image_filename,
                "registered_date": timestamp
            }

            visitors_data["visitors"].append(new_visitor)
            save_visitors_info(visitors_data)

            print()
            print("="*60)
            print("REGISTRATION COMPLETE")
            print("="*60)
            print(f"Name:        {name}")
            print(f"Destination: {destination}")
            print(f"ID:          {person_id}")
            print(f"Image:       {image_filename}")
            print("="*60)
            print()
            print("Face registered successfully!")
            print("You can now use the face recognition system.")

            return True
        else:
            print("Error: Failed to save image")
            return False
    else:
        print("No frame captured. Registration cancelled.")
        return False


def list_registered_faces():
    """List all registered faces"""
    visitors_data = load_visitors_info()
    visitors = visitors_data.get("visitors", [])

    if not visitors:
        print("\nNo registered faces found.")
        return

    print("\n" + "="*60)
    print(f"REGISTERED FACES ({len(visitors)} total)")
    print("="*60)

    for i, visitor in enumerate(visitors, 1):
        print(f"\n{i}. {visitor.get('name', 'Unknown')}")
        print(f"   Destination: {visitor.get('destination', 'Unknown')}")
        print(f"   ID: {visitor.get('id', 'Unknown')}")
        print(f"   Image: {visitor.get('image_filename', 'Unknown')}")
        print(f"   Registered: {visitor.get('registered_date', 'Unknown')}")

    print("="*60)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("FACE REGISTRATION SYSTEM (Auto-Cam Version)")
    print("="*60)

    while True:
        print("\nOptions:")
        print("  1. Register new face (Auto-capture)")
        print("  2. Register new face (Manual-capture)")
        print("  3. List registered faces")
        print("  4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            try:
                register_new_face(auto_mode=True)
            except KeyboardInterrupt:
                print("\n\nRegistration interrupted by user")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '2':
            try:
                register_new_face(auto_mode=False)
            except KeyboardInterrupt:
                print("\n\nRegistration interrupted by user")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3':
            list_registered_faces()

        elif choice == '4':
            print("\nExiting...")
            break

        else:
            print("\nInvalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
