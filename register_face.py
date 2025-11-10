"""
Face Registration Tool for Jetson Nano Orin
- Capture face from webcam
- Save to registered_faces/
- Update visitors_info.json
"""

import cv2
import json
import os
from datetime import datetime
from pathlib import Path


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


def register_new_face(camera_index=0):
    """
    Register a new face using webcam

    Args:
        camera_index: Camera device index (default: 0)
    """
    # Create registered_faces directory if not exists
    faces_dir = Path("registered_faces")
    faces_dir.mkdir(exist_ok=True)

    # Load existing visitors info
    visitors_data = load_visitors_info()

    # Get person information
    print("\n" + "="*50)
    print("FACE REGISTRATION")
    print("="*50)
    print()

    name = input("Enter name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    destination = input("Enter destination: ").strip()
    if not destination:
        print("Error: Destination cannot be empty")
        return

    # Generate unique ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_id = f"person_{timestamp}"
    image_filename = f"{person_id}.jpg"

    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print()
    print("="*50)
    print("Camera opened. Instructions:")
    print("  - Position your face in the frame")
    print("  - Press 'SPACE' to capture")
    print("  - Press 'q' to quit without saving")
    print("="*50)
    print()

    captured = False
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Display frame
        display_frame = frame.copy()

        # Draw instructions
        cv2.putText(display_frame, "Press SPACE to capture, Q to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Registration - Position your face', display_frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space bar
            captured_frame = frame.copy()
            captured = True
            print("Face captured!")

            # Show captured image for confirmation
            cv2.imshow('Captured - Press Y to save, N to retake', captured_frame)

            while True:
                confirm_key = cv2.waitKey(0) & 0xFF
                if confirm_key == ord('y') or confirm_key == ord('Y'):
                    break
                elif confirm_key == ord('n') or confirm_key == ord('N'):
                    captured = False
                    print("Retaking photo...")
                    cv2.destroyWindow('Captured - Press Y to save, N to retake')
                    break

            if captured:
                break

        elif key == ord('q') or key == ord('Q'):
            print("Registration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured and captured_frame is not None:
        # Save image
        image_path = faces_dir / image_filename
        cv2.imwrite(str(image_path), captured_frame)
        print(f"Image saved: {image_path}")

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
        print("="*50)
        print("REGISTRATION COMPLETE")
        print("="*50)
        print(f"Name: {name}")
        print(f"Destination: {destination}")
        print(f"ID: {person_id}")
        print(f"Image: {image_filename}")
        print("="*50)
        print()
        print("Face registered successfully!")
        print("You can now use the face recognition system.")
    else:
        print("No face captured. Registration cancelled.")


def main():
    """Main function"""
    try:
        register_new_face(camera_index=0)
    except KeyboardInterrupt:
        print("\n\nRegistration interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
