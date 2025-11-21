import cv2

def list_available_cameras(max_cameras=10):
    print(f"Scanning for cameras (indices 0 to {max_cameras-1})...")
    available_cameras = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCCESS] Camera found at index: {index}")
                print(f"  - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                available_cameras.append(index)
            else:
                print(f"[WARNING] Camera opened at index {index} but failed to read frame.")
            cap.release()
        else:
            pass
            # print(f"[INFO] No camera at index {index}")

    print("\n" + "="*30)
    if available_cameras:
        print(f"Available Camera Indices: {available_cameras}")
        print("Use one of these indices in your face recognition script.")
    else:
        print("No cameras found.")
    print("="*30)

if __name__ == "__main__":
    list_available_cameras()
