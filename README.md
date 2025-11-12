
# YOLOv8 + InceptionResnetV1 기반 실시간 방문자 얼굴 인식 시스템

## 1. 프로젝트 개요

이 프로젝트는 웹캠을 통해 실시간으로 방문자를 인식하는 시스템입니다. 최신 객체 탐지 모델인 **YOLOv8-Face**를 사용하여 얼굴을 빠르고 정확하게 탐지하고, **InceptionResnetV1 (FaceNet)** 모델을 이용해 신원을 확인합니다.

방문자를 사전에 등록하면, 시스템이 해당 방문자를 인식했을 때 이름과 방문 목적지를 화면에 표시해줍니다.

## 2. 주요 기능

- **실시간 얼굴 탐지**: YOLOv8-Face 모델을 사용하여 비디오 스트림에서 실시간으로 얼굴을 탐지합니다.
- **얼굴 인식**: 등록된 얼굴과 현재 탐지된 얼굴을 비교하여 신원을 확인합니다.
- **방문자 정보 표시**: 인식된 방문자의 이름과 목적지를 화면에 오버레이하여 표시합니다.
- **간편한 방문자 등록**: 간단한 CLI 스크립트를 통해 새로운 방문자의 얼굴과 정보를 쉽게 등록할 수 있습니다.

## 3. 동작 원리

시스템은 두 가지 주요 단계로 구성됩니다.

1.  **얼굴 탐지 (Face Detection)**
    - `ultralytics` 라이브러리의 **YOLOv8-Face** 모델을 사용합니다.
    - 이 모델은 비디오 프레임에서 얼굴 영역의 경계 상자(bounding box)를 찾아냅니다.
    - HuggingFace Hub에서 사전 훈련된 모델(`yolov8n-face.pt`)을 자동으로 다운로드하여 사용합니다.

2.  **얼굴 인식 (Face Recognition)**
    - `facenet-pytorch` 라이브러리의 **InceptionResnetV1** 모델 (VGGFace2 데이터셋으로 사전 훈련됨)을 사용합니다.
    - 탐지된 얼굴 영역을 160x160 크기로 변환한 뒤, 512차원의 임베딩(embedding) 벡터로 추출합니다. 이 벡터는 얼굴의 고유한 특징을 나타냅니다.
    - 시스템은 사전에 등록된 모든 얼굴의 임베딩 벡터를 미리 계산하여 저장해 둡니다.
    - 새로 탐지된 얼굴의 임베딩과 기존에 저장된 임베딩 간의 유클리드 거리(Euclidean distance)를 계산하여 가장 가까운 얼굴을 찾습니다.
    - 거리가 특정 임계값(`threshold`)보다 낮으면 동일인으로 판단하고, 등록된 정보를 화면에 표시합니다.

## 4. 디렉토리 구조

```
.
├── registered_faces/         # 등록된 얼굴 이미지 저장 폴더
│   └── person_xxxxxxxx.jpg
├── recognition_output/       # (미사용) 인식 결과 저장 폴더
├── face_recognition_yolo.py  # 메인 얼굴 인식 프로그램 (GUI)
├── register_face.py          # 얼굴 등록 프로그램 (CLI)
├── visitors_info.json        # 등록된 방문자 정보 (메타데이터)
├── requirements.txt          # 파이썬 의존성 파일
└── README.md                 # 프로젝트 설명 파일
```

## 5. 설치 및 환경 설정

1.  **프로젝트 클론**
    ```bash
    git clone https://github.com/your-username/YOLO_InceptionResnetV1_Repo.git
    cd YOLO_InceptionResnetV1_Repo
    ```

2.  **가상 환경 생성 및 활성화 (권장)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **의존성 설치**
    `requirements.txt` 파일에 명시된 모든 라이브러리를 설치합니다. GPU 사용 환경(NVIDIA, CUDA)을 적극 권장합니다.
    ```bash
    pip install -r requirements.txt
    ```
    *참고: `torch`와 `onnxruntime-gpu` 등 GPU 관련 라이브러리 설치 시 문제가 발생하면, 각 라이브러리의 공식 문서를 참고하여 환경에 맞는 버전을 직접 설치해야 할 수 있습니다.*

## 6. 사용 방법

### 6.1. 방문자 얼굴 등록

얼굴 인식을 수행하기 전에 먼저 한 명 이상의 방문자를 등록해야 합니다.

1.  터미널에서 아래 명령어를 실행합니다.
    ```bash
    python register_face.py
    ```

2.  안내에 따라 **이름(Name)**과 **방문 목적지(Destination)**를 입력합니다.

3.  웹캠 화면이 나타나면 얼굴을 프레임 중앙에 위치시킨 후 `SPACE` 키를 눌러 촬영합니다.

4.  `Y` 키를 눌러 저장하면 `registered_faces/` 폴더에 얼굴 이미지가 저장되고, `visitors_info.json` 파일에 관련 정보가 기록됩니다.

### 6.2. 실시간 얼굴 인식

방문자 등록을 완료한 후, 아래 명령어를 실행하여 실시간 인식을 시작합니다.

```bash
python face_recognition_yolo.py
```

- 웹캠이 켜지고 얼굴 인식이 시작됩니다.
- 등록된 방문자가 인식되면 초록색 상자와 함께 이름, 목적지, 신뢰도(%)가 표시됩니다.
- 미등록 방문자는 "Unknown"으로 표시됩니다.
- `q` 키를 누르면 프로그램이 종료됩니다.

## 7. 주요 파일 설명

- **`register_face.py`**: 웹캠을 통해 사용자의 이름, 목적지, 얼굴 사진을 등록하고 `visitors_info.json`과 `registered_faces/`에 저장하는 스크립트.
- **`face_recognition_yolo.py`**: 메인 애플리케이션. YOLOv8로 얼굴을 탐지하고 InceptionResnetV1으로 특징을 추출하여 등록된 얼굴과 비교 후 결과를 화면에 출력.
- **`*_no_gui.py`**: GUI(OpenCV 창) 없이 백그라운드에서 실행되는 버전의 스크립트들. (터미널 로그만 출력)
- **`visitors_info.json`**: 등록된 각 개인의 ID, 이름, 목적지, 이미지 파일명 등 메타데이터를 저장하는 JSON 파일.
- **`requirements.txt`**: `torch`, `torchvision`, `ultralytics`, `facenet-pytorch`, `opencv-python` 등 프로젝트 실행에 필요한 모든 파이썬 패키지 목록.

## 8. 라이선스 (License)

이 프로젝트는 **AGPL-3.0 라이선스**를 따릅니다. `ultralytics` 라이브러리(AGPL-3.0)를 사용하므로, 이 프로젝트의 소스 코드 또한 동일한 라이선스에 따라 공개되어야 합니다. 자세한 내용은 `LICENSE` 파일을 참고하십시오.

본 프로젝트는 교육 및 연구 목적으로 제작되었으며, 상업적 사용을 권장하지 않습니다.
