
# YOLOv8 + InceptionResnetV1 기반 실시간 방문자 얼굴 인식 시스템

## 1. 프로젝트 개요

이 프로젝트는 웹캠을 통해 실시간으로 방문자를 인식하는 시스템입니다. 최신 객체 탐지 모델인 **YOLOv8-Face**를 사용하여 얼굴을 빠르고 정확하게 탐지하고, **InceptionResnetV1 (FaceNet)** 모델을 이용해 신원을 확인합니다.

방문자를 사전에 등록하면, 시스템이 해당 방문자를 인식했을 때 이름과 방문 목적지를 화면에 표시해줍니다.

## 2. 주요 기능

- **실시간 얼굴 탐지**: YOLOv8-Face 모델을 사용하여 비디오 스트림에서 실시간으로 얼굴을 탐지합니다.
- **얼굴 인식**: 등록된 얼굴과 현재 탐지된 얼굴을 비교하여 신원을 확인합니다.
- **방문자 정보 표시**: 인식된 방문자의 이름과 목적지를 화면에 오버레이하여 표시합니다.
- **간편한 방문자 등록**: 간단한 CLI 스크립트를 통해 새로운 방문자의 얼굴과 정보를 쉽게 등록할 수 있습니다.
- **GUI / No-GUI 모드 지원**: 디스플레이가 있는 환경과 없는 환경(서버, 임베디드) 모두 지원합니다.
- **배치 처리**: No-GUI 모드에서 정해진 시간 동안 자동으로 얼굴을 인식하고 결과를 저장합니다.

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
├── registered_faces/              # 등록된 얼굴 이미지 저장 폴더
│   └── person_xxxxxxxx.jpg
├── recognition_output/            # 인식 결과 저장 폴더 (no_gui 버전에서 사용)
│   ├── capture_xxxxxxxx.jpg      # 캡처된 이미지
│   └── session_log_xxxxxxxx.json # 세션 로그
├── face_recognition_yolo.py       # 메인 얼굴 인식 프로그램 (GUI 버전)
├── face_recognition_yolo_no_gui.py # 얼굴 인식 프로그램 (No-GUI 버전)
├── register_face.py               # 얼굴 등록 프로그램 (GUI 버전)
├── register_face_no_gui.py        # 얼굴 등록 프로그램 (No-GUI 버전)
├── visitors_info.json             # 등록된 방문자 정보 (메타데이터)
├── requirements.txt               # 파이썬 의존성 파일
├── LICENSE                        # AGPL-3.0 라이선스 파일
└── README.md                      # 프로젝트 설명 파일
```

## 5. 설치 및 환경 설정

1.  **프로젝트 클론**
    ```bash
    git clone https://github.com/Soungtain4/YOLO_InceptionResnetV1_Repo.git
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

#### GUI 버전 (디스플레이가 있는 환경)

1.  터미널에서 아래 명령어를 실행합니다.
    ```bash
    python register_face.py
    ```

2.  안내에 따라 **이름(Name)**과 **방문 목적지(Destination)**를 입력합니다.

3.  웹캠 화면이 나타나면 얼굴을 프레임 중앙에 위치시킨 후 `SPACE` 키를 눌러 촬영합니다.

4.  `Y` 키를 눌러 저장하면 `registered_faces/` 폴더에 얼굴 이미지가 저장되고, `visitors_info.json` 파일에 관련 정보가 기록됩니다.

#### No-GUI 버전 (서버, 임베디드 환경)

디스플레이가 없는 환경(SSH 접속, Jetson Nano 등)에서 사용할 수 있습니다.

1.  터미널에서 아래 명령어를 실행합니다.
    ```bash
    python register_face_no_gui.py
    ```

2.  메뉴에서 옵션을 선택합니다:
    - **옵션 1**: 자동 캡처 모드 (3초 카운트다운 후 자동 촬영)
    - **옵션 2**: 수동 캡처 모드 (Enter 키를 눌러 촬영)
    - **옵션 3**: 등록된 얼굴 목록 조회

3.  이름과 목적지를 입력한 후, 카메라 앞에 위치합니다.

4.  자동 모드에서는 3초 카운트다운 후 자동으로 3장의 사진이 촬영되어 중간 사진이 저장됩니다.

### 6.2. 실시간 얼굴 인식

방문자 등록을 완료한 후, 아래 명령어를 실행하여 실시간 인식을 시작합니다.

#### GUI 버전

```bash
python face_recognition_yolo.py
```

- 웹캠이 켜지고 얼굴 인식이 시작됩니다.
- 등록된 방문자가 인식되면 초록색 상자와 함께 이름, 목적지, 신뢰도(%)가 표시됩니다.
- 미등록 방문자는 "Unknown"으로 표시됩니다.
- `q` 키를 누르면 프로그램이 종료됩니다.

#### No-GUI 버전

```bash
python face_recognition_yolo_no_gui.py
```

- 메뉴에서 옵션을 선택합니다:
  - **옵션 1**: 단일 이미지 처리
  - **옵션 2**: 웹캠 배치 모드 (지정된 시간 동안 자동 인식)
  - **옵션 3**: 종료

- 웹캠 배치 모드를 선택하면:
  - 인식 지속 시간(초)과 캡처 간격(초)을 설정합니다.
  - 설정된 간격마다 자동으로 얼굴을 인식하고 결과를 `recognition_output/` 폴더에 저장합니다.
  - 각 세션이 끝나면 JSON 형식의 로그 파일이 생성됩니다.

## 7. 주요 파일 설명

### 7.1. 얼굴 등록 스크립트

- **[register_face.py](register_face.py)**: 웹캠을 통해 사용자의 이름, 목적지, 얼굴 사진을 등록하는 GUI 버전. OpenCV 창을 통해 실시간으로 얼굴을 확인하며 SPACE 키로 촬영.

- **[register_face_no_gui.py](register_face_no_gui.py)**: 디스플레이가 없는 환경을 위한 No-GUI 버전. 자동 캡처 모드(카운트다운)와 수동 캡처 모드를 지원하며, 등록된 얼굴 목록 조회 기능 포함.

### 7.2. 얼굴 인식 스크립트

- **[face_recognition_yolo.py](face_recognition_yolo.py)**: 실시간 얼굴 인식 메인 프로그램 (GUI 버전). YOLOv8로 얼굴을 탐지하고 InceptionResnetV1으로 특징을 추출하여 등록된 얼굴과 비교 후 결과를 화면에 실시간 표시.

- **[face_recognition_yolo_no_gui.py](face_recognition_yolo_no_gui.py)**: No-GUI 버전 얼굴 인식 프로그램. 단일 이미지 처리 모드와 웹캠 배치 모드를 지원. 인식 결과를 이미지와 JSON 로그로 저장.

### 7.3. 데이터 및 설정 파일

- **`visitors_info.json`**: 등록된 각 방문자의 ID, 이름, 목적지, 이미지 파일명, 등록 날짜 등 메타데이터를 저장하는 JSON 파일.

- **[requirements.txt](requirements.txt)**: `torch`, `torchvision`, `ultralytics`, `facenet-pytorch`, `opencv-python` 등 프로젝트 실행에 필요한 모든 파이썬 패키지 목록.

### 7.4. 출력 폴더

- **`registered_faces/`**: 등록된 얼굴 이미지가 저장되는 폴더. 각 이미지는 `person_YYYYMMDD_HHMMSS.jpg` 형식으로 저장.

- **`recognition_output/`**: No-GUI 모드에서 인식 결과가 저장되는 폴더. 캡처된 이미지와 세션 로그(JSON)가 저장됨.

## 8. 라이선스 (License)

이 프로젝트는 **AGPL-3.0 라이선스**를 따릅니다. `ultralytics` 라이브러리(AGPL-3.0)를 사용하므로, 이 프로젝트의 소스 코드 또한 동일한 라이선스에 따라 공개되어야 합니다. 자세한 내용은 `LICENSE` 파일을 참고하십시오.

본 프로젝트는 교육 및 연구 목적으로 제작되었으며, 상업적 사용을 권장하지 않습니다.
