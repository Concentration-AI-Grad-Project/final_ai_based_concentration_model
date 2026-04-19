# 학습자의 집중도 기반 학습 지원 시스템

MediaPipe 얼굴 랜드마크 + RandomForest ML 모델로 학습자의 집중도를 실시간 측정하는 웹 시스템.

---

## 프로젝트 구조

```
concentration_project/
├── app.py                   # Flask 서버 (라우팅, SocketIO, DB)
├── concentration_model.py   # Feature 추출 + ML 추론 모듈
├── train_model.py           # RandomForest 학습 스크립트
├── requirements.txt
├── face_landmarker.task     # MediaPipe 모델 파일 (별도 다운로드)
├── models/
│   └── concentration_clf.pkl   # 학습 후 생성
├── dataset/
│   └── concentration_dataset.csv  # 학습 데이터 (학습 후 생성)
├── uploads/                 # 강의 영상
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── student.html
    ├── professor.html
    ├── watch.html
    └── upload.html
```

---

## 설치 방법

### 1. Python 3.11 또는 3.12 사용 (3.14 불가)

```bash
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. face_landmarker.task 다운로드

```bash
# 프로젝트 루트에 저장
curl -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 모델 학습 방법

### A. 합성 데이터로 빠른 프로토타입 학습 (즉시 실행 가능)

```bash
python train_model.py
```

합성 샘플 2000개로 RandomForest를 학습하고 `models/concentration_clf.pkl`을 생성합니다.

### B. 실제 데이터 수집 후 학습 (권장)

1. 서버 실행 후 강의 시청 페이지의 **"학습 데이터 수집"** 패널 사용
   - 집중 중일 때 ✅ 집중 버튼 클릭
   - 딴짓/졸음 상태일 때 ❌ 비집중 버튼 클릭
   - `dataset/concentration_dataset.csv`에 자동 저장됨

2. 충분한 샘플 수집 후 (권장: 각 클래스 200개 이상):

```bash
python train_model.py --csv dataset/random_dataset.csv
```

3. 서버에서 모델 갱신 (재시작 불필요):
   - 교수자 계정으로 로그인 → 브라우저 콘솔에서:
   ```javascript
   fetch('/api/reload_model', {method:'POST'})
   ```

### 학습 결과 예시 (합성 데이터 기준)

```
5-Fold CV F1: [0.97 0.96 0.97 0.97 0.96]  평균: 0.9680

테스트셋 평가 결과
===================================================
  Accuracy : 0.9725
  Precision: 0.9701
  Recall   : 0.9750
  F1-score : 0.9725

  Feature Importance:
  pitch              0.1823  ████████████████████████████████████████
  yaw                0.1612  ████████████████████████████████████
  l_pupil_dist       0.1401  ████████████████████████████████
  r_pupil_dist       0.1389  ███████████████████████████████
  left_ear           0.0921  █████████████████████
  right_ear          0.0887  ████████████████████
  ...
```

---

## 서버 실행 방법

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속.

### 기본 계정

| 역할    | 아이디   | 비밀번호 |
|---------|----------|----------|
| 교수자  | prof1    | pass     |
| 학생    | student1 | pass     |

---

## 사용 흐름

### 교수자
1. 로그인 → 강의 업로드 (mp4 등 동영상 파일)
2. 대시보드에서 강의별 집중도 분석 버튼 클릭
3. 타임스탬프별 평균 집중도 그래프 확인
4. CSV 다운로드로 원시 데이터 수출

### 학생
1. 로그인 → 강의 목록 → 시청하기
2. 카메라 켜기 클릭 → 웹캠 활성화
3. 영상 시청 중 자동으로 2초 간격 집중도 측정
4. 우측 패널에서 실시간 점수 및 추이 그래프 확인
5. (선택) 데이터 수집 패널로 ML 학습 데이터 기여

---

## ML 파이프라인 상세

### Feature 11개

| Feature       | 설명 |
|---------------|------|
| left_ear      | 왼쪽 Eye Aspect Ratio (눈 깜빡임/졸음 감지) |
| right_ear     | 오른쪽 Eye Aspect Ratio |
| l_pupil_dist  | 왼쪽 동공의 눈 중심 이탈 거리 (정규화) |
| l_pupil_dx    | 왼쪽 동공 수평 이탈량 |
| l_pupil_dy    | 왼쪽 동공 수직 이탈량 |
| r_pupil_dist  | 오른쪽 동공 이탈 거리 |
| r_pupil_dx    | 오른쪽 동공 수평 이탈량 |
| r_pupil_dy    | 오른쪽 동공 수직 이탈량 |
| pitch         | 고개 상하 각도 (°) |
| yaw           | 고개 좌우 각도 (°) |
| roll          | 고개 기울기 각도 (°) |

### 모델 구성
- **Pipeline**: StandardScaler → RandomForestClassifier(n_estimators=200)
- **학습/평가**: train_test_split(test=20%) + 5-Fold Cross Validation
- **불균형 처리**: class_weight="balanced"
- **출력**: predict_proba()[1] → 집중도 점수 0.0~1.0
- **스무딩**: 최근 5프레임 이동 평균으로 노이즈 완화
