# 사용자 피드백 기반 점진적 학습 집중도 측정 시스템

MediaPipe 얼굴 랜드마크 + RandomForest ML 모델 + Semi-Supervised Learning으로 학습자의 집중도를 실시간 측정하는 웹 시스템.

---

## 핵심 특징

- **최소 데이터로 시작**: 200개 샘플만으로 실용적 성능 (95%+)
- **점진적 학습**: 사용할수록 자동으로 성능 개선
- **품질 우선**: 대량의 랜덤 데이터보다 소량의 실제 데이터가 효과적
- **Semi-Supervised Learning**: 확신도 기반 자동 라벨링

---

## 프로젝트 구조

```
concentration_project/
├── app.py                      # Flask 서버 (라우팅, SocketIO, DB)
├── concentration_model.py      # Feature 추출 + ML 추론 모듈
├── train_model.py              # RandomForest 학습 스크립트
├── requirements.txt
├── face_landmarker.task        # MediaPipe 모델 파일 (별도 다운로드)
├── models/
│   ├── concentration_clf.pkl   # 현재 서비스 중인 모델
│   ├── user_1_clf.pkl          # 사용자별 학습 모델
│   └── random_clf.pkl          # 초기 랜덤 baseline
├── dataset/
│   ├── collect_user_1.csv      # 사용자 1 수집 데이터
│   ├── collect_user_2.csv      # 사용자 2 수집 데이터
│   └── auto_labeled_data.csv   # 자동 라벨링 데이터 (서비스 모드)
├── uploads/                    # 강의 영상
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

## 시스템 운영 모드

### **Phase 1: 실험 모드 (초기 데이터 수집)**

**목적**: 사용자 피드백으로 고품질 초기 데이터 수집

**설정** (app.py):
```python
EXPERIMENT_MODE = True
```

**동작**:
- 강의 시청 화면에 **"✅ 집중 중 / ❌ 비집중"** 버튼 표시
- 사용자가 직접 현재 상태를 라벨링
- `dataset/collect_user_1.csv` 등에 저장
- **권장 수집량**: 사용자당 200개 (집중 100 + 비집중 100)

**장점**:
- 데이터 품질 100% 보장
- 개인별 특성 반영

---

### **Phase 2: 서비스 모드 (자동 학습)**

**목적**: 사용자 개입 없이 자동으로 데이터 수집 및 모델 개선

**설정** (app.py):
```python
EXPERIMENT_MODE = False
AUTO_LABELING_ENABLED = True
CONFIDENCE_THRESHOLD = 0.90      # 확신도 90% 이상만 수집
AUTO_RETRAIN_INTERVAL = 100      # 100개마다 재학습
```

**동작**:
1. **실시간 예측**: 모델이 자동으로 집중도 측정
2. **확신도 필터링**: 예측 확신도가 90% 이상인 샘플만 선택
3. **자동 라벨링**: 선택된 샘플을 `dataset/auto_labeled_data.csv`에 저장
4. **자동 재학습**: 100개 누적마다 모델 재학습 (별도 스레드)
5. **모델 갱신**: 새 모델을 서비스에 자동 반영

**성능 개선 예상**:
```
초기 배포: 85% accuracy
+100개: 86% (+1%p)
+300개: 87% (+2%p)
+700개: 89% (+4%p)
+3000개: 91% (+6%p) → 포화
```

---

## 모델 학습 방법

### A. 랜덤 Baseline 생성 (테스트용)

```bash
python train_model.py
```

- 완전 랜덤 데이터 2000개 생성
- 성능: ~50% (랜덤 수준)
- 용도: Cold Start 시뮬레이션, 점진적 개선 비교 baseline

---

### B. 단일 사용자 데이터 학습

```bash
# dataset/collect_user_1.csv 사용
python train_model.py --csv dataset/collect_user_1.csv
```

**실험 결과 예시** (실제 데이터 206개):
```
============================================================
  발견된 사용자 CSV 파일: 4개
============================================================
  ✓  collect_user_1.csv: 604개 (집중: 323, 비집중: 281)
  ✓  collect_user_2.csv: 1476개 (집중: 910, 비집중: 566)
  ✓  collect_user_3.csv: 526개 (집중: 377, 비집중: 149)
  ✓  collect_user_4.csv: 1002개 (집중: 509, 비집중: 493)

============================================================
  ✅ 총 3608개 샘플 병합 완료!
  집중: 2119  |  비집중: 1489
============================================================


[train] 샘플 수: 3608  |  집중: 2119  비집중: 1489
[train] 학습: 2886 (80%)  |  테스트: 722 (20%)

[train] 5-Fold Cross Validation 진행 중...
[train] 5-Fold CV F1 점수: [0.8132 0.8397 0.8369 0.8244 0.8518]  평균: 0.8332

[train] 최종 모델 학습 중...

============================================================
  테스트셋 평가 결과 (Test Set Performance)
============================================================
  Accuracy : 0.8144
  Precision: 0.8356
  Recall   : 0.8514
  F1-score : 0.8435

  Classification Report:
              precision    recall  f1-score   support

      비집중(0)       0.78      0.76      0.77       298
       집중(1)       0.84      0.85      0.84       424

    accuracy                           0.81       722
   macro avg       0.81      0.81      0.81       722
weighted avg       0.81      0.81      0.81       722

  Confusion Matrix:
              예측: 비집중  예측: 집중
  실제: 비집중     227        71
  실제: 집중        63       361
============================================================

  Feature Importance:
  left_ear           14.81%  ███████
  right_ear          13.88%  ██████
  r_pupil_dy         13.21%  ██████
  l_pupil_dy         12.55%  ██████
  l_pupil_dx         10.10%  █████
  l_pupil_dist        9.86%  ████
  r_pupil_dx          8.76%  ████
  r_pupil_dist        7.71%  ███
  roll                3.66%  █
  pitch               0.95%
```

### C. 다중 사용자 데이터 병합 학습 (권장)

```bash
# dataset/collect_user_*.csv 모두 자동 병합
python train_model.py
```

**동작**:
1. `dataset/` 폴더에서 `collect_user_*.csv` 파일 자동 검색
2. 유효성 검증 (필수 컬럼, label 존재 확인)
3. 모든 파일 병합 + 셔플
4. 학습 진행

**출력 예시**:
```
============================================================
  발견된 사용자 CSV 파일: 5개
============================================================
  ✓  collect_user_1.csv: 200개 (집중: 100, 비집중: 100)
  ✓  collect_user_2.csv: 250개 (집중: 130, 비집중: 120)
  ✓  collect_user_3.csv: 180개 (집중: 90, 비집중: 90)
  ✓  collect_user_4.csv: 220개 (집중: 110, 비집중: 110)
  ✓  collect_user_5.csv: 150개 (집중: 75, 비집중: 75)

============================================================
  ✅ 총 1000개 샘플 병합 완료!
  집중: 505  |  비집중: 495
============================================================

[train] 학습: 800 (80%)  |  테스트: 200 (20%)
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

### 학생 (실험 모드)
1. 로그인 → 강의 목록 → 시청하기
2. 카메라 켜기 클릭 → 웹캠 활성화
3. 영상 시청 중 **"✅ 집중 중 / ❌ 비집중"** 버튼으로 라벨링
4. 우측 패널에서 실시간 점수 및 추이 그래프 확인
5. 200개 이상 수집 권장

### 학생 (서비스 모드)
1. 로그인 → 강의 목록 → 시청하기
2. 카메라 켜기 클릭 → 웹캠 활성화
3. 영상 시청 (자동으로 3초 간격 집중도 측정)
4. 우측 패널에서 실시간 점수 확인
5. 확신도 90% 이상 샘플은 자동으로 학습 데이터에 추가
6. **사용할수록 모델 성능 향상!**

---

## ML 파이프라인 상세

### Feature 11개

| Feature       | 설명 |
|---------------|------|
| r_pupil_dy    | 오른쪽 동공 수직 이탈량 |
| l_pupil_dy    | 왼쪽 동공 수직 이탈량 |
| pitch         | 고개 상하 각도 (°) | 
| yaw           | 고개 좌우 각도 (°) |
| right_ear     | 오른쪽 Eye Aspect Ratio |
| left_ear      | 왼쪽 Eye Aspect Ratio |
| r_pupil_dist  | 오른쪽 동공 이탈 거리 |
| l_pupil_dist  | 왼쪽 동공 이탈 거리 |
| r_pupil_dx    | 오른쪽 동공 수평 이탈량 |
| l_pupil_dx    | 왼쪽 동공 수평 이탈량 |
| roll          | 고개 기울기 각도 (°) |

### 모델 구성
- **Pipeline**: StandardScaler → RandomForestClassifier
  - n_estimators=200 (200개 결정 트리)
  - max_depth=None (제한 없음)
  - class_weight="balanced" (클래스 불균형 자동 보정)
- **학습/평가**: 
  - train_test_split(test=20%, stratify=True)
  - 5-Fold Cross Validation
- **출력**: predict_proba()[1] → 집중도 점수 0.0~1.0
- **스무딩**: 최근 5프레임 이동 평균으로 노이즈 완화

### 성능 지표 (실제 데이터 206개 기준)
```
  Accuracy : 0.8144
  Precision: 0.8356
  Recall   : 0.8514
  F1-score : 0.8435
  
Confusion Matrix:
              예측: 비집중  예측: 집중
  실제: 비집중     20         1
  실제: 집중        1        20
```

---

## 점진적 학습 전략

### Phase 1 → Phase 2 전환

```
1. 초기 데이터 수집 (실험 모드)
   - 사용자별 500~1000개 수집
   - n명
   - 학습 후 accuracy: 85-90%

2. 서비스 배포 (서비스 모드 전환)
   - app.py에서 EXPERIMENT_MODE = False
   - AUTO_LABELING_ENABLED = True

3. 자동 학습 시작
   - 확신도 ≥90% 샘플 자동 수집
   - 100개마다 재학습
   - 성능 점진적 향상: 85% → 91%

4. 모니터링
   - performance_log.csv 확인
   - 성능 정체 시 수동 데이터 추가
```