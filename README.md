# 점진적 학습을 활용한 온라인 강의 학습자 집중도 기반 강의 지원 시스템

## 프로젝트 개요

- **목적**: 온라인 강의 중 학습자의 집중도를 실시간으로 측정
- **기술**: MediaPipe Face Mesh + RandomForest Classifier
- **특징**: Semi-Supervised Learning 기반 점진적 학습 지원

---

## 주요 기능

### 1. 초기 데이터 수집 Human 개입 Supervised
- 11개의 특징 추출 (EAR, 동공 위치, 머리 각도)
- RandomForest 기반 200개 트리 분류 모델 사용
- 초기 정확도: 85.40%

### 2. Semi-Supervised 점진적 학습
- 확신도 90% 이상의 샘플 자동 수집
- 100개 샘플마다 자동 재학습 수행
- 재학습 후 정확도: **95.33%** (+11.63%p)


---

## 📁 프로젝트 구조

```plaintext
concentration_model/
├── app.py                              # Flask 서버 (메인)
├── concentration_model.py              # 특징 추출 및 예측
├── incremental_learner.py              # 점진적 학습 관리
├── train_model.py                      # 초기 모델 학습
├── compare_models.py                   # 모델 성능 비교
│
├── visualize_model_performance.py      # 성능 시각화 (3개 그래프)
├── dataset/
│   ├── collect_user_1.csv ~ 9.csv      # 초기 학습 데이터 (9명)
│   ├── merged_initial_data.csv         # 병합된 초기 데이터 (자동 생성)
│   ├── auto_labeled_data.csv           # 자동 라벨링 데이터
│   └── test_set.csv                    # 고정 테스트셋
│
├── models/
│   ├── model.pkl                       # 학습된 모델
│   └── model_backup.pkl                # 백업 모델
│
├── results/                            # 초기 모델 시각화 결과

```

---

## 설치 및 실행

### 1. 환경 설정

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python app.py
```

**접속 주소:** `http://localhost:8000`

---

## 성능 비교

### 모델 비교 실행

```bash
python compare_models.py
```

### 결과

| 지표 | 초기 모델 | 재학습 모델 | 향상률 |
|------|----------|------------|--------|
| **Accuracy** | 85.40% | 95.33% | **+11.63%** |
| **Precision** | 85.41% | 95.34% | +11.63% |
| **Recall** | 85.40% | 95.33% | +11.63% |
| **F1-Score** | 85.40% | 95.33% | +11.63% |

### 테스트 조건
- 테스트셋: 1,500개 샘플 (고정)
- 재학습 데이터: 실제 사용 2시간 동안 자동 수집
- 확신도 임계값: 90%

---

## 점진적 학습 프로세스

### 1. 자동 수집

확신도 90% 이상인 샘플만 저장


### 2. 자동 재학습

```plaintext
100개 샘플 수집 → 백그라운드 재학습 수행
├─ 초기 데이터 6,000개
├─ 자동 라벨링 데이터 100개
└─ 총 6,100개 데이터로 재학습
```

### 3. 성능 로그

```csv
performance_log.csv:
timestamp, total_samples, accuracy, f1_score
```

---

## 데이터 형식

### 특징 (11개)

```csv
left_ear, right_ear,
l_pupil_dist, l_pupil_dx, l_pupil_dy,
r_pupil_dist, r_pupil_dx, r_pupil_dy,
pitch, yaw, roll, label
```

### 라벨
- `0`: 비집중 (졸음, 딴짓, 고개 숙임)
- `1`: 집중 (정면 응시)

---

## 성능 수치

### 초기 모델
- 학습 데이터: 9명 × 833개 = 7,497개
- Train/Test 비율: 6,000 / 1,500 (80/20)
- 정확도: **85.40%**

### 재학습 모델
- 추가 데이터: 자동 라벨링 데이터 100개 (확신도 ≥ 90%)
- 총 학습 데이터: 6,100개
- 정확도: **95.33%** (+11.63%p)

### 클래스별 개선
- 비집중(0): Recall 85.25% → 94.78% (+9.53%p)
- 집중(1): Recall 85.56% → 95.91% (+10.35%p)

### 오분류 감소
- False Positive: 113 → 40 (-73개, -64.6%)
- False Negative: 106 → 30 (-76개, -71.7%)

---

## 작성자

- Kim Yujeong
- Hyewon Cheon