"""
train_model.py
--------------
1) dataset/concentration_dataset.csv 가 없으면 합성 데이터로 생성
2) RandomForestClassifier 학습 + 평가 지표 출력
3) models/concentration_clf.pkl 저장

실행:
    python train_model.py                       # 합성 데이터로 학습
    python train_model.py --csv path/to/file    # 직접 수집한 CSV 사용
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
CSV_PATH    = os.path.join(DATASET_DIR, "concentration_dataset.csv")
PKL_PATH    = os.path.join(MODEL_DIR,   "concentration_clf.pkl")

FEATURE_COLS = [
    "left_ear", "right_ear",
    "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
    "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
    "pitch", "yaw", "roll",
] #총 11개 특징

# ---------------------------------------------------------------------------
# 합성 데이터 생성 (실제 데이터 수집 전 임시 페이크 데이터)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    집중(label=1)과 비집중(label=0) 샘플을 각각 절반씩 생성.

    집중 상태:
      - Eye Aspect Ratio 정상 범위 (눈 뜸): 0.25~0.35
      - 동공 중앙 근처pupil: dist 0~0.15
      - 고개 정면: pitch/yaw ±10°
    비집중 상태:
      - Eye Aspect Ratio 낮거나(졸음) 또는 정상
      - 동공 치우침 pupil: dist 0.20~0.45
      - 고개 기울어짐: pitch/yaw ±25°
    """
    rng = np.random.default_rng(seed)
    rows = []

    half = n_samples // 2

    # ── 집중 (페이크)샘플 ──────────────────────────────────────────────────────────
    for _ in range(half):
        eye_aspect_ratio = rng.uniform(0.24, 0.36) 
        pupil_dist = rng.uniform(0.00, 0.18)
        pupil_dx = rng.uniform(-0.10, 0.10)
        pupil_dy = rng.uniform(-0.10, 0.10)
        pitch = rng.uniform(-12, 12)
        yaw = rng.uniform(-12, 12)
        roll = rng.uniform(-8, 8)
        rows.append([
            eye_aspect_ratio + rng.normal(0, 0.005),   # left_ear
            eye_aspect_ratio + rng.normal(0, 0.005),   # right_ear
            pupil_dist, pupil_dx, pupil_dy,           # left pupil
            pupil_dist + rng.uniform(-0.03, 0.03),
            pupil_dx   + rng.uniform(-0.03, 0.03),
            pupil_dy   + rng.uniform(-0.03, 0.03),
            pitch, yaw, roll, 1,
        ])

    # ── 비집중 샘플 ────────────────────────────────────────────────────────
    for _ in range(n_samples - half):
        # 비집중 유형을 3가지로 분류
        t = rng.integers(0, 3)
        if t == 0:   # 졸음 (Eye Aspect Ratio 낮음)
            eye_aspect_ratio = rng.uniform(0.08, 0.20)
            pupil_dist = rng.uniform(0.00, 0.25)
            pitch = rng.uniform(-15, 30)    # 고개 숙임
            yaw = rng.uniform(-10, 10)
        elif t == 1: # 딴짓 (동공 치우침 + 고개 회전)
            eye_aspect_ratio = rng.uniform(0.22, 0.34)
            pupil_dist = rng.uniform(0.25, 0.50)
            pitch = rng.uniform(-20, 20)
            yaw = rng.uniform(20,  45)
        else:        # 고개 아래 (필기 등)
            eye_aspect_ratio = rng.uniform(0.22, 0.34)
            pupil_dist = rng.uniform(0.10, 0.35)
            pitch = rng.uniform(20,  40)
            yaw = rng.uniform(-15, 15)

        pupil_dx = rng.uniform(-pupil_dist, pupil_dist)
        pupil_dy = rng.uniform(-pupil_dist, pupil_dist)
        roll = rng.uniform(-15, 15)
        rows.append([
            eye_aspect_ratio + rng.normal(0, 0.005),
            eye_aspect_ratio + rng.normal(0, 0.005),
            pupil_dist, pupil_dx, pupil_dy,
            pupil_dist + rng.uniform(-0.03, 0.03),
            pupil_dx   + rng.uniform(-0.03, 0.03),
            pupil_dy   + rng.uniform(-0.03, 0.03),
            pitch, yaw, roll, 0,
        ])

    cols = FEATURE_COLS + ["label"]
    df   = pd.DataFrame(rows, columns=cols)
    df   = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
#랜덤 데이터
# ---------------------------------------------------------------------------
def generate_random_dataset(n_samples=2000):
    rng = np.random.default_rng(42)
    rows = []

    for _ in range(n_samples):
        rows.append([
            rng.uniform(0, 1),  # left_ear
            rng.uniform(0, 1),  # right_ear
            rng.uniform(0, 1),  # l_pupil_dist
            rng.uniform(-1, 1), # l_pupil_dx
            rng.uniform(-1, 1), # l_pupil_dy
            rng.uniform(0, 1),  # r_pupil_dist
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            rng.uniform(-90, 90), # pitch
            rng.uniform(-90, 90), # yaw
            rng.uniform(-90, 90), # roll
            rng.integers(0, 2)   # label (0 or 1 랜덤)
        ])
    cols = FEATURE_COLS + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    return df

# ---------------------------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------------------------

def train(csv_path: str = CSV_PATH):
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── 데이터 로드 또는 합성 생성 ────────────────────────────────────────
    if os.path.exists(csv_path):
        print(f"[train] CSV 로드: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")
        print("[train] CSV 없음 → 합성 데이터 생성 (n=2000)")
        #df = generate_synthetic_dataset()
        df = generate_random_dataset(n_samples=2000)
        df.to_csv(csv_path, index=False)
        print(f"[train] 합성 데이터 저장: {csv_path}")

    print(f"[train] 샘플 수: {len(df)}  |  집중: {df['label'].sum()}  비집중: {(df['label']==0).sum()}")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── Train / Test 분리 ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] 학습: {len(X_train)}  테스트: {len(X_test)}")

    # ── Pipeline: StandardScaler + RandomForest ───────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",   # 클래스 불균형 보정
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # ── 5-Fold Cross Validation ───────────────────────────────────────────
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"\n[train] 5-Fold CV F1(성능): {cv_scores.round(4)}  평균: {cv_scores.mean():.4f}")

    # ── 최종 학습 ─────────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ── 평가 지표 ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  테스트셋 평가 결과")
    print("="*55)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["비집중(0)", "집중(1)"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*55)

    # ── Feature Importance ────────────────────────────────────────────────
    importances = pipeline.named_steps["clf"].feature_importances_
    print("\n  Feature Importance:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<18} {imp:.4f}  {bar}")

    # ── 모델 저장 ─────────────────────────────────────────────────────────
    with open(PKL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[train] 모델 저장 완료: {PKL_PATH}")
    return pipeline


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="집중도 분류 모델 학습")
    parser.add_argument("--csv", default=CSV_PATH, help="학습 데이터 CSV 경로")
    args = parser.parse_args()
    train(csv_path=args.csv)
