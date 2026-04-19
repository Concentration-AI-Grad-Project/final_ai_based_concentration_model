"""
train_model.py
--------------
1) dataset/random_dataset.csv 가 없으면 합성 데이터로 생성
2) RandomForestClassifier 학습 + 평가 지표 출력
3) models/random_clf.pkl 저장

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
CSV_PATH    = os.path.join(DATASET_DIR, "collect_user_1.csv")
PKL_PATH    = os.path.join(MODEL_DIR,   "user_1_clf.pkl")

FEATURE_COLS = [
    "left_ear", "right_ear",
    "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
    "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
    "pitch", "yaw", "roll",
] #총 11개 특징

# ---------------------------------------------------------------------------
# 합성 데이터 생성 (실제 데이터 수집 전 임시 random 페이크 데이터)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    완전 랜덤 합성 데이터 생성 (Cold Start 시뮬레이션)

    특징값과 라벨 간 아무런 상관관계 없음
    → 초기 모델 성능: ~50% (완전 랜덤 수준)
    → 사용자 피드백 수집 후 점진적 개선을 증명하기 위한 baseline
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        rows.append([
            rng.uniform(0.05, 0.40),  # left_ear (완전 랜덤)
            rng.uniform(0.05, 0.40),  # right_ear
            rng.uniform(0.0, 0.5),  # l_pupil_dist
            rng.uniform(-0.3, 0.3),  # l_pupil_dx
            rng.uniform(-0.3, 0.3),  # l_pupil_dy
            rng.uniform(0.0, 0.5),  # r_pupil_dist
            rng.uniform(-0.3, 0.3),  # r_pupil_dx
            rng.uniform(-0.3, 0.3),  # r_pupil_dy
            rng.uniform(-45, 45),  # pitch
            rng.uniform(-45, 45),  # yaw
            rng.uniform(-30, 30),  # roll
            rng.integers(0, 2)  # label (0 or 1 완전 랜덤!)
        ])

    cols = FEATURE_COLS + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
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
        #raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")
        print("[train] CSV 없음 → 합성 데이터 생성 (n=2000)")
        df = generate_synthetic_dataset()
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
