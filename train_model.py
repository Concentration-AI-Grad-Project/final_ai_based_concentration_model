"""
train_model.py
--------------
1) dataset/collect_user_*.csv 파일들을 자동으로 합침
2) RandomForestClassifier 학습 + 평가 지표 출력
3) models/4_users_clf.pkl 저장

실행:
    python train_model.py                       # collect_user_*.csv 모두 합치기
    python train_model.py --csv path/to/file    # 특정 CSV 파일만 사용
"""

import os
import glob
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
PKL_PATH    = os.path.join(MODEL_DIR,   "4_users_clf.pkl")

FEATURE_COLS = [
    "left_ear", "right_ear",
    "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
    "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
    "pitch", "yaw", "roll",
] #총 11개 특징

# ---------------------------------------------------------------------------
# 다중 사용자 CSV 합치기
# ---------------------------------------------------------------------------
def merge_user_csvs(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """
    dataset/collect_user_*.csv 파일들을 모두 찾아서 합침

    Returns:
        pd.DataFrame: 합쳐진 데이터프레임
    """
    pattern = os.path.join(dataset_dir, "collect_user_*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"[ERROR] {pattern} 패턴의 파일을 찾을 수 없습니다!")
        return None

    print(f"\n{'='*60}")
    print(f"  발견된 사용자 CSV 파일: {len(csv_files)}개")
    print(f"{'='*60}")

    dfs = []
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)

        # 필수 컬럼 체크
        missing_cols = [col for col in FEATURE_COLS + ["label"] if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️  {filename}: 필수 컬럼 누락 {missing_cols} - 스킵!")
            continue

        # label 값 확인
        if "label" not in df.columns or df["label"].isnull().all():
            print(f"  ⚠️  {filename}: label 컬럼이 없거나 모두 비어있음 - 스킵!")
            continue

        concentrated = (df["label"] == 1).sum()
        distracted = (df["label"] == 0).sum()

        print(f"  ✓  {filename}: {len(df)}개 (집중: {concentrated}, 비집중: {distracted})")
        dfs.append(df)

    if not dfs:
        print("\n  ❌ 유효한 CSV 파일이 없습니다!")
        return None

    # 합치기
    merged_df = pd.concat(dfs, ignore_index=True)

    # 셔플
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  ✅ 총 {len(merged_df)}개 샘플 병합 완료!")
    print(f"  집중: {(merged_df['label']==1).sum()}  |  비집중: {(merged_df['label']==0).sum()}")
    print(f"{'='*60}\n")

    return merged_df

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
# 랜덤 데이터
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

def train(csv_path: str = None, use_multi_user: bool = True):
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── 데이터 로드 또는 생성 ────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        # 특정 CSV 파일 지정된 경우
        print(f"[train] CSV 로드: {csv_path}")
        df = pd.read_csv(csv_path)
    elif use_multi_user:
        # 다중 사용자 CSV 자동 병합 (기본값)
        print("[train] 다중 사용자 CSV 병합 모드")
        df = merge_user_csvs(DATASET_DIR)
        if df is None:
            print("[ERROR] 유효한 CSV 파일이 없습니다. 학습을 중단합니다.")
            return None
    else:
        #error
        raise FileNotFoundError(f"no file {csv_path}")
        # 합성 데이터 생성
        print("[train] CSV 없음 → 랜덤 데이터 생성 (n=2000)")
        df = generate_random_dataset(n_samples=2000)
        df.to_csv(CSV_PATH, index=False)
        print(f"[train] 랜덤 데이터 저장: {CSV_PATH}")

    print(f"\n[train] 샘플 수: {len(df)}  |  집중: {df['label'].sum()}  비집중: {(df['label']==0).sum()}")

    # NaN 체크
    if df[FEATURE_COLS + ["label"]].isnull().any().any():
        print("[WARNING] 데이터에 NaN 값이 있습니다. 해당 행을 제거합니다.")
        df = df.dropna(subset=FEATURE_COLS + ["label"])
        print(f"[train] 정리 후 샘플 수: {len(df)}")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── Train / Test 분리 (80% / 20%) ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] 학습: {len(X_train)} (80%)  |  테스트: {len(X_test)} (20%)")

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
    print("\n[train] 5-Fold Cross Validation 진행 중...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"[train] 5-Fold CV F1 점수: {cv_scores.round(4)}  평균: {cv_scores.mean():.4f}")

    # ── 최종 학습 ─────────────────────────────────────────────────────────
    print("\n[train] 최종 모델 학습 중...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ── 평가 지표 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  테스트셋 평가 결과 (Test Set Performance)")
    print("="*60)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-score : {f1_score(y_test, y_pred):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["비집중(0)", "집중(1)"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              예측: 비집중  예측: 집중")
    print(f"  실제: 비집중    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  실제: 집중      {cm[1,0]:4d}      {cm[1,1]:4d}")
    print("="*60)

    # ── Feature Importance ────────────────────────────────────────────────
    importances = pipeline.named_steps["clf"].feature_importances_
    print("\n  Feature Importance:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:<18} {imp:6.2%}  {bar}")

    # ── 모델 저장 ─────────────────────────────────────────────────────────
    with open(PKL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[train] 모델 저장 완료: {PKL_PATH}")
    print("="*60 + "\n")

    return pipeline


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(csv_path=None, use_multi_user=True)