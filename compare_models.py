"""
compare_models.py
-----------------
초기 모델 vs 재학습 모델 성능 비교

사용법:
    python compare_models.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 경로
INITIAL_MODEL = os.path.join(BASE_DIR, "models", "initial_model.pkl")  # 초기 모델 (백업본)
RETRAINED_MODEL = os.path.join(BASE_DIR, "models", "model.pkl")  # 재학습 모델

# 테스트 데이터 경로
TEST_CSV = os.path.join(BASE_DIR, "dataset", "test_set.csv")  # 초기 학습 시 저장한 테스트셋

FEATURE_COLS = [
    "left_ear", "right_ear",
    "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
    "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
    "pitch", "yaw", "roll",
]


def load_test_data():
    """공통 테스트 데이터 로드"""
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] 테스트 데이터가 없습니다: {TEST_CSV}")
        print("\n해결 방법:")
        print("1. train_model.py 실행 시 테스트셋을 저장하거나")
        print("2. 아래 create_test_set() 함수로 새로 생성하세요")
        return None, None

    df = pd.read_csv(TEST_CSV)
    X_test = df[FEATURE_COLS].values
    y_test = df["label"].values

    print(f"\n[테스트 데이터]")
    print(f"  샘플 수: {len(df)}")
    print(f"  집중: {(y_test == 1).sum()}  |  비집중: {(y_test == 0).sum()}")

    return X_test, y_test


def evaluate_model(model_path, X_test, y_test):
    """모델 평가"""
    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 파일이 없습니다: {model_path}")
        return None

    # 모델 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 예측
    y_pred = model.predict(X_test)

    # 지표 계산
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["비집중(0)", "집중(1)"],
            output_dict=True
        )
    }

    return results


def print_comparison_table(initial_results, retrained_results):
    """비교 테이블 출력"""

    print("\n" + "=" * 80)
    print(" " * 20 + "모델 성능 비교 (Performance Comparison)")
    print("=" * 80)

    # 메인 지표
    print("\n【 주요 지표 (Main Metrics) 】")
    print("-" * 80)
    print(f"{'지표':<20} {'초기 모델':<20} {'재학습 모델':<20} {'향상률':<20}")
    print("-" * 80)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1")
    ]

    for name, key in metrics:
        initial_val = initial_results[key]
        retrained_val = retrained_results[key]
        improvement = ((retrained_val - initial_val) / initial_val) * 100

        print(f"{name:<20} {initial_val:>6.2%}{' ' * 14} {retrained_val:>6.2%}{' ' * 14} {improvement:>+6.2f}%")

    print("-" * 80)

    # 클래스별 성능
    print("\n【 클래스별 성능 (Per-Class Performance) 】")
    print("-" * 80)
    print(f"{'클래스':<15} {'지표':<15} {'초기 모델':<15} {'재학습 모델':<15} {'차이':<15}")
    print("-" * 80)

    classes = ["비집중(0)", "집중(1)"]
    for cls in classes:
        for metric in ["precision", "recall", "f1-score"]:
            initial_val = initial_results["classification_report"][cls][metric]
            retrained_val = retrained_results["classification_report"][cls][metric]
            diff = retrained_val - initial_val

            print(f"{cls:<15} {metric:<15} {initial_val:>6.2%}{' ' * 9} {retrained_val:>6.2%}{' ' * 9} {diff:>+6.2%}")
        print("-" * 80)

    # Confusion Matrix
    print("\n【 혼동 행렬 비교 (Confusion Matrix) 】")
    print("-" * 80)

    print("\n초기 모델:")
    cm_i = initial_results["confusion_matrix"]
    print(f"              예측: 비집중  예측: 집중")
    print(
        f"  실제: 비집중    {cm_i[0, 0]:4d}      {cm_i[0, 1]:4d}     (정확도: {cm_i[0, 0] / (cm_i[0, 0] + cm_i[0, 1]) * 100:.1f}%)")
    print(
        f"  실제: 집중      {cm_i[1, 0]:4d}      {cm_i[1, 1]:4d}     (정확도: {cm_i[1, 1] / (cm_i[1, 0] + cm_i[1, 1]) * 100:.1f}%)")

    print("\n재학습 모델:")
    cm_r = retrained_results["confusion_matrix"]
    print(f"              예측: 비집중  예측: 집중")
    print(
        f"  실제: 비집중    {cm_r[0, 0]:4d}      {cm_r[0, 1]:4d}     (정확도: {cm_r[0, 0] / (cm_r[0, 0] + cm_r[0, 1]) * 100:.1f}%)")
    print(
        f"  실제: 집중      {cm_r[1, 0]:4d}      {cm_r[1, 1]:4d}     (정확도: {cm_r[1, 1] / (cm_r[1, 0] + cm_r[1, 1]) * 100:.1f}%)")

    print("\n오분류 개선:")
    fp_diff = cm_r[0, 1] - cm_i[0, 1]  # False Positive 변화
    fn_diff = cm_r[1, 0] - cm_i[1, 0]  # False Negative 변화
    print(f"  False Positive (비집중→집중 오류): {cm_i[0, 1]} → {cm_r[0, 1]} ({fp_diff:+d})")
    print(f"  False Negative (집중→비집중 오류): {cm_i[1, 0]} → {cm_r[1, 0]} ({fn_diff:+d})")

    print("=" * 80 + "\n")


def save_comparison_csv(initial_results, retrained_results, save_path="comparison_results.csv"):
    """비교 결과 CSV 저장"""

    data = {
        "지표": [],
        "초기_모델": [],
        "재학습_모델": [],
        "향상률(%)": []
    }

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1")
    ]

    for name, key in metrics:
        initial_val = initial_results[key]
        retrained_val = retrained_results[key]
        improvement = ((retrained_val - initial_val) / initial_val) * 100

        data["지표"].append(name)
        data["초기_모델"].append(f"{initial_val:.4f}")
        data["재학습_모델"].append(f"{retrained_val:.4f}")
        data["향상률(%)"].append(f"{improvement:+.2f}")

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[저장] 비교 결과 CSV: {save_path}")


def create_test_set_from_multi_user():
    """
    기존 collect_user_*.csv에서 테스트셋 생성
    (초기 학습 시 저장 안 했을 경우)
    """
    from train_model import merge_user_csvs, DATASET_DIR, FEATURE_COLS
    from sklearn.model_selection import train_test_split

    print("\n[테스트셋 생성] collect_user_*.csv에서 추출 중...")

    df = merge_user_csvs(DATASET_DIR)
    if df is None:
        print("[ERROR] CSV 파일이 없습니다.")
        return

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # 80/20 분할 (train_model.py와 동일한 random_state)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 테스트셋 저장
    test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    test_df["label"] = y_test
    test_df.to_csv(TEST_CSV, index=False, encoding='utf-8')

    print(f"[저장] 테스트셋 생성 완료: {TEST_CSV}")
    print(f"  샘플 수: {len(test_df)}")


def main():
    """메인 실행"""

    print("\n" + "=" * 80)
    print(" " * 25 + "모델 성능 비교 스크립트")
    print("=" * 80)

    # 테스트 데이터 로드
    X_test, y_test = load_test_data()

    if X_test is None:
        # 테스트셋이 없으면 생성
        raise FileNotFoundError("테스트 셋 파일이 없음")

    # 모델 평가
    print("\n[평가] 초기 모델...")
    initial_results = evaluate_model(INITIAL_MODEL, X_test, y_test)

    if initial_results is None:
        raise FileNotFoundError("초기 모델이 없음")

    print("[평가] 재학습 모델...")
    retrained_results = evaluate_model(RETRAINED_MODEL, X_test, y_test)

    if retrained_results is None:
        print("\n재학습 모델을 찾을 수 없습니다!")
        return

    # 비교 테이블 출력
    print_comparison_table(initial_results, retrained_results)

    # CSV 저장
    save_comparison_csv(initial_results, retrained_results, "comparison_results.csv")


if __name__ == "__main__":
    main()
