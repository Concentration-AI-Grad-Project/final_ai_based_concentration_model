"""
visualize_model_performance.py
------------------------------
분류 모델의 성능을 다양한 방법으로 시각화

사용법:
    train_model.py 끝부분에서 호출:
    visualize_performance(pipeline, X_test, y_test, y_pred)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def visualize_performance(pipeline, X_test, y_test, y_pred, save_dir='results'):
    """
    분류 모델 성능을 5가지 방법으로 시각화

    Args:
        pipeline: 학습된 scikit-learn Pipeline
        X_test: 테스트 특징 데이터
        y_test: 테스트 실제 라벨
        y_pred: 테스트 예측 라벨
        save_dir: 이미지 저장 디렉토리
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 예측 확률 가져오기
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix Heatmap
    plot_confusion_matrix(y_test, y_pred, save_dir)

    # 2. ROC Curve
    plot_roc_curve(y_test, y_proba, save_dir)

    # 3. Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_proba, save_dir)

    print(f"\n[시각화] 모든 그래프가 '{save_dir}/' 폴더에 저장되었습니다!")


def plot_confusion_matrix(y_test, y_pred, save_dir):
    """1. Confusion Matrix Heatmap"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['비집중', '집중'],
                yticklabels=['비집중', '집중'],
                cbar_kws={'label': '샘플 수'})

    # 정확도 표시
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, weight='bold')
    plt.ylabel('실제 라벨', fontsize=12)
    plt.xlabel('예측 라벨', fontsize=12)

    # 각 셀에 비율 추가
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/1_confusion_matrix.png', dpi=300)
    plt.close()
    print(f"  ✓ Confusion Matrix 저장: {save_dir}/1_confusion_matrix.png")

def plot_roc_curve(y_test, y_proba, save_dir):
    """3. ROC Curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))

    # ROC Curve
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'모델 (AUC = {roc_auc:.3f})')

    # Random Baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='랜덤 (AUC = 0.500)')

    # 현재 threshold (0.5) 지점 표시
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    plt.scatter([fpr[idx_05]], [tpr[idx_05]], s=200, c='red',
                marker='o', edgecolors='black', linewidths=2,
                label=f'Threshold 0.5 (TPR={tpr[idx_05]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_roc_curve.png', dpi=300)
    plt.close()
    print(f"  ✓ ROC Curve 저장: {save_dir}/2_roc_curve.png")


def plot_precision_recall_curve(y_test, y_proba, save_dir):
    """4. Precision-Recall Curve"""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Average Precision
    from sklearn.metrics import average_precision_score
    ap_score = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=3,
             label=f'모델 (AP = {ap_score:.3f})')

    # Baseline (비율에 따른 평균 precision)
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Baseline (P = {baseline:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, weight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3_precision_recall_curve.png', dpi=300)
    plt.close()
    print(f"  ✓ Precision-Recall Curve 저장: {save_dir}/3_precision_recall_curve.png")

# 사용 예시
if __name__ == "__main__":
    print("이 파일은 train_model.py에서 import하여 사용합니다.")
    print("\n사용법:")
    print("  from visualize_model_performance import visualize_performance")
    print("  visualize_performance(pipeline, X_test, y_test, y_pred)")