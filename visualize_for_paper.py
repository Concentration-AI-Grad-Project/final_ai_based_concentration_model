"""
visualize_for_paper.py
----------------------
논문용 개별 시각화 함수 모음

그래프 제목과 라벨은 파일 최상단에서 한 번에 수정 가능
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ===========================================================================
# 📊 그래프 제목 설정 (여기서 한 번에 수정!)
# ===========================================================================
TITLE_ACCURACY = "모델 정확도 변화(랜덤데이터)"
TITLE_DATA_COLLECTION = "데이터 수집 추이(랜덤데이터)"
TITLE_FEATURE_IMPORTANCE = "Feature Importance(랜덤데이터)"
TITLE_CONFUSION_MATRIX = "Confusion Matrix(랜덤데이터)"
TITLE_METRICS_BAR = "모델 성능 지표(랜덤데이터)"
TITLE_USER_CONTRIBUTION = "사용자별 데이터 기여도(랜덤데이터)"
TITLE_DATA_DISTRIBUTION = "데이터 분포 (PCA)(랜덤데이터)"
TITLE_CLASS_BALANCE = "클래스 균형(랜덤데이터)"

# 축 라벨
XLABEL_DAYS = "경과 일수 (days)"
XLABEL_USER = "사용자"
XLABEL_IMPORTANCE = "중요도"
XLABEL_PREDICTED = "예측 라벨"

YLABEL_ACCURACY = "정확도 (Accuracy)"
YLABEL_DATA_COUNT = "누적 데이터 개수"
YLABEL_ACTUAL = "실제 라벨"
YLABEL_SCORE = "점수"
YLABEL_SAMPLE_COUNT = "샘플 수"

# 클래스 라벨
CLASS_LABELS = ['비집중', '집중']

# 색상
COLOR_PRIMARY = '#8bc34a'
COLOR_ACCENT = '#aed581'
COLOR_BLUE = '#42a5f5'
COLOR_ORANGE = '#ffa726'
COLOR_RED = '#ef5350'

# ===========================================================================
# 설정
# ===========================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. 정확도 시계열 그래프
# ===========================================================================
def plot_accuracy_over_time(days, accuracy, save_path=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(days, accuracy, marker='o', linewidth=3, 
            markersize=10, color=COLOR_PRIMARY, label='Accuracy')
    ax.fill_between(days, accuracy, alpha=0.2, color=COLOR_PRIMARY)
    ax.axhline(y=0.50, color=COLOR_RED, linestyle='--', 
               linewidth=2, label='랜덤 Baseline (50%)', alpha=0.7)
    
    ax.set_xlabel(XLABEL_DAYS, fontsize=13, fontweight='bold')
    ax.set_ylabel(YLABEL_ACCURACY, fontsize=13, fontweight='bold')
    ax.set_title(TITLE_ACCURACY, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, max(accuracy) + 0.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 2. 데이터 수집량 그래프
# ===========================================================================
def plot_data_collection(days, data_count, save_path=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(days, data_count, marker='s', linewidth=3,
            markersize=10, color=COLOR_BLUE)
    ax.fill_between(days, data_count, alpha=0.2, color=COLOR_BLUE)
    ax.text(days[-1], data_count[-1] + max(data_count)*0.03, 
            f'{data_count[-1]}개', fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlabel(XLABEL_DAYS, fontsize=13, fontweight='bold')
    ax.set_ylabel(YLABEL_DATA_COUNT, fontsize=13, fontweight='bold')
    ax.set_title(TITLE_DATA_COLLECTION, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 3. Feature Importance
# ===========================================================================
def plot_feature_importance(features, importances, save_path=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    
    sorted_idx = np.argsort(importances)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    colors = [COLOR_PRIMARY if imp > np.median(importances) else COLOR_ACCENT
              for imp in sorted_importances]
    
    bars = ax.barh(sorted_features, sorted_importances, color=colors, alpha=0.9)
    
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel(XLABEL_IMPORTANCE, fontsize=13, fontweight='bold')
    ax.set_title(TITLE_FEATURE_IMPORTANCE, fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 4. Confusion Matrix
# ===========================================================================
def plot_confusion_matrix(cm, save_path=None, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                cbar_kws={'label': '샘플 수'},
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title(TITLE_CONFUSION_MATRIX, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel(YLABEL_ACTUAL, fontsize=13, fontweight='bold')
    ax.set_xlabel(XLABEL_PREDICTED, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 5. 성능 지표 막대 그래프
# ===========================================================================
def plot_metrics_bar(metrics_dict, save_path=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, color=COLOR_PRIMARY, alpha=0.8, width=0.6)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(YLABEL_SCORE, fontsize=13, fontweight='bold')
    ax.set_title(TITLE_METRICS_BAR, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 6. 사용자별 데이터 기여도
# ===========================================================================
def plot_user_contribution(users, focused_counts, unfocused_counts, 
                          save_path=None, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(users))
    width = 0.7
    
    p1 = ax.bar(x, focused_counts, width, label='집중', 
                color=COLOR_PRIMARY, alpha=0.9)
    p2 = ax.bar(x, unfocused_counts, width, bottom=focused_counts,
                label='비집중', color=COLOR_ORANGE, alpha=0.9)
    
    for i, (f, u) in enumerate(zip(focused_counts, unfocused_counts)):
        total = f + u
        ax.text(i, total + max(focused_counts + unfocused_counts)*0.02, 
                f'{total}개', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel(XLABEL_USER, fontsize=13, fontweight='bold')
    ax.set_ylabel(YLABEL_SAMPLE_COUNT, fontsize=13, fontweight='bold')
    ax.set_title(TITLE_USER_CONTRIBUTION, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(users, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 7. 데이터 분포 (PCA)
# ===========================================================================
def plot_data_distribution_pca(data, labels, save_path=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    ax.scatter(data_pca[labels==0, 0], data_pca[labels==0, 1],
               c=COLOR_RED, alpha=0.6, s=50, label=CLASS_LABELS[0])
    ax.scatter(data_pca[labels==1, 0], data_pca[labels==1, 1],
               c=COLOR_PRIMARY, alpha=0.6, s=50, label=CLASS_LABELS[1])
    
    ax.set_xlabel('PC1', fontsize=13, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=13, fontweight='bold')
    ax.set_title(TITLE_DATA_DISTRIBUTION, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 8. 클래스 균형 파이 차트
# ===========================================================================
def plot_class_balance(focused_count, unfocused_count, save_path=None, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    
    sizes = [unfocused_count, focused_count]
    colors = [COLOR_ORANGE, COLOR_PRIMARY]
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=CLASS_LABELS,
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 13, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax.set_title(TITLE_CLASS_BALANCE, fontsize=15, fontweight='bold', pad=20)
    ax.legend([f'{CLASS_LABELS[0]} ({unfocused_count}개)',
               f'{CLASS_LABELS[1]} ({focused_count}개)'],
              loc='upper right', fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}")
    else:
        plt.show()
    plt.close()


# ===========================================================================
# 사용 예시
# ===========================================================================
if __name__ == "__main__":
    print("\n논문용 시각화 함수 테스트\n")
    
    # 1. 정확도 그래프
    plot_accuracy_over_time([0, 7, 14, 21], [0.51, 0.68, 0.78, 0.83], "accuracy.png")
    
    # 2. 데이터 수집
    plot_data_collection([0, 7, 14, 21], [0, 150, 350, 550], "data_collection.png")
    
    # 3. Feature Importance
    plot_feature_importance(['left_ear', 'right_ear', 'pitch', 'yaw'], 
                           [0.15, 0.14, 0.18, 0.12], "feature_importance.png")
    
    # 4. Confusion Matrix
    plot_confusion_matrix(np.array([[165, 35], [33, 167]]), "confusion_matrix.png")
    
    # 5. 성능 지표
    plot_metrics_bar({'Accuracy': 0.83, 'Precision': 0.81, 'Recall': 0.85}, "metrics.png")
    
    # 6. 사용자 기여도
    plot_user_contribution(['사용자1', '사용자2', '사용자3'], [30, 35, 28], [20, 15, 22], "users.png")
    
    # 7. 클래스 균형
    plot_class_balance(280, 220, "class_balance.png")
    
    print(f"\n✅ figures/ 폴더에 그래프 생성 완료!\n")
