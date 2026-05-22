"""
incremental_learner.py
----------------------
Semi-Supervised Learning 기반 점진적 학습 시스템

기능:
1. 확신도 ≥90% 샘플만 자동 라벨링
2. auto_labeled_data.csv에 누적 저장
3. 100개마다 백그라운드 재학습
4. 성능 로그 기록
"""

import os
import csv
import threading
import pickle
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


class IncrementalLearner:
    """
    점진적 학습 관리자
    """

    def __init__(
            self,
            base_dir: str = ".",
            confidence_threshold: float = 0.90,
            retrain_interval: int = 100,
            model_path: str = "models/4_users_clf.pkl"
    ):
        self.base_dir = base_dir
        self.confidence_threshold = confidence_threshold
        self.retrain_interval = retrain_interval
        self.model_path = os.path.join(base_dir, model_path)

        # 자동 라벨링 데이터 저장 경로
        self.auto_csv = os.path.join(base_dir, "dataset", "auto_labeled_data.csv")

        # 성능 로그
        self.performance_log = os.path.join(base_dir, "performance_log.csv")

        # 카운터
        self.sample_count = 0
        self.load_existing_count()

        # 재학습 락 (중복 방지)
        self.retrain_lock = threading.Lock()
        self.is_retraining = False

        print(f"\n{'=' * 60}")
        print(f"[IncrementalLearner] 초기화 완료")
        print(f"  확신도 임계값: {confidence_threshold * 100}%")
        print(f"  재학습 간격: {retrain_interval}개")
        print(f"  현재 누적 샘플: {self.sample_count}개")
        print(f"{'=' * 60}\n")

    def load_existing_count(self):
        """기존 auto_labeled_data.csv의 샘플 수 로드"""
        if os.path.exists(self.auto_csv):
            df = pd.read_csv(self.auto_csv)
            self.sample_count = len(df)
        else:
            self.sample_count = 0

    def record_sample(
            self,
            features: Dict[str, float],
            confidence: float,
            predicted_label: int,
            user: str = "anonymous"
    ) -> Dict:
        """
        확신도 기반 자동 라벨링 & 저장

        Args:
            features: 11개 특징 딕셔너리
            confidence: 예측 확신도 (0.0~1.0)
            predicted_label: 예측된 라벨 (0 or 1)
            user: 사용자 ID

        Returns:
            저장 결과 딕셔너리
        """
        # 확신도 체크
        if confidence < self.confidence_threshold:
            return {
                "saved": False,
                "reason": f"confidence {confidence:.2%} < {self.confidence_threshold:.0%}",
                "total_samples": self.sample_count
            }

        # CSV에 추가
        is_new = not os.path.exists(self.auto_csv)

        with open(self.auto_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 헤더 (최초 생성 시)
            if is_new:
                header = list(features.keys()) + ["label", "confidence", "user", "timestamp"]
                writer.writerow(header)

            # 데이터 행
            row = list(features.values()) + [
                predicted_label,
                round(confidence, 4),
                user,
                datetime.now().isoformat()
            ]
            writer.writerow(row)

        self.sample_count += 1

        # 재학습 트리거
        should_retrain = (self.sample_count % self.retrain_interval == 0)

        result = {
            "saved": True,
            "confidence": confidence,
            "label": predicted_label,
            "total_samples": self.sample_count,
            "will_retrain": should_retrain
        }

        if should_retrain and not self.is_retraining:
            # 백그라운드 재학습 시작
            threading.Thread(
                target=self._retrain_model,
                daemon=True
            ).start()

        return result

    def _retrain_model(self):
        """
        백그라운드 재학습
        """
        if not self.retrain_lock.acquire(blocking=False):
            print("[재학습] 이미 재학습 중입니다. 스킵.")
            return

        self.is_retraining = True

        try:
            print(f"\n{'=' * 60}")
            print(f"[재학습] 시작 - 누적 샘플: {self.sample_count}개")
            print(f"{'=' * 60}")

            # 데이터 로드
            df = pd.read_csv(self.auto_csv)

            feature_cols = [
                "left_ear", "right_ear",
                "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
                "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
                "pitch", "yaw", "roll"
            ]

            X = df[feature_cols].values
            y = df["label"].values

            print(f"[재학습] 데이터: {len(df)}개 (집중: {(y == 1).sum()}, 비집중: {(y == 0).sum()})")

            # Train/Test 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Pipeline 생성
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            # 학습
            pipeline.fit(X_train, y_train)

            # 평가
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"[재학습] 성능 - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # 모델 저장
            with open(self.model_path, 'wb') as f:
                pickle.dump(pipeline, f)

            print(f"[재학습] 모델 저장 완료: {self.model_path}")

            # 성능 로그 기록
            self._log_performance(len(df), accuracy, f1)

            # 모델 리로드 (concentration_model.py)
            from concentration_model import reload_model
            reload_model()
            print(f"[재학습] 모델 리로드 완료!")

            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"[재학습 오류] {e}")

        finally:
            self.is_retraining = False
            self.retrain_lock.release()

    def _log_performance(self, total_samples: int, accuracy: float, f1: float):
        """성능 로그 기록"""
        is_new = not os.path.exists(self.performance_log)

        with open(self.performance_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if is_new:
                writer.writerow([
                    "timestamp", "total_samples", "accuracy", "f1_score"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                total_samples,
                round(accuracy, 4),
                round(f1, 4)
            ])

    def get_stats(self) -> Dict:
        """현재 통계 반환"""
        stats = {
            "total_samples": self.sample_count,
            "next_retrain_at": self.retrain_interval - (self.sample_count % self.retrain_interval),
            "is_retraining": self.is_retraining
        }

        # 최근 성능 로그 (마지막 5개)
        if os.path.exists(self.performance_log):
            df = pd.read_csv(self.performance_log)
            stats["recent_performance"] = df.tail(5).to_dict('records')

        return stats


# 전역 인스턴스 (싱글톤)
_learner = None


def get_learner(
        confidence_threshold: float = 0.90,
        retrain_interval: int = 100
) -> IncrementalLearner:
    """전역 IncrementalLearner 인스턴스 반환"""
    global _learner
    if _learner is None:
        _learner = IncrementalLearner(
            confidence_threshold=confidence_threshold,
            retrain_interval=retrain_interval
        )
    return _learner
