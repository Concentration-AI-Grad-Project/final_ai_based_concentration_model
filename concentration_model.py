"""
concentration_model.py
----------------------
MediaPipe FaceLandmarker 기반 Feature 추출 + ML 모델 추론 모듈.
app.py 에서 import 하여 사용한다.
"""

import os
import math
import pickle
from PIL.ImageQt import rgb
import numpy as np
import mediapipe as mp
import cv2
import random

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
SKL_MODEL  = os.path.join(BASE_DIR, "models", "random_clf.pkl")

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker 초기화 (IMAGE 모드 — 프레임 단위 호출)
# ---------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
# ---------------------------------------------------------------------------
# 랜드마크 인덱스 (MediaPipe 478-point 기준)
# ---------------------------------------------------------------------------
# 왼쪽 눈 윤곽 (EAR 계산용)
L_EYE  = [33, 160, 158, 133, 153, 144]
# 오른쪽 눈 윤곽
R_EYE  = [362, 385, 387, 263, 373, 380]
# iris 중심
L_IRIS = 468
R_IRIS = 473
# 눈 윤곽 전체 (동공 거리 계산용)
L_EYE_FULL = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
R_EYE_FULL = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# ---------------------------------------------------------------------------
# Feature 추출 함수
# ---------------------------------------------------------------------------

def _lm_to_np(landmarks, w: int, h: int) -> np.ndarray:
    """landmark 리스트 → (N, 2) pixel 좌표 배열"""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def _ear(eye_pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio (Soukupová & Čech, 2016).
    eye_pts: shape (6, 2) — [p1, p2, p3, p4, p5, p6] 순서
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    a = np.linalg.norm(eye_pts[1] - eye_pts[5])
    b = np.linalg.norm(eye_pts[2] - eye_pts[4])
    c = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (a + b) / (2.0 * c + 1e-6)


def _pupil_offset(eye_full_pts: np.ndarray, iris_pt: np.ndarray):
    """
    동공이 눈 중심에서 얼마나 벗어났는지 (정규화된 거리).
    반환: (dist_ratio, dx_ratio, dy_ratio)
    """
    center   = np.mean(eye_full_pts, axis=0)
    max_dist = np.max(np.linalg.norm(eye_full_pts - center, axis=1)) + 1e-6
    diff     = iris_pt - center
    dist     = np.linalg.norm(diff)
    return dist / max_dist, diff[0] / max_dist, diff[1] / max_dist


def _rotation_to_euler(R: np.ndarray):
    """4×4 변환 행렬의 3×3 회전 부분 → (pitch, yaw, roll) in degrees"""
    r = R[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( r[2, 1],  r[2, 2]))
        yaw   = math.degrees(math.atan2(-r[2, 0],  sy))
        roll  = math.degrees(math.atan2( r[1, 0],  r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw   = math.degrees(math.atan2(-r[2, 0], sy))
        roll  = 0.0
    return pitch, yaw, roll


def extract_features(frame: np.ndarray):
    """
    RGB numpy array (H, W, 3) → feature dict 또는 None (얼굴 미감지).

    반환 dict 키:
        left_ear, right_ear,
        l_pupil_dist, l_pupil_dx, l_pupil_dy,
        r_pupil_dist, r_pupil_dx, r_pupil_dy,
        pitch, yaw, roll
    총 11개 feature.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face = result.multi_face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark])

    # ── EAR ──────────────────────────────────────────────────────────────
    l_ear = _ear(pts[L_EYE])
    r_ear = _ear(pts[R_EYE])

    # ── 동공 offset ───────────────────────────────────────────────────────
    l_dist, l_dx, l_dy = _pupil_offset(pts[L_EYE_FULL], pts[L_IRIS])
    r_dist, r_dx, r_dy = _pupil_offset(pts[R_EYE_FULL], pts[R_IRIS])

    # ── 고개 각도 ─────────────────────────────────────────────────────────
    pitch, yaw, roll = 0.0, 0.0, 0.0

    return dict(
        left_ear=l_ear,   right_ear=r_ear,
        l_pupil_dist=l_dist, l_pupil_dx=l_dx, l_pupil_dy=l_dy,
        r_pupil_dist=r_dist, r_pupil_dx=r_dx, r_pupil_dy=r_dy,
        pitch=pitch, yaw=yaw, roll=roll,
    )


FEATURE_COLS = [
    "left_ear", "right_ear",
    "l_pupil_dist", "l_pupil_dx", "l_pupil_dy",
    "r_pupil_dist", "r_pupil_dx", "r_pupil_dy",
    "pitch", "yaw", "roll",
]

# ---------------------------------------------------------------------------
# 모델 로드 및 추론
# ---------------------------------------------------------------------------

def _load_model():
    if os.path.exists(SKL_MODEL):
        with open(SKL_MODEL, "rb") as f:
            return pickle.load(f)
    return None


_clf = _load_model()          # 서버 시작 시 1회 로드
_prev_features = None         # 프레임 간 시선 변화 계산용 (옵션)


def reload_model():
    """train_model.py 학습 완료 후 서버 재시작 없이 모델 갱신."""
    global _clf
    _clf = _load_model()


# ---------------------------------------------------------------------------
# 스무딩 버퍼 (시간 기반 집중도 보정)
# ---------------------------------------------------------------------------
_score_buffer = []
SMOOTH_WINDOW = 5          # 최근 N 프레임 이동 평균


def _smooth(score: float) -> float:
    _score_buffer.append(score)
    if len(_score_buffer) > SMOOTH_WINDOW:
        _score_buffer.pop(0)
    return float(np.mean(_score_buffer))


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray) -> float:
    """
    app.py 에서 호출하는 진입점.
    frame : RGB numpy array (H, W, 3)
    return: 집중도 점수 0.0 ~ 1.0
    """
    feats = extract_features(frame)

    if feats is None:
        return _smooth(0.0)          # 얼굴 미감지 → 0점

    x = np.array([[feats[c] for c in FEATURE_COLS]])

    if _clf is not None:
        # ── ML 모델 추론 ────────────────────────────────────────────────
        proba = _clf.predict_proba(x)[0]
        # predict_proba 는 [P(class=0), P(class=1)] 반환
        score = float(proba[1])
    else:
        raise FileNotFoundError(f"no file")
        # ── 모델 미학습 시 규칙 기반 fallback ───────────────────────────
        score = _rule_based_fallback(feats)

    return _smooth(round(score, 4))


def _rule_based_fallback(feats: dict) -> float:
    """
    규칙기반
    
    # EAR: 0.25~0.35 정상 범위 → 1점, 너무 낮으면(눈 감음) 0점
    ear_avg = (feats["left_ear"] + feats["right_ear"]) / 2
    ear_score = float(np.clip((ear_avg - 0.10) / 0.20, 0, 1))

    # 동공 offset: 낮을수록(중앙에 가까울수록) 집중
    pupil_avg = (feats["l_pupil_dist"] + feats["r_pupil_dist"]) / 2
    pupil_score = float(np.clip(1.0 - pupil_avg / 0.35, 0, 1))

    # 고개 각도: pitch·yaw 절댓값 작을수록 집중
    head_score = float(np.clip(1.0 - (abs(feats["pitch"]) + abs(feats["yaw"])) / 60.0, 0, 1))
    
    return 0.3 * ear_score + 0.4 * pupil_score + 0.3 * head_score
    """
    # 완전 랜덤
    return round(random.random(), 4)
