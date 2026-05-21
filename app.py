"""
app.py
------
Flask + SocketIO 기반 학습 지원 서버.
"""

import os
import io
import base64
import csv as csv_module
from datetime import datetime
from collections import deque

import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, send_from_directory, jsonify, Response)
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from flask_login import (LoginManager, UserMixin,
                         login_user, login_required, logout_user, current_user)

from concentration_model import analyze_frame, extract_features, reload_model, FEATURE_COLS, get_model_status



# ---------------------------------------------------------------------------
# 앱 초기화
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_DIR,   exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"]                  = "dev-secret-change-in-prod"
app.config["SQLALCHEMY_DATABASE_URI"]     = "sqlite:///" + os.path.join(BASE_DIR, "database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"]              = UPLOAD_FOLDER

db        = SQLAlchemy(app)
socketio  = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
login_mgr = LoginManager(app)
login_mgr.login_view = "login"

# ---------------------------------------------------------------------------
# DB 모델
# ---------------------------------------------------------------------------

class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80),  unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role     = db.Column(db.String(20),  default="student")   # student | professor


class Lecture(db.Model):
    id          = db.Column(db.Integer,  primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    filename    = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime,    default=datetime.utcnow)


class ViewRecord(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"),    nullable=False)
    lecture_id  = db.Column(db.Integer, db.ForeignKey("lecture.id"), nullable=False)
    started_at  = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime)


class ConcentrationPoint(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    view_id   = db.Column(db.Integer, db.ForeignKey("view_record.id"), nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)   # 시청 시작부터 경과 초
    score     = db.Column(db.Float,   nullable=False)   # 0.0 ~ 1.0


@login_mgr.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def create_tables():
    db.create_all()
    if not User.query.filter_by(username="prof1").first():
        db.session.add_all([
            User(username="prof1",    password="pass", role="professor"),
            User(username="student1", password="pass", role="student"),
        ])
        db.session.commit()

# ---------------------------------------------------------------------------
# 인증
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            username=request.form["username"].strip(),
            password=request.form["password"].strip(),
        ).first()
        if user:
            login_user(user)
            return redirect(url_for("index"))
        flash("아이디 또는 비밀번호가 올바르지 않습니다.")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        role = request.form["role"].strip()

        if not username or not password:
            flash("아이디와 비밀번호는 공백일 수 없습니다.")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("이미 존재하는 아이디입니다.")
            return render_template("register.html")

        db.session.add(User(
            username=username,
            password=password,
            role=role,
        ))
        db.session.commit()
        flash("회원가입 완료. 로그인해주세요.")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

# ---------------------------------------------------------------------------
# 메인 / 역할별 대시보드
# ---------------------------------------------------------------------------



@app.route("/")
def index():

    if current_user.is_authenticated:
        return redirect(url_for("professor" if current_user.role == "professor" else "student"))
    return render_template("index.html")


@app.route("/student")
@login_required
def student():
    lectures = Lecture.query.order_by(Lecture.uploaded_at.desc()).all()
    return render_template("student.html", lectures=lectures)


@app.route("/professor")
@login_required
def professor():
    if current_user.role != "professor":
        flash("교수자만 접근 가능합니다.")
        return redirect(url_for("student"))
    lectures = Lecture.query.order_by(Lecture.uploaded_at.desc()).all()
    return render_template("professor.html", lectures=lectures)

# ---------------------------------------------------------------------------
# 강의 관리
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if current_user.role != "professor":
        flash("교수자만 업로드 가능합니다.")
        return redirect(url_for("index"))
    if request.method == "POST":
        f     = request.files["file"]
        title = request.form.get("title") or f.filename
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
        db.session.add(Lecture(title=title, filename=f.filename))
        db.session.commit()
        flash("강의가 업로드되었습니다.")
        return redirect(url_for("professor"))
    return render_template("upload.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/delete_lecture/<int:lecture_id>", methods=["DELETE", "POST"])
@login_required
def delete_lecture(lecture_id):
    if current_user.role != "professor":
        return jsonify({"ok": False, "error": "권한 없음"}), 403
    lecture = Lecture.query.get_or_404(lecture_id)
    fpath   = os.path.join(app.config["UPLOAD_FOLDER"], lecture.filename)
    if os.path.exists(fpath):
        os.remove(fpath)
    for view in ViewRecord.query.filter_by(lecture_id=lecture_id).all():
        ConcentrationPoint.query.filter_by(view_id=view.id).delete()
        db.session.delete(view)
    db.session.delete(lecture)
    db.session.commit()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# 강의 시청
# ---------------------------------------------------------------------------

@app.route("/watch/<int:lecture_id>")
@login_required
def watch(lecture_id):
    lecture = Lecture.query.get_or_404(lecture_id)
    view    = ViewRecord(user_id=current_user.id, lecture_id=lecture.id)
    db.session.add(view)
    db.session.commit()
    return render_template("watch.html", lecture=lecture, view_id=view.id)

# ---------------------------------------------------------------------------
# 집중도 API
# ---------------------------------------------------------------------------

@app.route("/api/concentration/view/<int:view_id>")
@login_required
def api_get_concentration(view_id):
    pts = (ConcentrationPoint.query
           .filter_by(view_id=view_id)
           .order_by(ConcentrationPoint.timestamp)
           .all())
    return jsonify([{"timestamp": p.timestamp, "score": p.score} for p in pts])


@app.route("/api/concentration/lecture/<int:lecture_id>/average")
@login_required
def api_lecture_average(lecture_id):
    """강의 전체 시청자의 타임스탬프별 평균 집중도."""
    views   = ViewRecord.query.filter_by(lecture_id=lecture_id).all()
    total   = {}
    count   = {}
    for v in views:
        for p in ConcentrationPoint.query.filter_by(view_id=v.id).all():
            total[p.timestamp] = total.get(p.timestamp, 0.0) + p.score
            count[p.timestamp] = count.get(p.timestamp, 0)   + 1
    result = [{"timestamp": ts, "avg": total[ts] / count[ts]}
              for ts in sorted(total)]
    return jsonify(result)


@app.route("/api/concentration/lecture/<int:lecture_id>/summary")
@login_required
def api_lecture_summary(lecture_id):
    """교수자 대시보드용 강의 요약."""
    views = ViewRecord.query.filter_by(lecture_id=lecture_id).all()
    scores = []
    for v in views:
        for p in ConcentrationPoint.query.filter_by(view_id=v.id).all():
            scores.append(p.score)
    if not scores:
        return jsonify({"count": 0, "mean": 0, "min": 0, "max": 0})
    return jsonify({
        "count":    len(scores),
        "mean":     round(float(np.mean(scores)),   3),
        "min":      round(float(np.min(scores)),    3),
        "max":      round(float(np.max(scores)),    3),
        "viewers":  len(views),
    })


@app.route("/api/reload_model", methods=["POST"])
@login_required
def api_reload_model():
    """train_model.py 실행 후 모델을 서버 재시작 없이 갱신."""
    if current_user.role != "professor":
        return jsonify({"ok": False}), 403
    reload_model()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# CSV 다운로드 (교수자용)
# ---------------------------------------------------------------------------

@app.route("/api/export/lecture/<int:lecture_id>")
@login_required
def export_lecture_csv(lecture_id):
    if current_user.role != "professor":
        return jsonify({"ok": False}), 403
    lecture = Lecture.query.get_or_404(lecture_id)
    views   = ViewRecord.query.filter_by(lecture_id=lecture_id).all()

    def generate():
        yield "view_id,user_id,timestamp,score\n"
        for v in views:
            for p in (ConcentrationPoint.query
                      .filter_by(view_id=v.id)
                      .order_by(ConcentrationPoint.timestamp).all()):
                yield f"{v.id},{v.user_id},{p.timestamp},{p.score}\n"

    headers = {
        "Content-Disposition": f'attachment; filename="lecture_{lecture_id}.csv"',
        "Content-Type": "text/csv; charset=utf-8",
    }
    return Response(generate(), headers=headers)

# ---------------------------------------------------------------------------
# SocketIO — 실시간 프레임 처리
# ---------------------------------------------------------------------------

@socketio.on("frame")
def handle_frame(data):
    """
    클라이언트가 보내는 데이터:
        { view_id: int, timestamp: int, image: "data:image/jpeg;base64,..." }
    """
    view_id  = data.get("view_id")
    ts       = data.get("timestamp", 0)
    img_data = data.get("image", "")

    if not img_data or not view_id:
        return

    try:
        # base64 디코딩 → PIL → numpy (RGB)
        _, b64 = img_data.split(",", 1)
        img    = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        frame  = np.array(img)

        score  = analyze_frame(frame)

        # DB 저장
        cp = ConcentrationPoint(view_id=view_id, timestamp=ts, score=float(score))
        db.session.add(cp)
        db.session.commit()

        # 클라이언트로 결과 반환
        emit("frame_result", {
            "view_id":   view_id,
            "timestamp": ts,
            "score":     float(score),
        })

    except Exception as e:
        print(f"[frame] 처리 오류: {e}")


@socketio.on('collect_sample')
def on_collect_sample(data, incremental_learner=None):
    """
    학생/교수자가 'focused' / 'unfocused' 버튼을 누를 때 호출.
    feature 추출 → CSV append → 50개 모이면 백그라운드 재학습.
    """
    import base64, numpy as np, cv2

    try:
        # data['image'] = "data:image/jpeg;base64,..."
        b64 = data['image'].split(',', 1)[-1]
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(b64), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            emit('collect_result', {'ok': False, 'reason': 'decode failed'})
            return

        feats = extract_features(img)
        if feats is None:
            emit('collect_result', {'ok': False, 'reason': 'face not detected'})
            return

        user = current_user.username if current_user.is_authenticated else 'anonymous'
        result = incremental_learner.record_sample(
            features=feats,
            label=int(data['label']),
            user=user,
        )
        emit('collect_result', {'ok': True, **result})
    except Exception as e:
        emit('collect_result', {'ok': False, 'reason': str(e)})

# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with app.app_context():
        create_tables()
    socketio.run(app, debug=True, host="127.0.0.1", port=8000)