import eventlet
eventlet.monkey_patch()

import logging
import os
from functools import wraps
from collections import deque
import time

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO

import database as db
from matcher import match_gesture
from gesture_emoji import emoji_from_samples, PRESET_EMOJIS

# --------------------------------------------------------------- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signvoice")

app = Flask(__name__)

# In production, set SIGNVOICE_SECRET_KEY as an environment variable.
# Falling back to a fixed dev key locally is fine for a project demo, but
# we log a warning so it's clear this isn't meant for real deployment.
app.secret_key = os.environ.get("SIGNVOICE_SECRET_KEY")
if not app.secret_key:
    app.secret_key = "dev-only-change-this-before-any-real-deployment"
    log.warning(
        "SIGNVOICE_SECRET_KEY not set — using an insecure development key. "
        "Fine for local testing/demo, not for real deployment."
    )

socketio = SocketIO(app, cors_allowed_origins="*")

db.init_db()

# Hand detection runs client-side in the browser via MediaPipe's JS Tasks
# Vision API (see static/js/hand_tracker.js) — the server only receives
# already-extracted landmark numbers, not video frames.

RECORD_FRAMES = 10      # landmark frames captured per registered sample
RECOGNIZE_FRAMES = 10   # sliding window size used during live recognition
MATCH_EVERY_N_FRAMES = 3  # run matching every 3rd incoming frame — matching on every
                           # single frame (the old value of 1) meant a full DTW pass
                           # against every registered sign on every frame, which is the
                           # main thing that was making live recognition feel laggy
COOLDOWN_SECONDS = 1.2  # suppress re-announcing the SAME sign within this window
NO_MATCH_EMIT_INTERVAL = 1.5  # seconds between "not recognized" notices, to avoid spamming

# Majority-vote confirmation: look at the last VOTE_WINDOW match attempts and
# require at least VOTE_THRESHOLD of them to agree before announcing. More
# forgiving than requiring N-in-a-row, so real signs trigger faster on
# average while brief incidental poses still get filtered out.
VOTE_WINDOW = 5
VOTE_THRESHOLD = 3

# Squared-distance jump threshold for detecting a sharp hand-pose change
# (i.e. switching from one sign to another) so the sliding window flushes
# immediately instead of lingering on stale frames. Tune with DEBUG_MATCH on.
POSE_JUMP_THRESHOLD_SQ = 999  # start disabled; enable once matching accuracy is solid

DEBUG_MATCH = os.environ.get("SIGNVOICE_DEBUG", "0") == "1"

# per-connection scratch state, keyed by socket session id
buffers = {}
frame_counters = {}
last_recognized = {}   # sid -> (gesture_id, timestamp)
vote_history = {}       # sid -> deque of recent match results (gesture_id or None)
no_match_last_emit = {}  # sid -> timestamp of last "no_match" notice sent


def _frame_distance_sq(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ------------------------------------------------------------ error pages --

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That page doesn't exist."), 404


@app.errorhandler(500)
def server_error(e):
    log.exception("Unhandled server error")
    return render_template(
        "error.html", code=500, message="Something went wrong on our end. Please try again."
    ), 500


# ---------------------------------------------------------------- pages ----

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        try:
            user_id = db.verify_user(username, password)
        except Exception:
            log.exception("Login failed due to a database error")
            return render_template(
                "login.html", error="Something went wrong. Please try again."
            ), 500
        if user_id:
            session["user_id"] = user_id
            session["username"] = username
            log.info("User '%s' logged in", username)
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Incorrect username or password.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return render_template("register_user.html", error="Fill in both fields.")
        if len(password) < 4:
            return render_template(
                "register_user.html", error="Password should be at least 4 characters."
            )
        try:
            created = db.create_user(username, password)
        except Exception:
            log.exception("Signup failed due to a database error")
            return render_template(
                "register_user.html", error="Something went wrong. Please try again."
            ), 500
        if created:
            log.info("New user registered: '%s'", username)
            return redirect(url_for("login"))
        return render_template("register_user.html", error="That username is taken.")
    return render_template("register_user.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    try:
        gestures = db.get_user_gestures(session["user_id"])
    except Exception:
        log.exception("Failed to load gestures for dashboard")
        gestures = []
    return render_template(
        "dashboard.html", gestures=gestures, username=session["username"]
    )


@app.route("/add-gesture", methods=["GET", "POST"])
@login_required
def add_gesture_page():
    if request.method == "POST":
        gesture_name = request.form.get("gesture_name", "").strip()
        text_message = request.form.get("text_message", "").strip()
        if not gesture_name or not text_message:
            return jsonify({"error": "Both fields are required."}), 400
        if len(gesture_name) > 50 or len(text_message) > 200:
            return jsonify({"error": "Name or message is too long."}), 400
        try:
            # Starts with the default seal — recording a sample right after
            # will replace it with one matching the actual hand shape (see
            # gesture_emoji.emoji_from_samples, called from recording_frame).
            gesture_id = db.add_gesture(session["user_id"], gesture_name, text_message)
        except Exception:
            log.exception("Failed to add gesture")
            return jsonify({"error": "Could not save the gesture. Please try again."}), 500
        return jsonify({"gesture_id": gesture_id})

    try:
        existing = db.get_user_gestures(session["user_id"])
        used_emojis = {g["emoji"] for g in existing if g["emoji"]}
    except Exception:
        log.exception("Failed to load existing seals for add-sign page")
        used_emojis = set()
    available_presets = [e for e in PRESET_EMOJIS if e not in used_emojis]

    return render_template(
        "add_gesture.html", record_frames=RECORD_FRAMES, available_presets=available_presets
    )


@app.route("/delete-gesture/<int:gesture_id>", methods=["POST"])
@login_required
def delete_gesture_route(gesture_id):
    try:
        db.delete_gesture(gesture_id, session["user_id"])
    except Exception:
        log.exception("Failed to delete gesture %s", gesture_id)
    return redirect(url_for("dashboard"))


def _available_presets_for_edit(user_id, gesture_id, current_emoji):
    """Same idea as the Add Sign page's available-seals list — seals already
    used by this user's OTHER signs are hidden — but the sign being edited
    keeps its own current seal in the list regardless, so it's never left
    without a valid, selectable option."""
    try:
        used_by_others = db.get_user_gesture_emojis(user_id, exclude_gesture_id=gesture_id)
    except Exception:
        log.exception("Failed to load seals for edit-sign page")
        used_by_others = set()
    presets = [e for e in PRESET_EMOJIS if e not in used_by_others]
    if current_emoji and current_emoji not in presets:
        presets.append(current_emoji)
    return presets


@app.route("/edit-gesture/<int:gesture_id>", methods=["GET", "POST"])
@login_required
def edit_gesture_page(gesture_id):
    gesture = db.get_gesture(gesture_id, session["user_id"])
    if gesture is None:
        return render_template("error.html", code=404, message="That sign doesn't exist."), 404

    if request.method == "POST":
        gesture_name = request.form.get("gesture_name", "").strip()
        text_message = request.form.get("text_message", "").strip()
        emoji = request.form.get("emoji", "✨").strip() or "✨"
        presets = _available_presets_for_edit(session["user_id"], gesture_id, emoji)
        if not gesture_name or not text_message:
            return render_template(
                "edit_gesture.html", gesture=gesture, presets=presets,
                error="Both fields are required.",
            )
        if len(gesture_name) > 50 or len(text_message) > 200:
            return render_template(
                "edit_gesture.html", gesture=gesture, presets=presets,
                error="Name or message is too long.",
            )
        try:
            db.update_gesture(gesture_id, session["user_id"], gesture_name, text_message, emoji)
        except Exception:
            log.exception("Failed to update gesture %s", gesture_id)
            return render_template(
                "edit_gesture.html", gesture=gesture, presets=presets,
                error="Could not save changes. Please try again.",
            )
        return redirect(url_for("dashboard"))

    presets = _available_presets_for_edit(session["user_id"], gesture_id, gesture["emoji"])
    return render_template("edit_gesture.html", gesture=gesture, presets=presets)


@app.route("/recognize")
@login_required
def recognize_page():
    return render_template("recognize.html")


@app.route("/seal-pack")
@login_required
def seal_pack_page():
    return render_template("seal_pack.html", presets=PRESET_EMOJIS)


@app.route("/legal")
def legal_page():
    return render_template("legal.html")


# ---------------------------------------------------------- socket events --

@socketio.on("connect")
def on_connect():
    buffers[request.sid] = []
    frame_counters[request.sid] = 0
    last_recognized[request.sid] = (None, 0)
    vote_history[request.sid] = deque(maxlen=VOTE_WINDOW)
    no_match_last_emit[request.sid] = 0


@socketio.on("disconnect")
def on_disconnect():
    buffers.pop(request.sid, None)
    frame_counters.pop(request.sid, None)
    last_recognized.pop(request.sid, None)
    vote_history.pop(request.sid, None)
    no_match_last_emit.pop(request.sid, None)


@socketio.on("start_recording_sample")
def start_recording_sample(data):
    buffers[request.sid] = []


@socketio.on("recording_frame")
def recording_frame(data):
    try:
        landmarks = data["landmarks"]
        hand_present = data["hand_present"]
    except (KeyError, TypeError):
        log.warning("Malformed recording_frame payload, ignoring")
        return

    if hand_present:
        buffers[request.sid].append(landmarks)

    progress = len(buffers[request.sid])
    socketio.emit(
        "recording_progress",
        {"progress": progress, "target": RECORD_FRAMES},
        room=request.sid,
    )

    if progress >= RECORD_FRAMES:
        gesture_id = data.get("gesture_id")
        try:
            db.add_sample(gesture_id, buffers[request.sid][:RECORD_FRAMES])
            socketio.emit("sample_complete", {}, room=request.sid)
        except Exception:
            log.exception("Failed to save sample for gesture %s", gesture_id)
            socketio.emit(
                "sample_error", {"message": "Could not save sample. Try again."}, room=request.sid
            )
            buffers[request.sid] = []
            return

        try:
            all_samples = db.get_samples_for_gesture(gesture_id)
            shape_emoji = emoji_from_samples(all_samples)
            if shape_emoji:
                db.update_gesture_emoji(gesture_id, shape_emoji)
        except Exception:
            # Non-critical — the gesture keeps whatever seal it already had.
            log.exception("Failed to classify hand shape for gesture %s", gesture_id)

        buffers[request.sid] = []


@socketio.on("recognize_frame")
def recognize_frame(data):
    user_id = session.get("user_id")
    if not user_id:
        return

    try:
        landmarks = data["landmarks"]
        hand_present = data["hand_present"]
    except (KeyError, TypeError):
        log.warning("Malformed recognize_frame payload, ignoring")
        return

    buf = buffers.setdefault(request.sid, [])
    hist = vote_history.setdefault(request.sid, deque(maxlen=VOTE_WINDOW))

    if hand_present:
        if buf:
            jump = _frame_distance_sq(landmarks, buf[-1])
            if jump > POSE_JUMP_THRESHOLD_SQ:
                buf.clear()
                hist.clear()
                last_recognized[request.sid] = (None, 0)
        buf.append(landmarks)
        if len(buf) > RECOGNIZE_FRAMES:
            buf.pop(0)
    else:
        buf.clear()
        hist.clear()
        last_recognized[request.sid] = (None, 0)

    if len(buf) < RECOGNIZE_FRAMES:
        return

    frame_counters[request.sid] = frame_counters.get(request.sid, 0) + 1
    if frame_counters[request.sid] % MATCH_EVERY_N_FRAMES != 0:
        return

    try:
        samples_map, lookup = db.get_gesture_samples_map(user_id)
        gesture_id, distance = match_gesture(user_id, buf, samples_map)
    except Exception:
        log.exception("Matching failed for user %s", user_id)
        return

    if DEBUG_MATCH:
        log.debug("match gesture_id=%s distance=%.3f", gesture_id, distance)

    hist.append(gesture_id)

    if len(hist) < VOTE_THRESHOLD:
        return

    counts = {}
    for gid in hist:
        if gid is not None:
            counts[gid] = counts.get(gid, 0) + 1

    if not counts:
        last_no_match = no_match_last_emit.get(request.sid, 0)
        now = time.time()
        if now - last_no_match > NO_MATCH_EMIT_INTERVAL:
            socketio.emit("no_match", {}, room=request.sid)
            no_match_last_emit[request.sid] = now
        return

    winner_id, winner_count = max(counts.items(), key=lambda kv: kv[1])
    if winner_count < VOTE_THRESHOLD:
        return

    prev_id, prev_time = last_recognized.get(request.sid, (None, 0))
    now = time.time()
    if winner_id == prev_id and (now - prev_time) < COOLDOWN_SECONDS:
        return

    info = lookup.get(winner_id)
    if info is None:
        # Gesture was deleted between being registered and being matched —
        # stale reference, just ignore this frame instead of crashing.
        log.warning("Matched gesture_id %s not found in lookup (deleted?)", winner_id)
        return

    socketio.emit(
        "recognized",
        {"text": info["text"], "name": info["name"], "emoji": info.get("emoji", "✨")},
        room=request.sid,
    )
    last_recognized[request.sid] = (winner_id, now)
    hist.clear()


if __name__ == "__main__":
        socketio.run(app, debug=DEBUG_MATCH, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
