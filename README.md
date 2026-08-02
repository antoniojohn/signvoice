# SignVoice — Customizable Sign Language Recognition

Register your own hand signs, assign each one a text message, and have it
spoken aloud automatically when recognized live from your webcam.

## How it works

- **MediaPipe Hand Landmarker** (the official JS/WASM version, GPU-accelerated) runs
  **directly in your browser** and extracts 21 landmark points per hand from every
  video frame locally. This is the current recommended MediaPipe API and tends to be
  more accurate across lighting conditions than the older Python bindings.
- Only the extracted landmark numbers (not video frames) are sent to the Flask
  backend over a WebSocket — this keeps things fast, since no image encoding,
  network image transfer, or server-side model inference is needed.
- Each registered sign stores a short **sequence** of landmarks (not just one frame),
  so it works for both static poses (thumbs up) and moving signs (waving hello).
- Recognition uses **Dynamic Time Warping (DTW)** to compare your live hand movement
  against your saved samples — no model training or retraining needed. Register a
  sign and it works immediately.
- Voice output uses the browser's built-in **Web Speech API**, so it needs no backend
  work and works offline once the page is loaded.

### Why it's fast

Earlier versions of this project sent full JPEG images to a Python backend for
MediaPipe processing there — that round trip (encode → network → decode → infer)
was the main source of lag. Now detection runs on your GPU in-browser, and the
server only receives small landmark arrays plus does DTW matching, throttled to
every few frames (`MATCH_EVERY_N_FRAMES` in `app.py`) so it doesn't run the
comparison against every saved sample on every single frame.

## Setup (Windows / Mac / Linux)

1. **Install Python 3.10 or 3.11** (MediaPipe does not yet support the very latest
   Python versions — 3.10/3.11 is the safest bet).

2. Open this folder in VS Code, then open a terminal (``Terminal → New Terminal``).

3. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the app:

   ```bash
   python app.py
   ```

6. Open your browser at **http://localhost:5000** — sign up, log in, and start
   registering signs.

## Using it

1. **Sign up** for an account (each user has their own private sign library).
2. Go to **Add sign** → type a name and the text/message it should mean →
   record 3 samples holding the sign in front of your camera.
3. Go to **Live recognize** → show any of your registered signs → the matched
   text appears as a caption and is spoken aloud.

## Tuning accuracy and speed

If signs are misrecognized or nothing matches:

- Open `matcher.py` and adjust the `threshold` value in `match_gesture()`
  (lower = stricter, fewer false positives; higher = more lenient).
- Record more samples per sign (edit `SAMPLES_NEEDED` in
  `static/js/register.js`) with slightly different hand angles and, ideally,
  under a couple of different lighting conditions.
- Make sure your hand is fully in frame.

If it still feels laggy:

- Raise `MATCH_EVERY_N_FRAMES` in `app.py` (e.g. from 4 to 8) — this runs the
  DTW comparison less often, trading a bit of responsiveness for speed.
- Raise `EMIT_INTERVAL_MS` in `static/js/recognize.js` (e.g. from 80 to 120) to
  send fewer landmark updates per second to the server.
- Close other GPU-heavy browser tabs/apps — the hand tracker uses your GPU
  via WebGL/WASM.

## Project structure

```
signvoice/
├── app.py                 Flask + SocketIO server, routes & socket events
├── database.py            SQLite helpers (users, gestures, samples)
├── gesture_utils.py        MediaPipe hand landmark extraction
├── matcher.py              DTW-based gesture matching
├── requirements.txt
├── templates/              Jinja2 HTML pages
└── static/
    ├── css/style.css
    └── js/register.js, recognize.js
```

## Ideas to extend

- Add per-gesture accuracy testing right in the dashboard.
- Support two-hand signs more richly (already captured, just not weighted specially).
- Add a "practice mode" that quizzes you on your own signs.
- Deploy with `eventlet`/`gunicorn` behind a real server for production use.
