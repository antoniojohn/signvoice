import { initHandLandmarker, extractLandmarks } from "/static/js/hand_tracker.js";

const socket = io();

const video = document.getElementById("video");
const handIndicator = document.getElementById("hand-indicator");
const captionBubble = document.getElementById("caption-bubble");
const statusLine = document.getElementById("status-line");
const connDot = document.getElementById("conn-dot");

let handLandmarker = null;
let captionTimeout = null;
let lastEmit = 0;
const EMIT_INTERVAL_MS = 90; // ~11 fps sent to the server — was 60ms (~16fps), which combined
                              // with matching on every frame server-side was overloading the
                              // socket/DTW pipeline and showing up as recognition lag

let isSpeaking = false;
let currentSpokenText = null;
let speechToken = 0; // bumped on every speak() call — lets a stale, cancelled
                      // utterance's onend/onerror recognize it's no longer
                      // the active one and skip touching the caption/state.

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240 },
    });
    video.srcObject = stream;
    await video.play();

    statusLine.textContent = "Loading hand tracker…";
    handLandmarker = await initHandLandmarker();
    statusLine.textContent = "Camera ready. Show a sign whenever you're ready.";

    detectionLoop();
  } catch (err) {
    statusLine.textContent =
      "Could not access camera or load hand tracker. Check camera permission and your internet connection.";
    console.error(err);
  }
}

function detectionLoop(timestamp) {
  const result = handLandmarker.detectForVideo(video, performance.now());
  const { landmarks, handPresent } = extractLandmarks(result);

  handIndicator.classList.toggle("active", handPresent);

  if (!lastEmit || timestamp - lastEmit >= EMIT_INTERVAL_MS) {
    lastEmit = timestamp;
    socket.emit("recognize_frame", { landmarks, hand_present: handPresent });
  }

  requestAnimationFrame(detectionLoop);
}

socket.on("connect", () => {
  if (connDot) connDot.classList.add("online");
});

socket.on("disconnect", () => {
  if (connDot) connDot.classList.remove("online");
  statusLine.textContent = "Connection lost — trying to reconnect…";
});

socket.on("recognized", (data) => {
  // If we're still reading THIS SAME sign's message aloud, let it finish —
  // don't cancel and restart it. A genuinely different sign is still free
  // to interrupt right away.
  if (isSpeaking && data.text === currentSpokenText) return;
  showCaption(data.text, "recognized", data.emoji);
  speak(data.text);
});

socket.on("no_match", () => {
  // Don't flash "not recognized" while a sign is still being read out —
  // brief hand jitter shouldn't interrupt what's already playing.
  if (isSpeaking) return;
  showCaption("Sign not recognized", "unknown");
});

function showCaption(text, state, emoji) {
  if (state === "recognized") {
    captionBubble.innerHTML =
      (emoji ? `<span class="cap-emoji">${emoji}</span>` : "") +
      `<span class="cap-text">${text}</span>`;
  } else {
    captionBubble.textContent = text;
  }
  captionBubble.classList.remove("recognized", "unknown", "show");
  // force reflow so the signature-reveal clip-path animation restarts cleanly
  void captionBubble.offsetWidth;
  captionBubble.classList.add("show", state);
  clearTimeout(captionTimeout);
  if (state === "unknown") {
    captionTimeout = setTimeout(() => {
      captionBubble.classList.remove("show");
    }, 1200);
  }
  // For "recognized", the caption stays up until speak() reports the
  // utterance has actually finished — see below — rather than a fixed timer.
}

function speak(text) {
  currentSpokenText = text;
  const token = ++speechToken; // this utterance's identity, captured now —
                                // before cancel() below can trigger the
                                // PREVIOUS utterance's onerror out from under us
  if (!("speechSynthesis" in window)) {
    // No speech support in this browser — fall back to a fixed display time.
    captionTimeout = setTimeout(() => {
      if (token !== speechToken) return; // a newer sign already took over
      captionBubble.classList.remove("show");
    }, 3500);
    return;
  }
  window.speechSynthesis.cancel(); // don't let utterances stack up
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  isSpeaking = true;
  utterance.onend = () => {
    if (token !== speechToken) return; // stale — a newer sign already replaced this one
    isSpeaking = false;
    captionBubble.classList.remove("show");
  };
  utterance.onerror = () => {
    if (token !== speechToken) return; // this is the cancel() from a newer sign, not a real error
    isSpeaking = false;
    captionBubble.classList.remove("show");
  };
  window.speechSynthesis.speak(utterance);
}

startCamera();
