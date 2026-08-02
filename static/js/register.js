import { initHandLandmarker, extractLandmarks } from "/static/js/hand_tracker.js";

const socket = io();

const video = document.getElementById("video");
const handIndicator = document.getElementById("hand-indicator");
const progressFill = document.getElementById("progress-fill");
const statusLine = document.getElementById("status-line");
const sampleCounter = document.getElementById("sample-counter");
const recordBtn = document.getElementById("record-btn");
const finishBtn = document.getElementById("finish-btn");

let gestureId = null;
let samplesRecorded = 0;
let recording = false;
let handLandmarker = null;
let detectionLoopId = null;

recordBtn.disabled = true;
recordBtn.textContent = "Loading hand tracker…";

// ---- Step 1: details form -> create gesture row in DB ----
document.getElementById("continue-btn").addEventListener("click", async () => {
  const continueBtn = document.getElementById("continue-btn");
  if (continueBtn.disabled) return; // already submitting, ignore extra clicks

  const gesture_name = document.getElementById("gesture_name").value.trim();
  const text_message = document.getElementById("text_message").value.trim();

  if (!gesture_name || !text_message) {
    alert("Please fill in both fields.");
    return;
  }

  continueBtn.disabled = true;
  const originalLabel = continueBtn.textContent;
  continueBtn.textContent = "Saving…";

  const form = new URLSearchParams();
  form.append("gesture_name", gesture_name);
  form.append("text_message", text_message);

  let data;
  try {
    const res = await fetch("/add-gesture", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
    });
    data = await res.json();
  } catch (err) {
    alert("Could not reach the server. Please try again.");
    continueBtn.disabled = false;
    continueBtn.textContent = originalLabel;
    return;
  }
  if (data.error) {
    alert(data.error);
    continueBtn.disabled = false;
    continueBtn.textContent = originalLabel;
    return;
  }

  gestureId = data.gesture_id;
  document.getElementById("step-details").style.display = "none";
  document.getElementById("step-camera").style.display = "block";
  document.getElementById("sample-title").textContent = `Recording: ${gesture_name}`;
  await startCamera();
});

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240 },
    });
    video.srcObject = stream;
    await video.play();

    handLandmarker = await initHandLandmarker();
    recordBtn.disabled = false;
    recordBtn.textContent = "Record sample";
    statusLine.textContent = 'Click "Record sample" and hold your sign.';

    detectionLoop();
  } catch (err) {
    statusLine.textContent =
      "Could not access camera or load hand tracker. Check camera permission and your internet connection.";
    console.error(err);
  }
}

// Runs continuously; while `recording` is true, emits landmarks to the server.
function detectionLoop() {
  const result = handLandmarker.detectForVideo(video, performance.now());
  const { landmarks, handPresent } = extractLandmarks(result);

  handIndicator.classList.toggle("active", handPresent);

  if (recording) {
    socket.emit("recording_frame", {
      landmarks,
      hand_present: handPresent,
      gesture_id: gestureId,
    });
  }

  detectionLoopId = requestAnimationFrame(detectionLoop);
}

recordBtn.addEventListener("click", () => {
  if (recording || samplesRecorded >= SAMPLES_NEEDED) return;
  recording = true;
  recordBtn.disabled = true;
  statusLine.textContent = "Recording... hold your sign steady!";
  socket.emit("start_recording_sample", {});
});

// The bar reflects overall progress across ALL samples, not just the one
// currently recording — so it grows in thirds (1 of 3, 2 of 3, 3 of 3)
// instead of filling to 100% and resetting back to 0% each time.
socket.on("recording_progress", (data) => {
  const withinSample = data.progress / data.target;
  const overallPct = Math.min(100, ((samplesRecorded + withinSample) / SAMPLES_NEEDED) * 100);
  progressFill.style.width = overallPct + "%";
});

socket.on("sample_complete", () => {
  recording = false;
  recordBtn.disabled = false;
  samplesRecorded += 1;
  sampleCounter.textContent = `${samplesRecorded} of ${SAMPLES_NEEDED} samples recorded`;
  progressFill.style.width = ((samplesRecorded / SAMPLES_NEEDED) * 100) + "%";

  if (samplesRecorded >= SAMPLES_NEEDED) {
    statusLine.textContent = "Great! You have enough samples for good accuracy.";
    recordBtn.style.display = "none";
    finishBtn.style.display = "inline-block";
    if (detectionLoopId) cancelAnimationFrame(detectionLoopId);
  } else {
    statusLine.textContent = "Sample saved. Click again to record another (vary your angle slightly).";
  }
});

// Enter key drives whichever step is currently visible:
// - step-details: same as clicking "Continue to camera"
// - step-camera: records the next sample, or finishes once all 3 are in
document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;

  const detailsVisible = document.getElementById("step-details").style.display !== "none";
  if (detailsVisible) {
    event.preventDefault();
    document.getElementById("continue-btn").click();
    return;
  }

  if (samplesRecorded >= SAMPLES_NEEDED) {
    event.preventDefault();
    finishBtn.click();
  } else if (!recording && !recordBtn.disabled) {
    event.preventDefault();
    recordBtn.click();
  }
});
