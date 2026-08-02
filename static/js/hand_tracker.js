// Runs MediaPipe's HandLandmarker directly in the browser (GPU-accelerated via WASM).
// This replaces server-side Python MediaPipe processing entirely — only the
// extracted landmark numbers get sent to the backend, not video frames.

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const LANDMARKS_PER_FRAME = 126; // 2 hands x 21 points x (x, y, z)

export async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  return handLandmarker;
}

// Scales by the hand's overall bounding box (computed across all 21 points)
// so noise in any single landmark barely moves the result — this is what
// makes similar-looking signs distinguishable and matching stable.
function normalizeHand(handPoints) {
  const wrist = handPoints[0];

  const xs = handPoints.map((p) => p.x);
  const ys = handPoints.map((p) => p.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const scale = Math.max(width, height) || 1;

  return handPoints.map((p) => ({
    x: (p.x - wrist.x) / scale,
    y: (p.y - wrist.y) / scale,
    z: (p.z - wrist.z) / scale,
  }));
}

export function extractLandmarks(result) {
  const landmarks = [];
  let handPresent = false;

  if (result.landmarks && result.landmarks.length > 0) {
    handPresent = true;
    const sortedHands = [...result.landmarks].sort(
      (a, b) => a[0].x - b[0].x
    );
    for (const hand of sortedHands.slice(0, 2)) {
      const normalized = normalizeHand(hand);
      for (const point of normalized) {
        landmarks.push(point.x, point.y, point.z);
      }
    }
  }

  while (landmarks.length < LANDMARKS_PER_FRAME) {
    landmarks.push(0.0);
  }

  return { landmarks: landmarks.slice(0, LANDMARKS_PER_FRAME), handPresent };
}
