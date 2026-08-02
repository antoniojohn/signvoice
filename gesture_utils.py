# NOTE: This file is no longer used by app.py. Hand detection now runs
# client-side in the browser via MediaPipe's JS Tasks Vision API
# (see static/js/hand_tracker.js) for lower latency and GPU acceleration.
# Kept here in case you want a server-side fallback for a non-browser client.

import base64
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# 2 hands x 21 landmarks x (x, y, z) = 126 numbers per frame
LANDMARKS_PER_FRAME = 126


class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def decode_base64_image(self, data_url):
        """data_url looks like 'data:image/jpeg;base64,<...>'"""
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    def extract_landmarks(self, frame):
        """Returns (landmarks_list_len_126, hand_present_bool)."""
        if frame is None:
            return [0.0] * LANDMARKS_PER_FRAME, False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks = []
        hand_present = False
        if result.multi_hand_landmarks:
            hand_present = True
            # Sort hands left-to-right so the same hand tends to fill the same slot
            sorted_hands = sorted(
                result.multi_hand_landmarks, key=lambda h: h.landmark[0].x
            )
            for hand in sorted_hands[:2]:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        while len(landmarks) < LANDMARKS_PER_FRAME:
            landmarks.append(0.0)

        return landmarks[:LANDMARKS_PER_FRAME], hand_present
