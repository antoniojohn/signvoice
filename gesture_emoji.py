import logging

log = logging.getLogger("signvoice")

# MediaPipe Hands landmark layout: 21 points per hand.
#   0            = wrist
#   1,2,3,4      = thumb  (4 = tip)
#   5,6,7,8      = index  (8 = tip)
#   9,10,11,12   = middle (12 = tip)
#   13,14,15,16  = ring   (16 = tip)
#   17,18,19,20  = pinky  (20 = tip)
WRIST = 0
FINGER_JOINTS = {
    "thumb": (2, 4),   # (pip-ish joint, tip)
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}


def _dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _hand_points(frame):
    """frame is 126 floats: 2 hands x 21 landmarks x (x, y, z), normalized in
    hand_tracker.js. Returns the first hand's 21 (x, y, z) points, or None if
    that hand slot is empty (all zeros -> no hand detected in this frame)."""
    hand = frame[:63]
    if all(v == 0.0 for v in hand):
        return None
    return [tuple(hand[i:i + 3]) for i in range(0, 63, 3)]


def classify_shape(frame):
    """Looks at one frame's hand landmarks and returns a best-guess emoji for
    the finger shape, or None if it doesn't confidently match a known shape."""
    points = _hand_points(frame)
    if points is None:
        return None

    wrist = points[WRIST]
    extended = {}
    for name, (mid_idx, tip_idx) in FINGER_JOINTS.items():
        if name == "thumb":
            continue  # handled separately below — this method is unreliable for the thumb
        extended[name] = _dist(points[tip_idx], wrist) > _dist(points[mid_idx], wrist) * 1.15

    # The thumb barely changes its distance from the wrist whether it's
    # tucked in or held out to the side — that's what made 1 vs 7 unreliable.
    # Instead, compare how far the thumb tip is from the base of the pinky
    # against the palm's own width: a thumb held out to the side ends up
    # much farther from the pinky-base than the palm itself is wide, while
    # a tucked thumb stays roughly within that span regardless of hand
    # rotation, size, or camera distance.
    palm_width = _dist(points[5], points[17])  # index base -> pinky base
    thumb_spread = _dist(points[4], points[17])  # thumb tip -> pinky base
    extended["thumb"] = palm_width > 0 and thumb_spread > palm_width * 1.15

    # OK sign is a special case: the thumb curls toward the INDEX finger
    # (not toward the palm/pinky side), so the general thumb check above
    # misreads it as "extended." Detect it directly via thumb-to-index-tip
    # distance instead, checked before the general lookup.
    if palm_width > 0:
        thumb_to_index = _dist(points[4], points[8])
        if thumb_to_index < palm_width * 0.5 and extended["middle"] and extended["ring"] and extended["pinky"]:
            return "👌"

    key = (
        extended["thumb"], extended["index"], extended["middle"],
        extended["ring"], extended["pinky"],
    )
    return SHAPE_PATTERNS.get(key)


# Each key is (thumb, index, middle, ring, pinky) — True where that finger
# is extended. Counting 6-9 builds up thumb, then thumb+index, etc.
SHAPE_PATTERNS = {
    (False, False, False, False, False): "✊",    # fist
    (False, True, False, False, False): "☝️",    # 1 — index only
    (False, True, True, False, False): "✌️",     # 2 — index + middle
    (False, True, True, True, False): "3️⃣",     # 3 — index + middle + ring
    (False, True, True, True, True): "4️⃣",      # 4 — four fingers, thumb tucked
    (True, True, True, True, True): "🖐️",        # 5 — all five spread
    (True, False, False, False, False): "👍",   # 6 — thumb only
    (True, True, False, False, False): "👆",    # 7 — thumb + index ("L" shape)
    (True, True, True, False, False): "8️⃣",     # 8 — thumb + index + middle
    (True, True, True, True, False): "9️⃣",      # 9 — thumb + index + middle + ring
    (True, True, False, False, True): "🤟",      # I love you — thumb + index + pinky
    # 👌 OK sign is handled above via thumb-to-index touch detection, not this table
    (False, True, False, False, True): "🤘",     # rock on — index + pinky
    (True, False, False, False, True): "🤙",     # call-me / shaka — thumb + pinky
}

# Every seal a sign can end up with — used to build the manual picker on the
# edit-gesture page. "✨" (the placeholder shown before any shape has been
# detected yet) is included first since it's a valid choice to reset back to.
# It's deliberately not a hand shape — 🖐️ (open palm, the "5" seal) already
# lives in this same pack, and reusing a hand for "not yet classified" made
# the two impossible to tell apart at a glance.
PRESET_EMOJIS = ["✨"] + list(dict.fromkeys(SHAPE_PATTERNS.values())) + ["👌", "🖕"]


def emoji_from_samples(sample_sequences):
    """sample_sequences: list of recorded sample sequences (each a list of
    landmark frames, as stored per gesture). Classifies several frames per
    sample (not just one) and returns the overall majority-vote emoji, or
    None if no frame gave a confident match."""
    votes = {}
    for seq in sample_sequences:
        if not seq:
            continue
        # Middle 60% of frames tends to be the held pose — the very start
        # and end of a recording often catch the hand still moving in/out.
        start = len(seq) // 5
        end = len(seq) - start
        for frame in seq[start:end] or seq:
            result = classify_shape(frame)
            if result:
                votes[result] = votes.get(result, 0) + 1

    if not votes:
        return None
    winner, _count = max(votes.items(), key=lambda kv: kv[1])
    return winner
