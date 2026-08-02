import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ---------------------------------------------------------------- cache ----
# Precompute and cache numpy arrays / mean vectors per user so we don't
# redo that work on every single incoming video frame. Invalidated by
# database.py whenever that user's gestures/samples change.
_precomputed = {}  # user_id -> {gesture_id: [(sample_arr, mean_vec), ...]}


def invalidate_user_cache(user_id):
    _precomputed.pop(user_id, None)


def _build_cache(user_id, samples_map):
    entry = {}
    for gesture_id, samples in samples_map.items():
        pairs = []
        for s in samples:
            arr = np.array(s)
            pairs.append((arr, np.mean(arr, axis=0)))
        entry[gesture_id] = pairs
    _precomputed[user_id] = entry
    return entry


def match_gesture(user_id, live_sequence, samples_map, threshold=12.0, margin_ratio=0.85):
    """
    user_id: cache key for precomputed sample data
    live_sequence: list of frames captured just now (each frame = 126 floats)
    samples_map: dict {gesture_id: [sample_sequence, ...]}
    threshold: lower = stricter match
    margin_ratio: best/second_best must be <= this to accept (ambiguity guard)

    Returns (best_gesture_id_or_None, best_distance)
    """
    if not samples_map:
        return None, float("inf")

    precomputed = _precomputed.get(user_id)
    if precomputed is None:
        precomputed = _build_cache(user_id, samples_map)

    live_arr = np.array(live_sequence)
    live_mean = np.mean(live_arr, axis=0)

    quick_scores = []
    for gesture_id, pairs in precomputed.items():
        best_quick = min(euclidean(live_mean, mean_vec) for _, mean_vec in pairs)
        quick_scores.append((best_quick, gesture_id))
    quick_scores.sort(key=lambda pair: pair[0])

    cutoff = quick_scores[0][0] * 3.0 + 0.05
    shortlist = [gid for score, gid in quick_scores if score <= cutoff]

    gesture_best = {}
    for gesture_id in shortlist:
        local_best = float("inf")
        for sample_arr, _ in precomputed[gesture_id]:
            distance, _ = fastdtw(live_arr, sample_arr, dist=euclidean)
            normalized = distance / max(len(sample_arr), 1)
            if normalized < local_best:
                local_best = normalized
        gesture_best[gesture_id] = local_best

    if not gesture_best:
        return None, float("inf")

    ranked = sorted(gesture_best.items(), key=lambda kv: kv[1])
    best_id, best_distance = ranked[0]

    if best_distance >= threshold:
        return None, best_distance

    if len(ranked) > 1:
        second_distance = ranked[1][1]
        if second_distance > 0 and (best_distance / second_distance) > margin_ratio:
            return None, best_distance

    return best_id, best_distance
