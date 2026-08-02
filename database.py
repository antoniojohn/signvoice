import sqlite3
import json
import logging
from werkzeug.security import generate_password_hash, check_password_hash

log = logging.getLogger("signvoice")

DB_PATH = "signlang.db"

# --------------------------------------------------------------- cache -----
_samples_cache = {}  # user_id -> (samples_map, lookup)


def invalidate_samples_cache(user_id):
    _samples_cache.pop(user_id, None)
    try:
        import matcher
        matcher.invalidate_user_cache(user_id)
    except ImportError:
        pass


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            gesture_name TEXT NOT NULL,
            text_message TEXT NOT NULL,
            emoji TEXT DEFAULT '✨',
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS gesture_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture_id INTEGER NOT NULL,
            sequence_json TEXT NOT NULL,
            FOREIGN KEY(gesture_id) REFERENCES gestures(id)
        );
        """
    )
    conn.commit()

    # Migration for DBs created before the emoji column existed.
    existing_cols = [row["name"] for row in conn.execute("PRAGMA table_info(gestures)")]
    if "emoji" not in existing_cols:
        conn.execute("ALTER TABLE gestures ADD COLUMN emoji TEXT DEFAULT '✨'")
        conn.commit()
        log.info("Migrated gestures table: added emoji column")

    conn.close()
    log.info("Database ready at %s", DB_PATH)


def create_user(username, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return row["id"]
    return None


def add_gesture(user_id, gesture_name, text_message, emoji="✨"):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO gestures (user_id, gesture_name, text_message, emoji) VALUES (?, ?, ?, ?)",
        (user_id, gesture_name, text_message, emoji or "✨"),
    )
    conn.commit()
    gesture_id = cur.lastrowid
    conn.close()
    invalidate_samples_cache(user_id)
    log.info("Gesture '%s' added for user %s", gesture_name, user_id)
    return gesture_id


def add_sample(gesture_id, sequence, user_id=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO gesture_samples (gesture_id, sequence_json) VALUES (?, ?)",
        (gesture_id, json.dumps(sequence)),
    )
    conn.commit()
    if user_id is None:
        row = conn.execute("SELECT user_id FROM gestures WHERE id = ?", (gesture_id,)).fetchone()
        user_id = row["user_id"] if row else None
    conn.close()
    if user_id is not None:
        invalidate_samples_cache(user_id)


def get_samples_for_gesture(gesture_id):
    """Returns all recorded landmark sequences for one gesture (decoded from JSON)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT sequence_json FROM gesture_samples WHERE gesture_id = ?", (gesture_id,)
    ).fetchall()
    conn.close()
    return [json.loads(r["sequence_json"]) for r in rows]


def update_gesture_emoji(gesture_id, emoji):
    """Overwrites a gesture's seal — used once we've looked at its actual
    recorded hand shape and worked out a better-fitting emoji."""
    conn = get_db()
    row = conn.execute("SELECT user_id FROM gestures WHERE id = ?", (gesture_id,)).fetchone()
    conn.execute("UPDATE gestures SET emoji = ? WHERE id = ?", (emoji, gesture_id))
    conn.commit()
    conn.close()
    if row is not None:
        invalidate_samples_cache(row["user_id"])


def update_gesture(gesture_id, user_id, gesture_name, text_message, emoji):
    """Edits a gesture's name, message, and seal — only if it belongs to
    user_id. Returns True if a row was actually updated."""
    conn = get_db()
    cur = conn.execute(
        "UPDATE gestures SET gesture_name = ?, text_message = ?, emoji = ? "
        "WHERE id = ? AND user_id = ?",
        (gesture_name, text_message, emoji, gesture_id, user_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    if updated:
        invalidate_samples_cache(user_id)
    return updated


def get_user_gesture_emojis(user_id, exclude_gesture_id=None):
    """Lightweight lookup of emojis in use across a user's signs — for
    building seal pickers. Skips the sample_count subquery that
    get_user_gestures() runs (not needed here), and excludes one gesture
    in SQL rather than filtering it out in Python afterward."""
    conn = get_db()
    if exclude_gesture_id is not None:
        rows = conn.execute(
            "SELECT emoji FROM gestures WHERE user_id = ? AND id != ? AND emoji IS NOT NULL",
            (user_id, exclude_gesture_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT emoji FROM gestures WHERE user_id = ? AND emoji IS NOT NULL", (user_id,)
        ).fetchall()
    conn.close()
    return {row["emoji"] for row in rows}


def get_user_gestures(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT g.*, (SELECT COUNT(*) FROM gesture_samples s WHERE s.gesture_id = g.id) as sample_count "
        "FROM gestures g WHERE g.user_id = ? ORDER BY g.id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def get_gesture_samples_map(user_id):
    """Returns (samples_by_gesture_id, info_by_gesture_id), cached per user."""
    cached = _samples_cache.get(user_id)
    if cached is not None:
        return cached

    conn = get_db()
    gestures = conn.execute("SELECT * FROM gestures WHERE user_id = ?", (user_id,)).fetchall()
    samples_map, lookup = {}, {}
    for g in gestures:
        rows = conn.execute(
            "SELECT sequence_json FROM gesture_samples WHERE gesture_id = ?", (g["id"],)
        ).fetchall()
        if rows:
            samples_map[g["id"]] = [json.loads(r["sequence_json"]) for r in rows]
            lookup[g["id"]] = {
                "name": g["gesture_name"],
                "text": g["text_message"],
                "emoji": g["emoji"] if "emoji" in g.keys() else "✨",
            }
    conn.close()

    _samples_cache[user_id] = (samples_map, lookup)
    return samples_map, lookup


def delete_gesture(gesture_id, user_id):
    conn = get_db()
    conn.execute("DELETE FROM gesture_samples WHERE gesture_id = ?", (gesture_id,))
    conn.execute("DELETE FROM gestures WHERE id = ? AND user_id = ?", (gesture_id, user_id))
    conn.commit()
    conn.close()
    invalidate_samples_cache(user_id)
    log.info("Gesture %s deleted for user %s", gesture_id, user_id)


def get_gesture(gesture_id, user_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM gestures WHERE id = ? AND user_id = ?", (gesture_id, user_id)
    ).fetchone()
    conn.close()
    return row
