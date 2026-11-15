"""
database.py - Database access layer for AURASKIN.
Manages Users, Sessions, and SkinAnalysisHistory tables with SQLite.
Ensures zero data loss by migrating existing users from users.json.
"""

import os
import json
import sqlite3
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(CURRENT_DIR, "auraskin.db")
USERS_JSON_FILE = os.path.join(CURRENT_DIR, "users.json")
SEED_USERS_FILE = os.path.join(CURRENT_DIR, "users.demo.json") 
UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")

os.makedirs(UPLOADS_DIR, exist_ok=True)


def get_db_connection():
    """Create a thread-safe connection to auraskin.db with foreign keys enabled."""
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """
    Initializes database schema and performs safe migration from users.json.
    Guarantees no existing user data is lost.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id TEXT PRIMARY KEY,
                fullName TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                createdAt TEXT NOT NULL
            );
        """)

        # 2. SkinAnalysisHistory table (with foreign key to Users)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SkinAnalysisHistory (
                id TEXT PRIMARY KEY,
                userId TEXT NOT NULL,
                imageUrl TEXT,
                detectedLabels TEXT,
                analysisResult TEXT NOT NULL,
                createdAt TEXT NOT NULL,
                FOREIGN KEY (userId) REFERENCES Users(id) ON DELETE CASCADE
            );
        """)

        # 3. Sessions table for secure backend authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Sessions (
                token TEXT PRIMARY KEY,
                userId TEXT NOT NULL,
                createdAt TEXT NOT NULL,
                expiresAt TEXT NOT NULL,
                FOREIGN KEY (userId) REFERENCES Users(id) ON DELETE CASCADE
            );
        """)

        # Create indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON Users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON SkinAnalysisHistory(userId);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON Sessions(token);")

        conn.commit()

    # 4. Safe migration from users.json if present
    migrate_from_users_json()


def migrate_from_users_json():
    """Nạp tài khoản từ users.json (máy cá nhân) và users.demo.json (tài khoản mẫu)
    vào bảng Users, không tạo trùng."""
    for path in (USERS_JSON_FILE, SEED_USERS_FILE):
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
        except Exception:
            continue

        with get_db_connection() as conn:
            cursor = conn.cursor()
            for u in data:
                uid = u.get("id") or f"usr_{int(time.time() * 1000)}"
                fname = u.get("fullName", "").strip()
                email = u.get("email", "").strip().lower()
                pwd = u.get("password", "")
                created = u.get("createdAt") or datetime.now(timezone.utc).isoformat()
                if email and pwd:
                    cursor.execute("""
                        INSERT OR IGNORE INTO Users (id, fullName, email, password, createdAt)
                        VALUES (?, ?, ?, ?, ?)
                    """, (uid, fname, email, pwd, created))
            conn.commit()


def sync_users_to_json():
    """Sync current Users table back to users.json for external backwards compatibility."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fullName, email, password, createdAt FROM Users ORDER BY createdAt ASC")
            rows = cursor.fetchall()
            users_list = [dict(r) for r in rows]

        with open(USERS_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(users_list, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ==========================================================
# USER & AUTH OPERATIONS
# ==========================================================

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    norm_email = email.strip().lower()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, fullName, email, password, createdAt FROM Users WHERE LOWER(email) = ?", (norm_email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, fullName, email, password, createdAt FROM Users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_user(full_name: str, email: str, password: str) -> Dict[str, Any]:
    user_id = f"usr_{int(time.time() * 1000)}_{secrets.token_hex(3)}"
    created_at = datetime.now(timezone.utc).isoformat()
    norm_email = email.strip().lower()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Users (id, fullName, email, password, createdAt)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, full_name.strip(), norm_email, password, created_at))
        conn.commit()

    sync_users_to_json()

    return {
        "id": user_id,
        "fullName": full_name.strip(),
        "email": norm_email,
        "createdAt": created_at
    }


def create_session(user_id: str, days_valid: int = 7) -> str:
    token = secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=days_valid)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Sessions (token, userId, createdAt, expiresAt)
            VALUES (?, ?, ?, ?)
        """, (token, user_id, now.isoformat(), expires.isoformat()))
        conn.commit()

    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.fullName, u.email, u.createdAt
            FROM Sessions s
            JOIN Users u ON s.userId = u.id
            WHERE s.token = ? AND s.expiresAt > ?
        """, (token.strip(), now_iso))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> bool:
    if not token:
        return False
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Sessions WHERE token = ?", (token.strip(),))
        conn.commit()
        return cursor.rowcount > 0


# ==========================================================
# SKIN ANALYSIS HISTORY OPERATIONS
# ==========================================================

def create_analysis_history(
    user_id: str,
    image_url: str,
    detected_labels: List[str],
    analysis_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Saves a skin analysis session for a specific user into SkinAnalysisHistory."""
    history_id = f"hist_{int(time.time() * 1000)}_{secrets.token_hex(3)}"
    created_at = datetime.now(timezone.utc).isoformat()
    labels_json = json.dumps(detected_labels, ensure_ascii=False)
    result_json = json.dumps(analysis_result, ensure_ascii=False)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO SkinAnalysisHistory (id, userId, imageUrl, detectedLabels, analysisResult, createdAt)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (history_id, user_id, image_url, labels_json, result_json, created_at))
        conn.commit()

    return {
        "id": history_id,
        "userId": user_id,
        "imageUrl": image_url,
        "detectedLabels": detected_labels,
        "analysisResult": analysis_result,
        "createdAt": created_at
    }


def get_user_analysis_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves analysis history for a specific user, strictly isolated from other users."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, userId, imageUrl, detectedLabels, analysisResult, createdAt
            FROM SkinAnalysisHistory
            WHERE userId = ?
            ORDER BY createdAt DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()

    history = []
    for r in rows:
        try:
            labels = json.loads(r["detectedLabels"]) if r["detectedLabels"] else []
        except Exception:
            labels = []
        try:
            res_obj = json.loads(r["analysisResult"]) if r["analysisResult"] else {}
        except Exception:
            res_obj = {}

        history.append({
            "id": r["id"],
            "userId": r["userId"],
            "imageUrl": r["imageUrl"],
            "detectedLabels": labels,
            "analysisResult": res_obj,
            "createdAt": r["createdAt"]
        })
    return history


def get_analysis_history_item(history_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single analysis history item strictly checking user ownership."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, userId, imageUrl, detectedLabels, analysisResult, createdAt
            FROM SkinAnalysisHistory
            WHERE id = ? AND userId = ?
        """, (history_id, user_id))
        r = cursor.fetchone()
        if not r:
            return None

    try:
        labels = json.loads(r["detectedLabels"]) if r["detectedLabels"] else []
    except Exception:
        labels = []
    try:
        res_obj = json.loads(r["analysisResult"]) if r["analysisResult"] else {}
    except Exception:
        res_obj = {}

    return {
        "id": r["id"],
        "userId": r["userId"],
        "imageUrl": r["imageUrl"],
        "detectedLabels": labels,
        "analysisResult": res_obj,
        "createdAt": r["createdAt"]
    }


def delete_analysis_history_item(history_id: str, user_id: str) -> bool:
    """Deletes a single history item verifying user ownership."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM SkinAnalysisHistory
            WHERE id = ? AND userId = ?
        """, (history_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


def clear_user_analysis_history(user_id: str) -> int:
    """Clears all history items belonging to the specified user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM SkinAnalysisHistory WHERE userId = ?", (user_id,))
        conn.commit()
        return cursor.rowcount
