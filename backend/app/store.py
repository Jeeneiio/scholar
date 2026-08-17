"""Persistent storage for session and file metadata.

The original project used PostgreSQL in production and dictionaries in
``memory`` mode.  SQLite is a good fit for the server setup used here:
it stores everything in one local file and does not require Docker or a
separate database service.
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Optional

from psycopg_pool import AsyncConnectionPool

_pool: Optional[AsyncConnectionPool] = None
_sqlite_conn: Optional[sqlite3.Connection] = None
_storage_mode = "memory"
_sqlite_lock = asyncio.Lock()

# These dictionaries are intentionally kept only for the explicit memory mode.
_sessions: dict[str, dict] = {}
_files: dict[str, dict] = {}

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)"""

_FILES_DDL = """
CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
)"""


async def init_store(
    pool: Optional[AsyncConnectionPool],
    sqlite_path: Optional[str] = None,
    mode: str = "memory",
):
    """Initialise the selected metadata backend.

    ``sqlite`` is deliberately separate from the PostgreSQL pool because the
    two databases use different SQL placeholder styles (``?`` vs ``%s``).
    """
    global _pool, _sqlite_conn, _storage_mode
    _pool = pool
    _storage_mode = mode

    if mode == "sqlite":
        if not sqlite_path:
            raise ValueError("sqlite_path is required when storage mode is sqlite")
        path = Path(sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_conn = sqlite3.connect(path, check_same_thread=False)
        _sqlite_conn.row_factory = sqlite3.Row
        _sqlite_conn.execute(_SESSIONS_DDL)
        _sqlite_conn.execute(_FILES_DDL)
        _sqlite_conn.commit()
        return

    _sqlite_conn = None
    if mode == "memory":
        _sessions.clear()
        _files.clear()
        return

    if pool is None:
        raise ValueError("PostgreSQL storage requires a connection pool")
    async with pool.connection() as conn:
        await conn.execute(_SESSIONS_DDL)
        await conn.execute(_FILES_DDL)
        await conn.commit()


def _sqlite() -> sqlite3.Connection:
    if _sqlite_conn is None:
        raise RuntimeError("SQLite store is not initialised")
    return _sqlite_conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


def _get_pool() -> AsyncConnectionPool:
    assert _pool is not None, "store not initialised — call init_store first"
    return _pool


# ── sessions ─────────────────────────────────────────────

async def create_session(session_id: str, title: str = "") -> dict:
    now = time.time()
    if _storage_mode == "memory":
        record = {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}
        _sessions.setdefault(session_id, record)
        return _sessions[session_id]
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            conn = _sqlite()
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return dict(row)
    async with _get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
            (session_id, title, now, now),
        )
        await conn.commit()
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


async def update_session(session_id: str, title: Optional[str] = None) -> bool:
    now = time.time()
    if _storage_mode == "memory":
        if session_id not in _sessions:
            return False
        _sessions[session_id]["updated_at"] = now
        if title is not None:
            _sessions[session_id]["title"] = title
        return True
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            conn = _sqlite()
            if title is None:
                cur = conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            else:
                cur = conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            conn.commit()
            return cur.rowcount > 0
    parts, vals = ["updated_at = %s"], [now]
    if title is not None:
        parts.extend(["title = %s"])
        vals.append(title)
    vals.append(session_id)
    async with _get_pool().connection() as conn:
        cur = await conn.execute(f"UPDATE sessions SET {', '.join(parts)} WHERE session_id = %s", vals)
        await conn.commit()
        return cur.rowcount > 0


async def list_sessions() -> list[dict]:
    if _storage_mode == "memory":
        return sorted(_sessions.values(), key=lambda x: x["updated_at"], reverse=True)
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            return _rows_to_dicts(_sqlite().execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall())
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        return [dict(zip([d.name for d in cur.description], row)) for row in await cur.fetchall()]


async def get_session(session_id: str) -> Optional[dict]:
    if _storage_mode == "memory":
        return _sessions.get(session_id)
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            return _row_to_dict(_sqlite().execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone())
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
        row = await cur.fetchone()
        return dict(zip([d.name for d in cur.description], row)) if row else None


async def delete_session(session_id: str) -> bool:
    if _storage_mode == "memory":
        return _sessions.pop(session_id, None) is not None
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            cur = _sqlite().execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            _sqlite().commit()
            return cur.rowcount > 0
    async with _get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        await conn.commit()
        return cur.rowcount > 0


# ── files ────────────────────────────────────────────────

async def add_file(
    file_id: str, filename: str, paper_id: str,
    size_bytes: int = 0, page_count: int = 0, chunk_count: int = 0,
) -> dict:
    now = time.time()
    record = {
        "file_id": file_id, "filename": filename, "paper_id": paper_id,
        "size_bytes": size_bytes, "page_count": page_count,
        "chunk_count": chunk_count, "created_at": now,
    }
    if _storage_mode == "memory":
        _files[file_id] = record
        return record
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            conn = _sqlite()
            conn.execute(
                "INSERT OR REPLACE INTO files "
                "(file_id, filename, paper_id, size_bytes, page_count, chunk_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, filename, paper_id, size_bytes, page_count, chunk_count, now),
            )
            conn.commit()
        return record
    async with _get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO files (file_id, filename, paper_id, size_bytes, page_count, chunk_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (file_id) DO UPDATE SET filename=EXCLUDED.filename, paper_id=EXCLUDED.paper_id, "
            "size_bytes=EXCLUDED.size_bytes, page_count=EXCLUDED.page_count, chunk_count=EXCLUDED.chunk_count",
            (file_id, filename, paper_id, size_bytes, page_count, chunk_count, now),
        )
        await conn.commit()
    return record


async def list_files() -> list[dict]:
    if _storage_mode == "memory":
        return sorted(_files.values(), key=lambda x: x["created_at"], reverse=True)
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            return _rows_to_dicts(_sqlite().execute("SELECT * FROM files ORDER BY created_at DESC").fetchall())
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files ORDER BY created_at DESC")
        return [dict(zip([d.name for d in cur.description], row)) for row in await cur.fetchall()]


async def get_file(file_id: str) -> Optional[dict]:
    if _storage_mode == "memory":
        return _files.get(file_id)
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            return _row_to_dict(_sqlite().execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone())
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        return dict(zip([d.name for d in cur.description], row)) if row else None


async def delete_file_record(file_id: str) -> Optional[dict]:
    if _storage_mode == "memory":
        return _files.pop(file_id, None)
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            conn = _sqlite()
            row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
            if row:
                conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
                conn.commit()
            return _row_to_dict(row)
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        if not row:
            return None
        record = dict(zip([d.name for d in cur.description], row))
        await conn.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
        await conn.commit()
        return record


async def clear_all_files() -> int:
    if _storage_mode == "memory":
        count = len(_files)
        _files.clear()
        return count
    if _storage_mode == "sqlite":
        async with _sqlite_lock:
            conn = _sqlite()
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.execute("DELETE FROM files")
            conn.commit()
            return count
    async with _get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM files")
        await conn.commit()
        return cur.rowcount
