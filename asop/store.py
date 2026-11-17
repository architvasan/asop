"""The campaign record store the deterministic DAG nodes read from and write to.

This is the "database" the operator-submitted functions work against. The whole
point (per the operator/engineer split): a production node does NOT ask an LLM to
decide what to run — it issues a *deterministic query* against this store
("designs for target X awaiting QC"), fans them across HPC via `run_batch`, and
writes the results straight back. No skill-following at run time.

SQLite keeps it dependency-free and works on a shared HPC filesystem (WAL mode).
Swap the backend by implementing the same three method groups; nodes depend only
on these signatures, never on SQLite.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
    design_id    TEXT PRIMARY KEY,
    target       TEXT NOT NULL,
    sequence     TEXT NOT NULL,
    backbone_path TEXT,
    method       TEXT,
    status       TEXT NOT NULL DEFAULT 'new',   -- new | queued | evaluated | culled
    created_ts   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS designs_scope ON designs(target, status);

CREATE TABLE IF NOT EXISTS metrics (
    design_id TEXT NOT NULL,
    name      TEXT NOT NULL,
    value     REAL NOT NULL,
    gate_pass INTEGER,                           -- 0/1/NULL
    run_id    TEXT,
    PRIMARY KEY (design_id, name, run_id)
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    step         TEXT NOT NULL,
    status       TEXT NOT NULL,                  -- submitted | complete | failed
    submitted_ts REAL NOT NULL
);
"""


@dataclass
class Store:
    """Thin, deterministic record store. All reads are plain SQL; no LLM, ever."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")       # safe for concurrent HPC writers
        c.execute("PRAGMA foreign_keys=ON")
        return c

    # -- designs ------------------------------------------------------------- #

    def put_designs(self, rows: list[dict[str, Any]]) -> int:
        now = time.time()
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO designs(design_id,target,sequence,backbone_path,method,status,created_ts)"
                " VALUES(:design_id,:target,:sequence,:backbone_path,:method,:status,:ts)",
                [{"backbone_path": None, "method": "", "status": "new", "ts": now, **r} for r in rows],
            )
            return c.total_changes

    def designs(self, target: str, status: str) -> list[dict[str, Any]]:
        """The deterministic work query a node runs before submitting a batch."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT design_id,target,sequence,backbone_path,method,status FROM designs"
                " WHERE target=? AND status=? ORDER BY design_id", (target, status))
            return [dict(r) for r in cur.fetchall()]

    def set_status(self, design_ids: list[str], status: str) -> None:
        with self._conn() as c:
            c.executemany("UPDATE designs SET status=? WHERE design_id=?",
                          [(status, d) for d in design_ids])

    # -- metrics ------------------------------------------------------------- #

    def put_metrics(self, rows: list[dict[str, Any]]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO metrics(design_id,name,value,gate_pass,run_id)"
                " VALUES(:design_id,:name,:value,:gate_pass,:run_id)",
                [{"gate_pass": None, "run_id": "", **r} for r in rows])

    def metrics_for(self, target: str, name: str | None = None) -> list[dict[str, Any]]:
        q = ("SELECT m.* FROM metrics m JOIN designs d ON d.design_id=m.design_id"
             " WHERE d.target=?")
        args: list[Any] = [target]
        if name is not None:
            q += " AND m.name=?"
            args.append(name)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    # -- batches (receipts) -------------------------------------------------- #

    def record_batch(self, batch_id: str, step: str, status: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO batches(batch_id,step,status,submitted_ts)"
                      " VALUES(?,?,?,?)", (batch_id, step, status, time.time()))
