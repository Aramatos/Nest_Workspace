"""
Resumable parameter-sweep manager.

Each parameter combination hashes to a deterministic ID.
Completed runs are stored append-only in JSONL (one line per sim).
Re-running skips combos that already have entries.

Place at:  NEST_Workspace/shared/sweep_manager.py
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone


def param_hash(params: dict) -> str:
    """Deterministic 12-char hex hash of a param dict."""
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


class SweepManager:

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.done: dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.done[rec["id"]] = rec

    def _append(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── public API ───────────────────────────────────────────────
    def is_done(self, params: dict) -> bool:
        return param_hash(params) in self.done

    def get(self, params: dict) -> dict | None:
        """Return stored record for these params, or None."""
        return self.done.get(param_hash(params))

    def record(self, params: dict, metrics: dict, duration_s: float,
               extra: dict | None = None):
        """Append one completed run to the log."""
        pid = param_hash(params)
        rec = {
            "id": pid,
            "params": params,
            "metrics": metrics,
            "duration_s": round(duration_s, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            rec["extra"] = extra
        self._append(rec)
        self.done[pid] = rec

    @property
    def n_done(self) -> int:
        return len(self.done)

    def all_records(self) -> list[dict]:
        return list(self.done.values())
