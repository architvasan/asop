"""How a deterministic node submits work at scale.

A node does not go through a pydantic-ai agent to run science. It calls a
`BatchRunner` directly — Python → HPC — the way loamlite's deployment server runs
`run_batch`, but with the LLM removed because an operator already chose the branch.

**The contract matches loamlite's `run_batch` exactly: one script, run ONCE PER
ROW of a JSONL input, each row in its own folder:**

    python <script.py> input.json         # cwd = a per-row working folder
      - input.json is that one row (a JSON object; carries an id field);
      - the script writes its outputs into the current folder, including a
        `result.json` object, and exits 0 on success;
      - a row that exits non-zero, or writes a `result.json` carrying an
        `"error"` field, is a failed row — it does not travel downstream.

Because `run_batch` copies only the one script file to the worker, a production
script must be SELF-CONTAINED (no local imports). `gpus` rows run side by side on
free tiles, so a one-device script (xpu:0) uses gpus=1 and 12 rows fill a node.

`LocalBatchRunner` runs the script per row in a subprocess (offline, deterministic).
`McpBatchRunner` issues the same script + JSONL to loamlite's running MCP server.
A node is written once against `BatchRunner` and runs in either place.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("asop.batch")

RESULT_FILE = "result.json"


@dataclass
class BatchReceipt:
    batch_id: str
    step: str
    rows: list[dict[str, Any]]      # per-row result dicts (parsed result.json), once complete
    ok: bool
    error: str = ""


def _row_failed(r: dict[str, Any]) -> bool:
    return "error" in r and bool(r.get("error"))


@runtime_checkable
class BatchRunner(Protocol):
    """The one dependency a production node has on the outside world."""

    def submit(self, *, step: str, script: str, rows: list[dict[str, Any]],
               id_field: str = "design_id", venv_path: str | None = None,
               resources: dict | None = None) -> BatchReceipt:
        """Stage the script, run it once per row, and return the completed receipt.

        Synchronous by design: the SOP executor is plain Python and a node blocks
        on its own batch (loamlite's rule: a batch is observed only when it reports
        both complete and ok). Checkpoint/resume lives in the script + the store.
        """
        ...


# --------------------------------------------------------------------------- #
# Local: runs the script per row in a subprocess. For dry runs, tests, laptops.
# --------------------------------------------------------------------------- #


@dataclass
class LocalBatchRunner:
    """Trusted, unsandboxed local execution (same caveat as loamlite local mode).
    Runs rows sequentially for determinism; `python_exe` defaults to the current
    interpreter so stub scripts just work."""

    workdir: Path | None = None
    python_exe: str = sys.executable
    timeout: float = 1800.0
    env: dict[str, str] | None = None     # extra env for the script (e.g. ASOP_QC_MOCK)

    def submit(self, *, step: str, script: str, rows: list[dict[str, Any]],
               id_field: str = "design_id", venv_path: str | None = None,
               resources: dict | None = None) -> BatchReceipt:
        import os

        batch_id = f"local-{step}-{uuid.uuid4().hex[:8]}"
        base = Path(self.workdir) if self.workdir else Path(tempfile.mkdtemp(prefix="asop-batch-"))
        batch_dir = base / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        script_path = batch_dir / "step.py"
        script_path.write_text(script)
        py = (str(Path(venv_path) / "bin" / "python") if venv_path else self.python_exe)
        child_env = {**os.environ, **(self.env or {})}

        results: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            rid = str(row.get(id_field, row.get("id", i)))
            rundir = batch_dir / f"row-{i:04d}"
            rundir.mkdir(exist_ok=True)
            (rundir / "input.json").write_text(json.dumps(row))
            try:
                proc = subprocess.run([py, str(script_path), "input.json"],
                                      capture_output=True, text=True, timeout=self.timeout,
                                      cwd=rundir, env=child_env)
            except subprocess.TimeoutExpired:
                results.append({id_field: rid, "error": "timeout"})
                continue
            res_path = rundir / RESULT_FILE
            if proc.returncode != 0 and not res_path.is_file():
                results.append({id_field: rid, "error": f"exit {proc.returncode}: {proc.stderr[-400:]}"})
                continue
            if not res_path.is_file():
                results.append({id_field: rid, "error": "no result.json written"})
                continue
            r = json.loads(res_path.read_text())
            r.setdefault(id_field, rid)
            results.append(r)

        ok = bool(results) and not any(_row_failed(r) for r in results)
        return BatchReceipt(batch_id, step, results, ok=ok,
                            error="" if ok else "one or more rows failed")


# --------------------------------------------------------------------------- #
# HPC: the same script + JSONL, issued to loamlite's deployment MCP server.
# --------------------------------------------------------------------------- #


@dataclass
class McpBatchRunner:
    """Direct MCP client to the loamlite-managed hpc-deployment server.

    Note the inversion vs. loamlite: there the *code-generator agent* calls
    `run_batch` as a tool. Here asop (plain Python, no agent) is the MCP client,
    because the branch decision was already made by the operator. The call shape is
    identical, so a node body is portable between the two.

    Thin, documented adapter — fastmcp is an optional (`[hpc]`) extra, imported
    lazily so the local path never needs it. The per-row result harvest reads each
    row folder's `result.json`; prefer binding the script with `register_tool`
    (its `expects` contract makes a row that skips its outputs count as failed).
    """

    mcp_url: str
    venv_path: str
    poll_seconds: float = 15.0
    max_wait: float = 6 * 3600.0
    gpus: float = 1

    def submit(self, *, step: str, script: str, rows: list[dict[str, Any]],
               id_field: str = "design_id", venv_path: str | None = None,
               resources: dict | None = None) -> BatchReceipt:
        import time

        from fastmcp import Client  # lazy: only needed in HPC mode

        batch_id = f"{step}-{uuid.uuid4().hex[:8]}"
        jsonl = "".join(json.dumps(r) + "\n" for r in rows)
        with Client(self.mcp_url) as mcp:
            # 1. stage script + rows (loamlite: write_files)
            mcp.call_tool("write_files", {
                "files": {f"{batch_id}/step.py": script, f"{batch_id}/inputs.jsonl": jsonl}})
            # 2. one run_batch: the script runs once per row across free tiles
            sub = mcp.call_tool("run_batch", {
                "script_path": f"{batch_id}/step.py",
                "inputs_path": f"{batch_id}/inputs.jsonl",
                "venv_path": venv_path or self.venv_path,
                "gpus": (resources or {}).get("gpus", self.gpus),
            }).data
            server_batch = sub.get("batch_id", batch_id)
            # 3. poll until complete (loamlite: observe only on complete && ok)
            deadline = time.time() + self.max_wait
            while time.time() < deadline:
                st = mcp.call_tool("batch_status", {"batch_id": server_batch}).data
                if st.get("complete"):
                    rows_out = self._harvest(mcp, server_batch, st, id_field)
                    ok = bool(st.get("ok")) and bool(rows_out) and not any(_row_failed(r) for r in rows_out)
                    return BatchReceipt(server_batch, step, rows_out, ok=ok,
                                        error="" if ok else st.get("error", "batch not ok"))
                time.sleep(self.poll_seconds)
        return BatchReceipt(batch_id, step, [], ok=False, error="max_wait exceeded")

    def _harvest(self, mcp, server_batch: str, status: dict, id_field: str) -> list[dict[str, Any]]:
        """Collect each row's result.json. If batch_status already carries the
        per-row results, use them; otherwise read result.json from each run dir."""
        if status.get("results"):
            return list(status["results"])
        rows: list[dict[str, Any]] = []
        for rundir in status.get("run_dirs", []):
            got = mcp.call_tool("inspect_filesystem", {"path": f"{rundir}/{RESULT_FILE}"}).data
            text = got.get("content") if isinstance(got, dict) else None
            if text:
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    rows.append({id_field: rundir, "error": "unparseable result.json"})
        return rows
