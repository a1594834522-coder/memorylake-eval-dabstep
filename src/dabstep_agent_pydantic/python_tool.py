from __future__ import annotations

import contextlib
import io
import sys
import traceback
import types
import threading
from pathlib import Path

from pydantic import BaseModel

from dabstep_agent_pydantic import dabstep_core


class ToolResult(BaseModel):
    ok: bool
    output: str = ""
    error: str = ""
    error_type: str | None = None


# Concurrent tasks run tool calls on executor threads (pydantic-ai runs sync
# tools off the event loop). Executing LLM-written pandas code concurrently
# against shared cached DataFrames triggers native-level races (observed as
# SIGSEGV on long runs), so code execution is globally serialized. Generation
# time dominates execution time, making the throughput cost negligible.
_EXEC_LOCK = threading.Lock()


class PythonWorkspace:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._globals: dict[str, object] = self._initial_globals()
        self._history: list[str] = []

    def _initial_globals(self) -> dict[str, object]:
        helpers = {
            name: getattr(dabstep_core, name)
            for name in getattr(dabstep_core, "__all__", ())
            if hasattr(dabstep_core, name)
        }
        helpers.update(
            {
                "assert_nonempty": assert_nonempty,
                "check_categorical": check_categorical,
            }
        )
        module = types.ModuleType("dabstep")
        for name, value in helpers.items():
            setattr(module, name, value)
        sys.modules["dabstep"] = module
        return {"__builtins__": __builtins__, **helpers}

    def execute(self, code: str) -> ToolResult:
        self._history.append(code.rstrip() + "\n")
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                with _EXEC_LOCK:
                    exec(compile(code, "<agent-python>", "exec"), self._globals, self._globals)
        except SystemExit as exc:
            # LLM code calling exit()/sys.exit() must terminate the snippet,
            # never the host process (observed: a bootstrap solve killed the
            # whole learn run with exit code 0, silently losing all results).
            return ToolResult(
                ok=False,
                output=stdout.getvalue(),
                error=f"SystemExit({exc.code}): do not call exit()/sys.exit(); just end the snippet.",
                error_type="SystemExit",
            )
        except Exception as exc:  # noqa: BLE001 - returned to the model as tool feedback.
            return ToolResult(
                ok=False,
                output=stdout.getvalue(),
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                error_type=type(exc).__name__,
            )
        return ToolResult(ok=True, output=stdout.getvalue())

    def save_generated_code(self, task_id: str) -> Path:
        path = self.workspace_dir / f"{task_id}_solution.py"
        path.write_text("\n".join(self._history))
        return path

    def reset(self) -> None:
        self._globals = self._initial_globals()
        self._history = []


def assert_nonempty(frame, label: str):
    if len(frame) == 0:
        raise ValueError(f"{label} is empty after filtering")
    return frame


def check_categorical(frame, column: str, values):
    if column not in frame.columns:
        raise ValueError(f"column {column!r} not found; available columns: {sorted(map(str, frame.columns))}")
    requested = [values] if isinstance(values, str) else list(values)
    available = sorted(str(value) for value in frame[column].dropna().unique())
    available_set = set(available)
    missing = [str(value) for value in requested if str(value) not in available_set]
    if missing:
        raise ValueError(
            f"{column!r} has no requested value(s) {missing}; available values: {available}"
        )
    return True
