"""WorldForge Studio Launch Script.

Starts:
1. FastAPI Python Backend at http://localhost:8000
2. Vite React Studio Frontend at http://localhost:5173

Automatically selects the project .venv Python interpreter so dependencies
like MuJoCo 3.17, FastAPI, and Stable-Baselines3 resolve seamlessly regardless
of system Python version.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_venv_python() -> str:
    """Find the virtual environment Python executable."""
    win_py = ROOT / ".venv" / "Scripts" / "python.exe"
    nix_py = ROOT / ".venv" / "bin" / "python"
    if win_py.exists():
        return str(win_py)
    if nix_py.exists():
        return str(nix_py)
    return sys.executable


VENV_PYTHON = get_venv_python()


def main():
    # If invoked using a system Python outside .venv (e.g. Python 3.14),
    # cleanly re-delegate to .venv Python (Python 3.12.9) where MuJoCo is installed.
    if Path(sys.executable).resolve() != Path(VENV_PYTHON).resolve() and Path(VENV_PYTHON).exists():
        print(f"[Launcher] Running WorldForge via virtual environment: {VENV_PYTHON}")
        sys.exit(subprocess.call([VENV_PYTHON] + sys.argv))

    print("=" * 65)
    print("  WorldForge Studio 2.0 Launch")
    print("  MuJoCo 3.17 Physics + Baseten meta-models/muse-spark-open-1026")
    print("=" * 65)
    print(f"  Python runtime: {VENV_PYTHON}")

    print("\n[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(ROOT),
    )

    time.sleep(2)

    print("[2/2] Starting React Vite frontend on http://localhost:5173 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(ROOT / "frontend"),
    )

    time.sleep(2)
    print("\n" + "=" * 65)
    print("  WorldForge Studio is LIVE!")
    print("  Frontend UI:  http://localhost:5173")
    print("  Backend API:  http://localhost:8000/docs")
    print("=" * 65)
    print("Press Ctrl+C to terminate both servers.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping WorldForge Studio...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
