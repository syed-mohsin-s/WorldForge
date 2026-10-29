"""WorldForge Studio Launch Script.

Starts:
1. FastAPI Python Backend at http://localhost:8000
2. Vite React Studio Frontend at http://localhost:5173
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def main():
    print("=" * 65)
    print("  WorldForge Studio 2.0 Launch")
    print("  MuJoCo 3.17 Physics + Baseten meta-models/muse-spark-open-1026")
    print("=" * 65)

    print("[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"],
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
    print("Press Ctrl+C to terminate both servers.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping WorldForge Studio...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
