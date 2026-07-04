"""Launcher for the CoED-GNN Cora reproduction."""

import os
import subprocess
import sys


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = os.environ.copy()
    env.setdefault("TL_BACKEND", "torch")
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "coed_trainer.py")] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd, env=env, cwd=root))


if __name__ == "__main__":
    main()
