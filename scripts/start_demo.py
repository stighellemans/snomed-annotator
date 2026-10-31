"""Start the built ZAS demo in the background; safe to run more than once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:5180"


def ready() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=2) as response:
            data = json.load(response)
        return data.get("app") == "snomed-annotation-zas" and data.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not ready():
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 5180)) == 0:
                raise SystemExit("Poort 5180 is al in gebruik door een andere server.")
        python = ROOT / "backend" / ".venv" / "bin" / "python"
        if not python.exists():
            raise SystemExit("Python-omgeving ontbreekt. Zie README.md voor de installatie.")
        if not (ROOT / "frontend" / "dist" / "index.html").exists():
            raise SystemExit("Frontend ontbreekt. Voer in frontend eerst npm ci en npm run build uit.")
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        log_path = runtime / "demo.log"
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        with log_path.open("a") as log:
            process = subprocess.Popen(
                [str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "5180"],
                cwd=ROOT / "backend", env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        (runtime / "demo.pid").write_text(str(process.pid))
        print("ZAS-demo wordt gestart; de zoekindex wordt geladen…", flush=True)
        for _ in range(120):
            if ready():
                break
            if process.poll() is not None:
                raise SystemExit(f"Opstarten mislukt. Zie {log_path}")
            time.sleep(1)
        else:
            raise SystemExit(f"Server is nog niet klaar. Zie {log_path}")
    print(f"ZAS-demo klaar: {URL}", flush=True)
    if not args.no_browser:
        webbrowser.open(URL)


if __name__ == "__main__":
    main()
