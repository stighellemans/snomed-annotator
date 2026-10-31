"""Check prerequisites, build the index when needed, and start the Docker app.

Only Python's standard library is needed on the host. No Python/Node environment
has to be installed for the application itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", *args], cwd=ROOT, check=check,
                          capture_output=capture, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Alleen vereisten controleren; niets bouwen of starten.")
    parser.add_argument("--rebuild-index", action="store_true", help="Index opnieuw opbouwen, bijvoorbeeld na een RF2-update.")
    parser.add_argument("--skip-llm-check", action="store_true", help="Geen synthetische API-test uitvoeren.")
    args = parser.parse_args()
    try:
        port = int(os.environ.get("APP_PORT", "5180"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        print("FOUT: APP_PORT moet een geldig poortnummer zijn.")
        return 1
    url = f"http://127.0.0.1:{port}"
    if not shutil.which("docker"):
        print("FOUT: installeer Docker met Docker Compose en start Docker.")
        return 1
    for command in [["docker", "info", "--format", "{{.ServerVersion}}"], ["docker", "compose", "version"]]:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print("FOUT: Docker met Compose is niet beschikbaar. Start Docker en probeer opnieuw.")
            return 1
    env_file = ROOT / "backend" / ".env"
    if not env_file.exists():
        if not args.check:
            shutil.copyfile(ROOT / "backend" / ".env.example", env_file)
            env_file.chmod(0o600)
        print("Vul eerst de LLM-instellingen in backend/.env in (zie backend/.env.example).")
        return 1
    releases = list((ROOT / "snomed").glob("*.zip"))
    if len(releases) != 1:
        print("FOUT: plaats precies één SNOMED CT International RF2-release-zip in snomed/.")
        return 1
    resolved = compose("config", "--format", "json", check=False, capture=True)
    if resolved.returncode:
        print("FOUT: Docker kan de configuratie niet lezen; controleer backend/.env en compose.yaml.")
        return 1
    # Read only the selected backend from resolved configuration; never print
    # this object because Compose includes credentials from env_file in it.
    index_backend = json.loads(resolved.stdout)["services"]["app"].get("environment", {}).get("DEFAULT_EMBEDDER", "biolord")
    if index_backend not in {"biolord", "openai"}:
        print("FOUT: DEFAULT_EMBEDDER moet biolord of openai zijn.")
        return 1
    with socket.socket() as probe:
        occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    if occupied:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=3) as response:
                identity = json.load(response)
            if identity.get("app") != "snomed-annotator":
                raise ValueError
        except (OSError, ValueError):
            print(f"FOUT: poort {port} wordt door een andere toepassing gebruikt. Kies bijvoorbeeld APP_PORT=5181.")
            return 1
    print("Docker, Compose, configuratiebestand, RF2-zip en poort: OK", flush=True)
    if args.check:
        print("Provider, model en index worden in de container gecontroleerd bij de installatie.")
        return 0
    try:
        compose("build", "app")
        config_check = ["run", "--rm", "--no-deps", "app", "python", "-m", "app.check", "--config-only"]
        if not args.skip_llm_check:
            config_check.append("--probe-llm")
        compose(*config_check)
        available = compose("run", "--rm", "--no-deps", "app", "python", "-m", "app.check", "--index-only", check=False, capture=True)
        if args.rebuild_index or available.returncode:
            if index_backend != "biolord":
                print("Bouw de optionele OpenAI-index apart volgens docs/installation.md; deze stap kan API-kosten geven.")
                return 1
            print("BioLORD-zoekindex wordt opgebouwd; een volledige release kan lang duren.", flush=True)
            command = ["run", "--rm", "index", "python", "-m", "indexing.build", "--embedders", "biolord"]
            if args.rebuild_index:
                command.append("--force-prepare")
            compose(*command)
        compose("up", "-d", "--wait", "app")
        for _ in range(30):
            try:
                with urllib.request.urlopen(url + "/api/readiness", timeout=3) as response:
                    if json.load(response).get("status") == "ready":
                        print(f"SNOMED CT Annotator klaar: {url}")
                        return 0
            except (OSError, ValueError):
                pass
            time.sleep(1)
        print("FOUT: app is niet klaar. Bekijk docker compose logs app en /api/readiness.")
        return 1
    except subprocess.CalledProcessError:
        print("Installatie gestopt bij bovenstaande fout; controleer de melding en voer dit commando opnieuw uit.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
