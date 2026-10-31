"""Check configuration, index files and optionally one synthetic LLM response."""
from __future__ import annotations

import argparse

from .config import settings
from .llm import call_structured, configuration_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--probe-llm", action="store_true")
    args = parser.parse_args()
    errors = [] if args.index_only else configuration_errors()
    if not args.config_only:
        from .vector_store import embedder_available
        if not embedder_available(settings.default_embedder):
            errors.append("Zoekindex ontbreekt: bouw de index met de indexservice.")
    if errors:
        for error in errors:
            print("FOUT: " + error)
        return 1
    if args.probe_llm:
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        try:
            result = call_structured("Return a JSON object with ok=true.", "Synthetic installation test; no clinical data.", "installation_check", schema)
            if result["ok"] is not True:
                raise RuntimeError("De API gaf ok=false terug.")
        except RuntimeError as exc:
            print("FOUT: " + str(exc))
            return 1
        print("LLM-verbinding en gestructureerd antwoord: OK (synthetische test)")
    print("Installatiecontrole: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
