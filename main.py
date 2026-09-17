import argparse
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# The reloader's child process does not inherit sys.path edits made here, so
# PYTHONPATH is the only way it can find src/. Normally redundant -- `uv sync`
# installs the package -- but it keeps `python main.py` working in an
# environment where that has not been run yet.
_existing = os.environ.get("PYTHONPATH", "")
if str(SRC) not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC), _existing) if p)

from permrag.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run the PermRAG API server.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (on by default outside production).",
    )
    args = parser.parse_args()

    # Reloading is a development convenience; it doubles the process count and
    # watches the filesystem, so it stays off anywhere but local.
    reload = settings.environment == "local" and not args.no_reload

    uvicorn.run(
        "permrag.api.main:app",
        host=args.host,
        port=args.port,
        reload=reload,
        # Watch only the package. Watching the repository root also reloads on
        # edits to ui/, scripts/ and tests/, none of which the server runs.
        reload_dirs=[str(SRC)] if reload else None,
    )


if __name__ == "__main__":
    main()
    