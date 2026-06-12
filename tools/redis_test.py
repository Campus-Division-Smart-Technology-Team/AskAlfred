import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.clients import get_redis  # noqa: E402


def main() -> int:
    """Run a manual Redis smoke test."""
    load_dotenv()
    client = get_redis()
    try:
        client.set("foo", "bar")
        print(client.get("foo"))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
