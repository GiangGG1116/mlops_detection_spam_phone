from __future__ import annotations

import sys

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
