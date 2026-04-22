from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve relative paths against project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
