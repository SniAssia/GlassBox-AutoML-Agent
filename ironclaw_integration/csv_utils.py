from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse


_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def _is_missing_token(value: str) -> bool:
    return value.strip().lower() in _MISSING_TOKENS


@dataclass(frozen=True)
class LoadedCSV:
    source_path: str
    header: list[str]
    rows: list[list[object]]  # object values, with missing as None


def _resolve_csv_path(file_path: str) -> tuple[Path, list[str]]:
    original = file_path

    if file_path.startswith("file://"):
        parsed = urlparse(file_path)
        path = unquote(parsed.path)
        # file:///C:/... comes through as /C:/...
        if path.startswith("/") and len(path) >= 3 and path[2] == ":":
            path = path[1:]
        file_path = path

    p = Path(file_path)

    tried: list[Path] = []
    if p.is_absolute():
        tried.append(p)
        return p, [str(t) for t in tried]

    # 1) current working directory (depends on how MCP launches us)
    tried.append(Path.cwd() / p)
    # 2) repo root (stable across environments)
    repo_root = Path(__file__).resolve().parents[1]
    tried.append(repo_root / p)
    # 3) ironclaw_integration package dir
    tried.append(Path(__file__).resolve().parent / p)

    for candidate in tried:
        if candidate.exists():
            return candidate, [str(t) for t in tried]

    # Fall back to the first candidate; open() will raise a clear error.
    return tried[0], [str(t) for t in tried]


def load_csv(file_path: str, *, max_rows: Optional[int] = None) -> LoadedCSV:
    """Load a CSV into an object matrix (list-of-rows), converting missing tokens to None."""

    resolved_path, tried = _resolve_csv_path(file_path)

    try:
        f = open(resolved_path, "r", newline="", encoding="utf-8")
    except FileNotFoundError as e:
        raise ValueError(
            f"CSV file not found: '{file_path}'. Tried: {tried}"
        ) from e

    with f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty")

        header = [h.strip().strip('"').strip("'") for h in header]
        rows: list[list[object]] = []

        for i, raw_row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if len(raw_row) == 0:
                continue

            # pad/truncate to header length
            if len(raw_row) < len(header):
                raw_row = raw_row + [""] * (len(header) - len(raw_row))
            elif len(raw_row) > len(header):
                raw_row = raw_row[: len(header)]

            row: list[object] = []
            for cell in raw_row:
                cell = cell.strip().strip('"').strip("'")
                if _is_missing_token(cell):
                    row.append(None)
                else:
                    row.append(cell)
            rows.append(row)

    if not rows:
        raise ValueError("CSV contains header but no data rows")

    return LoadedCSV(source_path=str(resolved_path), header=header, rows=rows)


def column_values(loaded: LoadedCSV, col_index: int) -> list[object]:
    return [r[col_index] for r in loaded.rows]


def iter_columns(loaded: LoadedCSV) -> Iterable[tuple[str, list[object]]]:
    for i, name in enumerate(loaded.header):
        yield name, column_values(loaded, i)
