"""Discover official generator outputs under generadores/*/outputs/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import EXPECTED_GENERATOR_COUNT, GENERATORS_ROOT
from io_utils import sha256_file


@dataclass(frozen=True)
class DiscoveredOutput:
    generator_id: str
    path: Path
    filename: str
    rows: int
    sha256: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryResult:
    ok: bool
    outputs: tuple[DiscoveredOutput, ...]
    errors: tuple[str, ...]


def _generator_dirs() -> list[Path]:
    if not GENERATORS_ROOT.is_dir():
        return []
    return sorted(
        path
        for path in GENERATORS_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def discover_outputs() -> DiscoveryResult:
    """Return exactly one parquet per generator or fail with explicit errors."""

    errors: list[str] = []
    discovered: list[DiscoveredOutput] = []

    generator_dirs = _generator_dirs()
    if len(generator_dirs) != EXPECTED_GENERATOR_COUNT:
        names = [path.name for path in generator_dirs]
        errors.append(
            f"Expected {EXPECTED_GENERATOR_COUNT} generator directories, found {len(generator_dirs)}: {names}"
        )

    for generator_dir in generator_dirs:
        generator_id = generator_dir.name
        outputs_dir = generator_dir / "outputs"
        if not outputs_dir.is_dir():
            errors.append(f"{generator_id}: missing outputs/ directory")
            continue

        parquets = sorted(outputs_dir.glob("*.parquet"))
        if len(parquets) == 0:
            errors.append(f"{generator_id}: no parquet output found")
            continue
        if len(parquets) > 1:
            names = [path.name for path in parquets]
            errors.append(
                f"{generator_id}: expected exactly one official parquet, found {len(parquets)}: {names}"
            )
            continue

        path = parquets[0]
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        schema = pq.read_schema(path)
        discovered.append(
            DiscoveredOutput(
                generator_id=generator_id,
                path=path,
                filename=path.name,
                rows=int(metadata.num_rows),
                sha256=sha256_file(path),
                columns=tuple(schema.names),
            )
        )

    if errors:
        return DiscoveryResult(ok=False, outputs=tuple(discovered), errors=tuple(errors))

    if len(discovered) != EXPECTED_GENERATOR_COUNT:
        errors.append(
            f"Expected {EXPECTED_GENERATOR_COUNT} discovered outputs, found {len(discovered)}"
        )
        return DiscoveryResult(ok=False, outputs=tuple(discovered), errors=tuple(errors))

    return DiscoveryResult(ok=True, outputs=tuple(discovered), errors=())
