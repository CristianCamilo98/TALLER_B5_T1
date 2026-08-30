"""Schema equivalence checks for synthetic outputs."""

from __future__ import annotations

from dataclasses import dataclass

from constants import CANONICAL_COLUMNS, FORBIDDEN_METADATA_COLUMNS


@dataclass(frozen=True)
class SchemaAssessment:
    ok: bool
    columns: tuple[str, ...]
    errors: tuple[str, ...]


def assess_schema(columns: tuple[str, ...]) -> SchemaAssessment:
    errors: list[str] = []
    column_set = set(columns)

    missing = [name for name in CANONICAL_COLUMNS if name not in column_set]
    if missing:
        errors.append(f"missing canonical columns: {missing}")

    forbidden = sorted(column_set & FORBIDDEN_METADATA_COLUMNS)
    if forbidden:
        errors.append(f"forbidden metadata columns present: {forbidden}")

    extra = sorted(column_set - set(CANONICAL_COLUMNS))
    if extra:
        errors.append(f"unexpected extra columns: {extra}")

    return SchemaAssessment(ok=not errors, columns=columns, errors=tuple(errors))


def assess_schema_equivalence(schemas: dict[str, tuple[str, ...]]) -> tuple[bool, tuple[str, ...]]:
    """All generators must expose the exact same column set."""

    if not schemas:
        return False, ("no schemas to compare",)

    reference = next(iter(schemas.values()))
    errors: list[str] = []
    for generator_id, columns in schemas.items():
        if columns != reference:
            errors.append(
                f"{generator_id}: column set {list(columns)} != reference {list(reference)}"
            )
    return not errors, tuple(errors)
