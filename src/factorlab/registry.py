"""Small JSONL factor registry for deduplication and research lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class FactorRecord:
    name: str
    formula: str
    intuition: str = ""
    source: str = "human"
    status: str = "candidate"
    data_version: str | None = None

    @property
    def formula_hash(self) -> str:
        normalized = "".join(self.formula.split()).lower().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:16]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["formula_hash"] = self.formula_hash
        return value


class FactorRegistry:
    """Append-only registry that refuses duplicate normalized formulas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
        return result

    def register(self, record: FactorRecord) -> dict[str, object]:
        existing = self.records()
        if any(item.get("formula_hash") == record.formula_hash for item in existing):
            raise ValueError("factor formula already exists in registry")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = record.as_dict()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        return value
