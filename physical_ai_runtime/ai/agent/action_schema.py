from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticAction:
    action: str

    object_id: Optional[str] = None

    reason: Optional[str] = None

    strategy: Optional[str] = None
