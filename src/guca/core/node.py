from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class Node:
    id: int
    state: str
    prior_state: Optional[str] = None
    parents_count: int = 0
    neighbors: Set[int] = field(default_factory=set)
    marked_new: bool = True
    marked_deleted: bool = False
    rule_index: int = 0  # for continuable mode

    # snapshot at iteration start
    saved_state: Optional[str] = None
    saved_parents_count: int = 0
    saved_degree: int = 0

    @property
    def degree(self) -> int:
        return len(self.neighbors)