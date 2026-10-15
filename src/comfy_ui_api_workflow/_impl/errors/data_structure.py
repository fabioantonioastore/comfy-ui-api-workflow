from typing import TYPE_CHECKING

from ..exceptions import DataStructureException

if TYPE_CHECKING:
    from ..data_structure import IncrementalDirectedAcyclicGraphNode


class CycleError(DataStructureException):
    def __init__(self, cycle: list["IncrementalDirectedAcyclicGraphNode"]) -> None:
        super().__init__(cycle)
        self.cycle = cycle

    @property
    def edge(
        self,
    ) -> tuple[
        "IncrementalDirectedAcyclicGraphNode", "IncrementalDirectedAcyclicGraphNode"
    ]:
        return self.cycle[0], self.cycle[1]

    @property
    def path(self) -> list["IncrementalDirectedAcyclicGraphNode"]:
        return self.cycle[1:]

    def __str__(self) -> str:
        source, target = self.edge
        if source == target:
            return f"Cannot add edge {source} -> {target}: a node cannot have an edge to itself."
        return (
            f"Cannot add edge {source} -> {target}: a path from {target} back to {source} already exists "
            f"({' -> '.join(map(str, self.path))}), "
            f"so the new edge would create the cycle {' -> '.join(map(str, self.cycle))}."
        )
