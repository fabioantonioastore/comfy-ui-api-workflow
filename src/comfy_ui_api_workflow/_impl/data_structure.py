from typing import Any

from .errors.data_structure import CycleError


class IncrementalDirectedAcyclicGraphNode:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data!r})"

    def __str__(self) -> str:
        return repr(self.data)


class IncrementalDirectedAcyclicGraph:
    def __init__(self) -> None:
        self._successors: dict[
            IncrementalDirectedAcyclicGraphNode,
            set[IncrementalDirectedAcyclicGraphNode],
        ] = {}
        self._predecessors: dict[
            IncrementalDirectedAcyclicGraphNode,
            set[IncrementalDirectedAcyclicGraphNode],
        ] = {}
        self._order: dict[IncrementalDirectedAcyclicGraphNode, int] = {}
        self._next = 0

    def add_node(self, node: IncrementalDirectedAcyclicGraphNode) -> None:
        if node in self._order:
            return
        self._successors[node] = set()
        self._predecessors[node] = set()
        self._order[node] = self._next
        self._next += 1

    def remove_node(self, node: IncrementalDirectedAcyclicGraphNode) -> None:
        if node not in self._order:
            raise KeyError(node)
        for successor in self._successors[node]:
            self._predecessors[successor].discard(node)
        for predecessor in self._predecessors[node]:
            self._successors[predecessor].discard(node)
        del self._successors[node], self._predecessors[node], self._order[node]

    def insert_between(
        self,
        a: IncrementalDirectedAcyclicGraphNode,
        b: IncrementalDirectedAcyclicGraphNode,
        node: IncrementalDirectedAcyclicGraphNode,
        remove_original: bool = True,
    ) -> None:
        new_nodes = [n for n in (a, b, node) if n not in self]
        new_edge = node not in self._successors.get(a, ())
        try:
            self.add_node(node=node)
            self.add_edge(x=a, y=node)
            self.add_edge(x=node, y=b)
        except CycleError:
            if new_edge and a in self and node in self:
                self.remove_edge(x=a, y=node)
            for new_node in new_nodes:
                if new_node in self:
                    self.remove_node(node=new_node)
            raise
        if remove_original:
            self.remove_edge(x=a, y=b)

    def add_edge(
        self,
        x: IncrementalDirectedAcyclicGraphNode,
        y: IncrementalDirectedAcyclicGraphNode,
    ) -> None:
        if x == y:
            raise CycleError([x, x])
        self.add_node(node=x)
        self.add_node(node=y)
        if y in self._successors[x]:
            return
        order_x = self._order[x]
        order_y = self._order[y]
        if order_x < order_y:
            self._link(x=x, y=y)
            return
        forward = self._forward_search(y=y, x=x, upper_bound=order_x)
        backward = self._backward_search(x=x, lower_bound=order_y)
        self._reorder(backward=backward, forward=forward)
        self._link(x=x, y=y)

    def remove_edge(
        self,
        x: IncrementalDirectedAcyclicGraphNode,
        y: IncrementalDirectedAcyclicGraphNode,
    ) -> None:
        self._successors[x].discard(y)
        self._predecessors[y].discard(x)

    def topological_order(self) -> list[IncrementalDirectedAcyclicGraphNode]:
        return sorted(self._order, key=self._order.__getitem__)

    def dependencies(
        self, node: IncrementalDirectedAcyclicGraphNode
    ) -> set[IncrementalDirectedAcyclicGraphNode]:
        return set(self._successors[node])

    def dependents(
        self, node: IncrementalDirectedAcyclicGraphNode
    ) -> set[IncrementalDirectedAcyclicGraphNode]:
        return set(self._predecessors[node])

    def __contains__(self, node: object) -> bool:
        return node in self._order

    def __len__(self) -> int:
        return len(self._order)

    def _link(
        self,
        x: IncrementalDirectedAcyclicGraphNode,
        y: IncrementalDirectedAcyclicGraphNode,
    ) -> None:
        self._successors[x].add(y)
        self._predecessors[y].add(x)

    def _forward_search(
        self,
        y: IncrementalDirectedAcyclicGraphNode,
        x: IncrementalDirectedAcyclicGraphNode,
        upper_bound: int,
    ) -> list[IncrementalDirectedAcyclicGraphNode]:
        parent: dict[
            IncrementalDirectedAcyclicGraphNode,
            IncrementalDirectedAcyclicGraphNode | None,
        ] = {y: None}
        stack = [y]
        while stack:
            node = stack.pop()
            for successor in self._successors[node]:
                if successor == x:
                    path = [node]
                    previous = parent[node]
                    while previous is not None:
                        path.append(previous)
                        previous = parent[previous]
                    path.reverse()
                    raise CycleError([x] + path + [x])
                if successor not in parent and self._order[successor] < upper_bound:
                    parent[successor] = node
                    stack.append(successor)
        return list(parent)

    def _backward_search(
        self,
        x: IncrementalDirectedAcyclicGraphNode,
        lower_bound: int,
    ) -> list[IncrementalDirectedAcyclicGraphNode]:
        visited = {x}
        stack = [x]
        while stack:
            node = stack.pop()
            for predecessor in self._predecessors[node]:
                if (
                    predecessor not in visited
                    and self._order[predecessor] > lower_bound
                ):
                    visited.add(predecessor)
                    stack.append(predecessor)
        return list(visited)

    def _reorder(
        self,
        backward: list[IncrementalDirectedAcyclicGraphNode],
        forward: list[IncrementalDirectedAcyclicGraphNode],
    ) -> None:
        key = self._order.__getitem__
        backward.sort(key=key)
        forward.sort(key=key)
        nodes = backward + forward
        slots = sorted(self._order[node] for node in nodes)
        for node, position in zip(nodes, slots):
            self._order[node] = position
