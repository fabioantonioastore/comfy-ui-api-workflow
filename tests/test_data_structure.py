import copy
import pickle
import random

import pytest

from comfy_ui_api_workflow._impl.data_structure import (
    IncrementalDirectedAcyclicGraph,
    IncrementalDirectedAcyclicGraphNode,
)
from comfy_ui_api_workflow._impl.errors.data_structure import CycleError
from comfy_ui_api_workflow._impl.exceptions import (
    ComfyUIAPIWorkflowException,
    DataStructureException,
)


def make_nodes(*names):
    return [IncrementalDirectedAcyclicGraphNode(name) for name in names]


def edges(graph):
    return {
        (node, dependency)
        for node in graph.topological_order()
        for dependency in graph.dependencies(node)
    }


def structure(graph):
    return set(graph.topological_order()), edges(graph)


def assert_valid_order(graph):
    position = {node: index for index, node in enumerate(graph.topological_order())}
    for node, dependency in edges(graph):
        assert position[node] < position[dependency]
        assert node in graph.dependents(dependency)


def with_edges(graph, new_edges):
    adjacency = {node: graph.dependencies(node) for node in graph.topological_order()}
    for source, target in new_edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    return adjacency


def has_cycle(adjacency):
    indegree = {node: 0 for node in adjacency}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(adjacency)


def chain(*nodes):
    graph = IncrementalDirectedAcyclicGraph()
    for node, dependency in zip(nodes, nodes[1:]):
        graph.add_edge(node, dependency)
    return graph


def test_add_node_is_idempotent():
    a, b = make_nodes("a", "b")
    graph = chain(a, b)
    graph.add_node(a)
    assert len(graph) == 2
    assert graph.topological_order() == [a, b]
    assert edges(graph) == {(a, b)}
    assert graph.dependents(b) == {a}


def test_add_edge_adds_missing_nodes():
    a, b = make_nodes("a", "b")
    graph = chain(a, b)
    assert a in graph and b in graph
    assert graph.dependencies(a) == {b}
    assert graph.dependents(b) == {a}


def test_add_edge_against_current_order_reorders_nodes():
    graph = IncrementalDirectedAcyclicGraph()
    a, b = make_nodes("a", "b")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_edge(b, a)
    assert graph.topological_order() == [b, a]


def test_add_edge_moves_the_whole_affected_region():
    graph = IncrementalDirectedAcyclicGraph()
    a, b, c, d = make_nodes("a", "b", "c", "d")
    for node in (a, b, c, d):
        graph.add_node(node)
    graph.add_edge(a, b)
    graph.add_edge(c, d)
    graph.add_edge(d, a)
    assert graph.topological_order() == [c, d, a, b]


def test_duplicate_edge_is_ignored():
    a, b = make_nodes("a", "b")
    graph = chain(a, b)
    order_before = graph.topological_order()
    graph.add_edge(a, b)
    assert graph.topological_order() == order_before
    assert edges(graph) == {(a, b)}


def test_edge_closing_a_cycle_is_rejected_with_the_cycle():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b, c)
    with pytest.raises(CycleError) as excinfo:
        graph.add_edge(c, a)
    assert excinfo.value.cycle == [c, a, b, c]


def test_rejected_edge_leaves_the_graph_unchanged():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b, c)
    order_before, edges_before = graph.topological_order(), edges(graph)
    with pytest.raises(CycleError):
        graph.add_edge(c, a)
    assert graph.topological_order() == order_before
    assert edges(graph) == edges_before


def test_self_loop_is_rejected():
    graph = IncrementalDirectedAcyclicGraph()
    (node,) = make_nodes("n")
    graph.add_node(node)
    with pytest.raises(CycleError) as excinfo:
        graph.add_edge(node, node)
    assert excinfo.value.cycle == [node, node]
    assert graph.dependencies(node) == set()


def test_self_loop_on_a_new_node_does_not_add_it():
    graph = IncrementalDirectedAcyclicGraph()
    (node,) = make_nodes("n")
    with pytest.raises(CycleError):
        graph.add_edge(node, node)
    assert node not in graph
    assert len(graph) == 0


def test_cycle_error_belongs_to_the_library_hierarchy():
    assert issubclass(CycleError, DataStructureException)
    assert issubclass(CycleError, ComfyUIAPIWorkflowException)


def test_cycle_error_is_caught_by_except_exception():
    graph = IncrementalDirectedAcyclicGraph()
    (node,) = make_nodes("n")
    caught = None
    try:
        graph.add_edge(node, node)
    except Exception as error:
        caught = error
    assert isinstance(caught, CycleError)


def test_cycle_error_message_explains_the_rejected_edge():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b, c)
    with pytest.raises(CycleError) as excinfo:
        graph.add_edge(c, a)
    assert str(excinfo.value) == (
        "Cannot add edge 'c' -> 'a': a path from 'a' back to 'c' already exists ('a' -> 'b' -> 'c'), "
        "so the new edge would create the cycle 'c' -> 'a' -> 'b' -> 'c'."
    )


def test_self_loop_error_message_explains_the_self_loop():
    graph = IncrementalDirectedAcyclicGraph()
    (node,) = make_nodes("n")
    with pytest.raises(CycleError) as excinfo:
        graph.add_edge(node, node)
    assert (
        str(excinfo.value)
        == "Cannot add edge 'n' -> 'n': a node cannot have an edge to itself."
    )


def test_cycle_error_exposes_the_rejected_edge_and_the_existing_path():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b, c)
    with pytest.raises(CycleError) as excinfo:
        graph.add_edge(c, a)
    assert excinfo.value.edge == (c, a)
    assert excinfo.value.path == [a, b, c]


@pytest.mark.parametrize(
    "clone",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy],
    ids=["pickle", "copy"],
)
def test_cycle_error_survives_pickle_and_copy(clone):
    a, b = make_nodes("a", "b")
    error = CycleError([a, b, a])
    restored = clone(error)
    assert str(restored) == str(error)
    assert [node.data for node in restored.cycle] == ["a", "b", "a"]
    assert [node.data for node in restored.edge] == ["a", "b"]


def test_remove_node_drops_its_edges():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b, c)
    graph.remove_node(b)
    assert b not in graph
    assert len(graph) == 2
    assert graph.dependencies(a) == set()
    assert graph.dependents(c) == set()


def test_remove_missing_node_raises_key_error():
    graph = IncrementalDirectedAcyclicGraph()
    (node,) = make_nodes("n")
    with pytest.raises(KeyError):
        graph.remove_node(node)


def test_remove_edge_keeps_both_nodes():
    a, b = make_nodes("a", "b")
    graph = chain(a, b)
    graph.remove_edge(a, b)
    assert edges(graph) == set()
    assert graph.dependents(b) == set()
    assert a in graph and b in graph


def test_removed_edge_allows_the_reverse_edge():
    a, b = make_nodes("a", "b")
    graph = chain(a, b)
    graph.remove_edge(a, b)
    graph.add_edge(b, a)
    assert graph.topological_order() == [b, a]


def test_insert_between_splices_the_node_into_the_edge():
    a, b, node = make_nodes("a", "b", "n")
    graph = chain(a, b)
    graph.insert_between(a, b, node)
    assert edges(graph) == {(a, node), (node, b)}
    assert graph.topological_order() == [a, node, b]


def test_insert_between_can_keep_the_original_edge():
    a, b, node = make_nodes("a", "b", "n")
    graph = chain(a, b)
    graph.insert_between(a, b, node, remove_original=False)
    assert edges(graph) == {(a, b), (a, node), (node, b)}


def test_insert_between_rejecting_a_cycle_leaves_the_graph_unchanged():
    a, b, c, node = make_nodes("a", "b", "c", "n")
    graph = chain(b, c, a)
    before = structure(graph)
    with pytest.raises(CycleError):
        graph.insert_between(a, b, node)
    assert structure(graph) == before
    assert node not in graph


def test_insert_between_error_names_the_edge_that_closes_the_cycle():
    a, b, c, node = make_nodes("a", "b", "c", "n")
    graph = chain(b, c, a)
    with pytest.raises(CycleError) as excinfo:
        graph.insert_between(a, b, node)
    assert excinfo.value.edge == (node, b)
    assert excinfo.value.cycle == [node, b, c, a, node]


def test_insert_between_rollback_removes_a_new_endpoint():
    a, b, m, node = make_nodes("a", "b", "m", "n")
    graph = chain(b, m, node)
    before = structure(graph)
    with pytest.raises(CycleError):
        graph.insert_between(a, b, node)
    assert structure(graph) == before
    assert a not in graph


def test_insert_between_rollback_keeps_a_preexisting_edge():
    a, b, node = make_nodes("a", "b", "n")
    graph = IncrementalDirectedAcyclicGraph()
    graph.add_edge(a, node)
    graph.add_edge(b, node)
    before = structure(graph)
    with pytest.raises(CycleError):
        graph.insert_between(a, b, node)
    assert structure(graph) == before


def test_dependencies_and_dependents_return_copies():
    a, b, c = make_nodes("a", "b", "c")
    graph = chain(a, b)
    graph.dependencies(a).add(c)
    graph.dependents(b).add(c)
    assert graph.dependencies(a) == {b}
    assert graph.dependents(b) == {a}


def test_node_repr_shows_its_data():
    assert (
        repr(IncrementalDirectedAcyclicGraphNode("a"))
        == "IncrementalDirectedAcyclicGraphNode('a')"
    )


def test_node_str_shows_only_its_data():
    assert str(IncrementalDirectedAcyclicGraphNode("a")) == "'a'"


@pytest.mark.parametrize("seed", range(10))
def test_random_operations_match_a_brute_force_cycle_check(seed):
    rng = random.Random(seed)
    graph = IncrementalDirectedAcyclicGraph()
    pool = make_nodes(*range(12))
    for _ in range(300):
        roll = rng.random()
        if roll < 0.55:
            source, target = rng.choice(pool), rng.choice(pool)
            before = structure(graph)
            if has_cycle(with_edges(graph, [(source, target)])):
                with pytest.raises(CycleError):
                    graph.add_edge(source, target)
                assert structure(graph) == before
            else:
                graph.add_edge(source, target)
                assert target in graph.dependencies(source)
        elif roll < 0.75:
            a, b, node = rng.choice(pool), rng.choice(pool), rng.choice(pool)
            remove_original = rng.random() < 0.5
            before = structure(graph)
            if has_cycle(with_edges(graph, [(a, node), (node, b)])):
                with pytest.raises(CycleError):
                    graph.insert_between(a, b, node, remove_original=remove_original)
                assert structure(graph) == before
            else:
                graph.insert_between(a, b, node, remove_original=remove_original)
                expected_edges = before[1] | {(a, node), (node, b)}
                if remove_original:
                    expected_edges.discard((a, b))
                assert edges(graph) == expected_edges
        elif roll < 0.9:
            current = sorted(
                edges(graph), key=lambda edge: (edge[0].data, edge[1].data)
            )
            if current:
                graph.remove_edge(*rng.choice(current))
        elif len(graph):
            graph.remove_node(
                rng.choice(
                    sorted(graph.topological_order(), key=lambda node: node.data)
                )
            )
        assert_valid_order(graph)
