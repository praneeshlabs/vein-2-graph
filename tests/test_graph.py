import numpy as np
from vein2graph import extract_graph, graph_summary


def skeleton(*points, shape=(25, 25)):
    a = np.zeros(shape, dtype=bool)
    for y, x in points:
        a[y, x] = True
    return a


def test_empty_skeleton_is_empty_graph():
    g = extract_graph(np.zeros((8, 8), dtype=bool))
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0
    assert graph_summary(g)["independent_cycles"] == 0


def test_straight_line_has_two_endpoints_one_edge():
    s = skeleton(*[(12, x) for x in range(4, 14)])
    g = extract_graph(s)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert sorted(d["kind"] for _, d in g.nodes(data=True)) == ["endpoint", "endpoint"]
    assert graph_summary(g)["independent_cycles"] == 0


def test_t_junction_becomes_one_junction_and_three_endpoints():
    pts = [(12, x) for x in range(5, 20)] + [(y, 12) for y in range(3, 13)]
    g = extract_graph(skeleton(*pts))
    kinds = [d["kind"] for _, d in g.nodes(data=True)]
    assert kinds.count("junction") == 1
    assert kinds.count("endpoint") == 3
    assert g.number_of_edges() == 3


def test_closed_square_loop_is_preserved_as_one_cycle():
    pts = []
    for x in range(5, 16):
        pts.append((5, x)); pts.append((15, x))
    for y in range(6, 15):
        pts.append((y, 5)); pts.append((y, 15))
    g = extract_graph(skeleton(*pts))
    assert g.number_of_nodes() == 1
    assert g.number_of_edges() == 1
    assert graph_summary(g)["independent_cycles"] == 1
    u, v, _ = next(iter(g.edges(data=True)))
    assert u == v


def test_disconnected_lines_count_components_and_cycles():
    pts = [(3, x) for x in range(2, 7)] + [(18, x) for x in range(15, 21)]
    g = extract_graph(skeleton(*pts))
    stats = graph_summary(g)
    assert stats["connected_components"] == 2
    assert stats["nodes"] == 4
    assert stats["edges"] == 2
    assert stats["independent_cycles"] == 0


def test_short_spur_pruning_preserves_main_line():
    from vein2graph import prune_short_spurs
    pts = [(12, x) for x in range(4, 21)] + [(y, 12) for y in range(9, 13)]
    s = skeleton(*pts)
    pruned = prune_short_spurs(s, max_length=3)
    assert pruned[12, 4] and pruned[12, 20]
    assert pruned.sum() < s.sum()
