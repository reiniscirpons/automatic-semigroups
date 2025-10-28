from libsemigroups_pybind11 import UNDEFINED, WordGraph, WordRange, word_graph

L = WordGraph(2, [[0, 1], [UNDEFINED, 1]])


def accepts(A: WordGraph, w: list[int]):
    return word_graph.follow_path(A, 0, w) != UNDEFINED


def are_equiv(
    A: WordGraph,
    L: WordGraph,
    a: int,
    k: int,
    u1: list[int],
    v1: list[int],
    u2: list[int],
    v2: list[int],
) -> bool:
    wr = WordRange().alphabet_size(L.out_degree()).max(k)
    for s in wr:
        for t in wr:
            cond1 = (
                accepts(L, u1 + s)
                and accepts(L, v1 + t)
                and accepts(A, u1 + s, v1 + t + [a])
            )
            cond2 = (
                accepts(L, u2 + s)
                and accepts(L, v2 + t)
                and accepts(A, u2 + s, v2 + t + [a])
            )
            if cond1 != cond2:
                return False
    return True
