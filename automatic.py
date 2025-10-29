from libsemigroups_pybind11 import (
    Paths,
    Presentation,
    ToddCoxeter,
    UNDEFINED,
    WordGraph,
    WordRange,
    congruence_kind,
    presentation,
    word_graph,
)
from itertools import product
from functools import cache

L = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
A = WordGraph(4, [[1, 2], [], [3], []])

p = Presentation([0, 1])
presentation.add_rule(p, [0, 1], [1, 0])

tc = ToddCoxeter(congruence_kind.twosided, p)
tc.run_until(lambda: tc.number_of_nodes_active() > 20)
tc.shrink_to_fit()
A = tc.current_word_graph()


class Uf:
    def __init__(self) -> None:
        self._parent = []

    def add_nodes(self, num: int):
        n = len(self._parent)
        self._parent.extend(list(range(n, n + num)))
        return self

    def find(self, node: int):
        if node == UNDEFINED:
            return node
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def union(self, node1: int, node2: int) -> None:
        assert node1 != UNDEFINED
        assert node2 != UNDEFINED

        node1, node2 = self.find(node1), self.find(node2)
        if node2 < node1:
            node1, node2 = node2, node1
        self._parent[node2] = node1
        return self

    def number_of_blocks(self) -> int:
        return len(self.reps())

    def reps(self) -> list[int]:
        return [x for i, x in enumerate(self._parent) if x == i]


class QuotientWordGraph:
    def __init__(self, num_nodes: int, out_degree: int) -> None:
        self._uf = Uf()
        self._uf.add_nodes(num_nodes)
        self._wg = WordGraph(num_nodes, out_degree)

    def target(self, s: int, a: int) -> int:
        return self._uf.find(self._wg.target(self._uf.find(s), a))

    def set_target(self, s: int, a: int, t: int):
        self._wg.target(self._uf.find(s), a, self._uf.find(t))
        return self

    def out_degree(self) -> int:
        return self._wg.out_degree()

    def number_of_nodes(self) -> int:
        return self._wg.number_of_nodes()

    def add_nodes(self, num_nodes: int):
        self._uf.add_nodes(num_nodes)
        self._wg.add_nodes(num_nodes)
        return self

    def merge_nodes(self, node1: int, node2: int):
        self._uf.union(node1, node2)
        return self

    def to_word_graph(self) -> WordGraph:
        lookup = {x: i for i, x in enumerate(self._uf.reps())}
        result = WordGraph(self._uf.number_of_blocks(), self._wg.out_degree())
        for node in lookup:
            for a in range(result.out_degree()):
                t = self.target(node, a)
                if t != UNDEFINED:
                    result.target(lookup[node], a, lookup[t])
        return result


def follow_path(wg: QuotientWordGraph, s: int, w: tuple[int, ...]) -> int:
    for a in w:
        if s == UNDEFINED:
            return s
        s = wg.target(s, a)
    return s


def accepts(A: WordGraph, w: tuple[int, ...]):
    return word_graph.follow_path(A, 0, w) != 4294967295


@cache
def readout(
    A: WordGraph,
    L: WordGraph,
    a: int,
    k: int,
    u: tuple[int, ...],
    v: tuple[int, ...],
) -> bool:
    result = []
    wr = WordRange().alphabet_size(L.out_degree()).max(k)
    for s in wr:
        for t in wr:
            vta = v + tuple(t)
            if a < A.out_degree():
                vta += (a,)
            result.append(
                accepts(L, u + tuple(s))
                and accepts(L, v + tuple(t))
                and accepts(A, u + tuple(s))
                and word_graph.follow_path(A, 0, u + tuple(s))
                == word_graph.follow_path(A, 0, vta)
            )
    return sum([b << i for i, b in enumerate(result)])


def merge_nodes(wg: QuotientWordGraph, node1: int, node2: int) -> WordGraph:
    """
    Merge the nodes ``node1`` and ``node2``.
    """

    kappa = [[node1, node2]]

    while len(kappa) != 0:
        node1, node2 = kappa.pop()
        if node1 == node2:
            continue
        if node1 > node2:
            node1, node2 = node2, node1
        for letter in range(wg.out_degree()):
            if wg.target(node2, letter) != UNDEFINED:
                if wg.target(node1, letter) == UNDEFINED:
                    wg.set_target(node1, letter, wg.target(node2, letter))
                else:
                    kappa.append((wg.target(node1, letter), wg.target(node2, letter)))
        print(
            f"quotienting by ({node1}, {node2}) = ({wg._uf.find(node1)}, {wg._uf.find(node2)}), number of nodes is {wg._uf.number_of_blocks() - 1}"
        )
        wg.merge_nodes(node1, node2)


def word_difference_automata(A: WordGraph, L: WordGraph, a: int, k: int) -> WordGraph:
    result = QuotientWordGraph(0, (A.out_degree() + 1) ** 2)
    readouts_to_node = dict()
    next_free_node = 0

    def node(u: tuple[int, ...], v: tuple[int, ...]) -> int:
        nonlocal next_free_node
        uv = readout(A, L, a, k, u, v)
        if uv not in readouts_to_node:
            readouts_to_node[uv] = next_free_node
            next_free_node += 1
        return readouts_to_node[uv]

    def label(b: int, c: int) -> int:
        return b * (A.out_degree() + 1) + c

    def def_edge(wg: QuotientWordGraph, s: int, x: int, t: int) -> None:
        if max(s, t) >= wg.number_of_nodes():
            wg.add_nodes(max(s, t) - wg.number_of_nodes() + 1)
        if wg.target(s, x) != UNDEFINED and wg.target(s, x) != t:
            merge_nodes(wg, t, wg.target(s, x))
        else:
            wg.set_target(s, x, t)

    L_lang = Paths(L).source(0).max(k)
    alphabet = list(range(A.out_degree()))
    epsilon = A.out_degree()
    for u, v in product(L_lang, L_lang):
        u, v = tuple(u), tuple(v)
        s = node(u, v)
        for b, c in product(alphabet, alphabet):
            t = node(u + (b,), v + (c,))
            def_edge(result, s, label(b, c), t)
        for b in alphabet:
            t = node(u + (b,), v)
            def_edge(result, s, label(b, epsilon), t)
            t = node(u, v + (b,))
            def_edge(result, s, label(epsilon, b), t)
    return result.to_word_graph()
