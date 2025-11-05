r"""An attempt at implementing the deterministic algorithm for a 3rd time.

This time using DP. The thrust of the matter is as follows:

.. math::

    (u_1, v_1) \sim_{a, k} (u_2, v_2) \Longleftrightarrow
    (u_1, v_1) \sim_{a, 0} (u_2, v_2) \wedge
    \bigwedge_{b, c\in A_\$} (u_1b, v_1c) \sim_{a, k-1} (u_2b, v_2c)

Hence we may determine if :math:`(u_1, v_1) \sim_{a, k} (u_2, v_2)`
by utilizing dynamic programming.
"""

from time import time
from datetime import timedelta
from enum import Enum, auto
from typing import Iterable
from warnings import warn
from libsemigroups_pybind11 import (
    UNDEFINED,
    Undefined,
    congruence_kind,
    Presentation,
    presentation,
    ToddCoxeter,
    WordGraph,
    word_graph,
    todd_coxeter,
)
from libsemigroups_pybind11.word_graph import dot
from det import (
    Automaton,
    Letter,
    Vertex,
    Word,
    direct_product_automaton,
    intersection_automaton,
    trim_automaton,
)
from itertools import chain


def label_from_pair(alphabet_size: int, b: Letter, c: Letter) -> Letter:
    return b * (alphabet_size + 1) + c


type PairVertex = tuple[Vertex, Vertex]


def right_multiply_pair_vertex(
    u: PairVertex, a: Letter, rep_automaton: Automaton, cayley_graph: WordGraph
) -> PairVertex | None:
    alphabet_size = cayley_graph.out_degree()
    if a == alphabet_size:
        return u

    beta, x = u
    xa = cayley_graph.target(x, a)
    if xa == UNDEFINED:
        warn(f"Warning, Cayley graph is too small!")
        return None
    if beta == UNDEFINED:
        gamma = UNDEFINED
    else:
        gamma = rep_automaton.word_graph.target(beta, a)
    # NOTE(reiniscirpons): gamma is explicitly allowed to be undefined to allow
    # for falling off the machine
    return (gamma, xa)


type QuadVertex = tuple[PairVertex, PairVertex]


def sim_a_0_helper(
    p: QuadVertex, a: Letter, rep_automaton: Automaton, cayley_graph: WordGraph
) -> bool:
    """Calculate whether (u in L) and (v in L) and (ua =_S v) holds.

    Implemented as a separate function to account for mutation of Cayley graph.
    """
    alphabet_size = cayley_graph.out_degree()
    (beta, x), (gamma, y) = p
    if (
        beta == UNDEFINED
        or gamma == UNDEFINED
        or beta not in rep_automaton.final_states
        or gamma not in rep_automaton.final_states
    ):
        return False
    xa = x
    if a != alphabet_size:
        xa = cayley_graph.target(x, a)
        if xa == UNDEFINED:
            warn(f"Warning, Cayley graph is too small! x={x}, a={a}")
    return xa == y


def is_sim_a_k_equivalent(
    p: QuadVertex,
    q: QuadVertex,
    a: Letter,
    k: int,
    rep_automaton: Automaton,
    cayley_graph: WordGraph,
    memo: dict[tuple[QuadVertex, QuadVertex, Letter], bool],
) -> bool:
    alphabet_size = cayley_graph.out_degree()
    if p == q:
        return True
    if k == 0:
        bool_p = sim_a_0_helper(p, a, rep_automaton, cayley_graph)
        bool_q = sim_a_0_helper(q, a, rep_automaton, cayley_graph)
        return (bool_p and bool_q) or (not bool_p and not bool_q)

    if (p, q, a) in memo:
        return memo[(p, q, a)]
    if (q, p, a) in memo:
        return memo[(q, p, a)]

    result = is_sim_a_k_equivalent(p, q, a, 0, rep_automaton, cayley_graph, memo)
    if not result:
        memo[(q, p, a)] = False
        return False

    pu, pv = p
    qu, qv = q
    for b in range(alphabet_size + 1):
        pub = right_multiply_pair_vertex(pu, b, rep_automaton, cayley_graph)
        qub = right_multiply_pair_vertex(qu, b, rep_automaton, cayley_graph)
        if pub is None or qub is None:
            result = False
            break
        for c in range(alphabet_size + 1):
            if b == c and c == alphabet_size:
                continue
            pvc = right_multiply_pair_vertex(pv, c, rep_automaton, cayley_graph)
            qvc = right_multiply_pair_vertex(qv, c, rep_automaton, cayley_graph)
            if pvc is None or qvc is None:
                result = False
                break
            result = result and is_sim_a_k_equivalent(
                (pub, pvc),
                (qub, qvc),
                a,
                k - 1,
                rep_automaton,
                cayley_graph,
                memo,
            )
            if not result:
                break
        if not result:
            break

    memo[(p, q, a)] = result
    return result


# TODO: implement Outcome tagging on result and incorporate in graph building.
# class Outcome(Enum):
#     SUCCESS = auto()
#     FAILURE_K = auto()
#     FAILURE_BALL = auto()


def multiplier_automaton(
    a: Letter,
    k_1: int,
    k_2: int,
    rep_automaton: Automaton,
    cayley_graph: WordGraph,
    memo: dict[tuple[QuadVertex, QuadVertex, Letter], bool],
) -> Automaton:
    alphabet_size = cayley_graph.out_degree()
    que: list[QuadVertex] = [
        ((rep_automaton.initial_state, 0), (rep_automaton.initial_state, 0))
    ]
    # parent_index[i] is the index j such that que[j] obtains que[i] by right
    # multiplying via the letter parent_letter[i]
    parent_index: list[int | None] = [None]
    parent_letter: list[tuple[Letter, Letter] | tuple[None, None]] = [(None, None)]
    # equivalent_to[i] is the smallest index j such that que[i] is sim_{a, k}
    # equivalent to que[j]
    equivalent_to: list[int] = []
    # For a vertex v, vertex_to_idx[v] = i if the vertex corresponds to the
    # equivalence class of que[i]
    vertex_to_idx: list[int] = []
    # Also track final states for later
    final_state = set()
    # Start by constructing all distinct vertices
    i = 0
    while i < len(que):
        p = que[i]
        for j in vertex_to_idx:
            q = que[j]
            if is_sim_a_k_equivalent(p, q, a, k_2, rep_automaton, cayley_graph, memo):
                equivalent_to.append(j)
                break
        else:
            # New unique equivalence class defined
            equivalent_to.append(i)
            vertex_to_idx.append(i)
            if sim_a_0_helper(p, a, rep_automaton, cayley_graph):
                # Is a final state
                final_state.add(i)
            # If already at max capacity, then skip further processing
            if len(que) >= k_1:
                i += 1
                continue
            # Iterate through children
            pu, pv = p
            for b in range(alphabet_size + 1):
                pub = right_multiply_pair_vertex(pu, b, rep_automaton, cayley_graph)
                if pub is None:
                    continue
                for c in range(alphabet_size + 1):
                    if b == c and c == alphabet_size:
                        continue
                    pvc = right_multiply_pair_vertex(pv, c, rep_automaton, cayley_graph)
                    if pvc is None:
                        continue
                    # Populate parent info and que
                    que.append((pub, pvc))
                    parent_index.append(i)
                    parent_letter.append((b, c))
        i += 1

    # Now construct graph
    result = WordGraph(len(vertex_to_idx), (alphabet_size + 1) ** 2)
    idx_to_vertex = {idx: vertex for vertex, idx in enumerate(vertex_to_idx)}
    for i in range(len(que)):
        j = parent_index[i]
        if j is None:
            continue
        b, c = parent_letter[i]
        assert b is not None and c is not None
        result.target(
            idx_to_vertex[equivalent_to[j]],
            label_from_pair(alphabet_size, b, c),
            idx_to_vertex[equivalent_to[i]],
        )
    return Automaton(
        result, 0, frozenset(idx_to_vertex[state] for state in final_state)
    )


def run_method(p: Presentation, rep_automaton: Automaton):
    pair_automaton = direct_product_automaton(rep_automaton)
    alphabet_size = len(p.alphabet())

    tc = ToddCoxeter(congruence_kind.twosided, p)

    # TODO: dynamically adjust k_1,  k_2 and ball size
    tc.strategy(tc.options.strategy.hlt)
    tc.run_for(timedelta(seconds=1))
    tc.strategy(tc.options.strategy.felsch)
    tc.run_for(timedelta(seconds=1))
    tc.strategy(tc.options.strategy.hlt)
    tc.run_for(timedelta(seconds=1))
    tc.strategy(tc.options.strategy.felsch)
    tc.run_for(timedelta(seconds=1))
    k_1 = 50
    k_2 = 7

    memo = {}
    multiplication_automata = [
        trim_automaton(
            intersection_automaton(
                multiplier_automaton(
                    a, k_1, k_2, rep_automaton, tc.current_word_graph(), memo
                ),
                pair_automaton,
            )
        )
        for a in range(alphabet_size + 1)
    ]

    # TODO: check automata correct, if not then raise k_1 and k_2 and rerun algo
    return multiplication_automata


if __name__ == "__main__":
    # Free commutative monoid on 2 generators
    # p = Presentation([0, 1])
    # presentation.add_rule(p, (0, 1), (1, 0))
    # wg = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
    # automaton = Automaton(wg, 0, frozenset({0, 1}))

    # Bicyclic monoid
    p = Presentation([0, 1])
    p.contains_empty_word(True)
    presentation.add_rule(p, (1, 0), ())
    wg = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
    automaton = Automaton(wg, 0, frozenset({0, 1}))

    multiplication_automata = run_method(p, automaton)

    # 0 (0, 0) #00ff00 lime green
    # 1 (0, 1) #ff00ff magenta
    # 2 (0, $) #007fff blue
    # 3 (1, 0) #ff7f00 orange
    # 4 (1, 1) #7fbf7f light green
    # 5 (1, $) #4604ac dark purple
    # 6 ($, 0) #de0328 red
    # 7 ($, 1) #19801d dark green
    # 8 ($, $) #d881f5 light purple
    dot(multiplication_automata[1].word_graph).view()
    print(multiplication_automata[1].word_graph)
    print(multiplication_automata[1].initial_state)
    print(multiplication_automata[1].final_states)

# def multiplier_automaton_temp(
#     a: Letter,
#     rep_automaton: Automaton,
#     alphabet: Iterable[Letter],
#     relations: Iterable[tuple[Word, Word]],
# ):
#     alphabet = tuple(alphabet)
#     relations = tuple(relations)
#
#     alphabet_size = len(alphabet)
#     padding_symbol = alphabet_size + 1
#     padded_alphabet = alphabet + (padding_symbol,)
#
#     if a not in padded_alphabet:
#         raise ValueError(
#             "Multiplier must be a letter in alphabet or the padding symbol!"
#         )
#
#     p = Presentation(list(alphabet))
#     p.contains_empty_word(True)
#     for u, v in relations:
#         presentation.add_rule(p, u, v)
#     tc = ToddCoxeter(congruence_kind.twosided, p)
#
#     result = WordGraph(0, (alphabet_size + 1) ** 2)
#     parent_letter = []
#     nr_vertices = 0
#     que: list[QuadVertex] = [
#         ((rep_automaton.initial_state, 0), (rep_automaton.initial_state, 0))
#     ]
#     new_que = []
#     k = 0
#     done = False
#     vertex_to_idx = []
#     while not done:
#         tc.run_for(timedelta(seconds=1))
#         i = 0
#         new_que.clear()
#         while i < len(que):
#             pass
#         new_que, que = que, new_que
#         i = 0
#         memo = {}
#         while i < len(que):
#             p = que[i]
#             for vertex, que_idx in enumerate(vertex_to_idx):
#                 q = que[que_idx]
#                 if is_sim_a_k_equivalent(
#                     p, q, a, k, rep_automaton, tc.current_word_graph(), memo
#                 ):
#                     parent, letter = parent_letter[i]
#                     parent_target = result.target(parent, letter)
#                     if parent_target == UNDEFINED:
#                         result.target(parent, letter, vertex)
#                     elif parent_target != vertex:
#                         # Got non-determinism, k is too small or ball is too
#                         # small
#                         k += 1
#                         break
