from dataclasses import dataclass
from datetime import timedelta
from itertools import cycle, islice, chain
from typing import Iterator, Iterable
from warnings import warn
from libsemigroups_pybind11 import (
    POSITIVE_INFINITY,
    UNDEFINED,
    Order,
    Paths,
    PositiveInfinity,
    WordGraph,
    word_graph as word_graph_helper,
    ToddCoxeter,
    congruence_kind,
    presentation,
    Presentation,
    WordRange,
)
from libsemigroups_pybind11.word_graph import dot


type Vertex = int
type Letter = int
type Word = tuple[Letter, ...]


def roundrobin(*iterables):
    "Visit input iterables in a cycle until each is exhausted."
    # roundrobin('ABC', 'D', 'EF') → A D E B F C
    # Algorithm credited to George Sakkis
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


@dataclass
class Automaton:
    word_graph: WordGraph
    initial_state: Vertex
    final_states: frozenset[Vertex]

    def accepts(self, word: Word) -> bool:
        target = word_graph_helper.follow_path(
            self.word_graph, self.initial_state, word
        )
        if target == UNDEFINED:
            return False
        return target in self.final_states

    def language(
        self, max_len: int | PositiveInfinity = POSITIVE_INFINITY
    ) -> Iterator[Word]:
        paths = []
        if self.initial_state in self.final_states:
            paths.append(())
        paths.extend(
            [
                Paths(self.word_graph)
                .source(self.initial_state)
                .target(final_state)
                .max(max_len + 1)
                .order(Order.lex)
                for final_state in self.final_states
            ]
        )
        return (tuple(word) for word in roundrobin(*(iter(path) for path in paths)))


def all_words(
    alphabet_size: int, max_size: int | PositiveInfinity = POSITIVE_INFINITY
) -> Iterator[Word]:
    return (
        tuple(word)
        for word in WordRange().alphabet_size(alphabet_size).min(0).max(max_size + 1)
    )


def compute_sim_fingerprint(
    alphabet_size: int,
    u: Word,
    v: Word,
    a: Letter,
    k: int,
    rep_automaton: Automaton,
    cayley_ball: WordGraph,
    all_words_of_length_up_to_k: Iterable[Word] | None = None,
    verbose: bool = True,
) -> tuple[bool, ...]:
    assert all_words_of_length_up_to_k is not None
    if all_words_of_length_up_to_k is None:
        all_words_of_length_up_to_k = tuple(all_words(alphabet_size, k))
    result = []
    for s_idx, s in enumerate(all_words_of_length_up_to_k):
        us = u + s
        if not rep_automaton.accepts(us):
            result.append(False)
            continue
        usa = us
        if a != alphabet_size:
            usa = us + (a,)
        target_usa = word_graph_helper.follow_path(cayley_ball, 0, usa)
        if target_usa == UNDEFINED:
            warn(f"Ball is too small, usa={usa} fell out!", RuntimeWarning)
            result.append(False)
            continue
        for t_idx, t in enumerate(all_words_of_length_up_to_k):
            vt = v + t
            if not rep_automaton.accepts(vt):
                result.append(False)
                continue
            target_vt = word_graph_helper.follow_path(cayley_ball, 0, vt)
            if target_vt == UNDEFINED:
                warn(f"Ball is too small, vt={vt} fell out!", RuntimeWarning)
                result.append(False)
                continue
            result.append(target_usa == target_vt)
    return tuple(result)


def label_from_pair(alphabet_size: int, b: int, c: int) -> int:
    return b * (alphabet_size + 1) + c


def compute_multiplication_automaton(
    alphabet_size: int,
    a: Letter,
    k_1: int,
    k_2: int,
    rep_automaton: Automaton,
    cayley_ball: Word,
    verbose: bool = True,
):
    word_pair_to_fingerprint: dict = {}
    multiplication_wg = WordGraph(0, (alphabet_size + 1) ** 2)
    nr_nodes_defined = 0

    all_words_of_length_up_to_k_1 = tuple(all_words(alphabet_size, k_1))
    all_words_of_length_up_to_k_2 = tuple(all_words(alphabet_size, k_2))

    fingerprint_to_index: dict[tuple[bool, ...], int] = {}
    for u_idx, u in enumerate(all_words_of_length_up_to_k_1):
        for v_idx, v in enumerate(all_words_of_length_up_to_k_1):
            print(
                100
                * (u_idx * len(all_words_of_length_up_to_k_1) + v_idx)
                / (
                    len(all_words_of_length_up_to_k_1)
                    * len(all_words_of_length_up_to_k_1)
                )
            )
            if (u, v) not in word_pair_to_fingerprint:
                word_pair_to_fingerprint[(u, v)] = compute_sim_fingerprint(
                    alphabet_size,
                    u,
                    v,
                    a,
                    k_2,
                    rep_automaton,
                    cayley_ball,
                    all_words_of_length_up_to_k_2,
                )
            fingerprint = word_pair_to_fingerprint[(u, v)]
            if fingerprint not in fingerprint_to_index:
                fingerprint_to_index[fingerprint] = nr_nodes_defined
                multiplication_wg.add_nodes(1)
                nr_nodes_defined += 1

            for b in range(alphabet_size + 1):
                for c in range(alphabet_size + 1):
                    if b == c and b == alphabet_size:
                        continue
                    ub = u
                    if b != alphabet_size:
                        ub = u + (b,)
                    vc = v
                    if c != alphabet_size:
                        vc = v + (c,)

                    if (ub, vc) not in word_pair_to_fingerprint:
                        word_pair_to_fingerprint[(ub, vc)] = compute_sim_fingerprint(
                            alphabet_size,
                            ub,
                            vc,
                            a,
                            k_2,
                            rep_automaton,
                            cayley_ball,
                            all_words_of_length_up_to_k_2,
                        )
                    new_fingerprint = word_pair_to_fingerprint[(ub, vc)]

                    if new_fingerprint not in fingerprint_to_index:
                        fingerprint_to_index[new_fingerprint] = nr_nodes_defined
                        multiplication_wg.add_nodes(1)
                        nr_nodes_defined += 1
                    multiplication_wg.target(
                        fingerprint_to_index[fingerprint],
                        label_from_pair(alphabet_size, b, c),
                        fingerprint_to_index[new_fingerprint],
                    )

    final_states = set()
    for u in rep_automaton.language(k_1):
        for v in rep_automaton.language(k_1):
            ua = u
            if a != alphabet_size:
                ua = u + (a,)
            target_ua = word_graph_helper.follow_path(cayley_ball, 0, ua)
            target_v = word_graph_helper.follow_path(cayley_ball, 0, v)
            if target_ua == target_v:
                if (u, v) not in word_pair_to_fingerprint:
                    word_pair_to_fingerprint[(u, v)] = compute_sim_fingerprint(
                        alphabet_size,
                        u,
                        v,
                        a,
                        k_2,
                        rep_automaton,
                        cayley_ball,
                        all_words_of_length_up_to_k_2,
                    )
                pp_temp = word_pair_to_fingerprint[(u, v)]
                final_states.add(fingerprint_to_index[pp_temp])

    multiplication_automaton = Automaton(
        multiplication_wg,
        fingerprint_to_index[
            compute_sim_fingerprint(
                alphabet_size,
                (),
                (),
                a,
                k_2,
                rep_automaton,
                cayley_ball,
                all_words_of_length_up_to_k_2,
            )
        ],
        frozenset(final_states),
    )
    return multiplication_automaton


def direct_product_automaton(automaton: Automaton) -> Automaton:
    """Given an automaton for L, return a synchronous automaton for L x L."""
    alphabet_size = automaton.word_graph.out_degree()
    result = WordGraph(1, (alphabet_size + 1) ** 2)
    padding_state = automaton.word_graph.number_of_nodes()
    que: list[tuple[Vertex, Vertex]] = [
        (automaton.initial_state, automaton.initial_state)
    ]
    state_pair_to_index: dict[tuple[Vertex, Vertex], Vertex] = {
        (automaton.initial_state, automaton.initial_state): 0
    }
    nr_vertices = 1

    i = 0
    while i < len(que):
        state1, state2 = que[i]
        source = state_pair_to_index[(state1, state2)]
        for b in range(alphabet_size + 1):
            # If we are in padding state, only proceed via padding symbol,
            # if we are not in a padding state, we can only accept a padding symbol
            # if the current state accepts
            if (state1 == padding_state and b != alphabet_size) or (
                state1 != padding_state
                and b == alphabet_size
                and state1 not in automaton.final_states
            ):
                continue
            # Assume we are transitioning to the padding state
            new_state1 = padding_state
            # This if does not trigger if either state1 is the padding state
            # (in which case we will remain in the padding state)
            # or if we are not in the padding state and b is a padding symbol
            # (but this case cannot occur since the previous if did not trigger)
            if state1 != padding_state and b != alphabet_size:
                new_state1 = automaton.word_graph.target(state1, b)
            if new_state1 == UNDEFINED:
                # Fell off the machine, so one of the two words is wrong
                continue
            for c in range(alphabet_size + 1):
                # Same story as before, but also check we aren't traversing via double padding
                if (
                    (state2 == padding_state and c != alphabet_size)
                    or (
                        state2 != padding_state
                        and c == alphabet_size
                        and state2 not in automaton.final_states
                    )
                    or (b == alphabet_size and c == alphabet_size)
                ):
                    continue
                new_state2 = padding_state
                if state2 != padding_state and c != alphabet_size:
                    new_state2 = automaton.word_graph.target(state2, c)
                if new_state2 == UNDEFINED:
                    continue

                if (new_state1, new_state2) not in state_pair_to_index:
                    state_pair_to_index[(new_state1, new_state2)] = nr_vertices
                    que.append((new_state1, new_state2))
                    result.add_nodes(1)
                    nr_vertices += 1
                target = state_pair_to_index[(new_state1, new_state2)]
                result.target(source, label_from_pair(alphabet_size, b, c), target)
        i += 1

    return Automaton(
        result,
        state_pair_to_index[(automaton.initial_state, automaton.initial_state)],
        frozenset(
            state_pair_to_index[(state_1, state_2)]
            for state_1 in chain(automaton.final_states, (padding_state,))
            for state_2 in chain(automaton.final_states, (padding_state,))
            if state_1 != padding_state or state_2 != padding_state
        ),
    )


# TODO: test this
def intersection_automaton(automaton1: Automaton, automaton2: Automaton) -> Automaton:
    assert automaton1.word_graph.out_degree() == automaton2.word_graph.out_degree()
    alphabet_size = automaton1.word_graph.out_degree()
    result = WordGraph(1, alphabet_size)
    state_pair_to_index: dict[tuple[Vertex, Vertex], Vertex] = {
        (automaton1.initial_state, automaton1.initial_state): 0
    }
    que = [(automaton1.initial_state, automaton2.initial_state)]
    nr_vertices = 1
    i = 0
    while i < len(que):
        state1, state2 = que[i]
        source = state_pair_to_index[(state1, state2)]
        for letter in range(alphabet_size):
            new_state1 = automaton1.word_graph.target(state1, letter)
            new_state2 = automaton2.word_graph.target(state2, letter)
            if new_state1 == UNDEFINED or new_state2 == UNDEFINED:
                continue
            if (new_state1, new_state2) not in state_pair_to_index:
                state_pair_to_index[(new_state1, new_state2)] = nr_vertices
                que.append((new_state1, new_state2))
                result.add_nodes(1)
                nr_vertices += 1
            target = state_pair_to_index[(new_state1, new_state2)]
            result.target(source, letter, target)
        i += 1

    return Automaton(
        result,
        state_pair_to_index[(automaton1.initial_state, automaton1.initial_state)],
        frozenset(
            state_pair_to_index[(state_1, state_2)]
            for state_1 in automaton1.final_states
            for state_2 in automaton2.final_states
            if (state_1, state_2) in state_pair_to_index
        ),
    )


# TODO: test this
def trim_automaton(automaton: Automaton) -> Automaton:
    alphabet_size = automaton.word_graph.out_degree()
    reverse_reach_graph = [[] for _ in range(automaton.word_graph.number_of_nodes())]
    for vertex in range(len(reverse_reach_graph)):
        for letter in range(alphabet_size):
            target = automaton.word_graph.target(vertex, letter)
            if target == UNDEFINED:
                continue
            reverse_reach_graph[target].append(vertex)

    seen = set(automaton.final_states)
    que = list(automaton.final_states)
    i = 0
    while i < len(que):
        vertex = que[i]
        for child in reverse_reach_graph[vertex]:
            if child not in seen:
                seen.add(child)
                que.append(child)
        i += 1

    assert automaton.initial_state in seen
    vertex_to_index = {automaton.initial_state: 0}
    nr_vertices = 1
    for vertex in que:
        if vertex not in vertex_to_index:
            vertex_to_index[vertex] = nr_vertices
            nr_vertices += 1

    result = WordGraph(nr_vertices, alphabet_size)
    for vertex in que:
        for letter in range(alphabet_size):
            target = automaton.word_graph.target(vertex, letter)
            if target not in seen:
                continue
            result.target(vertex_to_index[vertex], letter, vertex_to_index[target])
    return Automaton(
        result,
        vertex_to_index[automaton.initial_state],
        frozenset(vertex_to_index[state] for state in automaton.final_states),
    )


if __name__ == "__main__":
    p = Presentation([0, 1])
    presentation.add_rule(p, (0, 1), (1, 0))

    tc = ToddCoxeter(congruence_kind.twosided, p)
    tc.run_for(timedelta(seconds=1))

    wg = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
    automaton = Automaton(wg, 0, frozenset({0, 1}))

    # print(
    #     compute_sim_fingerprint(
    #         3, (0,), (0, 1), 1, 2, automaton, tc.current_word_graph(), 2
    #     )
    # )

    alphabet_size = 2
    k_1 = 3
    k_2 = k_1 * k_1

    multiplication_automata = [
        compute_multiplication_automaton(
            alphabet_size, a, k_1, k_2, automaton, tc.current_word_graph()
        )
        for a in range(alphabet_size + 1)
    ]

    dot(multiplication_automata[0].word_graph).view()
    print(multiplication_automata[0].word_graph)
    print(multiplication_automata[0].initial_state)
    print(multiplication_automata[0].final_states)
