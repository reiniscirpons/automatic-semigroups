from dataclasses import dataclass
from datetime import timedelta
from itertools import cycle, islice
from typing import Iterator, Iterable
from warnings import warn
from libsemigroups_pybind11 import (
    POSITIVE_INFINITY,
    UNDEFINED,
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
        paths = [
            Paths(self.word_graph)
            .source(self.initial_state)
            .target(final_state)
            .max(max_len + 1)
            for final_state in self.final_states
        ]
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
