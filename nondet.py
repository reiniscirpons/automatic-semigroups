"""Nondeterministic word graph computation implementation for automatic semigroups."""

from itertools import chain
from typing import Iterable, Callable, Iterator
from dataclasses import dataclass
import graphviz
from libsemigroups_pybind11 import (
    UNDEFINED,
    ToddCoxeter,
    Presentation,
    congruence_kind,
    presentation,
)
from warnings import warn
from datetime import timedelta

type Vertex = int
type Word[Letter] = Iterable[Letter]


def append_letter[Letter](word: Word[Letter], letter: Letter) -> Word[Letter]:
    return tuple(chain(word, (letter,)))


assert "".join(append_letter("abcd", "a")) == "abcda"
assert tuple(append_letter((1, 2, 3, 4, 5), 7)) == (1, 2, 3, 4, 5, 7)


def concatenate_words[Letter](*words: Word[Letter]) -> Word[Letter]:
    return tuple(chain(*words))


assert "".join(concatenate_words("abcd", "abba", "cda")) == "abcdabbacda"
assert tuple(
    concatenate_words(
        (1, 2, 3, 4, 5),
        (2, 3),
        (1, 1),
        (),
        (3, 4),
    )
) == (1, 2, 3, 4, 5, 2, 3, 1, 1, 3, 4)


class NondeterministicWordGraph[Letter]:
    def __init__(self, alphabet: Iterable[Letter]):
        self._alphabet: tuple[Letter, ...] = tuple(alphabet)
        seen = set()
        for letter in self._alphabet:
            if letter in seen:
                raise RuntimeError(f"Alphabet not duplicate free, alphabet={alphabet}")
            seen.add(letter)
        self._targets: list[dict[Letter, set[Vertex]]] = []

    @property
    def alphabet(self) -> Iterable[Letter]:
        return self._alphabet

    @property
    def vertices(self) -> Iterable[Vertex]:
        return range(len(self._targets))

    def new_node(self) -> Vertex:
        self._targets.append({letter: set() for letter in self._alphabet})
        return len(self._targets) - 1

    def targets(self, vertex: Vertex, letter: Letter) -> set[Vertex]:
        return self._targets[vertex][letter]

    def add_edge(self, source: Vertex, letter: Letter, target: Vertex) -> None:
        # print()
        # print(*self._targets, sep="\n")
        # print(source, letter, target)
        self._targets[source][letter].add(target)

    def follow(
        self, vertex: Vertex, word: Word[Letter], padding_symbol: Letter | None = None
    ) -> set[Vertex]:
        current_set = {vertex}
        new_set = set()
        for letter in word:
            if padding_symbol is not None and letter == padding_symbol:
                continue
            new_set.clear()
            for current_vertex in current_set:
                new_set |= self.targets(current_vertex, letter)
            current_set, new_set = new_set, current_set
        return current_set

    def __str__(self):
        return str(self._targets)

    def dot(self) -> graphviz.Digraph:
        _COLOR_NAMES = (
            "red",
            "blue",
            "orange",
            "green",
            "magenta",
            "cyan",
            "yellow",
            "purple",
            "brown",
            "teal",
        )
        result = graphviz.Digraph()
        for vertex in self.vertices:
            result.node(str(vertex))
        for source in self.vertices:
            for letter, color_name in zip(self.alphabet, _COLOR_NAMES):
                for target in self.targets(source, letter):
                    result.edge(str(source), str(target), color=color_name)
        print(*zip(self.alphabet, _COLOR_NAMES), sep="\n")
        return result


W_wg = NondeterministicWordGraph[str]("ab")
W_wg.new_node()
W_wg.new_node()
W_wg.add_edge(0, "a", 0)
W_wg.add_edge(0, "b", 1)
W_wg.add_edge(1, "b", 1)

assert W_wg.follow(0, "abba") == set()
assert W_wg.follow(0, "abbb") == {1}
assert W_wg.follow(0, "abbbb") == {1}
assert W_wg.follow(0, "aaabbbb") == {1}
assert W_wg.follow(0, "aaaabbbb") == {1}
assert W_wg.follow(0, "aa") == {0}
assert W_wg.follow(0, "aaaa") == {0}


def cayley_graph_ball[T](
    alphabet: Iterable[T],
    word_problem_oracle: Callable[[Word[T], Word[T]], bool],
    radius: int,
) -> NondeterministicWordGraph[T]:
    result = NondeterministicWordGraph[T](alphabet)
    initial = result.new_node()
    assert initial == 0
    shortlex_word: list[Word[T]] = [()]

    que = [(initial, 0)]
    i = 0
    while i < len(que):
        source, depth = que[i]
        if depth > radius:
            return result
        word = shortlex_word[source]
        for letter in alphabet:
            new_word = append_letter(word, letter)
            for target, target_word in enumerate(shortlex_word):
                if word_problem_oracle(new_word, target_word):
                    result.add_edge(source, letter, target)
                    break
            else:
                target = result.new_node()
                shortlex_word.append(new_word)
                result.add_edge(source, letter, target)
                que.append((target, depth + 1))
        i += 1
    # Finite semigroup
    return result


def letter_counts[T](word: Word[T]) -> dict[T, int]:
    result = {}
    for letter in word:
        if letter not in result:
            result[letter] = 0
        result[letter] += 1
    return result


def free_commutative_monoid_word_problem_oracle[T](
    word1: Word[T], word2: Word[T]
) -> bool:
    return letter_counts(word1) == letter_counts(word2)


ball = cayley_graph_ball("ab", free_commutative_monoid_word_problem_oracle, 2)
# ball.dot().view()


@dataclass
class NondeterministicAutomaton[Letter]:
    word_graph: NondeterministicWordGraph[Letter]
    initial_state: Vertex
    final_states: set[Vertex]

    def accepts(self, word: Word[Letter], padding_symbol: Letter | None = None) -> bool:
        return (
            len(
                self.word_graph.follow(self.initial_state, word, padding_symbol)
                & self.final_states
            )
            != 0
        )

    def language(self, max_len: int | None = None) -> Iterator[Word[Letter]]:
        que: list[tuple[Vertex, Letter | None, int | None]] = [
            (self.initial_state, None, None)
        ]
        i = 0
        while i < len(que):
            source, parent_letter, parent = que[i]
            if source in self.final_states:
                word = []
                while parent is not None:
                    assert parent_letter is not None
                    word.append(parent_letter)
                    _, parent_letter, parent = que[parent]
                if max_len is not None and len(word) > max_len:
                    return
                yield tuple(reversed(word))
            for letter in self.word_graph.alphabet:
                for target in self.word_graph.targets(source, letter):
                    que.append((target, letter, i))
            i += 1


W = NondeterministicAutomaton(W_wg, 0, {0, 1})
assert tuple("".join(x) for x in W.language(3)) == (
    "",
    "a",
    "b",
    "aa",
    "ab",
    "bb",
    "aaa",
    "aab",
    "abb",
    "bbb",
)

total_wg = NondeterministicWordGraph("ab")
total_wg.new_node()
total_wg.add_edge(0, "a", 0)
total_wg.add_edge(0, "b", 0)
total_automaton = NondeterministicAutomaton(total_wg, 0, {0})


def all_words[T](
    alphabet: Iterable[T], max_len: int | None = None
) -> Iterable[Word[T]]:
    total_wg = NondeterministicWordGraph(alphabet)
    total_wg.new_node()
    for letter in alphabet:
        total_wg.add_edge(0, letter, 0)
    total_automaton = NondeterministicAutomaton(total_wg, 0, {0})
    return total_automaton.language(max_len)


assert tuple("".join(x) for x in all_words("ab", 2)) == (
    "",
    "a",
    "b",
    "aa",
    "ab",
    "ba",
    "bb",
)


def compute_sim_fingerprint[T](
    u: Word[T],
    v: Word[T],
    a: T,
    k: int,
    rep_automaton: NondeterministicAutomaton[T],
    cayley_ball: NondeterministicWordGraph[T],
    padding_symbol: T,
    all_words_of_length_up_to_k: Iterable[Word[T]] | None = None,
) -> tuple[bool, ...]:
    if all_words_of_length_up_to_k is None:
        all_words_of_length_up_to_k = tuple(all_words(cayley_ball.alphabet, k))
    result = []
    for s in all_words_of_length_up_to_k:
        for t in all_words_of_length_up_to_k:
            us = tuple(concatenate_words(u, s))
            vt = tuple(concatenate_words(v, t))
            usa = us
            if a != padding_symbol:
                usa = tuple(append_letter(us, a))
            reach_usa = cayley_ball.follow(0, usa, padding_symbol)
            if len(reach_usa) == 0:
                warn(f"Ball is too small, usa={usa} fell out!", RuntimeWarning)
                result.append(False)
                continue
            reach_vt = cayley_ball.follow(0, vt, padding_symbol)
            if len(reach_vt) == 0:
                warn(f"Ball is too small, vt={vt} fell out!", RuntimeWarning)
                result.append(False)
                continue
            result.append(
                rep_automaton.accepts(us, padding_symbol)
                and rep_automaton.accepts(vt, padding_symbol)
                and len(reach_usa & reach_vt) != 0
            )
    return tuple(result)


k_1 = 2
k_2 = 2 * k_1
alphabet = "ab"
padding_symbol = "$"
padded_alphabet = alphabet + padding_symbol
ball = cayley_graph_ball(
    alphabet, free_commutative_monoid_word_problem_oracle, k_1 + k_2 + 2
)

all_words_of_length_up_to_3 = tuple(all_words(alphabet, 3))
print(
    compute_sim_fingerprint(
        "aa", "b", "a", 3 * 3, W, ball, "$", all_words_of_length_up_to_3
    )
)


def compute_multiplication_automaton[T](
    alphabet: Iterable[T],
    a: T,
    k_1: int,
    k_2: int,
    rep_automaton: NondeterministicAutomaton[T],
    cayley_ball: NondeterministicWordGraph[T],
    padding_symbol: T,
):
    padded_alphabet = tuple(append_letter(alphabet, padding_symbol))
    padded_pair_alphabet = tuple(
        (a, b) for a in padded_alphabet for b in padded_alphabet
    )
    multiplication_wg = NondeterministicWordGraph(padded_pair_alphabet)

    all_words_of_length_up_to_k_1 = tuple(all_words(alphabet, k_1))
    all_words_of_length_up_to_k_2 = tuple(all_words(alphabet, k_2))

    fingerprint_to_index: dict[tuple[bool, ...], int] = {}
    for u in all_words_of_length_up_to_k_1:
        for v in all_words_of_length_up_to_k_1:
            fingerprint = compute_sim_fingerprint(
                u,
                v,
                a,
                k_2,
                rep_automaton,
                cayley_ball,
                padding_symbol,
                all_words_of_length_up_to_k_2,
            )
            if fingerprint not in fingerprint_to_index:
                fingerprint_to_index[fingerprint] = multiplication_wg.new_node()
            for b, c in padded_pair_alphabet:
                if b == c and b == padding_symbol:
                    continue
                ub = tuple(append_letter(u, b))
                vc = tuple(append_letter(v, c))
                new_fingerprint = compute_sim_fingerprint(
                    ub,
                    vc,
                    a,
                    k_2,
                    rep_automaton,
                    cayley_ball,
                    padding_symbol,
                    all_words_of_length_up_to_k_2,
                )
                if new_fingerprint not in fingerprint_to_index:
                    fingerprint_to_index[new_fingerprint] = multiplication_wg.new_node()
                multiplication_wg.add_edge(
                    fingerprint_to_index[fingerprint],
                    (b, c),
                    fingerprint_to_index[new_fingerprint],
                )

    final_states = set()
    for u in rep_automaton.language(k_1):
        for v in rep_automaton.language(k_1):
            ua = tuple(append_letter(u, a))
            reach_ua = cayley_ball.follow(0, ua, padding_symbol)
            reach_v = cayley_ball.follow(0, v, padding_symbol)
            if len(reach_ua & reach_v) != 0:
                pp_temp = compute_sim_fingerprint(
                    u,
                    v,
                    a,
                    k_2,
                    rep_automaton,
                    cayley_ball,
                    padding_symbol,
                    all_words_of_length_up_to_k_2,
                )
                print(a, u, v)

                final_states.add(fingerprint_to_index[pp_temp])

    multiplication_automaton = NondeterministicAutomaton(
        multiplication_wg,
        fingerprint_to_index[
            compute_sim_fingerprint(
                (),
                (),
                a,
                k_2,
                rep_automaton,
                cayley_ball,
                padding_symbol,
                all_words_of_length_up_to_k_2,
            )
        ],
        final_states.copy(),
    )
    return multiplication_automaton


# def construct_pair_automaton[T](
#     W: NondeterministicAutomaton[T], padding_symbol: T
# ) -> NondeterministicAutomaton[tuple[T, T]]:
#     padded_alphabet = tuple(append_letter(W.word_graph.alphabet, padding_symbol))
#     padded_pair_alphabet = tuple(
#         (a, b) for a in padded_alphabet for b in padded_alphabet
#     )
#     result = NondeterministicAutomaton

multiplication_automata = {
    a: compute_multiplication_automaton(alphabet, a, k_1, k_2, W, ball, padding_symbol)
    for a in padded_alphabet
}

# multiplication_automata["$"].word_graph.dot().view()
# print(multiplication_automata["$"].word_graph)
# print(multiplication_automata["$"].initial_state)
# print(multiplication_automata["$"].final_states)

# multiplication_automata["a"].word_graph.dot().view()
print(multiplication_automata["a"].word_graph)
print(multiplication_automata["a"].initial_state)
print(multiplication_automata["a"].final_states)


def direct_product_automaton[T](
    automaton: NondeterministicAutomaton[T], padding_symbol: T
) -> NondeterministicAutomaton[tuple[T, T]]:
    padded_alphabet = tuple(
        append_letter(automaton.word_graph.alphabet, padding_symbol)
    )
    padded_pair_alphabet = tuple(
        (a, b) for a in padded_alphabet for b in padded_alphabet
    )
    result = NondeterministicWordGraph(padded_pair_alphabet)
    state_pair_to_index: dict[tuple[Vertex | None, Vertex | None], Vertex] = {}
    que: list[tuple[Vertex | None, Vertex | None]] = [
        (automaton.initial_state, automaton.initial_state)
    ]
    state_pair_to_index[(automaton.initial_state, automaton.initial_state)] = (
        result.new_node()
    )
    i = 0
    while i < len(que):
        state1, state2 = que[i]
        source = state_pair_to_index[(state1, state2)]
        for b, c in padded_pair_alphabet:
            if (
                (state1 is None and b != padding_symbol)
                or (state2 is None and c != padding_symbol)
                or (
                    state1 is not None
                    and b == padding_symbol
                    and state1 not in automaton.final_states
                )
                or (
                    state2 is not None
                    and c == padding_symbol
                    and state2 not in automaton.final_states
                )
                or (b == padding_symbol and c == padding_symbol)
            ):
                continue

            new_states_1 = [None]
            if state1 is not None and b != padding_symbol:
                new_states_1 = automaton.word_graph.targets(state1, b)

            new_states_2 = [None]
            if state2 is not None and c != padding_symbol:
                new_states_2 = automaton.word_graph.targets(state2, c)

            for new_state_1 in new_states_1:
                for new_state_2 in new_states_2:
                    if (new_state_1, new_state_2) not in state_pair_to_index:
                        state_pair_to_index[(new_state_1, new_state_2)] = (
                            result.new_node()
                        )
                        que.append((new_state_1, new_state_2))
                    target = state_pair_to_index[(new_state_1, new_state_2)]
                    result.add_edge(source, (b, c), target)
        i += 1

    return NondeterministicAutomaton(
        result,
        state_pair_to_index[(automaton.initial_state, automaton.initial_state)],
        set(
            state_pair_to_index[(state_1, state_2)]
            for state_1 in chain(automaton.final_states, (None,))
            for state_2 in chain(automaton.final_states, (None,))
            if state_1 is not None or state_2 is not None
        ),
    )


def deinterleave_word[T](pair_word: Iterable[tuple[T, T]]) -> tuple[Word[T], Word[T]]:
    return (tuple(a for a, _ in pair_word), tuple(a for _, a in pair_word))


rep_pair_automaton = direct_product_automaton(W, "$")
assert tuple(
    map(
        lambda w: tuple("".join(x) for x in deinterleave_word(w)),
        rep_pair_automaton.language(2),
    )
) == (
    ("", ""),
    ("a", "a"),
    ("a", "b"),
    ("a", "$"),
    ("b", "a"),
    ("b", "b"),
    ("b", "$"),
    ("$", "a"),
    ("$", "b"),
    ("aa", "aa"),
    ("aa", "ab"),
    ("aa", "a$"),
    ("ab", "aa"),
    ("ab", "ab"),
    ("ab", "a$"),
    ("a$", "aa"),
    ("a$", "ab"),
    ("aa", "bb"),
    ("aa", "b$"),
    ("ab", "bb"),
    ("ab", "b$"),
    ("a$", "bb"),
    ("aa", "$$"),
    ("ab", "$$"),
    ("bb", "aa"),
    ("bb", "ab"),
    ("bb", "a$"),
    ("b$", "aa"),
    ("b$", "ab"),
    ("bb", "bb"),
    ("bb", "b$"),
    ("b$", "bb"),
    ("bb", "$$"),
    ("$$", "aa"),
    ("$$", "ab"),
    ("$$", "bb"),
)


def automaton_intersection[T](
    automaton1: NondeterministicAutomaton[T], automaton2: NondeterministicAutomaton[T]
) -> NondeterministicAutomaton[T]:
    assert set(automaton1.word_graph.alphabet) == set(automaton2.word_graph.alphabet)
    result = NondeterministicWordGraph(automaton1.word_graph.alphabet)
    state_pair_to_index: dict[tuple[Vertex, Vertex], Vertex] = {}
    que: list[tuple[Vertex, Vertex]] = [
        (automaton1.initial_state, automaton1.initial_state)
    ]
    state_pair_to_index[(automaton1.initial_state, automaton1.initial_state)] = (
        result.new_node()
    )
    i = 0
    while i < len(que):
        state1, state2 = que[i]
        source = state_pair_to_index[(state1, state2)]
        for letter in result.alphabet:
            for new_state_1 in automaton1.word_graph.targets(state1, letter):
                for new_state_2 in automaton2.word_graph.targets(state2, letter):
                    if (new_state_1, new_state_2) not in state_pair_to_index:
                        state_pair_to_index[(new_state_1, new_state_2)] = (
                            result.new_node()
                        )
                        que.append((new_state_1, new_state_2))
                    target = state_pair_to_index[(new_state_1, new_state_2)]
                    result.add_edge(source, letter, target)
        i += 1
    return NondeterministicAutomaton(
        result,
        state_pair_to_index[(automaton1.initial_state, automaton2.initial_state)],
        set(
            state_pair_to_index[(state_1, state_2)]
            for state_1 in automaton1.final_states
            for state_2 in automaton2.final_states
            if (state_1, state_2) in state_pair_to_index
        ),
    )


intersection_W_ab_star = automaton_intersection(W, total_automaton)
print(tuple("".join(w) for w in intersection_W_ab_star.language(2)))


for letter in padded_alphabet:
    multiplication_automata[letter] = automaton_intersection(
        multiplication_automata[letter], rep_pair_automaton
    )

# multiplication_automata["a"].word_graph.dot().view()
print(multiplication_automata["a"].word_graph)
print(multiplication_automata["a"].initial_state)
print(multiplication_automata["a"].final_states)


def trim_automaton[T](
    automaton: NondeterministicAutomaton[T],
) -> NondeterministicAutomaton[T]:
    rev: list[dict[T, list[Vertex]]] = [
        {letter: [] for letter in automaton.word_graph.alphabet}
        for _ in automaton.word_graph.vertices
    ]
    for source in automaton.word_graph.vertices:
        for letter in automaton.word_graph.alphabet:
            for target in automaton.word_graph.targets(source, letter):
                rev[target][letter].append(source)

    que = list(automaton.final_states)
    seen = set()
    i = 0
    while i < len(que):
        state = que[i]
        for letter in automaton.word_graph.alphabet:
            for target in rev[state][letter]:
                if target not in seen:
                    que.append(target)
                    seen.add(target)
        i += 1

    final_reachable_set = set(que)
    assert automaton.initial_state in final_reachable_set
    result = NondeterministicWordGraph(automaton.word_graph.alphabet)
    state_to_index = {}
    state_to_index[automaton.initial_state] = result.new_node()
    que = [automaton.initial_state]
    i = 0
    while i < len(que):
        source = que[i]
        for letter in automaton.word_graph.alphabet:
            for target in automaton.word_graph.targets(source, letter):
                if target not in final_reachable_set:
                    continue
                if target not in state_to_index:
                    state_to_index[target] = result.new_node()
                    que.append(target)
                result.add_edge(state_to_index[source], letter, state_to_index[target])
        i += 1
    return NondeterministicAutomaton(
        result,
        state_to_index[automaton.initial_state],
        {state_to_index[state] for state in automaton.final_states},
    )


# trim_automaton(multiplication_automata["a"]).word_graph.dot().view()
for letter in padded_alphabet:
    multiplication_automata[letter] = trim_automaton(multiplication_automata[letter])

# multiplication_automata["b"].word_graph.dot().view()
print(multiplication_automata["b"].word_graph)
print(multiplication_automata["b"].initial_state)
print(multiplication_automata["b"].final_states)


def construct_reduced_multiplication_automata[T](
    alphabet: Iterable[T],
    padding_symbol: T,
    k_1: int,
    k_2: int,
    ball: NondeterministicWordGraph[T],
    rep_automaton: NondeterministicAutomaton[T],
) -> dict[T, NondeterministicAutomaton[tuple[T, T]]]:
    padded_alphabet = tuple(append_letter(alphabet, padding_symbol))
    multiplication_automata = {
        a: compute_multiplication_automaton(
            alphabet, a, k_1, k_2, rep_automaton, ball, padding_symbol
        )
        for a in padded_alphabet
    }
    rep_pair_automaton = direct_product_automaton(rep_automaton, padding_symbol)
    for letter in padded_alphabet:
        multiplication_automata[letter] = automaton_intersection(
            multiplication_automata[letter], rep_pair_automaton
        )
    for letter in padded_alphabet:
        multiplication_automata[letter] = trim_automaton(
            multiplication_automata[letter]
        )
    return multiplication_automata


# alphabet = "ab"
# k_1 = 2
# k_2 = k_1 * k_1
# rep_word_graph = NondeterministicWordGraph(alphabet)
# rep_word_graph.new_node()
# rep_word_graph.new_node()
# rep_word_graph.add_edge(0, "a", 0)
# rep_word_graph.add_edge(0, "b", 1)
# rep_word_graph.add_edge(1, "b", 1)
# rep_automaton = NondeterministicAutomaton(rep_word_graph, 0, {0, 1})
# multiplication_automata = construct_reduced_multiplication_automata(
#     "ab",
#     "$",
#     k_1,
#     k_2,
#     cayley_graph_ball(
#         alphabet, free_commutative_monoid_word_problem_oracle, k_1 + k_2 + 2
#     ),
#     rep_automaton,
# )

# multiplication_automata["a"].word_graph.dot().view()
# print(multiplication_automata["a"].word_graph)
# print(multiplication_automata["a"].initial_state)
# print(multiplication_automata["a"].final_states)


def todd_coxeter_to_nondet_word_graph[T](
    alphabet: Iterable[T], tc: ToddCoxeter
) -> NondeterministicWordGraph[T]:
    tcwg = tc.current_word_graph()
    wg = NondeterministicWordGraph(alphabet)
    letter_to_index = {letter: idx for idx, letter in enumerate(alphabet)}
    for _ in range(tcwg.number_of_nodes()):
        # assumes nodes homogeneous
        wg.new_node()
    # assumes initial node 0
    for vertex in tcwg.nodes():
        for letter in alphabet:
            target = tcwg.target(vertex, letter_to_index[letter])
            if target != UNDEFINED:
                wg.add_edge(vertex, letter, target)
    return wg


# alphabet = "ab"
# k_1 = 4
# k_2 = 8
# rep_word_graph = NondeterministicWordGraph(alphabet)
# rep_word_graph.new_node()
# rep_word_graph.new_node()
# rep_word_graph.new_node()
# rep_word_graph.add_edge(0, "a", 0)
# rep_word_graph.add_edge(0, "b", 1)
# rep_word_graph.add_edge(1, "b", 1)
# rep_word_graph.add_edge(1, "a", 2)
# rep_word_graph.add_edge(2, "a", 0)
# rep_automaton = NondeterministicAutomaton(rep_word_graph, 0, {0, 1, 2})
# relations = [("aba", "bab")]
#
#
# p = Presentation(alphabet)
# presentation.add_rule(p, "aba", "bab")
# tc = ToddCoxeter(word=str)
# tc.init(congruence_kind.twosided, p)
# tc.strategy(tc.options.strategy.felsch)
# tc.run_until(lambda: tc.number_of_nodes_active() > 10000)
#
#
# ball = todd_coxeter_to_nondet_word_graph(alphabet, tc)
# multiplication_automata = construct_reduced_multiplication_automata(
#     alphabet,
#     "$",
#     k_1,
#     k_2,
#     ball,
#     rep_automaton,
# )
#
# multiplication_automata["$"].word_graph.dot().view()
# print(multiplication_automata["$"].word_graph)
# print(multiplication_automata["$"].initial_state)
# print(multiplication_automata["$"].final_states)

alphabet = "abAB"
k_1 = 1
k_2 = 1
rep_word_graph = NondeterministicWordGraph(alphabet)
rep_word_graph.new_node()
rep_word_graph.new_node()
rep_word_graph.new_node()
rep_word_graph.new_node()
rep_word_graph.new_node()
rep_word_graph.add_edge(0, "a", 1)
rep_word_graph.add_edge(0, "A", 2)
rep_word_graph.add_edge(1, "a", 1)
rep_word_graph.add_edge(2, "A", 2)
rep_word_graph.add_edge(1, "b", 3)
rep_word_graph.add_edge(1, "B", 4)
rep_word_graph.add_edge(2, "b", 3)
rep_word_graph.add_edge(2, "B", 4)
rep_word_graph.add_edge(3, "b", 3)
rep_word_graph.add_edge(4, "B", 4)
rep_automaton = NondeterministicAutomaton(rep_word_graph, 0, {0, 1, 2, 3, 4})
relations = [("ab", "ba"), ("Aa", ""), ("aA", ""), ("Bb", ""), ("bB", "")]
p = Presentation(alphabet)
p.contains_empty_word(True)
presentation.add_rule(p, "ab", "ba")
presentation.add_rule(p, "aA", "")
presentation.add_rule(p, "Aa", "")
presentation.add_rule(p, "bB", "")
presentation.add_rule(p, "Bb", "")
tc = ToddCoxeter(word=str)
tc.init(congruence_kind.twosided, p)
tc.strategy(tc.options.strategy.felsch)
tc.run_until(lambda: tc.number_of_nodes_active() > 1000)


ball = todd_coxeter_to_nondet_word_graph(alphabet, tc)
multiplication_automata = construct_reduced_multiplication_automata(
    alphabet,
    "$",
    k_1,
    k_2,
    ball,
    rep_automaton,
)

multiplication_automata["$"].word_graph.dot().view()
print(multiplication_automata["$"].word_graph)
print(multiplication_automata["$"].initial_state)
print(multiplication_automata["$"].final_states)
