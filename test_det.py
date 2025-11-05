from typing import Iterable
from libsemigroups_pybind11 import WordGraph, UNDEFINED
from libsemigroups_pybind11.word_graph import dot
from det import Automaton, all_words, direct_product_automaton


def test_automaton():
    wg = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
    automaton = Automaton(wg, 0, frozenset({0, 1}))
    assert automaton.accepts((0, 0))
    assert automaton.accepts(())
    assert automaton.accepts((0, 0, 1))
    assert automaton.accepts((0, 0, 1, 1, 1))
    assert automaton.accepts((1, 1, 1))
    assert not automaton.accepts((0, 0, 1, 1, 1, 0))
    assert not automaton.accepts((1, 1, 1, 0))

    assert list(sorted(automaton.language(3))) == [
        (),
        (0,),
        (0, 0),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1),
        (0, 1, 1),
        (1,),
        (1, 1),
        (1, 1, 1),
    ]


def test_all_words():
    assert list(all_words(2, 3)) == [
        (),
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]


def letter_to_letter_pair(alphabet_size: int, letter: int) -> tuple[int, int]:
    return letter // (alphabet_size + 1), letter % (alphabet_size + 1)


def pair_word_to_word_pair(
    alphabet_size: int,
    words: Iterable[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    result = []
    for word in words:
        u = []
        v = []
        for letter in word:
            a, b = letter_to_letter_pair(alphabet_size, letter)
            if a != alphabet_size:
                u.append(a)
            if b != alphabet_size:
                v.append(b)
        result.append((tuple(u), tuple(v)))
    return tuple(result)


# 0 (0, 0) #00ff00 lime green
# 1 (0, 1) #ff00ff magenta
# 2 (0, $) #007fff blue
# 3 (1, 0) #ff7f00 orange
# 4 (1, 1) #7fbf7f light green
# 5 (1, $) #4604ac dark purple
# 6 ($, 0) #de0328 red
# 7 ($, 1) #19801d dark green
# 8 ($, $) #d881f5 light purple

# static constexpr std::array<std::string_view, 24> colors
#         = {"#00ff00", "#ff00ff", "#007fff", "#ff7f00", "#7fbf7f", "#4604ac",
#            "#de0328", "#19801d", "#d881f5", "#00ffff", "#ffff00", "#00ff7f",
#            "#ad5867", "#85f610", "#84e9f5", "#f5c778", "#207090", "#764ef3",
#            "#7b4c00", "#0000ff", "#b80c9a", "#601045", "#29b7c0", "#839f12"};


def test_direct_product_automaton():
    wg = WordGraph(2, [[0, 1], [UNDEFINED, 1]])
    automaton = Automaton(wg, 0, frozenset({0, 1}))
    prod_automaton = direct_product_automaton(automaton)
    dot(prod_automaton.word_graph).view()
    print(prod_automaton.word_graph)
    print(prod_automaton.initial_state)
    print(prod_automaton.final_states)

    words = list(automaton.language(4))
    word_pairs = list(sorted((u, v) for u in words for v in words))

    assert (
        list(sorted(pair_word_to_word_pair(2, prod_automaton.language(4))))
        == word_pairs
    )
