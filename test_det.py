from libsemigroups_pybind11 import WordGraph, UNDEFINED
from det import Automaton, all_words


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
        (0),
        (0, 0),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1),
        (0, 1, 1),
        (1),
        (1, 1),
        (1, 1, 1),
    ]


def test_all_words():
    assert list(all_words(2, 3)) == [
        (),
        (0),
        (1),
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
