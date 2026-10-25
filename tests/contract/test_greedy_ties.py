"""The greedy-token comparison's bfloat16 tie tolerance (tests/greedy_ties.py), on synthetic scores."""
from types import SimpleNamespace

import numpy as np
import pytest

from monolith.bench import bf16_ulp_diff
from monolith.formats.fp import bf16_to_f32
from tests.greedy_ties import agree

TIE = [0x3F00, 0x3F80, 0x3F7F, 0x3F7E]     # 0.5, 1.0 (the argmax), one and two bfloat16 steps below it
EQ = [0x3F00, 0x3F80, 0x3F80, 0x3F7E]      # two equal best scores
FAR = [0x3F80, 0x3F00, 0x3F00, 0x3F00]     # token 0 far ahead


def _reference(rows):
    bits, calls, reads = np.array(rows, np.uint16).tobytes(), [], []

    def read(name):
        assert name == 'logits'
        reads.append(calls[-1])
        return bits

    def generate(ids, n):
        calls.append((list(ids), n))
        return SimpleNamespace(tokens=[1] * n)
    return SimpleNamespace(read=read, generate=generate, calls=calls, reads=reads, prefill_chunk_size=8,
                           model=SimpleNamespace(config=SimpleNamespace(vocab_size=4)))


def _rows_with(at, row):
    return [row if k == at else FAR for k in range(8)]


@pytest.mark.parametrize('row', [TIE, [0xC000, 0xBF80, 0xBF81, 0xBF82]])   # the second below zero: the sign mapping
def test_one_bfloat16_step_is_a_tie_and_two_are_not(row):
    agree(_reference([row] * 8), [0] * 5, [2], [1], 1)
    for got, expected, ties in (([3], [1], 1), ([1], [2], 1), ([2], [1], 0)):   # two steps apart; not the argmax; no budget
        with pytest.raises(AssertionError):
            agree(_reference([row] * 8), [0] * 5, got, expected, ties)


@pytest.mark.parametrize('n,got,right,wrong', [(5, [2], 4, 0),        # token 0: the prompt's last row, (5 - 1) % 8
                                               (5, [1, 2], 0, 4),     # a later token: the decode step's row 0
                                               (13, [2], 4, 0)])      # a prompt past one chunk: its last chunk's row, 12 % 8
def test_the_scores_come_from_the_step_that_chose_the_token(n, got, right, wrong):
    agree(_reference(_rows_with(right, TIE)), [0] * n, got, [1] * len(got), 1)
    with pytest.raises(AssertionError):
        agree(_reference(_rows_with(wrong, TIE)), [0] * n, got, [1] * len(got), 1)


def test_equal_scores_lengths_and_what_follows_a_tie():
    agree(_reference([EQ] * 8), [0] * 5, [2], [1], 1)                     # equal best scores are a tie
    with pytest.raises(AssertionError, match='tokens from 0 on: 1 against 2'):
        agree(_reference([TIE] * 8), [0] * 5, [2], [1, 1], 1)            # lengths must match
    ref = _reference([TIE] * 8)
    agree(ref, [0] * 5, [2, 1], [1, 1], 1)
    assert ([0] * 5 + [2], 1) in ref.calls                                   # the rest is regenerated from the session's token
    with pytest.raises(AssertionError, match='token 1: 3 against 1, 2 bfloat16 steps apart'):
        agree(_reference([TIE] * 8), [0] * 5, [2, 3], [1, 1], 2)         # a two-step gap after a tie fails with budget left


def test_the_rest_is_compared_with_the_rerun_from_the_session_token():
    # a continuation that follows the prompt's last token: the rerun after the session's 2 continues with 2, expected with 1
    ref = _reference([TIE] * 8)
    record = ref.generate
    ref.generate = lambda ids, n: SimpleNamespace(tokens=[ids[-1]] * len(record(ids, n).tokens))
    agree(ref, [0] * 5, [2, 2], [1, 1], 1)
    with pytest.raises(AssertionError, match=r'tokens from 1 on: \[1\] against \[2\]'):
        agree(ref, [0] * 5, [2, 1], [1, 1], 1)


def test_the_tie_budget():
    agree(_reference([TIE] * 8), [0] * 5, [2, 2], [1, 1], 2)                # two ties within a budget of two
    with pytest.raises(AssertionError):
        agree(_reference([TIE] * 8), [0] * 5, [2, 2], [1, 1], 1)           # the second exceeds a budget of one


@pytest.mark.parametrize('got,at', [([2], 0), ([1, 2], 1)])
def test_the_scores_are_read_after_a_rerun_that_stops_at_the_differing_token(got, at):
    ref = _reference([TIE] * 8)
    agree(ref, [0] * 5, got, [1] * len(got), 1)
    assert ref.reads == [([0] * 5, at + 1)], ref.calls                       # the step that chose expected[at] ran last


@pytest.mark.parametrize('token', [4, -2])                                  # -2 would index token 2, one step below the argmax
def test_a_token_outside_the_vocabulary_is_never_a_tie(token):
    with pytest.raises(AssertionError, match='outside the vocabulary'):
        agree(_reference([TIE] * 8), [0] * 5, [token], [1], 1)


def test_bf16_ulp_diff_counts_bfloat16_steps_across_zero():
    x = bf16_to_f32(np.array([0xBF81, 0xBF80, 0x8000, 0x0000, 0x3F80, 0x3F81], np.uint16))   # -1.0078, -1, -0, +0, 1, 1.0078
    assert list(bf16_ulp_diff(x[:-1], x[1:])) == [1, 128 * 127, 0, 128 * 127, 1]
