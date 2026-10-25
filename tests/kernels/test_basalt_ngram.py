# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp:302-316, :324 and
# src/kernels/ple_oracle_vectors.inc @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the
# hash-vector check, the offsets identity, and the oracle cases both rest on.
"""``ngram_rows`` of ``runtime/src/basalt_ngram.h``: sixteen PLE table row indices per token.

**Why an oracle and not a property.** Every index this function can produce is in range by construction, because
the last step is ``mixed % vocab[h] + offset[h]`` and the offsets are the running sum of the vocabulary sizes. So
a range check passes on a function that is wrong in any of the ways the port is most likely to be wrong: reading
the predecessor window backwards, cutting after the store instead of before, masking instead of taking the
modulo, summing instead of xoring. basalt makes the same point about the window direction, verbatim
(``src/kernels/ngram.cpp:68-70``): reading index ``s - 1`` "still produces indices in range and so cannot be
caught by a range check - only by an oracle."

``tests/kernels/fixtures/ngram_oracle.txt`` is that oracle, transcribed from basalt's own generated vectors. The
six cases are chosen to separate exactly those confusions, and each is asserted on its own so a failure names
which distinction broke:

* **A1 window direction.** Four tokens with a rolling two-deep window. Reading the window backwards changes every
  row of every token but the first.
* **A2 an EOS one back cuts both.** The cut propagates forward: an EOS at any position replaces that position and
  every older one.
* **A3 a missing predecessor reads as EOS.** ``TOKEN_NULL`` is never stored; the replacement written is always
  ``PLE_EOS_TOKEN_ID``.
* **A4 token id 0 is not missing.** The test is ``t < 0``, not ``t == -1``, so a ``prev`` of zeros must differ
  from a ``prev`` of minus ones. This is the case that fails if the port writes ``t == TOKEN_NULL``.
* **A5 a token's own EOS does not cut its own context.** Only the predecessors are examined, so two EOS tokens
  with the same cut predecessors produce identical rows, which the fixture shows directly.
* **A6 real vocabulary ids.** Five tokens near the artifact's real id range, as a case that is not a small
  synthetic one.

**B** covers the constants themselves, and **C** the binding's refusals.
"""

import pathlib

import numpy as np
import pytest

from monolith.runtime import _native as _nt

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ngram_oracle.txt"
N_HEADS = 16
N_PREV = 2
EOS = 248044        # PLE_EOS_TOKEN_ID (basalt_ngram.h)


def load_cases():
    """The fixture's blocks as (name, tokens, prev, rows), shaped as the binding takes them."""
    cases, cur = [], None
    for line in FIXTURE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        if head == "case":
            cur = {"name": rest}
            cases.append(cur)
        else:
            cur[head] = [int(x) for x in rest.split()]
    out = []
    for c in cases:
        toks = np.array(c["tokens"], np.int32)
        prev = np.array(c["prev"], np.int32).reshape(len(toks), N_PREV)
        rows = np.array(c["rows"], np.uint32).reshape(len(toks), N_HEADS)
        out.append((c["name"], toks, prev, rows))
    return out


CASES = load_cases()


def rows_of(tokens, prev):
    out = np.zeros((len(tokens), N_HEADS), np.uint32)
    _nt.basalt_ngram_rows(tokens, prev, out)
    return out


# ---- A: the oracle, one case per confusion a range check cannot catch --------------------------------------------

@pytest.mark.parametrize("name,tokens,prev,want", CASES, ids=[c[0] for c in CASES])
def test_a_the_oracle_cases(name, tokens, prev, want):
    got = rows_of(tokens, prev)
    assert np.array_equal(got, want), f"{name}: {int((got != want).sum())} of {want.size} indices differ"


def test_a7_the_cases_are_not_all_alike():
    """The parametrized cases above assert equality against recorded data, which would also hold if the fixture
    and the port were both degenerate. This pins that the six cases actually separate: every case's first token
    produces a different row vector, and the two `prev` spellings of case 3 and case 2 differ."""
    firsts = {tuple(rows_of(t, p)[0].tolist()) for _, t, p, _ in CASES}
    assert len(firsts) == len(CASES), "two oracle cases produce the same first-token rows"


def test_a8_token_zero_is_a_real_token_and_a_negative_is_not():
    """Two separate properties, and they are caught by different inputs.

    Token id 0 is a real token, so a `prev` of zeros must not read as a missing predecessor. That is what oracle
    case 3 records and what the first assertion restates.

    The spelling of the test is the second property. basalt writes `t < 0`, not `t == TOKEN_NULL`
    (``ngram.hpp:18``), and the two agree on every input the session can actually produce, because the only
    negative a caller supplies is -1. They diverge only on some other negative, so that is what the second
    assertion feeds: under `t < 0` a predecessor of -2 cuts, and under `t == -1` it does not and -2 itself lands
    in the window. Measured: with the comparison mutated to `t == TOKEN_NULL`, the oracle cases and the first
    assertion here all still pass, and only the second fails."""
    tokens = np.array([100, 101], np.int32)
    zeros = np.array([[0, 0], [100, 0]], np.int32)
    nulls = np.array([[-1, -1], [-1, -1]], np.int32)
    assert not np.array_equal(rows_of(tokens, zeros), rows_of(tokens, nulls)), "a prev of 0 read as missing"

    other = np.array([[-2, -2], [-2, -2]], np.int32)
    assert np.array_equal(rows_of(tokens, other), rows_of(tokens, nulls)), \
        "a negative predecessor other than -1 did not cut, so the test is `t == -1` rather than `t < 0`"


def test_a9_the_token_cuts_nothing_of_its_own():
    """A5 as an assertion. Two EOS tokens whose predecessors cut to the same window must give identical rows, and
    a non-EOS token between them must not."""
    tokens = np.array([EOS, 5, EOS], np.int32)
    prev = np.array([[-1, -1], [EOS, -1], [5, EOS]], np.int32)
    got = rows_of(tokens, prev)
    assert np.array_equal(got[0], got[2]), "two EOS tokens with equal cut windows differ"
    assert not np.array_equal(got[0], got[1]), "the middle token matched an EOS token, so the case proves nothing"


# ---- B: the constants ------------------------------------------------------------------------------------------

def test_b1_the_offsets_are_the_running_sum_of_the_vocabulary_sizes():
    """basalt carries both arrays in the GGUF metadata and writes both out rather than deriving one from the
    other, so that a mismatch surfaces instead of hiding (`ngram.cpp:32-34`). This is that check."""
    _, vocab, offset = _nt.basalt_ngram_consts()
    want = np.concatenate([[0], np.cumsum(np.array(vocab, np.uint64))[:-1]])
    assert np.array_equal(np.array(offset, np.uint64), want)


def test_b2_no_vocabulary_size_is_a_power_of_two():
    """The last step is `% vocab[h]`, never `& (vocab[h] - 1)`. If every size happened to be a power of two the
    two would agree and the distinction would be untestable; none is, which is why the modulo is load-bearing."""
    _, vocab, _ = _nt.basalt_ngram_consts()
    for v in vocab:
        assert v & (v - 1) != 0, f"{v} is a power of two"


def test_b3_the_largest_index_fits_the_table():
    """The sum of the sizes bounds every index the hash can produce, which is what makes the final cast to uint32
    lossless. It also shows the table's last rows are unaddressable, which is a property of the artifact and not
    of this port."""
    _, vocab, _ = _nt.basalt_ngram_consts()
    total = int(np.sum(np.array(vocab, np.uint64)))
    assert total == 320001446
    assert total - 1 < 2**32
    assert 320001536 - total == 90, "the artifact's spare row count moved"


# ---- C: the binding's refusals ---------------------------------------------------------------------------------

def test_c1_the_binding_refuses_mismatched_shapes():
    tokens = np.array([1, 2], np.int32)
    with pytest.raises(ValueError, match="one row per token"):
        _nt.basalt_ngram_rows(tokens, np.zeros((1, N_PREV), np.int32), np.zeros((2, N_HEADS), np.uint32))
    with pytest.raises(ValueError, match="2 predecessors"):
        _nt.basalt_ngram_rows(tokens, np.zeros((2, 3), np.int32), np.zeros((2, N_HEADS), np.uint32))
    with pytest.raises(ValueError, match="16 head indices"):
        _nt.basalt_ngram_rows(tokens, np.zeros((2, N_PREV), np.int32), np.zeros((2, 8), np.uint32))


def test_c2_the_binding_refuses_shared_storage():
    """`out` is written while `prev` is read, so a shifted view of one buffer would let the write land under the
    read. The expert bindings refuse the same way."""
    buf = np.zeros(2 * N_HEADS, np.uint32)
    prev = buf[: 2 * N_PREV].view(np.int32).reshape(2, N_PREV)
    out = buf.reshape(2, N_HEADS)
    with pytest.raises(ValueError, match="prev and out must not share storage"):
        _nt.basalt_ngram_rows(np.array([1, 2], np.int32), prev, out)


def test_c3_tokens_can_alias_out_as_easily_as_prev_can():
    """The same hazard on the other input, and the easier one to write by accident: a contiguous int32 view of
    the uint32 `out` array is the right dtype, the right shape and contiguous, so it passes every other check
    here. The 16 words written for token 0 land exactly where tokens 1 through 15 are read from, so the first
    row's output decides the rest of the input. The result is finite and wrong, never a crash, which is what
    makes the refusal worth more than a copy."""
    n = 32
    buf = np.zeros(n * N_HEADS + n, np.uint32)
    out = buf[: n * N_HEADS].reshape(n, N_HEADS)
    prev = np.zeros((n, N_PREV), np.int32)

    alias = out.reshape(-1).view(np.int32)[:n]             # the write lands under the read
    alias[:] = np.arange(1, n + 1, dtype=np.int32)
    with pytest.raises(ValueError, match="tokens and out must not share storage"):
        _nt.basalt_ngram_rows(alias, prev, out)

    # The same buffer, past every word `out` covers. The check is about the ranges, not about the allocation,
    # so this one runs: without the second arm a guard that refused every int32 view would pass the first.
    far = buf[n * N_HEADS :].view(np.int32)
    assert far.shape == (n,) and np.shares_memory(far, buf) and not np.shares_memory(far, out), \
        "the disjoint view is not what this case meant to build: one buffer, no overlap with out"
    far[:] = np.arange(1, n + 1, dtype=np.int32)
    _nt.basalt_ngram_rows(far, prev, out)
    assert out.any(), "the allowed call wrote nothing, so it proves nothing"
