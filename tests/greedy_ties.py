"""Greedy-token comparison that tolerates bfloat16 near ties (no GPU, so the contract tier tests it)."""
import numpy as np

from monolith.bench import bf16_ulp_diff
from monolith.formats.fp import bf16_to_f32


def score_row(reference, ids, i):
    """The scores ``reference`` chose its token ``i`` after ``ids`` by, rerun to that step: its prompt pass's last row for
    token 0, its decode step's row 0 for a later token. ``reference`` is a plain session (no drafter or prefix cache)."""
    reference.generate(ids, i + 1)                      # the step that chose token i ran last: read its scores
    bits = np.frombuffer(reference.read('logits'), dtype=np.uint16).reshape(-1, reference.model.config.vocab_size)
    return bf16_to_f32(bits[(len(ids) - 1) % reference.prefill_chunk_size if i == 0 else 0])


def agree(reference, ids, got, expected, ties, at=0):
    """Greedy tokens equal to the reference's. An accelerator session (the tile rounds differently from the reference's
    shader GEMVs) may differ at up to ``ties`` tokens, each scoring at most one bfloat16 step below the reference's
    argmax in that step's scores (:func:`score_row`); the comparison continues from the session's token after each.
    ``at`` is ``got[0]``'s index in the caller's comparison, so a failure names the caller's token."""
    assert len(got) == len(expected), f'tokens from {at} on: {len(got)} against {len(expected)} ({got} against {expected})'
    for i, (a, b) in enumerate(zip(got, expected)):
        if a != b:
            assert ties, f'tokens from {at} on: {got} against {expected}'
            row = score_row(reference, ids, i)
            assert 0 <= a < len(row) and row[b] == row.max(), f'token {at + i}: {a} outside the vocabulary or {b} not the argmax'
            steps = bf16_ulp_diff(row[a], row[b])
            assert steps <= 1, f'token {at + i}: {a} against {b}, {steps} bfloat16 steps apart'
            rest = ids + got[:i + 1]
            return agree(reference, rest, got[i + 1:], reference.generate(rest, len(got) - i - 1).tokens, ties - 1, at + i + 1)
