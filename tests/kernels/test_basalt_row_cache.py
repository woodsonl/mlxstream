# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/ngram/ple_reader.cpp:30-71, :544-545 and
# src/ngram/ple_reader_test.cpp:130-131 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# the cache structure and the two invariants its self-test asserts.
"""The PLE row cache of ``runtime/src/basalt_row_cache.h``: rows already fetched, kept so a row recurring within
one prompt is not read from the device twice.

**Why this suite asserts structure rather than values.** The cache is bit-neutral. It stores and returns the raw
on-disk row bytes, and the dequantization happens afterwards and identically either way, so changing the mix, the
way count or the replacement policy moves the hit rate and the device traffic and can never move a produced
value. There is no output to hold against an oracle. What a port can still get wrong is the structure, and each
of those errors is invisible unless a test names it:

* **A1, the mix is a 64-bit multiply.** A 32-bit row index times a 64-bit constant promotes, so the product is
  truncated at 64 bits rather than 32. Doing it in 32 bits gives a different, still-valid set index.
* **A2, the set index is a modulo.** At the production capacity the set count is a power of two and a mask would
  agree, which is exactly what makes the substitution look safe. This suite uses a capacity where they differ.
* **B, replacement is per-set round robin.** Not clock, which is what basalt's own header says, and not
  least-recently-used. A hit updates nothing.
* **C, the fill counter counts first fills.** It never decreases and it saturates at the capacity, which rounds
  down to a multiple of the way count.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt

WAYS = 8
RB = 90                       # the PLE table's row stride
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def row_bytes(seed, n=RB):
    return np.frombuffer(np.random.default_rng(seed).bytes(n), np.uint8)


def get(cache, row):
    """The row's bytes, or None when absent."""
    out = np.zeros(RB, np.uint8)
    return out if cache.get(row, out) else None


# ---- A: the index arithmetic -----------------------------------------------------------------------------------

@pytest.mark.parametrize("row", [0, 1, 2, 12345, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFE, 320001535])
def test_a1_mix_is_a_64_bit_multiply_then_a_shift_xor(row):
    """Held against the arithmetic written out in Python, which has no 32-bit truncation to accidentally match."""
    x = (row * GOLDEN) & MASK64
    assert _nt.RowCache.mix(row) == (x ^ (x >> 29))


def test_a2_a_32_bit_multiply_would_differ():
    """The mutation this pins: truncating the product at 32 bits. If no row separated the two, A1 would be
    asserting against a coincidence."""
    differ = [r for r in range(1, 4096)
              if (((r * GOLDEN) & MASK64) ^ (((r * GOLDEN) & MASK64) >> 29))
              != ((((r * GOLDEN) & 0xFFFFFFFF)) ^ (((r * GOLDEN) & 0xFFFFFFFF) >> 29))]
    assert len(differ) > 4000 - 100, "a 32-bit multiply would agree too often for A1 to be meaningful"


def test_a3_the_set_index_is_a_modulo_not_a_mask():
    """A capacity whose set count is not a power of two, so the two differ. At the production capacity of 2^20
    rows the set count is 2^17 and they agree, which is what makes a mask look correct."""
    cache = _nt.RowCache(rows=8 * 100, row_bytes=RB)       # 100 sets, not a power of two
    assert cache.sets == 100
    disagreed = 0
    for row in range(5000):
        want = _nt.RowCache.mix(row) % 100
        assert cache.set_of(row) == want
        if want != (_nt.RowCache.mix(row) & 63):
            disagreed += 1
    assert disagreed > 4000, "the modulo and a mask agreed too often to separate them"


def test_a4_store_and_retrieve_at_a_non_power_of_two_capacity():
    """A3 holds the index arithmetic; this holds the path that uses it. Insert and look up at a capacity whose
    set count is not a power of two, so any index that disagrees with the one the insert used misses. Without
    this case a mask substitution survives, because every other capacity in this file is a power of two and
    there the mask and the modulo agree."""
    cache = _nt.RowCache(rows=8 * 100, row_bytes=RB)      # 100 sets
    rows = [r for r in range(0, 20000, 7)][:500]
    for r in rows:
        cache.insert(r, row_bytes(r))
    found = sum(1 for r in rows if get(cache, r) is not None)
    assert found > 300, f"only {found} of {len(rows)} rows came back; the lookup index disagrees with the insert"
    for r in rows:
        got = get(cache, r)
        if got is not None:
            assert np.array_equal(got, row_bytes(r)), f"row {r} came back as another row's bytes"


def test_a5_the_production_capacity_is_where_a_mask_hides():
    """Stated as its own case because it is the reason a1 through a3 exist: at 2^20 rows the two agree on every
    row, so a masked port passes any test run only at that size."""
    cache = _nt.RowCache(rows=1 << 20, row_bytes=RB)
    assert cache.sets == 1 << 17
    for row in (0, 1, 12345, 320001535, 0xFFFFFFFE):
        assert cache.set_of(row) == (_nt.RowCache.mix(row) & ((1 << 17) - 1))


# ---- B: storage and replacement --------------------------------------------------------------------------------

def test_b1_a_stored_row_comes_back_byte_for_byte():
    cache = _nt.RowCache(rows=8 * 16, row_bytes=RB)
    want = row_bytes(1)
    cache.insert(7, want)
    assert np.array_equal(get(cache, 7), want)
    assert get(cache, 8) is None, "an absent row reported as present"


def test_b2_a_set_fills_all_eight_ways_before_evicting():
    """Eight rows that land in one set all survive; the ninth evicts the first, because the pointer wrapped."""
    cache = _nt.RowCache(rows=8 * 4, row_bytes=RB)         # 4 sets, so collisions are easy to produce
    target = cache.set_of(0)
    same_set, row = [], 0
    while len(same_set) < 9:
        if cache.set_of(row) == target:
            same_set.append(row)
        row += 1

    for i, r in enumerate(same_set[:WAYS]):
        cache.insert(r, row_bytes(r))
    for r in same_set[:WAYS]:
        assert get(cache, r) is not None, f"row {r} evicted before the set was full"

    cache.insert(same_set[WAYS], row_bytes(same_set[WAYS]))
    assert get(cache, same_set[0]) is None, "the ninth insert did not evict the first way"
    for r in same_set[1:WAYS]:
        assert get(cache, r) is not None, f"the ninth insert evicted {r} as well as the first"


def test_b3_replacement_is_round_robin_not_least_recently_used():
    """The distinction that matters. After filling a set, touch the first way, then insert. Round robin evicts
    the first way anyway, because a hit moves nothing; least-recently-used would evict the second."""
    cache = _nt.RowCache(rows=8 * 4, row_bytes=RB)
    target = cache.set_of(0)
    same_set, row = [], 0
    while len(same_set) < 9:
        if cache.set_of(row) == target:
            same_set.append(row)
        row += 1
    for r in same_set[:WAYS]:
        cache.insert(r, row_bytes(r))

    assert get(cache, same_set[0]) is not None               # touch it: under LRU this would protect it
    cache.insert(same_set[WAYS], row_bytes(same_set[WAYS]))
    assert get(cache, same_set[0]) is None, "a hit protected the entry, so this is not round robin"
    assert get(cache, same_set[1]) is not None, "the second way went, which is what LRU would have evicted"


def test_b4_inserting_a_row_already_present_changes_nothing():
    """`insert` early-returns on a hit, so neither the bytes nor the replacement pointer move. If the pointer
    moved, the eviction order below would shift by one."""
    cache = _nt.RowCache(rows=8 * 4, row_bytes=RB)
    target = cache.set_of(0)
    same_set, row = [], 0
    while len(same_set) < 9:
        if cache.set_of(row) == target:
            same_set.append(row)
        row += 1
    for r in same_set[:WAYS]:
        cache.insert(r, row_bytes(r))

    before = cache.size
    cache.insert(same_set[0], row_bytes(999))                # different bytes, same row
    assert cache.size == before, "a re-insert counted as a fill"
    assert np.array_equal(get(cache, same_set[0]), row_bytes(same_set[0])), "a re-insert overwrote the bytes"

    cache.insert(same_set[WAYS], row_bytes(same_set[WAYS]))
    assert get(cache, same_set[0]) is None, "the re-insert moved the replacement pointer"


# ---- C: capacity and the fill counter ---------------------------------------------------------------------------

def test_c1_capacity_rounds_down_to_a_multiple_of_the_ways():
    for rows, want in ((1 << 20, 1 << 20), (100, 96), (8, 8), (15, 8)):
        assert _nt.RowCache(rows=rows, row_bytes=RB).capacity == want


def test_c2_fewer_than_eight_rows_disables_the_cache():
    """Zero sets: both lookup and insert do nothing. Zero is basalt's documented off value, and anything under
    one way's worth lands in the same place through the integer division."""
    for rows in (0, 1, 7):
        cache = _nt.RowCache(rows=rows, row_bytes=RB)
        assert cache.sets == 0 and cache.capacity == 0
        cache.insert(5, row_bytes(5))
        assert get(cache, 5) is None, "an off cache stored a row"
        assert cache.size == 0


def test_c3_the_fill_counter_never_decreases_and_stays_within_capacity():
    """It counts first fills of an empty way, so it rises to the capacity and stops. It is not a live-entry
    count; the two coincide only because entries are never removed."""
    cache = _nt.RowCache(rows=8 * 4, row_bytes=RB)
    last = 0
    for r in range(400):
        cache.insert(r, row_bytes(r))
        assert cache.size >= last, "the fill counter went backwards"
        assert cache.size <= cache.capacity, "the fill counter passed the capacity"
        last = cache.size
    assert cache.size == cache.capacity, "400 rows into 32 ways left the cache unfilled"


def test_c4_the_vacancy_marker_cannot_collide_with_a_real_row():
    """The marker is 0xFFFFFFFF and the table has 320,001,536 rows, so no live key can equal it. This holds the
    gap rather than the marker, since a port that grew the table past 2^32 would break it silently."""
    assert 320001536 < 0xFFFFFFFF
