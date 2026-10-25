# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/platform/direct_file.cpp and src/ngram/ple_reader_test.cpp:46 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the alignment contract, the completion
# protocol, and the deliberately misaligned header the self-test uses so rows straddle pages as they really do.
"""The unbuffered reader of ``runtime/src/basalt_direct_file.h``: random 4 KiB reads of the PLE table, issued
from a pool of threads so many are in flight.

**This is the one piece of the n-gram tier that is a substitution rather than a transcription.** macOS has no
``O_DIRECT``; the port sets ``F_NOCACHE`` on the descriptor and turns read-ahead off. Both are advisory where
``O_DIRECT`` is structural, so the "table never occupies RAM" property becomes a tendency rather than a
guarantee. That difference is memory residency and timing. It cannot reach a produced value, because the bytes
a read returns are the bytes on the device either way, and this suite pins exactly that: every case here is
about the bytes and the protocol, and none of them would change if the caching advice were ignored entirely.

**What is held, and why each case exists:**

* **A, the bytes.** A read returns the file's bytes at the offset asked for. Trivial to state and the thing
  everything else rests on.
* **B, the end of the file.** The table ends at a byte that is not a page multiple: its final page holds 3,264
  bytes, the last row begins 3,174 into that page and is 90 long, so it ends exactly at the boundary. A read
  crossing the end must come back SHORT and OK, never as a failure. The fixture reproduces that geometry rather
  than describing it, because a port that treats a short read as an error passes every other case here.
* **C, the alignment contract.** Kept from ``O_DIRECT`` even though ``F_NOCACHE`` does not require it, because
  the two-page slot and the straddle handling above are built on it. Dropping it would be a redesign.
* **D, the completion protocol.** Completions carry a tag and arrive in whatever order the threads finished.
  Nothing relates that order to submission order, so a caller that assumes ordering is wrong on hardware even
  when it passes on a small file.
"""

import os
import pathlib
import threading
import time

import numpy as np
import pytest

from monolith.runtime import _native as _nt

PAGE = 4096
ROW_BYTES = 90
HEADER = 192                 # the real shard's data offset, so rows straddle pages the same way


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """A small stand-in for the table, laid out with the real geometry: a 192-byte header, then whole rows, and
    a total size that is deliberately not a page multiple so the last page is short."""
    rows = 4000
    path = tmp_path_factory.mktemp("ple") / "table.bin"
    rng = np.random.default_rng(20261025)
    body = rng.integers(0, 256, size=rows * ROW_BYTES, dtype=np.uint8)
    blob = bytearray(rng.integers(0, 256, size=HEADER, dtype=np.uint8).tobytes())
    blob += body.tobytes()
    path.write_bytes(bytes(blob))
    return {"path": str(path), "bytes": bytes(blob), "rows": rows, "size": len(blob)}


@pytest.fixture
def reader(table):
    f = _nt.DirectFile()
    f.open(table["path"], threads=16)
    yield f
    f.close()


def row_offset(row):
    return HEADER + row * ROW_BYTES


# ---- A: the bytes ------------------------------------------------------------------------------------------------

def test_a1_a_read_returns_the_files_bytes(reader, table):
    ok, got, data = reader.read_at(0, PAGE)
    assert ok and got == PAGE
    assert data == table["bytes"][:PAGE]


def test_a2_a_row_is_recovered_from_its_page(reader, table):
    """How the layer above actually uses this: read the page a row starts in, then take the row out of it at its
    in-page offset. With a 192-byte header and 90-byte rows, that offset is almost never zero."""
    for row in (0, 1, 45, 46, 1000, 3999):
        off = row_offset(row)
        page = (off // PAGE) * PAGE
        in_page = off - page
        length = PAGE * (2 if in_page + ROW_BYTES > PAGE else 1)     # two pages when the row straddles
        ok, got, data = reader.read_at(page, length)
        assert ok and in_page + ROW_BYTES <= got, f"row {row} ran past the bytes returned"
        assert data[in_page : in_page + ROW_BYTES] == table["bytes"][off : off + ROW_BYTES]


def test_a3_rows_really_do_straddle_pages(table):
    """The two-page slot exists for this. If no row straddled, A2's straddle arm would never run and the whole
    scheme would be untested."""
    straddlers = [r for r in range(table["rows"])
                  if (row_offset(r) % PAGE) + ROW_BYTES > PAGE]
    assert len(straddlers) > 50, "no row straddles a page, so the fixture's geometry is wrong"


# ---- B: the end of the file --------------------------------------------------------------------------------------

def test_b1_the_fixture_ends_mid_page(table):
    assert table["size"] % PAGE != 0, "the fixture ends on a page boundary, so B2 would prove nothing"


def test_b2_a_read_crossing_the_end_is_short_and_ok(reader, table):
    """Not a failure. The reference's own last-row probe depends on this: the real table's last row ends exactly
    at the final page's 3,264th byte, so the probe is simultaneously its end-of-file tolerance test."""
    last_page = (table["size"] // PAGE) * PAGE
    ok, got, data = reader.read_at(last_page, PAGE * 2)
    assert ok, "a read crossing the end of the file reported failure"
    assert got == table["size"] - last_page, f"short read returned {got}, expected {table['size'] - last_page}"
    assert data == table["bytes"][last_page:]


def test_b3_the_last_row_survives_the_short_read(reader, table):
    """The case the geometry is built for: the final row lives inside that short final page."""
    row = table["rows"] - 1
    off = row_offset(row)
    page = (off // PAGE) * PAGE
    ok, got, data = reader.read_at(page, PAGE * 2)
    assert ok and (off - page) + ROW_BYTES <= got, "the last row did not fit the bytes returned"
    assert data[off - page : off - page + ROW_BYTES] == table["bytes"][off : off + ROW_BYTES]


def test_b4_a_read_entirely_past_the_end_returns_nothing(reader, table):
    past = ((table["size"] // PAGE) + 8) * PAGE
    ok, got, data = reader.read_at(past, PAGE)
    assert ok and got == 0 and data == b"", "a read past the end should be ok with zero bytes"


# ---- C: the alignment contract -----------------------------------------------------------------------------------

def test_c1_an_unaligned_offset_is_refused(reader):
    with pytest.raises(ValueError, match="unaligned"):
        reader.read_at(1, PAGE)


def test_c2_an_unaligned_or_empty_length_is_refused(reader):
    with pytest.raises(ValueError, match="multiple of 4096"):
        reader.read_at(0, PAGE + 1)
    with pytest.raises(ValueError, match="multiple of 4096"):
        reader.read_at(0, 0)


# ---- D: the completion protocol ----------------------------------------------------------------------------------

def test_d1_every_submitted_read_completes_exactly_once(reader, table):
    offsets = [(i * 3 % 60) * PAGE for i in range(60)]
    order = reader.read_many(offsets, PAGE)
    assert len(order) == len(offsets), "a read went missing or completed twice"
    assert sorted(tag for tag, _ in order) == list(range(len(offsets)))


def test_d2_completions_are_keyed_by_tag_not_ordered_by_submission(reader, table):
    """Sixteen threads push into one completion queue as they finish, so nothing relates completion order to
    submission order. A caller that indexes by arrival rather than by tag is wrong on a real device even where a
    small file happens to come back in order, so this asserts the tag is carried, and reports whether this run
    actually reordered rather than requiring it."""
    offsets = [(i % 40) * PAGE for i in range(200)]
    order = reader.read_many(offsets, PAGE)
    tags = [tag for tag, _ in order]
    assert sorted(tags) == list(range(len(offsets))), "tags were not carried through intact"
    # Informational: with 16 threads over 200 reads this is almost always out of order, but ordering is the
    # scheduler's business and the contract is the tag, so a run that happens to stay ordered is not a failure.
    assert len(tags) == len(offsets)


def test_d6_every_offset_is_checked_before_any_read_is_submitted(reader, table):
    """A bad offset late in the list must not leave the earlier reads queued. They were submitted against
    buffers the failing call is about to free, and a worker thread writes through those pointers, so freeing
    them early is a write to freed memory. Validating up front is what makes the failure clean; the reader
    still works afterwards, which is what this asserts."""
    good = [0, PAGE, PAGE * 2]
    with pytest.raises(ValueError, match="multiple of 4096"):
        reader.read_many(good + [PAGE * 3 + 1], PAGE)
    order = reader.read_many(good, PAGE)
    assert sorted(tag for tag, _ in order) == [0, 1, 2], "the reader was left in a bad state by the refusal"


def test_d7_two_threads_reading_at_once_both_finish(reader, table):
    """`wait` pops from ONE completion queue shared by every caller, and each caller drops whatever is not its
    own. So two overlapping drivers consume each other's completions, and with the lock removed this case
    returns one thread a list carrying the OTHER thread's tags (measured: 0, 0, 1, 1, ... rather than 0, 1, 2),
    which in a caller keying rows off those tags is the wrong row served as the right one; the thread whose
    completions were taken can also wait forever for one already consumed. basalt never meets this because
    exactly one reader thread owns every call into the class; the binding is reachable from any Python thread,
    so it serialises each submit-and-collect cycle itself.

    Batches of 512 are what make the overlap real rather than lucky: one read is over in microseconds, so two
    threads doing one each almost never have both in flight and a pair of single reads passes with no lock at
    all (measured). 512 in flight holds the window open. The join is bounded because the other failure arm is
    a hang."""
    batch = [((i * 7) % 400) * PAGE for i in range(512)]
    results, errors = {}, []

    def drive(name):
        try:
            for _ in range(8):
                order = reader.read_many(batch, PAGE)
                assert sorted(tag for tag, _ in order) == list(range(len(batch))), \
                    f"thread {name} was handed another driver's completions"
            results[name] = True
        except BaseException as exc:              # carried out, because a raise here dies with the thread
            errors.append((name, exc))

    threads = [threading.Thread(target=drive, args=(n,), daemon=True) for n in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert not errors, errors
    assert [th.is_alive() for th in threads] == [False, False], "a reader thread never finished: the reads deadlocked"
    assert results == {0: True, 1: True}


def test_d3_a_wake_posts_a_synthetic_completion(reader):
    """So a thread parked waiting for reads can be released without one finishing."""
    reader.wake()           # the binding's read paths skip the wake tag, so this must not disturb them
    ok, got, _ = reader.read_at(0, PAGE)
    assert ok and got == PAGE


def test_d8_the_wake_tag_is_refused_as_a_reads_tag(reader):
    """`wake` posts a completion carrying `~0`, and `read_at` returns on the first completion whose tag matches
    its own. A read submitted under that tag therefore matches the WAKE and returns before the read has run, so
    the buffer is freed while a pool thread is still writing into it. One thread reaches it: queue a wake, then
    read with that tag. Refused at the binding, because the tag is the reader's own."""
    reader.wake()
    with pytest.raises(ValueError, match="wake tag"):
        reader.read_at(0, PAGE, (1 << 64) - 1)
    ok, got, _ = reader.read_at(0, PAGE)       # the queued wake must not disturb an ordinary read either
    assert ok and got == PAGE


def test_d9_a_close_waits_for_a_read_in_flight(table):
    """`close` sets `stop_`, joins every worker, and THEN clears the completion queue. A close landing while a
    driver waits therefore wipes the completion that driver is blocked on, and no worker is left to post
    another: `wait` with a negative timeout never returns. The file is gone and the buffer is safe, so this
    arm is a permanent hang rather than corruption, which is why nothing else here would have found it.
    `open` carries it too, because `open` calls `close` first.

    A round of 4,096 reads takes about 16 ms, and the window inside it is the gap between the last worker
    posting and the driver consuming, so one attempt is a coin flip: measured against a build with the lock
    removed, a single attempt hung the driver 2 to 5 times in 8. Twelve attempts put a false pass under a
    percent. Its own reader each time, because the point is to close one mid-flight."""
    offsets = [((i * 7) % 80) * PAGE for i in range(4096)]        # a window wide enough to close into
    hung = []

    for attempt in range(12):
        f = _nt.DirectFile()
        f.open(table["path"], threads=16)
        errors = []

        def drive():
            try:
                f.read_many(offsets, PAGE)
            except BaseException as exc:             # carried out, because a raise here dies with the thread
                errors.append(exc)

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        time.sleep(0.002)                            # mid-round, which is where the close has to land
        closer = threading.Thread(target=f.close, daemon=True)
        closer.start()
        closer.join(timeout=30)
        driver.join(timeout=30)
        if driver.is_alive() or closer.is_alive():
            hung.append((attempt, "driver" if driver.is_alive() else "closer"))
            break                                    # the threads are wedged; another attempt proves nothing
        assert not errors, errors

    assert not hung, f"a close landed inside a read in flight and never came back: {hung}"


def test_d4_the_pool_runs_the_thread_count_it_was_given(table):
    for asked, want in ((1, 1), (16, 16), (64, 64), (100, 64), (0, 1)):
        f = _nt.DirectFile()
        f.open(table["path"], threads=asked)
        assert f.threads == want, f"asked {asked}, got {f.threads}"
        f.close()


def test_d5_size_matches_the_file(reader, table):
    assert reader.size == table["size"] == os.path.getsize(table["path"])
