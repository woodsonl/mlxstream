"""Separate large-prefill/small-decode programs preserve sequence state across chunk boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contract'))
from test_nn_lowering import _checkpoint
from dspark_synth import write_checkpoint as write_dspark, build
from greedy_ties import agree as _agree, score_row
from lm_synth import write_checkpoint as write_lm
from monolith.generate import Session
from monolith.formats import PackLayout
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.nn.pack_plan import pack_model
from monolith.spec.lm import LMDrafter


@pytest.mark.parametrize('draft_kind', [None, 'lm', 'dspark'])
@pytest.mark.parametrize('accelerator', ['off', 'on'])
@pytest.mark.parametrize('prefix_cache', [False, True])
@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('exact', [False, True])
def test_large_prefill_shares_state_with_small_decode(tmp_path, draft_kind, accelerator, prefix_cache, resident, exact):
    if resident and draft_kind != 'dspark':
        pytest.skip('Resident verification prefill requires a speculative decoder recipe')
    target, draft = tmp_path / 'target', tmp_path / 'draft'
    target.mkdir(); draft.mkdir()
    _checkpoint(target)
    def model():
        return Qwen3_5Model.from_checkpoint(str(target), max_context=512)
    pack_model(model(), str(target), str(target / 'pack'), PackLayout(rows=16))
    reference = Session(model(), str(target / 'pack'), prefill_chunk_size=8, eos=-1, autotune=False,
                        accelerator='off', attention='v1')
    ties = 1 if accelerator == 'on' else 0                 # the one-step ties each comparison may take
    m = model()
    options = {}
    if draft_kind == 'lm':
        write_lm(draft, vocab_size=50)
        d = LMDrafter.from_checkpoint(str(draft), target_lm_head=m.lm_head, max_context=512, gamma=3)
    elif draft_kind == 'dspark':
        write_dspark(draft, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1])
        d, _, _, _ = build(draft, target_lm_head=m.lm_head, max_context=512)
    if draft_kind:
        pack_model(d, str(draft), str(draft / 'pack'), PackLayout(rows=16))
        options = dict(drafter=d, drafter_pack=str(draft / 'pack'), verify='fixed', verify_length=3)
    session = Session(m, str(target / 'pack'), eos=-1, autotune=False, attention='v1', accelerator=accelerator,
                      prefix_cache=prefix_cache, decoder_kernel_config={} if resident else None, **options,
                      **(dict(prefill_chunk_size=512, prefill_exact=True) if exact else {}))
    # Small-to-large allocation growth, exact boundary, partial chunk, several chunks, then reuse the small graph.
    # 137 and 265 leave one token in a 128-row chunk before the eight rows the resident decoder ingests.
    for count in (5, 128, 129, 259, 137, 265, 5):
        ids = np.random.default_rng(count).integers(0, 50, count).tolist()
        expected = reference.generate(ids, 12).tokens
        result = session.generate(ids, 12)
        _agree(reference, ids, result.tokens, expected, ties)
        dec = session.engine(0 if draft_kind else 1)
        if not resident:
            pre = session.prefill_engine(count)
            assert pre is not dec
            assert pre.buffers['step_state'] is dec.buffers['step_state']
            for name, spec in dec.program.buffers.items():
                if spec.role in ('state', 'weights', 'ring'):
                    assert pre.buffers[name] is dec.buffers[name]
        assert dec.state()['position'] == count + 11
        assert dec.state()['error'] == 0
        vocab = m.config.vocab_size
        assert dec.program.buffers['logits'].nbytes == (8 if draft_kind else 1) * vocab * 2
        assert session.decode_t_max == 8
        assert session.prefill_chunk_size == (512 if exact else 128)
        if prefix_cache:
            # Reuse a checkpoint while replaying only the final input token.
            repeated = session.generate(ids, 12)
            assert repeated.tokens == result.tokens
            assert repeated.cached_prompt_tokens == count - 1
            if resident:
                assert session.engine(0) is dec
            changed = ids[:-1] + [(ids[-1] + 1) % 50]
            _agree(reference, changed, session.generate(changed, 12).tokens, reference.generate(changed, 12).tokens, ties)
    if resident:
        # A one-token answer ends before decode is allocated. The next short
        # request must not inherit either its state or original weight windows.
        session.generate(list(range(50)) * 3, 1)
        ids = [7, 3, 1, 8, 4]
        _agree(reference, ids, session.generate(ids, 12).tokens, reference.generate(ids, 12).tokens, ties)
        if prefix_cache:
            ids = list(range(50)) * 3
            session.prefix_cache.items.clear()
            session.generate(ids, 12, cache_prefix_tokens=120)
            # The prefill/decode boundary is not another cache checkpoint: it
            # would evict a large shared prefix under the byte budget.
            assert [item.tokens for item in session.prefix_cache.items] == [tuple(ids[:120])]
            changed = ids[:120] + [3, 7, 4, 2]
            result = session.generate(changed, 12, cache_prefix_tokens=120)
            assert result.cached_prompt_tokens == 120
            _agree(reference, changed, result.tokens, reference.generate(changed, 12).tokens, ties)
            # Preserve the earlier system/tool checkpoint when project context
            # inside the last message changes, while preferring the deep hit
            # for a new question with otherwise identical context.
            session.prefix_cache.items.clear()
            session.generate(ids, 12, cache_prefix_tokens=[100, 120])
            assert [len(item.tokens) for item in session.prefix_cache.items] == [100, 120]
            question = ids[:120] + [9, 8, 7]
            reused = session.generate(question, 12, cache_prefix_tokens=[100, 120])
            assert reused.cached_prompt_tokens == 120
            _agree(reference, question, reused.tokens, reference.generate(question, 12).tokens, ties)
            project_changed = ids.copy()
            project_changed[110] = (project_changed[110] + 1) % 50
            reused = session.generate(project_changed, 12, cache_prefix_tokens=[100, 120])
            assert reused.cached_prompt_tokens == 100
            _agree(reference, project_changed, reused.tokens, reference.generate(project_changed, 12).tokens, ties)


@pytest.mark.parametrize('accelerator', ['off', 'on'])
@pytest.mark.parametrize('attention', ['v1', 'auto'])
def test_exact_prompt_passes_give_the_scores_of_128_row_passes(tmp_path, attention, accelerator):
    # 512-row exact passes against 128-row passes, the last prompt row's scores bit for bit; the synthetic checkpoint's
    # head dim is 32, so auto runs v3 in both programs, with the projections on the tile when the accelerator is on
    _checkpoint(tmp_path)
    pack_model(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path), str(tmp_path / 'pack'),
               PackLayout(rows=16))

    def session(**options):
        return Session(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path / 'pack'), eos=-1,
                       autotune=False, attention=attention, accelerator=accelerator, **options)
    reference, exact = session(prefill_chunk_size=128), session(prefill_chunk_size=512, prefill_exact=True)
    for count in (260, 300, 400, 700):                     # 700: a second 512-row pass, from position 512
        ids = np.random.default_rng(count).integers(0, 50, count).tolist()
        tokens = reference.generate(ids, 12).tokens
        assert exact.generate(ids, 12).tokens == tokens, count
        rows = [score_row(s, ids, 0) for s in (reference, exact)]
        assert exact._last_engine is exact.engines['prefill.512']
        assert rows[0][tokens[0]] == rows[0].max()
        assert np.array_equal(rows[1].view(np.uint32), rows[0].view(np.uint32)), (count, int((rows[1] != rows[0]).sum()))
    core = 'gqa_decode_v3' if attention == 'auto' else 'gqa_decode'
    assert [k.macros.get('EXACT_SUB') for k in exact.engines['prefill.512'].program.kernels.values() if k.function == core] == ['128u']


@pytest.mark.parametrize('accelerator', ['off', 'on'])
@pytest.mark.parametrize('chunk', [512, 128])            # 128: serve's auto-exact where the chip measured no larger size
def test_exact_requests_restored_off_the_grid_give_the_bits_of_cold_runs(tmp_path, chunk, accelerator):
    # A request restored from a checkpoint off the 128-row grid gives the scores of its last pass's rows and its tokens
    # as the same request run cold, bit for bit. auto runs v3 at the synthetic checkpoint's head dim 32, and v3 pairs the
    # keys cached before a pass, so a row depends on where its pass starts unless the exact form anchors it at the
    # 128-row passes from token 0. The checkpoint's one attention layer is its last, so only a row's own fold reaches its
    # scores, and the rows compared must start their passes elsewhere warm than cold: turn 1 checkpoints its first 300
    # tokens; turn 2 keeps a shorter message prefix (250, as serve keeps a system prompt's) and restores 300, where the
    # cold run splits at 250. A pair and two single folds sum alike unless the running maximum moves between them, so the
    # starts lie far apart: at 428 tokens the last row's 128-row pass, counted from each run's pass start, begins at 300
    # warm and 378 cold (384 from token 0); at 429 the warm run's one-token chunk sends a 512-row session to its 128-row
    # graph while the cold run stays on the large one, as does a cold run's one-token chunk from the default checkpoint
    # at p - 1. A cold run whose passes all start on the grid gives what plain 128-row passes give, and only the exact
    # session's prompt programs take the exact form.
    _checkpoint(tmp_path)
    pack_model(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path), str(tmp_path / 'pack'),
               PackLayout(rows=16))

    def session(**options):
        return Session(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path / 'pack'), eos=-1,
                       autotune=False, attention='auto', accelerator=accelerator, **options)
    exact = session(prefill_chunk_size=chunk, prefill_exact=True, prefix_cache=True, prefix_cache_min_tokens=128)
    plain = session(prefill_chunk_size=128)
    ids = np.random.default_rng(81).integers(0, 50, 800).tolist()
    exact.generate(ids[:300] + ids[700:], 1, cache_prefix_tokens=300)
    turn1 = list(exact.prefix_cache.items)
    assert [len(item.tokens) for item in turn1] == [300]

    def last_pass(s, n, prefix, restore=False):
        """The first position and the scores of the last pass's rows, and 12 greedy tokens, each from its own request
        with the same cache."""
        outputs = []
        for steps in (1, 12):
            if s.prefix_cache is not None:
                s.prefix_cache.items = list(turn1) if restore else []
            g = s.generate(ids[:n], steps, cache_prefix_tokens=prefix)
            assert g.cached_prompt_tokens == g.prefill_timings[0]['position'] == (300 if restore else 0)
            if steps > 1:
                return outputs + [g.tokens]
            last = g.prefill_timings[-1]
            logits = np.frombuffer(s.read('logits'), dtype=np.uint16).reshape(-1, s.model.config.vocab_size)
            outputs.append((last['position'], logits[:last['tokens']].copy()))

    def agree(a, b, n, passes):
        """Two requests' last passes (first position, rows) as expected, their tokens, and the scores of every row both
        last passes hold."""
        ((pa, rows_a), tokens_a), ((pb, rows_b), tokens_b) = a, b
        assert [(pa, len(rows_a)), (pb, len(rows_b))] == passes, n
        lo = max(pa, pb)
        rows_a, rows_b = rows_a[lo - pa:n - pa], rows_b[lo - pb:n - pb]
        assert tokens_a == tokens_b, n
        assert np.array_equal(rows_a, rows_b), (n, lo, int((rows_a != rows_b).sum()))

    big = chunk == 512
    agree(last_pass(exact, 428, 250, restore=True), last_pass(exact, 428, 250), 428, [(300, 128), (250, 178) if big else (378, 50)])
    agree(last_pass(exact, 429, 250, restore=True), last_pass(exact, 429, 250), 429, [(428, 1), (250, 179) if big else (378, 51)])
    # the default checkpoint at p - 1 makes a one-token chunk; 100 is below the cache's minimum, no checkpoint
    agree(last_pass(exact, 429, 0), last_pass(exact, 429, [100]), 429, [(428, 1), (0, 429) if big else (384, 45)])
    agree(last_pass(exact, 700, 256), last_pass(plain, 700, 256), 700, [(256, 444) if big else (640, 60), (640, 60)])
    assert 'prefill.128' in exact._programs
    for s in (exact, plain):
        for key, prog in s._programs.items():
            forms = {k.macros.get('EXACT_SUB') for k in prog.kernels.values() if k.function == 'gqa_decode_v3'}
            assert forms == ({'128u'} if s is exact and str(key).startswith('prefill.') else {None}), (s is exact, key)


def test_exact_128_row_sessions_give_v1_its_plain_bits(tmp_path):
    # An exact 128-row session compiles its prompt programs with exact_attention, v1's included: over at most 128 rows
    # v1's EXACT_SUB form is one sub-pass, the plain kernel's work, so the same request restored off the grid (a pass of
    # 128 rows from 300) gives a plain and an exact session the same last prompt row's scores, bit for bit. pick_chunk
    # picks 16-key chunks at every context here, so this checks the form's blocks and fold, not v1's chunk choice.
    _checkpoint(tmp_path)
    pack_model(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path), str(tmp_path / 'pack'),
               PackLayout(rows=16))
    ids = np.random.default_rng(81).integers(0, 50, 800).tolist()
    rows = []
    for exact in (False, True):
        s = Session(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=1024), str(tmp_path / 'pack'), eos=-1,
                    autotune=False, attention='v1', accelerator='off', prefill_chunk_size=128, prefill_exact=exact,
                    prefix_cache=True, prefix_cache_min_tokens=128)
        s.generate(ids[:300] + ids[700:], 1, cache_prefix_tokens=300)
        g = s.generate(ids[:428], 1, cache_prefix_tokens=250)
        assert g.cached_prompt_tokens == 300 and [t['tokens'] for t in g.prefill_timings] == [128]
        rows.append(np.frombuffer(s.read('logits'), dtype=np.uint16).reshape(-1, s.model.config.vocab_size)[127].copy())
    assert np.array_equal(*rows), int((rows[0] != rows[1]).sum())
    v1 = [k for k in s._programs['prefill.128'].kernels.values() if k.function in ('gqa_decode', 'gqa_merge')]
    assert [k.macros.get('EXACT_SUB') for k in v1] == ['128u', '128u']


def test_tie_rows_hold_the_scores_the_reference_chose_by(tmp_path):
    # greedy_ties.score_row, the scores agree reads: the prompt pass's row for token 0 and the decode step's row 0 for a
    # later token, with the steps queued past the last leaving that row alone
    _checkpoint(tmp_path)
    model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=512)
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout(rows=16))
    reference = Session(Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=512), str(tmp_path / 'pack'),
                        prefill_chunk_size=8, eos=-1, autotune=False, accelerator='off', attention='v1')
    for count in (3, 5, 13, 16):                       # the 4- and 8-row prompt programs, a partial last chunk, a whole one
        ids = np.random.default_rng(count).integers(0, 50, count).tolist()
        tokens = reference.generate(ids, 6).tokens
        for n in (1, 2, 6):
            row = score_row(reference, ids, n - 1)
            assert row[tokens[n - 1]] == row.max(), (count, n)
            if n > 1:   # a repeated token is a later step's maximum too: the row must match its own position's prompt row
                same, after = (score_row(reference, ids + tokens[:k], 0) for k in (n - 1, n))
                assert np.abs(row - same).max() < np.abs(row - after).max(), (count, n)
