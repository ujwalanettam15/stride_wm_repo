"""scripts/plan_log.py: planning rows split per writer and merged for the reports, without losing a row."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'plan_log.py'


def row(model, arch, t0, k=1, cost='latent'):
    return dict(model=model, arch=arch, k=k, cost=cost, offset=40, t0=t0, horizon=40, samples=100, iters=10, seed=0)


def call(*args):
    subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=True, capture_output=True)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_split_then_merge_keeps_every_complete_row(tmp_path):
    rows = [row('chunk_k8_s0', 'chunk', 1), row('oracle_sim', 'oracle', 2, cost='true_cube'),
            row('expert_replay', 'replay', 3, k=0, cost='')]
    (tmp_path / 'plan.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows) + '{"model": "cut off')
    call('split', tmp_path)
    assert [r['t0'] for r in lines(tmp_path / 'plan_main.jsonl')] == [1, 3]
    assert [r['t0'] for r in lines(tmp_path / 'plan_sim.jsonl')] == [2]

    # a second split changes nothing, even after new rows were appended
    with open(tmp_path / 'plan_sim.jsonl', 'a') as f:
        f.write(json.dumps(row('oracle_sim', 'oracle', 4, cost='true_cube')) + '\n{"model": "being writ')
    call('split', tmp_path)
    assert len(lines(tmp_path / 'plan_main.jsonl')) == 2

    call('merge', tmp_path)  # the row still being written is skipped, not truncated away
    assert sorted(r['t0'] for r in lines(tmp_path / 'plan.jsonl')) == [1, 2, 3, 4]
    assert (tmp_path / 'plan_sim.jsonl').read_text().endswith('"being writ')


def test_split_starts_from_the_first_run_when_there_is_no_plan_yet(tmp_path):
    first = tmp_path / 'first.jsonl'
    first.write_text(json.dumps(row('chunk_k1_s0', 'chunk', 7)) + '\n')
    res = tmp_path / 'res'
    call('split', res, '--first', first)
    assert [r['t0'] for r in lines(res / 'plan_main.jsonl')] == [7]
    assert (res / 'plan_sim.jsonl').read_text() == ''


def test_merge_includes_per_process_files(tmp_path):
    (tmp_path / 'plan_main.jsonl').write_text(json.dumps(row('chunk_k1_s0', 'chunk', 1)) + '\n')
    (tmp_path / 'plan_main_s1.jsonl').write_text(json.dumps(row('chunk_k1_s1', 'chunk', 1)) + '\n')
    (tmp_path / 'plan_sim.jsonl').write_text(json.dumps(row('oracle_sim', 'oracle', 2, cost='true_cube')) + '\n')
    (tmp_path / 'plan_main_s1.jsonl.tmp').write_text('not a results file\n')
    call('merge', tmp_path)
    assert sorted(r['model'] for r in lines(tmp_path / 'plan.jsonl')) == ['chunk_k1_s0', 'chunk_k1_s1', 'oracle_sim']
