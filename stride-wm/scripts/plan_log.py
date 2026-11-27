"""Planning rows live in one file per writer while a run is going, merged into plan.jsonl for the reports.

  <res>/plan_main.jsonl  learned-model planners, expert replay and random actions (eval_plan.py, GPU stages)
  <res>/plan_sim.jsonl   the simulator planner (eval_oracle.py), which runs in the background at the same time
  <res>/plan.jsonl       both, rebuilt by `merge`; nothing writes to it directly

Why not one file: each planning script starts by dropping a half-written last row (stride.planning.load_done),
which would cut off a row the other process is writing at that moment.

  python scripts/plan_log.py split <res> [--first results_a100/plan.jsonl]
      once: divide <res>/plan.jsonl (or, if there is none yet, the first run's rows) between the two files.
      Does nothing when plan_main.jsonl already exists.
  python scripts/plan_log.py merge <res>
      rebuild <res>/plan.jsonl from the two files (and any plan_main_<tag>.jsonl / plan_sim_<tag>.jsonl from planners
      running side by side), skipping a row that is still being written."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.planning import row_key


def read_rows(path):
    """Complete rows of a JSON-lines file; a line that does not parse (cut off mid-write) is skipped."""
    path, out = Path(path), []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def write_rows(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    tmp.replace(path)  # atomic: a reader never sees a half-written file


def split(res, first=None):
    res = Path(res)
    main, sim = res / 'plan_main.jsonl', res / 'plan_sim.jsonl'
    if main.exists():
        print(f'{main} exists: planning rows already split', flush=True)
        return
    src = res / 'plan.jsonl'
    if not src.exists() and first and Path(first).exists():
        src = Path(first)
    rows = read_rows(src)
    old_sim = read_rows(sim)
    seen = {row_key(r) for r in old_sim}
    new_sim = [r for r in rows if r.get('arch') == 'oracle' and row_key(r) not in seen]
    res.mkdir(parents=True, exist_ok=True)
    write_rows(sim, old_sim + new_sim)
    write_rows(main, [r for r in rows if r.get('arch') != 'oracle'])  # last: its existence marks the split as done
    print(f'split {src}: {len(rows) - len(new_sim)} rows -> {main}, {len(old_sim) + len(new_sim)} -> {sim}', flush=True)


def merge(res):
    """plan.jsonl = plan_main.jsonl + plan_sim.jsonl, plus any plan_main_<tag>.jsonl / plan_sim_<tag>.jsonl written by
    processes that run at the same time (one file each, for the same reason as above)."""
    res = Path(res)
    main = [r for f in sorted(res.glob('plan_main*.jsonl')) for r in read_rows(f)]
    sim = [r for f in sorted(res.glob('plan_sim*.jsonl')) for r in read_rows(f)]
    write_rows(res / 'plan.jsonl', main + sim)
    print(f'{res / "plan.jsonl"}: {len(main)} model/replay/random rows + {len(sim)} simulator-planner rows', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('split', 'merge'))
    p.add_argument('res')
    p.add_argument('--first', default=None, help='rows to start from when <res>/plan.jsonl does not exist')
    a = p.parse_args()
    split(a.res, a.first) if a.action == 'split' else merge(a.res)
