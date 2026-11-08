"""Full-layout conformance: random programs on Anza's solana-sbpf (ground truth), the Python
reference and the hand-set weights. Writes results/full.json."""
import json, os, random, sys, time, collections
import multiprocessing as mp
import torch
from svllm import sref, scheck, svm
from svllm.srandprog import gen

N = int(os.environ.get('N', 600))
CU = int(os.environ.get('CU', 600))
SEED = int(os.environ.get('SEED', 2026))


def _init():
    torch.set_num_threads(1)


def _w(c):
    t = time.time()
    r = svm.run(*c)
    r['secs'] = time.time() - t
    return r


def main():
    rng = random.Random(SEED)
    cases = []
    for _ in range(N):
        p, m, f = gen(rng)
        cases.append((p, m, CU, rng.random() < 0.7, f))
    xs = scheck.run_many(cases)
    refs = [sref.run(*c) for c in cases]
    t0 = time.time()
    with mp.Pool(os.cpu_count() or 1, initializer=_init) as pool:
        ws = pool.map(_w, cases, chunksize=4)
    wall = time.time() - t0
    rx = sum(scheck.same(r, x) for r, x in zip(refs, xs))
    wx = sum(scheck.same(w, x) for w, x in zip(ws, xs))
    fails = [{'i': i, 'weights': (w['status'], w['cu'], w['r0']), 'anza': (x['status'], x['cu'], x.get('r0'))}
             for i, (w, x) in enumerate(zip(ws, xs)) if not scheck.same(w, x)]
    steps = sum(w['cu'] for w in ws)
    rep = {'cases': N, 'cu_limit': CU, 'seed': SEED, 'ref_vs_anza': f'{rx}/{N}', 'weights_vs_anza': f'{wx}/{N}',
           'statuses': dict(collections.Counter(x['status'] for x in xs)),
           'with_calls': sum(bool(c[4]) for c in cases), 'gaps_on': sum(c[3] for c in cases),
           'steps': steps, 'cpu_seconds': round(sum(w['secs'] for w in ws), 1), 'wall_seconds': round(wall, 1),
           'ms_per_step_one_thread': round(1000 * sum(w['secs'] for w in ws) / max(steps, 1), 2),
           'failures': fails[:30]}
    os.makedirs('results', exist_ok=True)
    json.dump(rep, open(os.environ.get('OUT', 'results/full.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'failures'}), 'failures', len(fails))


if __name__ == '__main__':
    main()
