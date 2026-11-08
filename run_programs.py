"""Real Solana programs in the weights: TinySVM's six programs, compiled by cargo-build-sbf,
loaded by Agave's loader, run on the hand-set weights and on LiteSVM (Agave 4.3.2 runtime).
Writes results/programs.json."""
import json, os, random, time, collections
import multiprocessing as mp
import torch
from svllm import solana, sref, svm

N = int(os.environ.get('N', 120))
SEED = int(os.environ.get('SEED', 11))
KIND = os.environ.get('KIND', 'tiny')
DEPLOY = os.environ.get('DEPLOY', '/home/claude/verafyharness/packages/tinysvm/programs/target/deploy')


def _init():
    torch.set_num_threads(1)


def judge(r, meta, acc, cu):
    accts, err = (solana.deserialize(r['input'], meta, acc) if r['status'] == 'ok' and r['r0'] == 0 else (None, None))
    return solana.outcome(r['status'], r['r0'], r['cu'], r['logs'], accts, err, limit=cu)


def _w(job):
    prog, elf, acc, data, cu = job
    img = solana.image(elf)
    inp, meta = solana.serialize(acc, data)
    t = time.time()
    r = svm.run_program(img, inp, cu, program_id=solana.PROGRAM_ID, program_id_b58=sref.b58(solana.PROGRAM_ID))
    o = judge(r, meta, acc, cu)
    o['secs'], o['steps'] = time.time() - t, r['steps']
    return o


def main():
    rng = random.Random(SEED)
    jobs = []
    for _ in range(N):
        prog, elf, acc, data = solana.gen_exerciser(rng) if KIND == 'ex' else solana.gen_case(rng, DEPLOY)
        cu = 200_000 if rng.random() < 0.85 else rng.choice([1, 50, 100, 300, 600, 900, 1200])
        jobs.append((prog, elf, acc, data, cu))
    truth = solana.truth_many([(e, a, d, cu) for _, e, a, d, cu in jobs])
    refs = []
    for (prog, elf, acc, data, cu) in jobs:
        inp, meta = solana.serialize(acc, data)
        refs.append(judge(sref.run_program(solana.image(elf), inp, cu), meta, acc, cu))
    t0 = time.time()
    with mp.Pool(os.cpu_count() or 1, initializer=_init) as pool:
        ws = pool.map(_w, jobs, chunksize=1)
    wall = time.time() - t0
    per = collections.defaultdict(lambda: {'cases': 0, 'weights_match': 0, 'ok': 0, 'cu': 0, 'steps': 0, 'secs': 0.0})
    fails = []
    for (prog, elf, acc, data, cu), t, r, w in zip(jobs, truth, refs, ws):
        p = per[prog]
        p['cases'] += 1; p['ok'] += t['ok']; p['cu'] += t['cu']; p['steps'] += w['steps']; p['secs'] += w['secs']
        m = solana.same(w, t)
        p['weights_match'] += m
        if not m:
            fails.append({'program': prog, 'data': data.hex(), 'cu_limit': cu,
                          'weights': {k: w.get(k) for k in ('ok', 'cu', 'err', 'logs')},
                          'truth': {k: t.get(k) for k in ('ok', 'cu', 'err', 'logs')}})
    rep = {
        'truth': 'LiteSVM 0.18.2 (Agave 4.3.2 runtime, mainnet feature set)',
        'loader': 'Agave 4.3.2 ELF loader (truth/src/bin/elfimage.rs)',
        'programs_built_with': 'cargo-build-sbf 2.1.21, platform-tools v1.43',
        'cases': N, 'seed': SEED,
        'ref_vs_truth': f"{sum(solana.same(r, t) for r, t in zip(refs, truth))}/{N}",
        'weights_vs_truth': f"{sum(solana.same(w, t) for w, t in zip(ws, truth))}/{N}",
        'truth_outcomes': dict(collections.Counter('ok' if t['ok'] else t['err'] for t in truth)),
        'cases_with_logs': sum(bool(t['logs']) for t in truth),
        'per_program': {k: dict(v, secs=round(v['secs'], 1)) for k, v in sorted(per.items())},
        'total_cu': sum(t['cu'] for t in truth), 'total_steps': sum(w['steps'] for w in ws),
        'cpu_seconds': round(sum(w['secs'] for w in ws), 1), 'wall_seconds': round(wall, 1),
        'failures': fails[:20],
    }
    json.dump(rep, open(os.environ.get('OUT', 'results/programs.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('failures', 'per_program')}), 'failures', len(fails))


if __name__ == '__main__':
    main()
