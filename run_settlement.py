"""The settlement core's suites and measurements (Part H of the Verafy Harness spec).

    python run_settlement.py            # N_SETTLE random calls per program (default 300)

For every named and random call, four runs must agree:
  oracle   svllm/settle_oracle.py, a transcription of runtime.js's settle.js
  ref      the reduced-sBPF settlement programs on the reference interpreter
  sbpf     the same programs on Anza's solana-sbpf 0.12 (xcheck/, if built)
  weights  the same programs on the hand-built weights machine (lazy mode)
plus full mode (every circuit of the subset machine evaluated every step) equal to lazy
mode on a subset. Then measures size and latency. Writes results/settlement.json.
"""
import json, os, platform, statistics, subprocess, sys, time
import multiprocessing as mp
import torch
from svllm import settle, settle_oracle, ref, xcheck, machine
from svllm.settle_cases import named_cases, random_cases

N = int(os.environ.get('N_SETTLE', 300))
N_FULL = int(os.environ.get('N_SETTLE_FULL', 4))
LAT_REPS = int(os.environ.get('N_SETTLE_LAT', 20))


def strip(r):
    return {k: v for k, v in r.items() if k != 'cu'}


def _init():
    torch.set_num_threads(1)


def _weights(args):
    p, c, lazy = args
    return settle.run_program(p, c, 'weights', lazy=lazy)


def machine_name():
    try:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout.strip()) // 2**30
        return f'{chip}, {mem} GB, {platform.system()} {platform.release()}'
    except Exception:                                          # noqa: BLE001
        return platform.platform()


def flops_per_step(program, call, lazy):
    """Dense multiply-accumulate FLOPs (torch FlopCounterMode: matmul, mv) per executed step."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        r = settle.run_program(program, call, 'weights', lazy=lazy)
    return fc.get_total_flops() / r['cu'], r['cu']


def main():
    torch.set_num_threads(1)
    pool = mp.Pool(min(8, os.cpu_count() or 1), initializer=_init)
    named = named_cases()
    cases = [(n, p, c) for n, (p, c) in named.items()] + [(f'random/{i}/{p}', p, c) for i, (p, c) in enumerate(random_cases(N))]
    report = {'date': time.strftime('%Y-%m-%d'), 'machine': machine_name(), 'torch': torch.__version__,
              'python': platform.python_version(), 'window_bytes': settle.WIN,
              'programs': {p: {'instructions': len(b) // 8} for p, b in settle.PROGRAMS.items()}}
    unsupported, rows = [], []
    for n, p, c in cases:
        try:
            settle.encode(p, c)
            rows.append((n, p, c))
        except settle.Unsupported as e:
            unsupported.append((n, str(e)))
    fails = []
    t0 = time.time()
    oracle = [settle_oracle.run_program(p, c) for _, p, c in rows]
    refs = [settle.run_program(p, c, 'ref') for _, p, c in rows]
    ok_ref = sum(strip(a) == b for a, b in zip(refs, oracle))
    fails += [('ref-vs-oracle', n) for (n, *_), a, b in zip(rows, refs, oracle) if strip(a) != b]
    # solana-sbpf: same bytes, same memory image, same windows
    sb = None
    if os.path.exists(xcheck.BIN):
        mems = [settle.encode(p, c)[0] for _, p, c in rows]
        xs = xcheck.run_many([(settle.PROGRAMS[p], m, settle.CU_LIMIT) for (_, p, _), m in zip(rows, mems)], win=settle.WIN)
        sb = 0
        for (n, p, c), m, x in zip(rows, mems, xs):
            rr = ref.run(settle.PROGRAMS[p], m, settle.CU_LIMIT, win=settle.WIN)
            if x['status'] == rr['status'] == 'ok' and x['r0'] == rr['r0'] and x['cu'] == rr['cu'] and x['mem'] == rr['mem'][:settle.WIN]:
                sb += 1
            else:
                fails.append(('sbpf-vs-ref', n))
    ws = pool.map(_weights, [(p, c, True) for _, p, c in rows], chunksize=1)
    ok_w = sum(a == b for a, b in zip(ws, refs))
    fails += [('weights-vs-ref', n) for (n, *_), a, b in zip(rows, ws, refs) if a != b]
    full_sub = [r for r in rows if r[0].startswith('random/')][:N_FULL * 5]
    fl = pool.map(_weights, [(p, c, False) for _, p, c in full_sub], chunksize=1)
    la = pool.map(_weights, [(p, c, True) for _, p, c in full_sub], chunksize=1)
    ok_full = sum(a == b for a, b in zip(fl, la))
    pool.close()
    per = {}
    for (n, p, c), o, w in zip(rows, oracle, ws):
        d = per.setdefault(p, {'calls': 0, 'ok': 0, 'refused': 0, 'cu': []})
        d['calls'] += 1
        d['ok' if o['ok'] else 'refused'] += 1
        d['cu'].append(w['cu'])
    report['suite'] = {
        'calls': len(rows), 'named': sum(1 for r in rows if not r[0].startswith('random/')), 'random_per_program': N,
        'unsupported_named': unsupported,
        'ref_vs_oracle': f'{ok_ref}/{len(rows)}',
        'solana_sbpf_vs_ref': f'{sb}/{len(rows)}' if sb is not None else 'skipped: xcheck not built',
        'weights_vs_ref': f'{ok_w}/{len(rows)}',
        'weights_vs_oracle': f'{sum(strip(a) == b for a, b in zip(ws, oracle))}/{len(rows)}',
        'full_mode_equals_lazy': f'{ok_full}/{len(full_sub)}',
        'seconds': round(time.time() - t0, 1),
        'per_program': {p: {'calls': d['calls'], 'ok': d['ok'], 'refused': d['refused'],
                            'cu_min': min(d['cu']), 'cu_mean': round(statistics.mean(d['cu']), 1), 'cu_max': max(d['cu'])}
                        for p, d in per.items()},
        'failures': fails[:50],
    }
    print('suite', json.dumps({k: v for k, v in report['suite'].items() if k != 'per_program'}), flush=True)
    # ---------------- size
    report['size'] = settle.weights_inventory()
    # ---------------- latency: one representative call per program, single thread, lazy and full
    reps = {'graded_payout': named['graded/43pct'], 'ranked_split': named['ranked/three-of-four'],
            'milestone_escrow': named['milestone/first'], 'bond': named['bond/slash'], 'ledger': named['ledger/move']}
    lat = {}
    for p, (pp, c) in reps.items():
        settle.run_program(pp, c, 'weights')                   # warm
        ts = []
        for _ in range(LAT_REPS):
            t = time.perf_counter(); r = settle.run_program(pp, c, 'weights'); ts.append(time.perf_counter() - t)
        tf = []
        for _ in range(max(3, LAT_REPS // 5)):
            t = time.perf_counter(); settle.run_program(pp, c, 'weights', lazy=False); tf.append(time.perf_counter() - t)
        tr = []
        for _ in range(LAT_REPS):
            t = time.perf_counter(); settle.run_program(pp, c, 'ref'); tr.append(time.perf_counter() - t)
        f_lazy, cu = flops_per_step(pp, c, True)
        f_full, _ = flops_per_step(pp, c, False)
        lat[p] = {'call': [k for k, v in named.items() if v is reps[p]][0], 'steps': cu,
                  'weights_lazy_ms_median': round(1000 * statistics.median(ts), 2),
                  'weights_full_ms_median': round(1000 * statistics.median(tf), 2),
                  'ref_interpreter_ms_median': round(1000 * statistics.median(tr), 3),
                  'dense_flops_per_step_lazy': round(f_lazy), 'dense_flops_per_step_full': round(f_full),
                  'dense_flops_total_full': round(f_full * cu)}
        print('latency', p, lat[p], flush=True)
    report['latency'] = {'label': f'measured, {report["date"]}, {report["machine"]}, one CPU thread, torch {torch.__version__}',
                         'repetitions': LAT_REPS, 'per_program': lat}
    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/settlement.json', 'w'), indent=1, default=str)
    print('size', report['size']['settlement_machine_params'], 'params,', report['size']['settlement_machine_bytes_float32'], 'bytes float32')
    print('fails', len(fails))
    print('wrote results/settlement.json')
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
