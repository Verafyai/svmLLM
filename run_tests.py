"""Run conformance + demo + random suites on: reference interpreter, solana-sbpf (Rust), weights machine."""
import json, random, sys, time, os
import multiprocessing as mp
import torch
from svllm import ref, machine, xcheck
from svllm.programs import demo_cases, conformance_cases, DEMOS
from svllm.randprog import gen
from svllm.isa import disasm_one, dec

N_RANDOM = int(os.environ.get('N_RANDOM', 300))
N_FULL = int(os.environ.get('N_FULL', 25))


def same(a, b, regs=True, memlen=256):
    if a['status'] != b['status'] or a['cu'] != b['cu']:
        return False
    if a['status'] != 'ok':
        return True
    if a['r0'] != b['r0'] or a['mem'][:memlen] != b['mem'][:memlen]:
        return False
    return (a['regs'] == b['regs']) if regs else True


def _init():
    torch.set_num_threads(1)


def _wrun(args):
    p, m, lim, lazy = args
    return machine.run(p, m, lim, lazy=lazy)


def main():
    pool = mp.Pool(min(8, os.cpu_count() or 1), initializer=_init)
    rng = random.Random(1337)
    suites = {'conformance': conformance_cases(), 'demo': demo_cases(),
              'random': [(f'random/{i}', *gen(rng), 200) for i in range(N_RANDOM)]}
    report, fails = {}, []
    for sname, cases in suites.items():
        t0 = time.time()
        refs = [ref.run(p, m, lim) for _, p, m, lim in cases]
        xc_idx = [i for i, (n, *_ ) in enumerate(cases) if 'call' not in n]
        xcs = xcheck.run_many([cases[i][1:] for i in xc_idx])
        rx = sum(same(refs[i], x, regs=False, memlen=128) for i, x in zip(xc_idx, xcs))
        for i, x in zip(xc_idx, xcs):
            if not same(refs[i], x, regs=False, memlen=128):
                fails.append(('ref-vs-rbpf', cases[i][0], refs[i]['status'], x['status'], refs[i]['cu'], x['cu']))
        mw = 0
        ws = pool.map(_wrun, [(p, m, lim, True) for _, p, m, lim in cases], chunksize=1)
        for (n, p, m, lim), r, w in zip(cases, refs, ws):
            if same(r, w):
                mw += 1
            else:
                fails.append(('weights-vs-ref', n, r['status'], w['status'], r['r0'], w['r0']))
        report[sname] = {'cases': len(cases), 'ref_vs_solana_sbpf': f'{rx}/{len(xc_idx)}',
                         'weights_vs_ref': f'{mw}/{len(cases)}', 'seconds': round(time.time() - t0, 1),
                         'statuses': {s: sum(r['status'] == s for r in refs) for s in set(r['status'] for r in refs)}}
        report[sname]['total_steps'] = sum(r['cu'] for r in refs)
        print(sname, report[sname], flush=True)
    # full mode (no lazy branch skipping) on a subset
    sub = suites['conformance'][:N_FULL] + suites['random'][:N_FULL]
    fa = pool.map(_wrun, [(p, m, lim, False) for _, p, m, lim in sub], chunksize=1)
    la = pool.map(_wrun, [(p, m, lim, True) for _, p, m, lim in sub], chunksize=1)
    fm = sum(same(a, b) for a, b in zip(fa, la))
    report['full_mode_equals_lazy'] = f'{fm}/{len(sub)}'
    report['weights_param_count'] = machine.PARAMS['count']
    report['failures'] = fails[:50]
    print('full mode', report['full_mode_equals_lazy'], 'fails', len(fails))
    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/results.json', 'w'), indent=1, default=str)
    # weights-run traces for the 10 site demos
    traces = {}
    for name, p, m, lim in suites['demo']:
        tr = []
        w = machine.run(p, m, lim, trace=tr)
        traces[name] = {'status': w['status'], 'r0': w['r0'], 'cu': w['cu'],
                        'steps': [{'pc': s['pc'], 'cu': s['cu'], 'insn': disasm_one(p, s['pc']),
                                   'regs': [hex(v) for v in s['regs']]} for s in tr]}
    json.dump(traces, open('results/weights_traces.json', 'w'))
    # judgement-filter examples: VM run on the weights machine, then the rule judge (not weights)
    from svllm.judge import run_examples
    jw, jr = run_examples('weights'), run_examples('ref')
    for a, b in zip(jw, jr):
        assert a['vm']['status'] == b['vm']['status'] and a['vm']['r0'] == b['vm']['r0'] and a['vm']['cu'] == b['vm']['cu'], a['id']
    json.dump(jw, open('results/judge_examples.json', 'w'), indent=1)
    print('judge', [(e['id'], e['verdict']['decision']) for e in jw])
    print('wrote results/')


if __name__ == '__main__':
    main()
