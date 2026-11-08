"""Speed: instructions per second in the weights (one CPU thread, lazy mode) against Anza's
solana-sbpf interpreter on the same machine. Writes results/speed.json."""
import json, os, subprocess, time
import torch
from svllm import svm, scheck
from svllm.isa import assemble

torch.set_num_threads(1)

# Each loop body runs N times; the class is what dominates the body.
KERNELS = {
    'alu (add, xor, shifts)': 'mov64 r6, {n}\nloop:\nadd64 r2, 7\nxor64 r3, r2\nlsh64 r3, 1\nrsh64 r2, 3\nsub64 r6, 1\njne r6, 0, loop\nexit',
    'mul': 'mov64 r6, {n}\nmov64 r2, 3\nloop:\nmul64 r2, 1000003\nmul64 r3, r2\nsub64 r6, 1\njne r6, 0, loop\nexit',
    'div and mod': 'mov64 r6, {n}\nlddw r2, 0x123456789abcdef\nloop:\ndiv64 r2, 3\nmod64 r3, 7\nadd64 r2, 99999\nsub64 r6, 1\njne r6, 0, loop\nexit',
    'memory (stack load/store)': 'mov64 r6, {n}\nloop:\nstxdw [r10-8], r6\nldxdw r2, [r10-8]\nstxb [r10-100], r2\nldxb r3, [r10-100]\nsub64 r6, 1\njne r6, 0, loop\nexit',
    'calls': 'mov64 r6, {n}\nloop:\ncall 7\nsub64 r6, 1\njne r6, 0, loop\nexit\nadd64 r0, 1\nexit',
}
FUNCS = {'calls': {7: 5}}


def weights_rate(name, n):
    p = assemble(KERNELS[name].format(n=n))
    m = svm.Machine(p, b'', 10**7, True, FUNCS.get(name, {}))
    steps, t = 0, time.time()
    while not m.halted:
        m.step(); steps += 1
    return steps, time.time() - t, m.status


def anza_rate(name, n):
    p = assemble(KERNELS[name].format(n=n))
    case = (p, b'', 10**9, True, FUNCS.get(name, {}))
    line = scheck.run_many([case])                       # warm the binary
    t = time.time()
    for _ in range(3):
        r = scheck.run_many([case])[0]
    return r['cu'], (time.time() - t) / 3, r['status']


def main():
    out = {'machine': f'{os.cpu_count()} CPU cores, no GPU; weights on one thread, lazy mode', 'kernels': {}}
    for name in KERNELS:
        steps, secs, st = weights_rate(name, 40)
        a_steps, a_secs, a_st = anza_rate(name, 2_000_000)
        out['kernels'][name] = {
            'weights_ms_per_insn': round(1000 * secs / steps, 2), 'weights_insn_per_s': round(steps / secs, 1),
            'anza_ns_per_insn': round(1e9 * a_secs / a_steps, 1), 'anza_insn_per_s': round(a_steps / a_secs),
            'slowdown': round((secs / steps) / (a_secs / a_steps)), 'status': [st, a_st]}
        print(name, out['kernels'][name], flush=True)
    json.dump(out, open('results/speed.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
