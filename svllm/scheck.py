"""Run cases on Anza's solana-sbpf with the full memory map (xcheck/src/bin/svmcheck.rs)."""
import subprocess, os
BIN = os.path.join(os.path.dirname(__file__), '..', 'xcheck', 'target', 'release', 'svmcheck')
NORM = {'StackAccessViolation': 'AccessViolation', 'UnsupportedInstruction': 'InvalidInstruction'}


def run_many(cases):
    """cases: (prog, input, cu_limit, gaps, funcs{key: pc})."""
    def fl(f):
        return ','.join(f'{k}:{v}' for k, v in f.items()) or '-'
    inp = ''.join(f"{p.hex()} {(m.hex() or '-')} {lim} {int(g)} {fl(f)}\n" for p, m, lim, g, f in cases)
    out = subprocess.run([BIN], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
    res = []
    for line in out:
        f = line.split()
        if f[0] == 'ok':
            res.append({'status': 'ok', 'r0': int(f[1]), 'cu': int(f[2]), 'regs': [int(x, 16) for x in f[3].split(',')],
                        'hashes': tuple(int(x, 16) for x in f[4:7])})
        else:
            res.append({'status': NORM.get(f[1], f[1]), 'cu': int(f[2])})
    return res


def same(a, b):
    """Equal status and CU; on success also r0, r1..r10 and the stack, heap and input hashes."""
    if a['status'] != b['status'] or a['cu'] != b['cu']:
        return False
    if a['status'] != 'ok':
        return True
    return a['r0'] == b['r0'] and a['regs'][1:] == b['regs'][1:] and a['hashes'] == b['hashes']
