"""Run programs on Anza's solana-sbpf interpreter (Rust, SBPF v0) via xcheck/ binary."""
import subprocess, os
BIN = os.path.join(os.path.dirname(__file__), '..', 'xcheck', 'target', 'release', 'xcheck')

NORM = {'StackAccessViolation': 'AccessViolation', 'UnsupportedInstruction': 'InvalidInstruction'}


def run_many(cases, win=None):
    """cases: list of (prog_bytes, mem128_bytes, cu_limit). Returns list of dicts.
    win: map `win` bytes of input and of stack instead of 128 (mem is then win bytes)."""
    w = f' {win}' if win else ''
    inp = ''.join(f"{p.hex()} {(m.hex() or '-')} {lim}{w}\n" for p, m, lim in cases)
    out = subprocess.run([BIN], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
    res = []
    for line in out:
        f = line.split()
        if f[0] == 'ok':
            regs = [int(x, 16) for x in f[3].split(',')]
            res.append({'status': 'ok', 'r0': int(f[1]), 'cu': int(f[2]), 'regs': regs,
                        'mem': bytes.fromhex(f[4]) if f[4] != '-' else b''})
        else:
            res.append({'status': NORM.get(f[1], f[1]), 'cu': int(f[2])})
    return res
