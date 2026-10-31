"""Random reduced-sBPF program generator (valid encodings, bounded loops via CU limit)."""
import random
from .isa import enc, ALU_OPS, JMP_OPS, SIZES, SIZE_BYTES, LDDW, EXIT

WREGS = [0, 2, 3, 4, 5, 6, 7, 8, 9]          # never write r1 / r10 (bases)
VALS = [0, 1, 2, 3, 7, 63, 64, 255, 0x7fffffff, -1, -2, -0x80000000, 1000000007]


def rand_imm(rng):
    return rng.choice(VALS) if rng.random() < 0.6 else rng.randint(-2**31, 2**31 - 1)


def gen(rng, n=None):
    n = n or rng.randint(6, 28)
    out = []
    # seed registers with lddw so values are rich
    for r in rng.sample(WREGS, 4):
        v = rng.getrandbits(64) if rng.random() < 0.7 else rng.choice([0, 1, 2**63, 2**64 - 1])
        out += [enc(LDDW, r, imm=v & 0xffffffff), enc(0, imm=v >> 32)]
    body = []
    while len(body) < n:
        k = rng.random()
        d = rng.choice(WREGS)
        if k < 0.55:
            name = rng.choice(list(ALU_OPS))
            h = ALU_OPS[name]
            if name == 'neg':
                body.append(('i', enc(0x87, d)))
            elif rng.random() < 0.5:
                body.append(('i', enc(0x0f | h, d, rng.choice(WREGS + [1, 10]))))
            else:
                imm = rand_imm(rng)
                if name in ('lsh', 'rsh', 'arsh'):
                    imm = rng.randint(0, 63)
                if name in ('div', 'mod') and imm == 0:
                    imm = 3
                body.append(('i', enc(0x07 | h, d, imm=imm)))
        elif k < 0.75:
            name = rng.choice(list(JMP_OPS))
            off = rng.randint(1, 4) if rng.random() < 0.85 else -rng.randint(1, 4)
            if name == 'ja':
                body.append(('j', 0x05, 0, 0, off, 0))
            elif rng.random() < 0.5:
                body.append(('j', 0x0d | JMP_OPS[name], rng.choice(WREGS), rng.choice(WREGS), off, 0))
            else:
                body.append(('j', 0x05 | JMP_OPS[name], rng.choice(WREGS), 0, off, rand_imm(rng)))
        else:
            s = rng.choice(list(SIZES)); nb = SIZE_BYTES[s]
            if rng.random() < 0.5:
                base, off = 1, rng.randint(0, 128 - nb) if rng.random() < 0.95 else 128 - nb + rng.randint(1, 4)
            else:
                base, off = 10, -rng.randint(nb, 128) if rng.random() < 0.97 else -rng.randint(129, 140)
            kind = rng.random()
            if kind < 0.4:
                body.append(('i', enc(0x61 | SIZES[s], d, base, off)))
            elif kind < 0.8:
                body.append(('i', enc(0x63 | SIZES[s], base, rng.choice(WREGS), off)))
            else:
                body.append(('i', enc(0x62 | SIZES[s], base, 0, off, rand_imm(rng))))
    # resolve jumps so targets stay within [0, len] (len = the final exit)
    L = len(body)
    for i, b in enumerate(body):
        if b[0] == 'j':
            _, op, d, s, off, imm = b
            off = max(-i - 1, min(off, L - i - 1))
            out.append(enc(op, d, s, off, imm))
        else:
            out.append(b[1])
    out.append(enc(EXIT))
    return b''.join(out), bytes(rng.getrandbits(8) for _ in range(128))
