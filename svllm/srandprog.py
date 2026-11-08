"""Random programs for the full memory map: every region, frame gaps, BPF-to-BPF calls, callx,
recursion and the error paths. Returns (prog, input, funcs)."""
import random
from .isa import enc, ALU_OPS, JMP_OPS, SIZES, SIZE_BYTES, LDDW, EXIT, CALL, CALLX

WREGS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
VALS = [0, 1, 2, 3, 7, 63, 64, 255, 0x7fffffff, -1, -2, -0x80000000, 1000000007]
BASES = {'ro': 1 << 32, 'heap': 3 << 32, 'input': 4 << 32}


def rand_imm(rng):
    return rng.choice(VALS) if rng.random() < 0.6 else rng.randint(-2**31, 2**31 - 1)


def lddw(r, v):
    v &= 2**64 - 1
    return [('i', enc(LDDW, r, imm=v & 0xffffffff)), ('i', enc(0, imm=v >> 32))]


def body(rng, n, nfuncs, inlen):
    out = []
    while len(out) < n:
        k = rng.random()
        d = rng.choice(WREGS)
        if k < 0.03:
            out.append(('i', enc(0xd4 if rng.random() < 0.5 else 0xdc, d, imm=rng.choice([16, 32, 64]))))
        elif k < 0.42:
            name = rng.choice(list(ALU_OPS)); h = ALU_OPS[name]
            w32 = rng.random() < 0.35
            ci, cr = (0x04, 0x0c) if w32 else (0x07, 0x0f)
            if name == 'neg':
                out.append(('i', enc(ci | 0x80, d)))
            elif rng.random() < 0.5:
                out.append(('i', enc(cr | h, d, rng.choice(WREGS + [1, 10]))))
            else:
                imm = rand_imm(rng)
                if name in ('lsh', 'rsh', 'arsh'):
                    imm = rng.randint(0, 31 if w32 else 63)
                if name in ('div', 'mod') and (imm & 0xffffffff if w32 else imm) == 0:
                    imm = 3
                out.append(('i', enc(ci | h, d, imm=imm)))
        elif k < 0.55:
            name = rng.choice(list(JMP_OPS))
            off = rng.randint(1, 4) if rng.random() < 0.85 else -rng.randint(1, 4)
            if name == 'ja':
                out.append(('j', 0x05, 0, 0, off, 0))
            elif rng.random() < 0.5:
                out.append(('j', 0x0d | JMP_OPS[name], rng.choice(WREGS), rng.choice(WREGS), off, 0))
            else:
                out.append(('j', 0x05 | JMP_OPS[name], rng.choice(WREGS), 0, off, rand_imm(rng)))
        elif k < 0.85:
            s = rng.choice(list(SIZES)); nb = SIZE_BYTES[s]
            where = rng.random()
            if where < 0.4:                               # stack, near the frame top and the gaps
                base = 10
                off = -rng.randint(nb, 4096) if rng.random() < 0.9 else rng.choice([-4096 - nb, 0, -nb + 2, 4092, 8000])
            elif where < 0.6:                             # input, near its end
                base = 1
                off = rng.randint(0, max(0, inlen - nb)) if rng.random() < 0.85 else inlen - nb + rng.randint(1, 3)
            else:                                         # heap or read-only program memory
                rg = 'heap' if rng.random() < 0.6 else 'ro'
                base = rng.choice([2, 3, 4, 5])
                out += lddw(base, BASES[rg])
                hi = 32768 if rg == 'heap' else 64
                off = rng.randint(0, hi - nb) if rng.random() < 0.85 else hi - nb + rng.randint(1, 3)
                off = min(off, 32767)
            t = rng.random()
            if t < 0.4:
                out.append(('i', enc(0x61 | SIZES[s], d, base, off)))
            elif t < 0.8:
                out.append(('i', enc(0x63 | SIZES[s], base, rng.choice(WREGS), off)))
            else:
                out.append(('i', enc(0x62 | SIZES[s], base, 0, off, rand_imm(rng))))
        elif nfuncs:
            f = rng.randrange(nfuncs)
            r = rng.random()
            if r < 0.75:
                out.append(('call', f))
            elif r < 0.92:
                out.append(('callx', f, rng.choice([2, 3, 4, 5])))
            else:
                out.append(('badcall',))
    return out


def gen(rng):
    nfuncs = rng.choice([0, 1, 2, 3])
    inlen = rng.choice([0, 8, 40, 128, 300])
    blocks = [body(rng, rng.randint(6, 24), nfuncs, inlen)]
    for _ in range(nfuncs):
        blocks.append(body(rng, rng.randint(3, 12), nfuncs, inlen))
    # lay out: each block ends with exit; expand callx into lddw + callx
    flat, starts = [], []
    for b in blocks:
        exp = []
        for item in b:
            if item[0] == 'callx':
                exp += [('xaddr', item[1], item[2]), ('xpad',), ('i', enc(CALLX, imm=item[2]))]
            else:
                exp.append(item)
        exp.append(('i', enc(EXIT)))
        starts.append(len(flat))
        flat.append(exp)
    pcs, pc = [], 0
    for b in flat:
        pcs.append(pc); pc += len(b)
    keys = rng.sample(range(1, 2**32), nfuncs)
    funcs = {keys[i]: pcs[i + 1] for i in range(nfuncs)}
    out = []
    for bi, b in enumerate(flat):
        L = len(b)
        for i, item in enumerate(b):
            if item[0] == 'j':
                _, op, d, s, off, imm = item
                off = max(-i - 1, min(off, L - i - 1))
                out.append(enc(op, d, s, off, imm))
            elif item[0] == 'call':
                out.append(enc(CALL, imm=keys[item[1]]))
            elif item[0] == 'badcall':
                out.append(enc(CALL, imm=rng.choice([0, 12345])))
            elif item[0] == 'xaddr':
                tgt = (1 << 32) + 8 * pcs[item[1] + 1] + rng.choice([0, 0, 0, 0, 3, 8 * 100000])
                out.append(enc(LDDW, item[2], imm=tgt & 0xffffffff))
            elif item[0] == 'xpad':
                out.append(enc(0, imm=1))
            else:
                out.append(item[1])
    return b''.join(out), bytes(rng.getrandbits(8) for _ in range(inlen)), funcs
