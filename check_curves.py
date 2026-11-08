"""Curve syscalls through the exerciser: random calls to sol_curve_validate_point, sol_curve_group_op,
sol_curve_multiscalar_mul (Edwards and Ristretto), sol_secp256k1_recover and the BN254 / Poseidon /
BLS12-381 families, with valid points, invalid points, non-canonical scalars and bad pointers,
replayed by the reference VM (or the weights) and compared with LiteSVM.

Account 0 is 4096 bytes: inputs at [128, 512) and [1024, 4096), syscall results at [512, ...),
output buffers at [1024 + ...)."""
import hashlib, random, struct, sys, collections, time
from svllm import solana, sref, curves as C

O = solana.OUT_VM
op = lambda t, *a: bytes([t]) + b''.join((x % 2**64).to_bytes(8, 'little') for x in (list(a) + [0] * 5)[:5])
B = C.ed_decompress(bytes.fromhex('58' + '66' * 31))


def ed_points(rng):
    k = rng.randrange(1, C.L)
    pt = C.ed_mul(k, B)
    c = rng.random()
    if c < 0.55:
        return C.ed_compress(pt), C.ristretto_encode(pt)
    if c < 0.7:                                              # small-order / torsion components
        t = C.ed_decompress(bytes(32)) if rng.random() < 0.5 else C.ed_decompress(bytes.fromhex('ec' + 'ff' * 30 + '7f'))
        return C.ed_compress(C.ed_add(pt, t)) if t else C.ed_compress(pt), C.ristretto_encode(pt)
    raw = bytes(rng.getrandbits(8) for _ in range(32))
    if rng.random() < 0.3:
        raw = raw[:31] + bytes([raw[31] | 0x80])
    return raw, raw


def scalar_bytes(rng):
    c = rng.random()
    if c < 0.75:
        return rng.randrange(C.L).to_bytes(32, 'little')
    if c < 0.85:
        return rng.choice([0, 1, C.L - 1]).to_bytes(32, 'little')
    return rng.randrange(C.L, 2**256).to_bytes(32, 'little')


def secp_case(rng):
    d = rng.randrange(1, C.KN)
    h = bytes(rng.getrandbits(8) for _ in range(32))
    k = rng.randrange(1, C.KN)
    R = C.k_mul(k, C.KG)
    r = R[0] % C.KN
    s = C.inv(k, C.KN) * (int.from_bytes(h, 'big') + r * d) % C.KN
    recid = (R[1] & 1) | (2 if R[0] >= C.KN else 0)
    if rng.random() < 0.3:
        s = C.KN - s; recid ^= 1
    c = rng.random()
    if c < 0.1:
        r = rng.choice([0, C.KN, C.KN + 5, 2**256 - 1])
    elif c < 0.2:
        s = rng.choice([0, C.KN, 2**256 - 1])
    elif c < 0.3:
        r = rng.randrange(1, C.KN)
    sig = r.to_bytes(32, 'big') + s.to_bytes(32, 'big')
    if rng.random() < 0.1:
        recid = rng.choice([2, 3, 4, 255, 256, 2**63])
    return h, sig, recid


from py_ecc import optimized_bn128 as PB


def bn_aff(P, g2):
    if P is None or PB.is_inf(P):
        return None
    x, y = PB.normalize(P)
    return ((x.coeffs[0] % C.BP, x.coeffs[1] % C.BP), (y.coeffs[0] % C.BP, y.coeffs[1] % C.BP)) if g2 else (x.n, y.n)


def bn_enc(P, g2, le):
    F = C.BN_G2 if g2 else C.BN_G1
    b = F.out(P)
    return b if le else C.be_to_le(b, 64 if g2 else 32)


def bn_point(rng, g2, le):
    F = C.BN_G2 if g2 else C.BN_G1
    c = rng.random()
    if c < 0.65:
        k = rng.randrange(1, C.BR)
        P = bn_aff(PB.multiply(PB.G2 if g2 else PB.G1, k), g2)
        return bn_enc(P, g2, le)
    if c < 0.72:
        return bytes(2 * F.nb())
    if c < 0.8 and g2:                                    # on the curve, outside the subgroup
        while True:
            x = (rng.randrange(C.BP), rng.randrange(C.BP))
            y = F.sqrt(F.add(F.mul(F.mul(x, x), x), F.b))
            if y:
                return bn_enc((x, y), True, le)
    b = bytearray(bn_enc(bn_aff(PB.multiply(PB.G2 if g2 else PB.G1, rng.randrange(1, 99)), g2), g2, True))
    k = rng.random()
    if k < 0.3:
        b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)      # off the curve
    elif k < 0.5:
        b[-1] |= rng.choice([0x40, 0x80, 0xc0])               # flags
    elif k < 0.7:
        b[:32] = (C.BP + rng.randrange(5)).to_bytes(32, 'little')
    else:
        b = bytearray(rng.getrandbits(8) for _ in range(len(b)))
    b = bytes(b)
    return b if le else C.be_to_le(b, 64 if g2 else 32)


def bn_scalar(rng, le):
    k = rng.choice([rng.randrange(C.BR), rng.randrange(2**256), 0, 1, C.BR, C.BR - 1])
    return k.to_bytes(32, 'little' if le else 'big')


def bn_compress_input(rng, g2, le):
    F = C.BN_G2 if g2 else C.BN_G1
    full = bn_point(rng, g2, True)
    try:
        c = F.compress(full)
    except C.Fail:
        c = bytes(rng.getrandbits(8) for _ in range(F.nb()))
    if rng.random() < 0.2:
        c = bytearray(c); c[-1] ^= rng.choice([0x40, 0x80, 0xc0, 1]); c = bytes(c)
    return c if le else C.be_to_le(c, 64 if g2 else 32)


def mutate_len(rng, b):
    return b if rng.random() < 0.9 else b[:rng.randrange(len(b) + 1)] + bytes(rng.choice([0, 0, 1]))


def bn_pairing_identity(rng, le):
    a, b = rng.randrange(1, C.BR), rng.randrange(1, C.BR)
    P1 = bn_aff(PB.multiply(PB.G1, a), False); Q1 = bn_aff(PB.multiply(PB.G2, b), True)
    P2 = bn_aff(PB.neg(PB.multiply(PB.G1, a * b % C.BR)), False); Q2 = bn_aff(PB.G2, True)
    return bn_enc(P1, False, le) + bn_enc(Q1, True, le) + bn_enc(P2, False, le) + bn_enc(Q2, True, le)


from py_ecc import optimized_bls12_381 as PL


def bls_aff(P, g2):
    if PL.is_inf(P):
        return None
    x, y = PL.normalize(P)
    return ((x.coeffs[0] % C.QP, x.coeffs[1] % C.QP), (y.coeffs[0] % C.QP, y.coeffs[1] % C.QP)) if g2 else (int(x) % C.QP, int(y) % C.QP)


def bls_enc(P, g2, be):
    b = C.bls_out(bls_aff(P, g2), g2)
    return b if be else C.bls_from_be(b, g2)


def bls_point(rng, g2, be, _):
    W = C.BLS_G2 if g2 else C.BLS_G1
    c = rng.random()
    if c < 0.6:
        return bls_enc(PL.multiply(PL.G2 if g2 else PL.G1, rng.randrange(1, C.QR)), g2, be)
    if c < 0.67:
        b = C.bls_out(None, g2)
    elif c < 0.8:                                      # on the curve, outside the subgroup
        while True:
            x = (rng.randrange(C.QP), rng.randrange(C.QP)) if g2 else rng.randrange(C.QP)
            y = W.sqrt(W.add(W.mul(W.mul(x, x), x), W.b))
            if y is not None:
                b = C.bls_out((x, y), g2); break
    else:
        b = bytearray(C.bls_out(bls_aff(PL.multiply(PL.G2 if g2 else PL.G1, rng.randrange(1, 50)), g2), g2))
        k = rng.random()
        if k < 0.35:
            b[rng.randrange(1, len(b))] ^= 1 << rng.randrange(8)
        elif k < 0.6:
            b[0] |= rng.choice([0x80, 0x40, 0x20])
        elif k < 0.75:
            b = bytearray(len(b)); b[0] = 0x40; b[rng.randrange(1, len(b))] = 1
        else:
            b[:48] = (C.QP + 1).to_bytes(48, 'big')
        b = bytes(b)
    return b if be else C.bls_from_be(b, g2)


def bls_comp(rng, g2, be):
    W = C.BLS_G2 if g2 else C.BLS_G1
    n = 96 if g2 else 48
    c = rng.random()
    if c < 0.08:
        b = bytes([0xc0]) + bytes(n - 1)
    else:
        if c < 0.7:
            P = bls_aff(PL.multiply(PL.G2 if g2 else PL.G1, rng.randrange(1, C.QR)), g2)
        else:
            while True:
                x = (rng.randrange(C.QP), rng.randrange(C.QP)) if g2 else rng.randrange(C.QP)
                y = W.sqrt(W.add(W.mul(W.mul(x, x), x), W.b))
                if y is not None or rng.random() < 0.3:
                    P = (x, y if y is not None else W.zero); break
        x, y = P
        b = bytearray(C._enc(x, g2))
        b[0] |= 0x80
        if W.key(y) > W.key(W.neg(y)):
            b[0] |= 0x20
        if rng.random() < 0.1:
            b[0] ^= rng.choice([0x80, 0x40, 0x20])
        b = bytes(b)
    return b if be else C.bls_from_be(b, g2)


def gen(rng, families):
    data = bytearray(4096)
    slots = []                                 # 32-byte inputs at [1024, 3072): 64 slots
    def put(bs):
        o = 1024 + 32 * len(slots)
        data[o:o + len(bs)] = bs
        slots.extend([None] * ((len(bs) + 31) // 32))
        return O + o
    ops = []
    bad = lambda: rng.choice([0, 0x100000000 + 8, O + 4096 - 8, O + 4096 + 10240 + 100, 0x500000000])
    out = lambda: (O + 3072 + 64 * len(ops)) if rng.random() < 0.93 else bad()
    for _ in range(rng.randint(1, 3)):
        fam = rng.choice(families)
        if fam == 'c25519':
            cid = rng.choice([0, 0, 1, 1, 2, 4, 9, 5, 6, 0x85])
            kind = rng.choice(['v', 'g', 'g', 'msm'])
            pts = [ed_points(rng)[0 if cid != 1 else 1] for _ in range(4)]
            if kind == 'v':
                ops.append(op(28, cid, put(pts[0]) if rng.random() < 0.95 else bad()))
            elif kind == 'g':
                g = rng.choice([0, 1, 2, 2, 3])
                left = put(scalar_bytes(rng)) if g == 2 else put(pts[0])
                ops.append(op(29, cid, g, left if rng.random() < 0.95 else bad(), put(pts[1]), out()))
            else:
                n = rng.choice([0, 1, 2, 3, 600])
                m = min(n, 3)
                sa = put(b''.join(scalar_bytes(rng) for _ in range(m))) if m else O + 1024
                pa = put(b''.join(ed_points(rng)[0 if cid != 1 else 1] for _ in range(m))) if m else O + 1024
                ops.append(op(30, cid, sa, pa, n, out()))
        elif fam == 'bn':
            le = 0x80 if rng.random() < 0.5 else 0
            o = rng.choice([0, 0, 2, 2, 4, 6, 3, 3, 1, 5, 7, 'c0', 'c1', 'c2', 'c3', 'c4'])
            if isinstance(o, str):
                cop = int(o[1])
                g2 = cop in (2, 3)
                pt = bn_point(rng, g2, le)
                if cop in (1, 3):
                    pt = bn_compress_input(rng, g2, le)
                inp = mutate_len(rng, pt)
                ops.append(op(36, cop | le, put(inp) if inp else O + 1024, len(inp) if rng.random() < 0.95 else rng.choice([0, 2**40]), out()))
            else:
                if o in (0, 1):
                    inp = bn_point(rng, False, le) + bn_point(rng, False, le)
                    if not le and rng.random() < 0.3:
                        inp = inp[:rng.choice([0, 1, 64, 100])]
                elif o == 2:
                    inp = bn_point(rng, False, le) + bn_scalar(rng, le)
                    if not le and rng.random() < 0.2:
                        inp = inp[:rng.choice([0, 64, 80])]
                elif o in (4, 5):
                    inp = bn_point(rng, True, le) + bn_point(rng, True, le)
                elif o == 6:
                    inp = bn_point(rng, True, le) + bn_scalar(rng, le)
                elif o == 3:
                    inp = b''.join(bn_point(rng, False, le) + bn_point(rng, True, le) for _ in range(rng.choice([0, 1, 2, 2])))
                    if rng.random() < 0.4:
                        inp = bn_pairing_identity(rng, le)
                else:
                    inp = bytes(64)
                if rng.random() < 0.08:
                    inp = inp + b'\x00'
                ops.append(op(35, o | le, put(inp) if inp else O + 1024, len(inp), out()))
        elif fam == 'poseidon':
            n = rng.choice([0, 1, 2, 3, 5, 12, 13])
            le = rng.choice([0, 1, 1, 0, 2]) if rng.random() < 0.97 else 0
            prm = 0 if rng.random() < 0.97 else 1
            vals = []
            for _ in range(min(n, 12)):
                x = rng.choice([rng.randrange(C.BR), rng.randrange(C.BR), C.BR, 2**256 - 1, 0])
                v = x.to_bytes(32, 'little' if le == 1 else 'big')
                if rng.random() < 0.05:
                    v = v[:rng.choice([0, 31])]
                vals.append(v)
            desc = b''
            for v in vals:
                desc += struct.pack('<QQ', put(v) if v else O + 1024, len(v))
            da = put(desc) if desc else O + 1024
            if rng.random() < 0.05:
                da += 4
            ops.append(op(34, prm, le, da, n, out()))
        elif fam == 'bls':
            be = 0x80 if rng.random() < 0.5 else 0
            g2 = rng.random() < 0.5
            cid = (6 if g2 else 5) | be
            kind = rng.choice(['v', 'add', 'sub', 'mul', 'mul', 'dec', 'dec', 'pair', 'pair', 'badop'])
            enc = lambda P, g: bls_point(rng, g, be, P)
            if kind == 'v':
                ops.append(op(28, cid, put(enc(None, g2))))
            elif kind in ('add', 'sub', 'badop'):
                g = {'add': 0, 'sub': 1, 'badop': 3}[kind]
                ops.append(op(29, cid, g, put(enc(None, g2)), put(enc(None, g2)), out()))
            elif kind == 'mul':
                k = rng.choice([rng.randrange(C.QR), 0, 1, C.QR, 2**256 - 1])
                ops.append(op(29, cid, 2, put(k.to_bytes(32, 'big' if be else 'little')), put(enc(None, g2)), out()))
            elif kind == 'dec':
                ops.append(op(31, cid if rng.random() < 0.95 else rng.choice([0, 4]), put(bls_comp(rng, g2, be)), out()))
            else:
                n = rng.choice([0, 1, 1, 2, 3, 9])
                g1s = b''.join(bls_point(rng, False, be, None) for _ in range(n))
                g2s = b''.join(bls_point(rng, True, be, None) for _ in range(n))
                if n == 2 and rng.random() < 0.5:
                    a_ = rng.randrange(1, C.QR)
                    g1s = bls_enc(PL.multiply(PL.G1, a_), False, be) + bls_enc(PL.neg(PL.G1), False, be)
                    g2s = bls_enc(PL.G2, True, be) + bls_enc(PL.multiply(PL.G2, a_), True, be)
                pc = (4 | be) if rng.random() < 0.95 else rng.choice([5, 0])
                ops.append(op(32, pc, n, put(g1s) if g1s else O + 1024, put(g2s) if g2s else O + 1024, O + 3072 + 64 * len(ops) - 2048 if rng.random() < 0.5 else O + 3200))
        elif fam == 'secp':
            h, sig, recid = secp_case(rng)
            ops.append(op(33, put(h) if rng.random() < 0.97 else bad(), recid, put(sig), out()))
    acc = [{'data': bytes(data), 'owner': 'program', 'writable': True}]
    return b''.join(ops), acc


EX_ELF = 'programs/exerciser/target/deploy/exerciser.so'


def ref_run(img, acc, data, cu):
    inp, meta = solana.serialize(acc, data)
    r = sref.run_program(img, inp, cu)
    accts, err = (solana.deserialize(r['input'], meta, acc) if r['status'] == 'ok' and r['r0'] == 0 else (None, None))
    return solana.outcome(r['status'], r['r0'], r['cu'], r['logs'], accts, err, limit=cu)


def weights_run(img, acc, data, cu):
    from svllm import svm
    inp, meta = solana.serialize(acc, data)
    r = svm.run_program(img, inp, cu, program_id=solana.PROGRAM_ID, program_id_b58=sref.b58(solana.PROGRAM_ID))
    accts, err = (solana.deserialize(r['input'], meta, acc) if r['status'] == 'ok' and r['r0'] == 0 else (None, None))
    return solana.outcome(r['status'], r['r0'], r['cu'], r['logs'], accts, err, limit=cu)


if __name__ == '__main__':
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    families = sys.argv[3].split(',') if len(sys.argv) > 3 else ['c25519', 'secp']
    weights = len(sys.argv) > 4 and sys.argv[4] == 'weights'
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        data, acc = gen(rng, families)
        cases.append((EX_ELF, acc, data, 200000 if rng.random() < 0.9 else rng.choice([300, 3000, 26000])))
    truth = solana.truth_many(cases)
    img = solana.image(EX_ELF)
    ok, bad, ex, t0 = 0, collections.Counter(), [], time.time()
    for (e, acc, data, cu), t in zip(cases, truth):
        o = (weights_run if weights else ref_run)(img, acc, data, cu)
        s = solana.same(o, t); ok += s
        if not s:
            bad[(o.get('err'), t.get('err'), o['cu'] == t['cu'])] += 1
            if len(ex) < 4:
                ex.append((data[:1].hex(), [data[i] for i in range(0, len(data), 41)], o['cu'], t['cu'], o.get('err'), t.get('err'), (t.get('raw_logs') or [''])[-1][-80:]))
    print(('weights' if weights else 'reference') + ' vs litesvm', ok, '/', n, f'({time.time() - t0:.0f}s)')
    print(collections.Counter('ok' if t['ok'] else t['err'] for t in truth).most_common(6))
    for k, v in bad.most_common(8): print(v, k)
    for x in ex: print(x)
