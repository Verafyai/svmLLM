"""The curve syscalls' mathematics as circuits: the same steps as curves.py (the reference), with
every field operation done by fastbig's table layers on 4-bit limbs. Inputs and outputs are bytes
as one-hot [2, 16] pairs, the machine's memory format.

Control flow (which branch of an addition formula, whether a decode failed) is decided by reading
circuit outputs, the way the rest of the syscall microprograms decide a translation fault or a
bump-seed search. The formulas are chosen to need few such decisions: Edwards points use complete
extended-coordinate addition (no branches at all); short Weierstrass points use Jacobian
coordinates with the doubling / infinity cases decided once per addition.
"""
from . import fastbig as F
from . import curves as R


class Fp:
    """A prime field's operations on limb tensors, plus the constants a curve needs."""
    def __init__(self, p):
        self.F = F.Field(p)
        self.p = p
        f = self.F
        self.mul, self.sq, self.add, self.sub, self.neg = f.mul, f.sq, f.add, f.sub, f.neg
        self.zero, self.one = f.n(0), f.n(1)

    def c(self, v):
        return self.F.n(v)

    def pow(self, a, e):
        return self.F.pow(a, e)

    def inv(self, a):
        return self.F.inv(a)

    def eq(self, a, b):
        return F.eq(a, b)

    def is_zero(self, a):
        return F.is_zero(a)

    def odd(self, a):
        return F.bit_of(a, 0)

    def from_int_limbs(self, limbs):
        """limbs < 16^k, possibly >= p -> canonical."""
        return self.F.canonical(limbs)


class Fp2:
    """Fp[u] / (u^2 + 1) over a circuit field: pairs of limb tensors."""
    def __init__(self, fp):
        self.b = fp
        self.zero, self.one = (fp.zero, fp.zero), (fp.one, fp.zero)

    def c(self, v):
        return (self.b.c(v[0]), self.b.c(v[1]))

    def add(self, x, y): return (self.b.add(x[0], y[0]), self.b.add(x[1], y[1]))
    def sub(self, x, y): return (self.b.sub(x[0], y[0]), self.b.sub(x[1], y[1]))
    def neg(self, x): return (self.b.neg(x[0]), self.b.neg(x[1]))

    def mul(self, x, y):                       # Karatsuba: three base multiplies
        b = self.b
        t0, t1 = b.mul(x[0], y[0]), b.mul(x[1], y[1])
        t2 = b.mul(b.add(x[0], x[1]), b.add(y[0], y[1]))
        return (b.sub(t0, t1), b.sub(b.sub(t2, t0), t1))

    def sq(self, x):
        return self.mul(x, x)

    def inv(self, x):
        b = self.b
        n = b.inv(b.add(b.sq(x[0]), b.sq(x[1])))
        return (b.mul(x[0], n), b.neg(b.mul(x[1], n)))

    def pow(self, x, e):
        r = None
        for bit in bin(e)[2:]:
            if r is not None:
                r = self.sq(r)
            if bit == '1':
                r = x if r is None else self.mul(r, x)
        return r if r is not None else self.one

    def eq(self, x, y):
        return F.eq(x[0], y[0]) and F.eq(x[1], y[1])

    def is_zero(self, x):
        return F.is_zero(x[0]) and F.is_zero(x[1])

    def sqrt(self, a):
        """The Fp2 square root used by curves.py (p = 3 mod 4), or None."""
        p = self.b.p
        if self.is_zero(a):
            return self.zero
        a1 = self.pow(a, (p - 3) // 4)
        alpha = self.mul(self.mul(a1, a1), a)
        minus1 = self.c((p - 1, 0))
        if self.eq(self.mul(self.pow(alpha, p), alpha), minus1):
            return None
        x0 = self.mul(a1, a)
        if self.eq(alpha, minus1):
            x = self.mul(self.c((0, 1)), x0)
        else:
            x = self.mul(self.pow(self.add(self.one, alpha), (p - 1) // 2), x0)
        return x if self.eq(self.sq(x), a) else None


def _lt_limbs(a, b):
    return bool(F.sub(a, b)[1])


def _lex_gt(x, y):
    """(c1, c0) lexicographic x > y, each comparison a borrow circuit."""
    if not F.eq(x[1], y[1]):
        return _lt_limbs(y[1], x[1])
    return _lt_limbs(y[0], x[0])


def fp_gt(x, y):
    return _lt_limbs(y, x)


# ------------------------------------------------------------------ Ed25519 / Ristretto
E = Fp(R.P)
D = E.c(R.D)
D2 = E.c(2 * R.D)
SQRT_M1 = E.c(R.SQRT_M1)
INVSQRT_A_MINUS_D = E.c(R.INVSQRT_A_MINUS_D)


def ed_ext(x, y):
    return (x, y, E.one, E.mul(x, y))


ED_ID = (E.zero, E.one, E.one, E.zero)


def ed_add(P, Q):
    """add-2008-hwcd-3: complete for a = -1 with non-square d."""
    X1, Y1, Z1, T1 = P
    X2, Y2, Z2, T2 = Q
    A = E.mul(E.sub(Y1, X1), E.sub(Y2, X2))
    B = E.mul(E.add(Y1, X1), E.add(Y2, X2))
    C = E.mul(E.mul(T1, D2), T2)
    Dd = E.mul(E.add(Z1, Z1), Z2)
    Ee, Ff, Gg, Hh = E.sub(B, A), E.sub(Dd, C), E.add(Dd, C), E.add(B, A)
    return (E.mul(Ee, Ff), E.mul(Gg, Hh), E.mul(Ff, Gg), E.mul(Ee, Hh))


def ed_neg(P):
    return (E.neg(P[0]), P[1], P[2], E.neg(P[3]))


def ed_mul(k_limbs, P, bits=253):
    r = ED_ID
    for i in range(bits - 1, -1, -1):
        r = ed_add(r, r)
        if F.bit_of(k_limbs, i):
            r = ed_add(r, P)
    return r


def sqrt_ratio_m1(u, v):
    v3 = E.mul(E.sq(v), v)
    v7 = E.mul(E.sq(v3), v)
    r = E.mul(E.mul(u, v3), E.pow(E.mul(u, v7), (R.P - 5) // 8))
    check = E.mul(v, E.sq(r))
    correct = E.eq(check, u)
    flipped = E.eq(check, E.neg(u))
    flipped_i = E.eq(check, E.mul(E.neg(u), SQRT_M1))
    if flipped or flipped_i:
        r = E.mul(r, SQRT_M1)
    if E.odd(r):
        r = E.neg(r)
    return correct or flipped, r


def ed_decompress(bs):
    limbs = F.from_bytes(bs).clone()
    sign = F.bit_of(limbs, 255)
    limbs[63] = F.AND7.ix(limbs[63])               # the low 255 bits
    y = E.from_int_limbs(limbs)
    yy = E.sq(y)
    u, v = E.sub(yy, E.one), E.add(E.mul(D, yy), E.one)
    ok, x = sqrt_ratio_m1(u, v)
    if not ok:
        return None
    if sign:
        x = E.neg(x)
    return ed_ext(x, y)


def ed_compress(P):
    X, Y, Z, _ = P
    zi = E.inv(Z)
    x, y = E.mul(X, zi), E.mul(Y, zi)
    out = y.clone()
    if E.odd(x):
        out[63] = F.OR8.ix(out[63])
    return F.to_bytes(out, 32)


def ristretto_decode(bs):
    s = F.from_bytes(bs)
    if not E.F.lt(s, R.P) or F.bit_of(s, 0):
        return None
    ss = E.sq(s)
    u1, u2 = E.sub(E.one, ss), E.add(E.one, ss)
    u2s = E.sq(u2)
    v = E.sub(E.neg(E.mul(D, E.sq(u1))), u2s)
    ok, isr = sqrt_ratio_m1(E.one, E.mul(v, u2s))
    dx = E.mul(isr, u2)
    dy = E.mul(E.mul(isr, dx), v)
    x = E.mul(E.add(s, s), dx)
    if E.odd(x):
        x = E.neg(x)
    y = E.mul(u1, dy)
    t = E.mul(x, y)
    if not ok or E.odd(t) or E.is_zero(y):
        return None
    return (x, y, E.one, t)


def ristretto_encode(P):
    x0, y0, z0, t0 = P
    u1 = E.mul(E.add(z0, y0), E.sub(z0, y0))
    u2 = E.mul(x0, y0)
    _, isr = sqrt_ratio_m1(E.one, E.mul(u1, E.sq(u2)))
    den1, den2 = E.mul(isr, u1), E.mul(isr, u2)
    z_inv = E.mul(E.mul(den1, den2), t0)
    rotate = E.odd(E.mul(t0, z_inv))
    if rotate:
        x, y = E.mul(y0, SQRT_M1), E.mul(x0, SQRT_M1)
        den_inv = E.mul(den1, INVSQRT_A_MINUS_D)
    else:
        x, y, den_inv = x0, y0, den2
    if E.odd(E.mul(x, z_inv)):
        y = E.neg(y)
    s = E.mul(den_inv, E.sub(z0, y))
    if E.odd(s):
        s = E.neg(s)
    return F.to_bytes(s, 32)


CODEC = {0: (ed_decompress, ed_compress), 1: (ristretto_decode, ristretto_encode)}


def scalar25519(bs):
    k = F.from_bytes(bs)
    return k if E.F.lt(k, R.L) else None


def c25519_validate(cid, bs):
    return CODEC[cid][0](bs) is not None


def c25519_group(cid, op, left, right):
    dec, enc = CODEC[cid]
    if op in (0, 1):
        a, b = dec(left), dec(right)
        if a is None or b is None:
            return None
        return enc(ed_add(a, b if op == 0 else ed_neg(b)))
    k, a = scalar25519(left), dec(right)
    if k is None or a is None:
        return None
    return enc(ed_mul(k, a))


def c25519_msm(cid, scalars, points):
    dec, enc = CODEC[cid]
    ks = [scalar25519(s) for s in scalars]
    ps = [dec(p) for p in points]
    if any(k is None for k in ks) or any(p is None for p in ps):
        return None
    r = ED_ID
    for k, a in zip(ks, ps):
        r = ed_add(r, ed_mul(k, a))
    return enc(r)


# ------------------------------------------------------------------ short Weierstrass, a = 0, Jacobian
class Jac:
    def __init__(self, K, b):
        self.K, self.b = K, b

    def dbl(self, P):
        if P is None:
            return None
        K = self.K
        X, Y, Z = P
        if K.is_zero(Y):
            return None
        A, B = K.sq(X), K.sq(Y)
        C = K.sq(B)
        t = K.sub(K.sub(K.sq(K.add(X, B)), A), C)
        Dd = K.add(t, t)
        Ee = K.add(K.add(A, A), A)
        Ff = K.sq(Ee)
        X3 = K.sub(Ff, K.add(Dd, Dd))
        C8 = K.add(C, C); C8 = K.add(C8, C8); C8 = K.add(C8, C8)
        Y3 = K.sub(K.mul(Ee, K.sub(Dd, X3)), C8)
        YZ = K.mul(Y, Z)
        return (X3, Y3, K.add(YZ, YZ))

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        K = self.K
        X1, Y1, Z1 = P
        X2, Y2, Z2 = Q
        Z1Z1, Z2Z2 = K.sq(Z1), K.sq(Z2)
        U1, U2 = K.mul(X1, Z2Z2), K.mul(X2, Z1Z1)
        S1, S2 = K.mul(K.mul(Y1, Z2), Z2Z2), K.mul(K.mul(Y2, Z1), Z1Z1)
        H, r = K.sub(U2, U1), K.sub(S2, S1)
        if K.is_zero(H):
            return self.dbl(P) if K.is_zero(r) else None
        HH = K.sq(H)
        HHH = K.mul(H, HH)
        V = K.mul(U1, HH)
        X3 = K.sub(K.sub(K.sq(r), HHH), K.add(V, V))
        Y3 = K.sub(K.mul(r, K.sub(V, X3)), K.mul(S1, HHH))
        return (X3, Y3, K.mul(K.mul(Z1, Z2), H))

    def mul(self, k_limbs, P, bits):
        r = None
        for i in range(bits - 1, -1, -1):
            r = self.dbl(r)
            if F.bit_of(k_limbs, i):
                r = self.add(r, P)
        return r

    def affine(self, P):
        if P is None:
            return None
        K = self.K
        zi = K.inv(P[2])
        zi2 = K.sq(zi)
        return (K.mul(P[0], zi2), K.mul(K.mul(P[1], zi2), zi))

    def jac(self, A):
        return None if A is None else (A[0], A[1], self.K.one)

    def on_curve(self, A):
        K = self.K
        x, y = A
        return K.eq(K.sq(y), K.add(K.mul(K.sq(x), x), self.b))


# ------------------------------------------------------------------ secp256k1 recovery
KP = Fp(R.KP)
KN = Fp(R.KN)
SECP = Jac(KP, KP.c(7))
KG = (KP.c(R.KG[0]), KP.c(R.KG[1]), KP.one)


def secp256k1_recover(h_bytes, recid, sig_bytes):
    """h, sig: one-hot byte lists (big-endian); recid an int. -> (code, 64 one-hot bytes or None)."""
    if recid > 255 or recid >= 4:
        return R.SECP_INVALID_RECID, None
    r = F.from_bytes(sig_bytes[:32], big=True)
    s = F.from_bytes(sig_bytes[32:], big=True)
    if not KN.F.lt(r, R.KN) or not KN.F.lt(s, R.KN):
        return R.SECP_INVALID_SIG, None
    if F.is_zero(r) or F.is_zero(s):
        return R.SECP_INVALID_SIG, None
    x = r
    if recid & 2:
        if not KP.F.lt(x, R.KP - R.KN):
            return R.SECP_INVALID_SIG, None
        x, _ = F.add(x, F.num(R.KN, 64))
    y2 = KP.add(KP.mul(KP.sq(x), x), KP.c(7))
    y = KP.pow(y2, (R.KP + 1) // 4)
    if not KP.eq(KP.sq(y), y2):
        return R.SECP_INVALID_SIG, None
    if KP.odd(y) != (recid & 1):
        y = KP.neg(y)
    e = KN.from_int_limbs(F.from_bytes(h_bytes, big=True))
    rn = KN.inv(r)
    u1 = KN.neg(KN.mul(rn, e))
    u2 = KN.mul(rn, s)
    Q = SECP.add(SECP.mul(u2, (x, y, KP.one), 256), SECP.mul(u1, KG, 256))
    if Q is None:
        return R.SECP_INVALID_SIG, None
    qx, qy = SECP.affine(Q)
    return 0, F.to_bytes(qx, 32, big=True) + F.to_bytes(qy, 32, big=True)


# ------------------------------------------------------------------ helpers shared by BN254 and BLS12-381
class Fail(Exception):
    pass


def limbs_of(bs, big=False):
    return F.from_bytes(bs, big)


def bytes_of(A, n, big=False):
    return F.to_bytes(A, n, big)


class CW:
    """Circuit counterpart of curves.BnField / curves.Wc for one group: K is Fp or Fp2, nb the
    encoded size of one coordinate."""
    def __init__(self, K, b, g2, nb):
        self.K, self.g2, self.nb = K, g2, nb
        self.J = Jac(K, b)
        self.b = b

    def key_gt(self, a, c):
        return _lex_gt(a, c) if self.g2 else fp_gt(a, c)

    def on_curve(self, A):
        return self.J.on_curve(A)

    def add(self, A, B):
        return self.J.affine(self.J.add(self.J.jac(A), self.J.jac(B)))

    def mul(self, k, A, bits):
        return self.J.affine(self.J.mul(k, self.J.jac(A), bits))

    def in_subgroup(self, A, r):
        return self.J.mul(F.num(r, 64), self.J.jac(A), r.bit_length()) is None

    def neg(self, A):
        return None if A is None else (A[0], self.K.neg(A[1]))

    def sqrt(self, a):
        K = self.K
        if self.g2:
            return K.sqrt(a)
        r = K.pow(a, (K.p + 1) // 4)
        return r if K.eq(K.sq(r), a) else None


# ------------------------------------------------------------------ BN254
BNF = Fp(R.BP)
BNF2 = Fp2(BNF)
BN1 = CW(BNF, BNF.c(3), False, 32)
BN2 = CW(BNF2, BNF2.c(R.f2mul((3, 0), R.f2inv((9, 1)))), True, 64)


def _zero_bytes(bs):
    return all(bool(b[0][0]) and bool(b[1][0]) for b in bs)


def bn_fe(bs, g2, flags=False):
    """Little-endian field element(s); the flags are bits 254/255 of the last 32-byte limb group."""
    def fp(b32, fl):
        L = limbs_of(b32).clone()
        f = None
        if fl:
            hi = (int(L[63]) >> 2) & 3            # the top nibble's upper two bits, read off its one-hot
            if hi == 3:
                raise Fail()
            f = {0: None, 1: 'inf', 2: 'neg'}[hi]
            L[63] = F.AND3.ix(L[63])
        if not BNF.F.lt(L, R.BP):
            raise Fail()
        return L, f
    if not g2:
        return fp(bs, flags)
    c0, _ = fp(bs[:32], False)
    c1, f = fp(bs[32:], flags)
    return (c0, c1), f


def bn_enc(v, g2):
    return bytes_of(v[0], 32) + bytes_of(v[1], 32) if g2 else bytes_of(v, 32)


def bn_parse(bs, g2, subgroup, check_curve=True):
    W = BN2 if g2 else BN1
    if _zero_bytes(bs):
        return None
    n = W.nb
    x, _ = bn_fe(bs[:n], g2)
    y, f = bn_fe(bs[n:], g2, True)
    if f == 'inf':
        return None
    P = (x, y)
    if check_curve and not W.on_curve(P):
        raise Fail()
    if subgroup and g2 and not W.in_subgroup(P, R.BR):
        raise Fail()
    return P


def bn_out(P, g2):
    W = BN2 if g2 else BN1
    if P is None:
        return [F.to_onehot(F.zeros(2))] * (2 * W.nb)
    return bn_enc(P[0], g2) + bn_enc(P[1], g2)


def be_le(bs, chunk):
    return [b for i in range(0, len(bs), chunk) for b in reversed(bs[i:i + chunk])]


def bn_group_op(op, inp):
    """inp: one-hot bytes. Mirrors curves.bn_group_op except the pairing, which is not here."""
    le = bool(op & 0x80); o = op & 0x7f
    Z = F.to_onehot(F.zeros(2))
    try:
        if o == 0:
            if len(inp) > 128 or (le and len(inp) != 128):
                raise Fail()
            inp = inp + [Z] * (128 - len(inp))
            cv = (lambda b: b) if le else (lambda b: be_le(b, 32))
            P, Qp = bn_parse(cv(inp[:64]), False, True), bn_parse(cv(inp[64:]), False, True)
            r = bn_out(BN1.add(P, Qp), False)
            return 0, r if le else be_le(r, 32)
        if o == 2:
            if len(inp) > 96 or (le and len(inp) != 96):
                raise Fail()
            inp = inp + [Z] * (96 - len(inp))
            P = bn_parse(inp[:64] if le else be_le(inp[:64], 32), False, True)
            k = limbs_of(inp[64:], big=not le)
            r = bn_out(BN1.mul(k, P, 256), False)
            return 0, r if le else be_le(r, 32)
        if o == 4:
            if len(inp) != 256:
                raise Fail()
            cv = (lambda b: b) if le else (lambda b: be_le(b, 64))
            P, Qp = bn_parse(cv(inp[:128]), True, False), bn_parse(cv(inp[128:]), True, False)
            r = bn_out(BN2.add(P, Qp), True)
            return 0, r if le else be_le(r, 64)
        if o == 6:
            if len(inp) != 160:
                raise Fail()
            P = bn_parse(inp[:128] if le else be_le(inp[:128], 64), True, True)
            k = limbs_of(inp[128:], big=not le)
            r = bn_out(BN2.mul(k, P, 256), True)
            return 0, r if le else be_le(r, 64)
    except Fail:
        return 1, None
    raise NotImplementedError('alt_bn128 pairing is not in the weights')


def bn_compression(op, inp):
    le = bool(op & 0x80)
    g2, kind = R.BN_COMPRESSION[op & 0x7f]
    W = BN2 if g2 else BN1
    K = W.K
    ch = 64 if g2 else 32
    need = (2 if kind == 'c' else 1) * W.nb
    if len(inp) != need:
        return 1, None
    bs = inp if le else be_le(inp, ch)
    Z = F.to_onehot(F.zeros(2))
    try:
        if kind == 'c':
            if _zero_bytes(bs):
                out = [Z] * W.nb
            else:
                n = W.nb
                x, _ = bn_fe(bs[:n], g2)
                y, f = bn_fe(bs[n:], g2, True)
                if f == 'inf':
                    out = [Z] * (n - 1) + [F.to_onehot(F.num(0x40, 2))]
                else:
                    out = bn_enc(x, g2)
                    if W.key_gt(y, K.neg(y)):
                        last = F.from_bytes([out[-1]]).clone()
                        last[1] = F.OR8.ix(last[1])                          # set bit 7
                        out = out[:-1] + F.to_bytes(last, 1)
        else:
            if _zero_bytes(bs):
                out = [Z] * (2 * W.nb)
            else:
                x, f = bn_fe(bs, g2, True)
                if f == 'inf':
                    out = [Z] * (2 * W.nb)
                else:
                    y = W.sqrt(K.add(K.mul(K.sq(x), x), W.b))
                    if y is None:
                        raise Fail()
                    ny = K.neg(y)
                    lo, hi = (ny, y) if W.key_gt(y, ny) else (y, ny)
                    out = bn_enc(x, g2) + bn_enc(hi if f == 'neg' else lo, g2)
    except Fail:
        return 1, None
    return 0, out if le else be_le(out, ch)


# ------------------------------------------------------------------ Poseidon
FR = Fp(R.BR)
_PC = {}


def poseidon_hashv(endian, vals):
    """vals: lists of one-hot bytes. -> (code, 32 one-hot bytes)."""
    if not 1 <= len(vals) <= 12:
        return 1, None
    xs = []
    for v in vals:
        if len(v) != 32:
            return 1, None
        L = limbs_of(v, big=not endian)
        if not FR.F.lt(L, R.BR):
            return 1, None
        xs.append(L)
    t = len(xs) + 1
    if t not in _PC:
        ark, mds, partial = R.poseidon_params(t)
        _PC[t] = ([FR.c(a) for a in ark], [[FR.c(m) for m in row] for row in mds], partial)
    ark, mds, partial = _PC[t]
    s = [FR.zero] + xs
    for rnd in range(8 + partial):
        s = [FR.add(v, ark[rnd * t + i]) for i, v in enumerate(s)]
        full = rnd < 4 or rnd >= 4 + partial
        s = [FR.mul(FR.sq(FR.sq(v)), v) if (full or i == 0) else v for i, v in enumerate(s)]
        new = []
        for row in mds:
            acc = FR.zero
            for m_, v in zip(row, s):
                acc = FR.add(acc, FR.mul(m_, v))
            new.append(acc)
        s = new
    return 0, bytes_of(s[0], 32, big=not endian)


# ------------------------------------------------------------------ BLS12-381 (all but the pairing)
QF = Fp(R.QP)
QF2 = Fp2(QF)
QRF = F.Field(R.QR)
BL1 = CW(QF, QF.c(4), False, 48)
BL2 = CW(QF2, QF2.c((4, 4)), True, 96)


def _byte_val(b):
    return int(b[0].argmax()) | (int(b[1].argmax()) << 4)


def _fq_be(bs, g2):
    if not g2:
        L = limbs_of(bs, big=True)
        if not QF.F.lt(L, R.QP):
            raise Fail()
        return L
    return (_fq_be(bs[48:], False), _fq_be(bs[:48], False))


def bls_uncompressed(bs, g2, subgroup):
    W = BL2 if g2 else BL1
    n = len(bs) // 2
    b0 = _byte_val(bs[0])                          # the flag bits, read off byte 0's one-hots
    if b0 & 0xa0:
        raise Fail()
    if b0 & 0x40:
        if b0 & 0x3f == 0 and _zero_bytes(bs[1:]):
            return None
        raise Fail()
    P = (_fq_be(bs[:n], g2), _fq_be(bs[n:], g2))
    if not W.on_curve(P):
        raise Fail()
    if subgroup and not W.in_subgroup(P, R.QR):
        raise Fail()
    return P


def bls_out(P, g2):
    n = 96 if g2 else 48
    Z = F.to_onehot(F.zeros(2))
    if P is None:
        return [F.to_onehot(F.num(0x40, 2))] + [Z] * (2 * n - 1)
    enc = (lambda v: bytes_of(v[1], 48, True) + bytes_of(v[0], 48, True)) if g2 else (lambda v: bytes_of(v, 48, True))
    return enc(P[0]) + enc(P[1])


def bls_compressed(bs, g2):
    W = BL2 if g2 else BL1
    K = W.K
    b0 = _byte_val(bs[0])
    if not b0 & 0x80:
        raise Fail()
    if b0 & 0x40:
        if b0 & 0x3f == 0 and _zero_bytes(bs[1:]):
            return None
        raise Fail()
    big = bool(b0 & 0x20)
    raw = [F.to_onehot(F.num(b0 & 0x1f, 2))] + list(bs[1:])
    x = _fq_be(raw, g2)
    y = W.sqrt(K.add(K.mul(K.sq(x), x), W.b))
    if y is None:
        raise Fail()
    ny = K.neg(y)
    lo, hi = (ny, y) if W.key_gt(y, ny) else (y, ny)
    P = (x, hi if big else lo)
    if not W.in_subgroup(P, R.QR):
        raise Fail()
    return P


def bls_perm(bs, g2):
    """The syscall's LE layout <-> zcash BE (an involution): reverse each 48-byte limb, swap c0/c1."""
    b = [x for i in range(0, len(bs), 48) for x in reversed(bs[i:i + 48])]
    if g2:
        b = [x for i in range(0, len(b), 96) for x in (b[i + 48:i + 96] + b[i:i + 48])]
    return b


def bls_syscall(kind, cid, args):
    le = not cid & 0x80
    g2 = (cid & 0x7f) == 6
    W = BL2 if g2 else BL1
    cv = (lambda x, g: bls_perm(x, g)) if le else (lambda x, g: x)
    try:
        if kind == 'validate':
            bls_uncompressed(cv(args[0], g2), g2, True)
            return 0, None
        if kind in ('add', 'sub'):
            P = bls_uncompressed(cv(args[0], g2), g2, False)
            Qp = bls_uncompressed(cv(args[1], g2), g2, False)
            Rr = W.add(P, Qp if kind == 'add' else W.neg(Qp))
            return 0, cv(bls_out(Rr, g2), g2)
        if kind == 'mul':
            k = limbs_of(args[0], big=not le)
            P = bls_uncompressed(cv(args[1], g2), g2, True)
            if not QRF.lt(k, R.QR):
                raise Fail()
            return 0, cv(bls_out(W.mul(k, P, 255), g2), g2)
        if kind == 'decompress':
            return 0, cv(bls_out(bls_compressed(cv(args[0], g2), g2), g2), g2)
    except Fail:
        return 1, None
    raise NotImplementedError('BLS12-381 pairing is not in the weights')
