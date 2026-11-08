"""Reference elliptic-curve arithmetic for the curve syscalls, in plain Python integers, written to
the semantics of the libraries Agave 4.3.2 calls (curve25519-dalek 4, libsecp256k1). This is the
reference side; the weights side builds the same operations from the field circuits in bignum.py.
"""

# ------------------------------------------------------------------ curve25519 (dalek semantics)
P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493           # prime subgroup order
D = (-121665 * pow(121666, P - 2, P)) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)
INVSQRT_A_MINUS_D = 54469307008909316920995813868745141605393597292927456921205312896311721017578
ED_ID = (0, 1)


def inv(x, p=P):
    return pow(x, p - 2, p)


def neg_(x):                     # IS_NEGATIVE: the low bit of the canonical encoding
    return (x % P) & 1


def ed_add(a, b):
    """Complete twisted Edwards addition (a = -1), affine."""
    (x1, y1), (x2, y2) = a, b
    t = D * x1 * x2 * y1 * y2 % P
    return ((x1 * y2 + y1 * x2) * inv(1 + t) % P, (y1 * y2 + x1 * x2) * inv(1 - t) % P)


def ed_neg(a):
    return ((-a[0]) % P, a[1])


def ed_mul(k, a):
    r = ED_ID
    for bit in bin(k)[2:] if k else '':
        r = ed_add(r, r)
        if bit == '1':
            r = ed_add(r, a)
    return r


def ed_decompress(bs):
    """CompressedEdwardsY::decompress: y from the low 255 bits (not required to be reduced), x the
    root of (y^2 - 1) / (d y^2 + 1) with the sign taken from the top bit; None if no root exists."""
    n = int.from_bytes(bs, 'little')
    y = (n & (2**255 - 1)) % P
    u, v = (y * y - 1) % P, (D * y * y + 1) % P
    ok, x = sqrt_ratio_m1(u, v)
    if not ok:
        return None
    if n >> 255:
        x = (-x) % P
    return (x, y)


def ed_compress(a):
    x, y = a
    return (y | (neg_(x) << 255)).to_bytes(32, 'little')


def sqrt_ratio_m1(u, v):
    """RFC 9496 SQRT_RATIO_M1: (was_square, the nonnegative root of u/v, or of i*u/v)."""
    v3 = v * v % P * v % P
    v7 = v3 * v3 % P * v % P
    r = u * v3 % P * pow(u * v7 % P, (P - 5) // 8, P) % P
    check = v * r % P * r % P
    correct, flipped, flipped_i = check == u % P, check == (-u) % P, check == (-u * SQRT_M1) % P
    if flipped or flipped_i:
        r = r * SQRT_M1 % P
    if neg_(r):
        r = (-r) % P
    return correct or flipped, r


def ristretto_decode(bs):
    s = int.from_bytes(bs, 'little')
    if s >= P or neg_(s):
        return None
    ss = s * s % P
    u1, u2 = (1 - ss) % P, (1 + ss) % P
    u2s = u2 * u2 % P
    v = (-(D * u1 % P * u1) - u2s) % P
    ok, isr = sqrt_ratio_m1(1, v * u2s % P)
    dx = isr * u2 % P
    dy = isr * dx % P * v % P
    x = 2 * s * dx % P
    if neg_(x):
        x = (-x) % P
    y = u1 * dy % P
    if not ok or neg_(x * y % P) or y == 0:
        return None
    return (x, y)


def ristretto_encode(a):
    x0, y0 = a
    z0, t0 = 1, x0 * y0 % P
    u1 = (z0 + y0) * (z0 - y0) % P
    u2 = x0 * y0 % P
    _, isr = sqrt_ratio_m1(1, u1 * u2 % P * u2 % P)
    den1, den2 = isr * u1 % P, isr * u2 % P
    z_inv = den1 * den2 % P * t0 % P
    rotate = neg_(t0 * z_inv)
    x, y = (y0 * SQRT_M1 % P, x0 * SQRT_M1 % P) if rotate else (x0, y0)
    den_inv = den1 * INVSQRT_A_MINUS_D % P if rotate else den2
    if neg_(x * z_inv):
        y = (-y) % P
    s = den_inv * (z0 - y) % P
    if neg_(s):
        s = (-s) % P
    return s.to_bytes(32, 'little')


def scalar(bs):
    """Scalar::from_canonical_bytes: None unless < l (and so the top bit is clear)."""
    k = int.from_bytes(bs, 'little')
    return k if k < L else None


CURVES = {0: (ed_decompress, ed_compress), 1: (ristretto_decode, ristretto_encode)}


def c25519_validate(cid, bs):
    return CURVES[cid][0](bs) is not None


def c25519_group(cid, op, left, right):
    dec, enc = CURVES[cid]
    if op in (0, 1):
        a, b = dec(left), dec(right)
        if a is None or b is None:
            return None
        return enc(ed_add(a, b if op == 0 else ed_neg(b)))
    k, a = scalar(left), dec(right)
    if k is None or a is None:
        return None
    return enc(ed_mul(k, a))


def c25519_msm(cid, scalars, points):
    dec, enc = CURVES[cid]
    ks = [scalar(s) for s in scalars]
    ps = [dec(p) for p in points]
    if None in ks or None in ps:
        return None
    r = ED_ID
    for k, a in zip(ks, ps):
        r = ed_add(r, ed_mul(k, a))
    return enc(r)


# ------------------------------------------------------------------ secp256k1 recovery (libsecp256k1)
KP = 2**256 - 2**32 - 977
KN = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
KG = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def k_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    (x1, y1), (x2, y2) = a, b
    if x1 == x2:
        if (y1 + y2) % KP == 0:
            return None
        m = 3 * x1 * x1 * inv(2 * y1, KP) % KP
    else:
        m = (y2 - y1) * inv(x2 - x1, KP) % KP
    x3 = (m * m - x1 - x2) % KP
    return (x3, (m * (x1 - x3) - y1) % KP)


def k_mul(k, a):
    r = None
    for bit in bin(k)[2:] if k else '':
        r = k_add(r, r)
        if bit == '1':
            r = k_add(r, a)
    return r


SECP_INVALID_HASH, SECP_INVALID_RECID, SECP_INVALID_SIG = 1, 2, 3


def secp256k1_recover(h, recid, sig):
    """-> (0, 64-byte x || y) or (error code, None). Signature::parse_standard rejects r or s >= n;
    recovery rejects zero r or s, an x that is not on the curve, and the point at infinity."""
    if recid > 255 or recid >= 4:
        return SECP_INVALID_RECID, None
    r, s = int.from_bytes(sig[:32], 'big'), int.from_bytes(sig[32:], 'big')
    if r >= KN or s >= KN:
        return SECP_INVALID_SIG, None
    if r == 0 or s == 0:
        return SECP_INVALID_SIG, None
    x = r
    if recid & 2:
        if x >= KP - KN:
            return SECP_INVALID_SIG, None
        x += KN
    y2 = (x ** 3 + 7) % KP
    y = pow(y2, (KP + 1) // 4, KP)
    if y * y % KP != y2:
        return SECP_INVALID_SIG, None
    if (y & 1) != (recid & 1):
        y = KP - y
    e = int.from_bytes(h, 'big') % KN
    rn = inv(r, KN)
    q = k_add(k_mul(rn * s % KN, (x, y)), k_mul((-rn * e) % KN, KG))
    if q is None:
        return SECP_INVALID_SIG, None
    return 0, q[0].to_bytes(32, 'big') + q[1].to_bytes(32, 'big')


# ------------------------------------------------------------------ BN254 (arkworks 0.5 semantics)
BP = 21888242871839275222246405745257275088696311157297823662689037894645226208583
BR = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class Fail(Exception):
    pass


def f2(a, b=0):
    return (a % BP, b % BP)


def f2add(a, b): return ((a[0] + b[0]) % BP, (a[1] + b[1]) % BP)
def f2sub(a, b): return ((a[0] - b[0]) % BP, (a[1] - b[1]) % BP)
def f2neg(a): return ((-a[0]) % BP, (-a[1]) % BP)
def f2mul(a, b): return ((a[0] * b[0] - a[1] * b[1]) % BP, (a[0] * b[1] + a[1] * b[0]) % BP)


def f2inv(a):
    n = inv((a[0] * a[0] + a[1] * a[1]) % BP, BP)
    return (a[0] * n % BP, (-a[1]) * n % BP)


def f2pow(a, e):
    r = (1, 0)
    while e:
        if e & 1:
            r = f2mul(r, a)
        a = f2mul(a, a); e >>= 1
    return r


def f2key(a):                    # arkworks Ord: c1 first, then c0
    return (a[1], a[0])


def f2sqrt(a):
    if a == (0, 0):
        return (0, 0)
    a1 = f2pow(a, (BP - 3) // 4)
    alpha = f2mul(f2mul(a1, a1), a)
    a0 = f2mul(f2pow(alpha, BP), alpha)
    if a0 == (BP - 1, 0):
        return None
    x0 = f2mul(a1, a)
    x = f2mul((0, 1), x0) if alpha == (BP - 1, 0) else f2mul(f2pow(f2add((1, 0), alpha), (BP - 1) // 2), x0)
    return x if f2mul(x, x) == a else None


class BnField:
    """The arithmetic one curve needs: G1 over Fp (ints) or G2 over Fp2 (pairs)."""
    def __init__(self, g2):
        self.g2 = g2
        if g2:
            self.add, self.sub, self.mul, self.neg, self.inv = f2add, f2sub, f2mul, f2neg, f2inv
            self.zero, self.one, self.b = (0, 0), (1, 0), f2mul((3, 0), f2inv((9, 1)))
            self.key, self.sqrt = f2key, f2sqrt
        else:
            self.add = lambda a, b: (a + b) % BP
            self.sub = lambda a, b: (a - b) % BP
            self.mul = lambda a, b: a * b % BP
            self.neg = lambda a: (-a) % BP
            self.inv = lambda a: inv(a, BP)
            self.zero, self.one, self.b = 0, 1, 3
            self.key = lambda a: a
            self.sqrt = lambda a: (lambda r: r if r * r % BP == a % BP else None)(pow(a, (BP + 1) // 4, BP))

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return self.mul(y, y) == self.add(self.mul(self.mul(x, x), x), self.b)

    def padd(self, A, B):
        if A is None: return B
        if B is None: return A
        (x1, y1), (x2, y2) = A, B
        if x1 == x2:
            if self.add(y1, y2) == self.zero:
                return None
            m = self.mul(self.mul((3, 0) if self.g2 else 3, self.mul(x1, x1)), self.inv(self.add(y1, y1)))
        else:
            m = self.mul(self.sub(y2, y1), self.inv(self.sub(x2, x1)))
        x3 = self.sub(self.sub(self.mul(m, m), x1), x2)
        return (x3, self.sub(self.mul(m, self.sub(x1, x3)), y1))

    def pmul(self, k, A):
        r = None
        for bit in bin(k)[2:] if k else '':
            r = self.padd(r, r)
            if bit == '1':
                r = self.padd(r, A)
        return r

    # ---- byte formats (little-endian "pod" layout; big-endian callers convert first)
    def nb(self):
        return 64 if self.g2 else 32

    def fe(self, bs, flags=False):
        """A field element; with flags, the top two bits of the last byte are SWFlags.
        Returns (element, flag) where flag is None, 'inf' or 'neg'."""
        def fp(b, fl):
            v = int.from_bytes(b, 'little')
            f = None
            if fl:
                hi = v >> 254
                if hi == 3:
                    raise Fail()
                f = {0: None, 1: 'inf', 2: 'neg'}[hi]
                v &= 2**254 - 1
            if v >= BP:
                raise Fail()
            return v, f
        if not self.g2:
            return fp(bs, flags)
        c0, _ = fp(bs[:32], False)
        c1, f = fp(bs[32:], flags)
        return (c0, c1), f

    def enc(self, v):
        if self.g2:
            return v[0].to_bytes(32, 'little') + v[1].to_bytes(32, 'little')
        return v.to_bytes(32, 'little')

    def parse(self, bs, subgroup, check_curve=True):
        """Uncompressed point (x || y), all-zero = identity. Fails on bad field elements or flags;
        checks the curve (and the prime-order subgroup when asked)."""
        if bs == bytes(len(bs)):
            return None
        n = self.nb()
        x, _ = self.fe(bs[:n])
        y, f = self.fe(bs[n:], True)
        if f == 'inf':
            return None
        P = (x, y)
        if check_curve and not self.on_curve(P):
            raise Fail()
        if subgroup and self.g2 and self.pmul(BR, P) is not None:
            raise Fail()
        return P

    def out(self, P):
        return bytes(2 * self.nb()) if P is None else self.enc(P[0]) + self.enc(P[1])

    def compress(self, bs):
        if bs == bytes(len(bs)):
            return bytes(self.nb())
        n = self.nb()
        x, _ = self.fe(bs[:n])
        y, f = self.fe(bs[n:], True)
        if f == 'inf':
            c = bytearray(n); c[-1] = 0x40
            return bytes(c)
        c = bytearray(self.enc(x))
        if self.key(y) > self.key(self.neg(y)):
            c[-1] |= 0x80
        return bytes(c)

    def decompress(self, bs):
        if bs == bytes(len(bs)):
            return bytes(2 * self.nb())
        x, f = self.fe(bs, True)
        if f == 'inf':
            return bytes(2 * self.nb())
        y = self.sqrt(self.add(self.mul(self.mul(x, x), x), self.b))
        if y is None:
            raise Fail()
        lo, hi = sorted([y, self.neg(y)], key=self.key)
        return self.enc(x) + self.enc(hi if f == 'neg' else lo)


BN_G1, BN_G2 = BnField(False), BnField(True)


def be_to_le(bs, chunk):
    return b''.join(bs[i:i + chunk][::-1] for i in range(0, len(bs), chunk))


def bn_pairing_is_one(pairs):
    from py_ecc import optimized_bn128 as b
    acc = b.FQ12.one()
    for P, Q in pairs:
        if P is None or Q is None:
            continue
        Pp = (b.FQ(P[0]), b.FQ(P[1]), b.FQ(1))
        Qp = (b.FQ2([Q[0][0], Q[0][1]]), b.FQ2([Q[1][0], Q[1][1]]), b.FQ2.one())
        acc = acc * b.pairing(Qp, Pp, final_exponentiate=False)
    return b.final_exponentiate(acc) == b.FQ12.one()


def bn_group_op(op, inp):
    """sol_alt_bn128_group_op after the syscall's own checks: (0, bytes) or (1, None)."""
    le = bool(op & 0x80); o = op & 0x7f
    try:
        if o == 0:                                            # G1 add
            if len(inp) > 128 or (le and len(inp) != 128):
                raise Fail()
            inp = inp + bytes(128 - len(inp))
            cv = (lambda b: b) if le else (lambda b: be_to_le(b, 32))
            P, Q = BN_G1.parse(cv(inp[:64]), True), BN_G1.parse(cv(inp[64:]), True)
            r = BN_G1.out(BN_G1.padd(P, Q))
            return 0, r if le else be_to_le(r, 32)
        if o == 2:                                            # G1 mul (V1: 96-byte input)
            if len(inp) > 96 or (le and len(inp) != 96):
                raise Fail()
            inp = inp + bytes(96 - len(inp))
            P = BN_G1.parse(inp[:64] if le else be_to_le(inp[:64], 32), True)
            k = int.from_bytes(inp[64:], 'little' if le else 'big')
            r = BN_G1.out(BN_G1.pmul(k, P))
            return 0, r if le else be_to_le(r, 32)
        if o == 4:                                            # G2 add (on-curve only)
            if len(inp) != 256:
                raise Fail()
            cv = (lambda b: b) if le else (lambda b: be_to_le(b, 64))
            P, Q = BN_G2.parse(cv(inp[:128]), False), BN_G2.parse(cv(inp[128:]), False)
            r = BN_G2.out(BN_G2.padd(P, Q))
            return 0, r if le else be_to_le(r, 64)
        if o == 6:                                            # G2 mul (subgroup checked)
            if len(inp) != 160:
                raise Fail()
            P = BN_G2.parse(inp[:128] if le else be_to_le(inp[:128], 64), True)
            k = int.from_bytes(inp[128:], 'little' if le else 'big')
            r = BN_G2.out(BN_G2.pmul(k, P))
            return 0, r if le else be_to_le(r, 64)
        if o == 3:                                            # pairing
            if len(inp) % 192:
                raise Fail()
            pairs = []
            for i in range(0, len(inp), 192):
                c = inp[i:i + 192]
                P = BN_G1.parse(c[:64] if le else be_to_le(c[:64], 32), True)
                Q = BN_G2.parse(c[64:] if le else be_to_le(c[64:], 64), True)
                pairs.append((P, Q))
            v = 1 if bn_pairing_is_one(pairs) else 0
            return 0, v.to_bytes(32, 'little' if le else 'big')
    except Fail:
        return 1, None
    raise ValueError(op)


BN_COMPRESSION = {0: (False, 'c'), 1: (False, 'd'), 2: (True, 'c'), 3: (True, 'd')}


def bn_compression(op, inp):
    le = bool(op & 0x80)
    g2, kind = BN_COMPRESSION[op & 0x7f]
    F = BN_G2 if g2 else BN_G1
    ch = 64 if g2 else 32
    need = (2 if kind == 'c' else 1) * F.nb()
    if len(inp) != need:
        return 1, None
    try:
        if kind == 'c':
            r = F.compress(inp if le else be_to_le(inp, ch))
        else:
            r = F.decompress(inp if le else be_to_le(inp, ch))
    except Fail:
        return 1, None
    return 0, r if le else be_to_le(r, ch)


# ------------------------------------------------------------------ Poseidon (light-poseidon, circom BN254 x5)
_POS = None


def poseidon_params(t):
    global _POS
    if _POS is None:
        import json, os
        raw = json.load(open(os.path.join(os.path.dirname(__file__), 'poseidon_bn254_x5.json')))
        PR = [56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64]
        _POS = {int(k): ([int(x, 16) for x in v['ark']], [[int(x, 16) for x in r] for r in v['mds']], PR[v['partial_idx']])
                for k, v in raw.items()}
    return _POS.get(t)


def poseidon_fr(xs):
    t = len(xs) + 1
    ark, mds, partial = poseidon_params(t)
    s = [0] + list(xs)
    half = 4
    for rnd in range(8 + partial):
        s = [(v + ark[rnd * t + i]) % BR for i, v in enumerate(s)]
        if rnd < half or rnd >= half + partial:
            s = [pow(v, 5, BR) for v in s]
        else:
            s[0] = pow(s[0], 5, BR)
        s = [sum(m * v for m, v in zip(row, s)) % BR for row in mds]
    return s[0]


def poseidon_hashv(params, endian, vals):
    """-> (0, 32 bytes) or (1, None). params and endian are already validated by the caller."""
    if not 1 <= len(vals) <= 12:
        return 1, None
    xs = []
    for v in vals:
        if len(v) != 32:
            return 1, None
        x = int.from_bytes(v, 'little' if endian else 'big')
        if x >= BR:
            return 1, None
        xs.append(x)
    h = poseidon_fr(xs)
    return 0, h.to_bytes(32, 'little' if endian else 'big')


# ------------------------------------------------------------------ BLS12-381 (blst / blstrs semantics, zcash encoding)
QP = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
QR = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


class Wc:
    """y^2 = x^3 + b over Fp (ints) or Fp2 = Fp[u]/(u^2 + 1) (pairs), modulus p."""
    def __init__(self, p, b, g2):
        self.p, self.g2 = p, g2
        if g2:
            self.add = lambda a, c: ((a[0] + c[0]) % p, (a[1] + c[1]) % p)
            self.sub = lambda a, c: ((a[0] - c[0]) % p, (a[1] - c[1]) % p)
            self.mul = lambda a, c: ((a[0] * c[0] - a[1] * c[1]) % p, (a[0] * c[1] + a[1] * c[0]) % p)
            self.neg = lambda a: ((-a[0]) % p, (-a[1]) % p)
            self.zero, self.three = (0, 0), (3, 0)
            self.key = lambda a: (a[1], a[0])
        else:
            self.add = lambda a, c: (a + c) % p
            self.sub = lambda a, c: (a - c) % p
            self.mul = lambda a, c: a * c % p
            self.neg = lambda a: (-a) % p
            self.zero, self.three = 0, 3
            self.key = lambda a: a
        self.b = b

    def inv(self, a):
        p = self.p
        if self.g2:
            n = pow((a[0] * a[0] + a[1] * a[1]) % p, p - 2, p)
            return (a[0] * n % p, (-a[1]) * n % p)
        return pow(a, p - 2, p)

    def pw(self, a, e):
        r = (1, 0) if self.g2 else 1
        while e:
            if e & 1:
                r = self.mul(r, a)
            a = self.mul(a, a); e >>= 1
        return r

    def sqrt(self, a):
        p = self.p
        if a == self.zero:
            return self.zero
        if not self.g2:
            r = pow(a, (p + 1) // 4, p)
            return r if r * r % p == a else None
        a1 = self.pw(a, (p - 3) // 4)
        alpha = self.mul(self.mul(a1, a1), a)
        if self.mul(self.pw(alpha, p), alpha) == (p - 1, 0):
            return None
        x0 = self.mul(a1, a)
        x = self.mul((0, 1), x0) if alpha == (p - 1, 0) else self.mul(self.pw(self.add((1, 0), alpha), (p - 1) // 2), x0)
        return x if self.mul(x, x) == a else None

    def on_curve(self, P):
        return P is None or self.mul(P[1], P[1]) == self.add(self.mul(self.mul(P[0], P[0]), P[0]), self.b)

    def padd(self, A, B):
        if A is None: return B
        if B is None: return A
        (x1, y1), (x2, y2) = A, B
        if x1 == x2:
            if self.add(y1, y2) == self.zero:
                return None
            m = self.mul(self.mul(self.three, self.mul(x1, x1)), self.inv(self.add(y1, y1)))
        else:
            m = self.mul(self.sub(y2, y1), self.inv(self.sub(x2, x1)))
        x3 = self.sub(self.sub(self.mul(m, m), x1), x2)
        return (x3, self.sub(self.mul(m, self.sub(x1, x3)), y1))

    def pmul(self, k, A):
        r = None
        for bit in bin(k)[2:] if k else '':
            r = self.padd(r, r)
            if bit == '1':
                r = self.padd(r, A)
        return r

    def pneg(self, A):
        return None if A is None else (A[0], self.neg(A[1]))


BLS_G1, BLS_G2 = Wc(QP, 4, False), Wc(QP, (4, 4), True)


def _fe48(bs):                   # one big-endian Fp element, 48 bytes; None if not reduced
    v = int.from_bytes(bs, 'big')
    return v if v < QP else None


def _fq(bs, g2):                 # zcash order: Fp2 is c1 || c0
    if not g2:
        return _fe48(bs)
    c1, c0 = _fe48(bs[:48]), _fe48(bs[48:])
    return None if c1 is None or c0 is None else (c0, c1)


def _enc(v, g2):
    return (v[1].to_bytes(48, 'big') + v[0].to_bytes(48, 'big')) if g2 else v.to_bytes(48, 'big')


def bls_uncompressed(bs, g2, subgroup):
    """blst_p1/p2_deserialize on uncompressed bytes (after the syscall's 0xa0 pre-check), then the
    optional torsion check. Returns ('ok', point or None for infinity) or ('fail', None)."""
    W = BLS_G2 if g2 else BLS_G1
    n = len(bs) // 2
    if bs[0] & 0xa0:
        return 'fail', None
    if bs[0] & 0x40:
        if bs[0] & 0x3f == 0 and not any(bs[1:]):
            return 'ok', None
        return 'fail', None
    x, y = _fq(bs[:n], g2), _fq(bs[n:], g2)
    if x is None or y is None or not W.on_curve((x, y)):
        return 'fail', None
    P = (x, y)
    if subgroup and W.pmul(QR, P) is not None:
        return 'fail', None
    return 'ok', P


def bls_out(P, g2):
    n = 96 if g2 else 48
    if P is None:
        return bytes([0x40]) + bytes(2 * n - 1)
    return _enc(P[0], g2) + _enc(P[1], g2)


def bls_compressed(bs, g2):
    W = BLS_G2 if g2 else BLS_G1
    if not bs[0] & 0x80:
        return 'fail', None
    if bs[0] & 0x40:
        if bs[0] & 0x3f == 0 and not any(bs[1:]):
            return 'ok', None
        return 'fail', None
    big = bool(bs[0] & 0x20)
    raw = bytes([bs[0] & 0x1f]) + bs[1:]
    x = _fq(raw, g2)
    if x is None:
        return 'fail', None
    y = W.sqrt(W.add(W.mul(W.mul(x, x), x), W.b))
    if y is None:
        return 'fail', None
    lo, hi = sorted([y, W.neg(y)], key=W.key)
    P = (x, hi if big else lo)
    if W.pmul(QR, P) is not None:
        return 'fail', None
    return 'ok', P


def bls_to_be(bs, g2):
    """The syscall's LE layout to zcash BE: reverse each 48-byte limb, then swap c0/c1 for G2."""
    b = bytearray(b''.join(bs[i:i + 48][::-1] for i in range(0, len(bs), 48)))
    if g2:
        for i in range(0, len(b), 96):
            b[i:i + 96] = b[i + 48:i + 96] + b[i:i + 48]
    return bytes(b)


bls_from_be = bls_to_be                  # the same permutation both ways


def bls_gt(pairs):
    """Product of pairings, final-exponentiated, serialized as blst's fp12 (Fp2 -> Fp6 -> Fp12
    tower, each Fp little-endian), 576 bytes."""
    from py_ecc import optimized_bls12_381 as b
    acc = b.FQ12.one()
    for P, Q in pairs:
        if P is None or Q is None:
            continue
        Pp = (b.FQ(P[0]), b.FQ(P[1]), b.FQ(1))
        Qp = (b.FQ2([Q[0][0], Q[0][1]]), b.FQ2([Q[1][0], Q[1][1]]), b.FQ2.one())
        acc = acc * b.pairing(Qp, Pp, final_exponentiate=False)
    f = b.final_exponentiate(acc) if pairs else acc
    # blst's Miller loop runs over the negative BLS parameter and its final exponentiation uses the
    # 3 * (p^4 - p^2 + 1) / r hard part, so its value is py_ecc's e(P, Q) to the power -3
    # (measured against LiteSVM: e(G1, G2) matches only for k = -3 in k in {+-1, +-2, +-3}).
    f = (f ** 3).inv()
    c = [int(x) % QP for x in f.coeffs]
    out = b''
    for i in range(2):                    # fp6 index: the w power parity
        for j in range(3):                # fp2 index: v^j = w^(2j)
            k = 2 * j + i
            c1 = c[k + 6]
            c0 = (c[k] + c[k + 6]) % QP
            out += c0.to_bytes(48, 'little') + c1.to_bytes(48, 'little')
    return out


def bls_syscall(kind, cid, args):
    """kind in validate/add/sub/mul/decompress/pairing; cid is the curve id (endianness in bit 7).
    args are raw byte strings; returns (0, output bytes or None) or (1, None)."""
    le = not cid & 0x80
    g2 = (cid & 0x7f) == 6
    be = (lambda x, g: bls_to_be(x, g)) if le else (lambda x, g: x)
    back = (lambda x, g: bls_from_be(x, g)) if le else (lambda x, g: x)
    if kind == 'validate':
        st, _ = bls_uncompressed(be(args[0], g2), g2, True)
        return (0, None) if st == 'ok' else (1, None)
    if kind in ('add', 'sub'):
        W = BLS_G2 if g2 else BLS_G1
        s1, P = bls_uncompressed(be(args[0], g2), g2, False)
        s2, Q = bls_uncompressed(be(args[1], g2), g2, False)
        if s1 != 'ok' or s2 != 'ok':
            return 1, None
        R = W.padd(P, Q if kind == 'add' else W.pneg(Q))
        return 0, back(bls_out(R, g2), g2)
    if kind == 'mul':
        W = BLS_G2 if g2 else BLS_G1
        k = int.from_bytes(args[0], 'little' if le else 'big')
        st, P = bls_uncompressed(be(args[1], g2), g2, True)
        if st != 'ok' or k >= QR:
            return 1, None
        return 0, back(bls_out(W.pmul(k, P), g2), g2)
    if kind == 'decompress':
        st, P = bls_compressed(be(args[0], g2), g2)
        if st != 'ok':
            return 1, None
        return 0, back(bls_out(P, g2), g2)
    if kind == 'pairing':
        g1s, g2s = args
        if len(g1s) > 8:
            return 1, None
        pairs = []
        for a, c in zip(g1s, g2s):
            s1, P = bls_uncompressed(be(a, False), False, True)
            s2, Q = bls_uncompressed(be(c, True), True, True)
            if s1 != 'ok' or s2 != 'ok':
                return 1, None
            pairs.append((P, Q))
        out = bls_gt(pairs)
        return 0, out if le else out[::-1]
    raise ValueError(kind)
