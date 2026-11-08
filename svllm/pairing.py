"""Optimal ate pairings for BN254 and BLS12-381, written once over a backend.

The backend supplies the arithmetic. IntBackend uses Python integers and exists to check the
algorithm against py_ecc. CircuitBackend (ccurves) runs the same steps on fastbig's table layers,
with whole Fp12 products batched into one pass.

Fp12 is flat: Fp2[w] / (w^6 - xi), an element is six Fp2 coefficients c_0 .. c_5. This is the
same field with the same w as py_ecc's Fp[w] / (w^12 - 2 Re(xi) w^6 + |xi|^2), and its coefficient
order is blst's Fp2 -> Fp6 -> Fp12 tower flattened (blst's a_ij is c_{2j+i}).

Lines are evaluated at P up to factors in Fp4 = Fp2(w^3), which the final exponentiation
(p^12 - 1) / r removes, since p^4 - 1 divides it. That lets every line skip its denominator.
"""


class Curve:
    def __init__(self, name, p, r, xi, b2, loop, dtype, final_frob):
        self.name, self.p, self.r, self.xi, self.b2 = name, p, r, xi, b2
        self.loop, self.dtype, self.final_frob = loop, dtype, final_frob
        e = (p ** 4 - p ** 2 + 1) // r
        self.hard = []                              # base-p digits of the hard exponent
        while e:
            self.hard.append(e % p); e //= p


def _f2pow(a, e, p):
    r = (1, 0)
    while e:
        if e & 1:
            r = ((r[0] * a[0] - r[1] * a[1]) % p, (r[0] * a[1] + r[1] * a[0]) % p)
        a = ((a[0] * a[0] - a[1] * a[1]) % p, (2 * a[0] * a[1]) % p)
        e >>= 1
    return r


def _f2inv(a, p):
    n = pow(a[0] * a[0] + a[1] * a[1], p - 2, p)
    return (a[0] * n % p, -a[1] * n % p)


def _f2mul(a, b, p):
    return ((a[0] * b[0] - a[1] * b[1]) % p, (a[0] * b[1] + a[1] * b[0]) % p)


BN = Curve('bn254', 21888242871839275222246405745257275088696311157297823662689037894645226208583,
           21888242871839275222246405745257275088548364400416034343698204186575808495617,
           (9, 1), None, 29793968203157093288, 'D', True)
BN.b2 = _f2mul((3, 0), _f2inv((9, 1), BN.p), BN.p)
BLS = Curve('bls12-381', 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab,
            0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001,
            (1, 1), (4, 4), 0xd201000000010000, 'M', False)


def frob_consts(C):
    """gamma_i = xi^(i (p - 1) / 6): w^p = gamma_1 w; and the twist Frobenius constants."""
    p = C.p
    g = [_f2pow(C.xi, i * (p - 1) // 6, p) for i in range(6)]
    return g


def miller(B, C, pairs):
    """Product of the Miller functions f_{loop, Q}(P) over pairs (P affine over Fp, Q affine over
    Fp2 on the twist), with BN's two closing lines. Pairs with an identity are skipped by the caller."""
    f = B.f12_one()
    Ts = [B.jac(Q) for _, Q in pairs]
    bits = bin(C.loop)[3:]
    for bit in bits:
        f = B.f12_sq(f)
        for k, (P, Q) in enumerate(pairs):
            line, Ts[k] = B.dbl_line(C, Ts[k], P)
            f = B.f12_mul(f, line)
        if bit == '1':
            for k, (P, Q) in enumerate(pairs):
                line, Ts[k] = B.add_line(C, Ts[k], Q, P)
                f = B.f12_mul(f, line)
    if C.final_frob:
        for k, (P, Q) in enumerate(pairs):
            Q1 = B.twist_frob(C, Q, 1)
            Q2 = B.twist_frob(C, Q, 2)
            Q2 = (Q2[0], B.f2_neg(Q2[1]))
            line, Ts[k] = B.add_line(C, Ts[k], Q1, P)
            f = B.f12_mul(f, line)
            line, Ts[k] = B.add_line(C, Ts[k], Q2, P)
            f = B.f12_mul(f, line)
    return f


def final_exp(B, C, f):
    f = B.f12_mul(B.f12_conj(f), B.f12_inv(f))          # ^(p^6 - 1)
    f = B.f12_mul(B.f12_frob(f, 2), f)                  # ^(p^2 + 1)
    bases = [f]                                         # hard part: prod frob^i(f)^d_i
    for i in range(1, len(C.hard)):
        bases.append(B.f12_frob(bases[-1], 1))
    n = len(bases)
    table = {}
    for mask in range(1, 1 << n):                       # every subset product, once
        lo = mask & -mask
        i = lo.bit_length() - 1
        table[mask] = bases[i] if mask == lo else B.f12_mul(table[mask ^ lo], bases[i])
    acc = None
    for bit in range(max(d.bit_length() for d in C.hard) - 1, -1, -1):
        if acc is not None:
            acc = B.f12_sq(acc)
        mask = sum(((d >> bit) & 1) << i for i, d in enumerate(C.hard))
        if mask:
            acc = table[mask] if acc is None else B.f12_mul(acc, table[mask])
    return acc


def pairing_product(B, C, pairs):
    pairs = [(P, Q) for P, Q in pairs if P is not None and Q is not None]
    if not pairs:
        return B.f12_one()
    return final_exp(B, C, miller(B, C, pairs))


class IntBackend:
    """Python integers, for checking the algorithm against py_ecc."""
    def __init__(self, C):
        self.C, self.p = C, C.p
        self.g = frob_consts(C)

    # Fp2
    def f2_add(self, a, b): return ((a[0] + b[0]) % self.p, (a[1] + b[1]) % self.p)
    def f2_sub(self, a, b): return ((a[0] - b[0]) % self.p, (a[1] - b[1]) % self.p)
    def f2_neg(self, a): return ((-a[0]) % self.p, (-a[1]) % self.p)
    def f2_mul(self, a, b): return _f2mul(a, b, self.p)
    def f2_sq(self, a): return _f2mul(a, a, self.p)
    def f2_conj(self, a): return (a[0], (-a[1]) % self.p)
    def f2_fp(self, a, s): return (a[0] * s % self.p, a[1] * s % self.p)     # Fp2 times an Fp element
    def f2_small(self, a, k): return (a[0] * k % self.p, a[1] * k % self.p)

    # Fp12
    def f12_one(self):
        return [(1, 0)] + [(0, 0)] * 5

    def f12_mul(self, x, y):
        s = [(0, 0)] * 11
        for i in range(6):
            for j in range(6):
                s[i + j] = self.f2_add(s[i + j], self.f2_mul(x[i], y[j]))
        return [self.f2_add(s[i], self.f2_mul(self.C.xi, s[i + 6])) if i < 5 else s[5] for i in range(6)]

    def f12_sq(self, x):
        return self.f12_mul(x, x)

    def f12_conj(self, x):
        return [c if i % 2 == 0 else self.f2_neg(c) for i, c in enumerate(x)]

    def f12_frob(self, x, k):
        for _ in range(k):
            x = [self.f2_mul(self.f2_conj(c), self.g[i]) for i, c in enumerate(x)]
        return x

    def f6_mul(self, a, b):
        xi = self.C.xi
        s = [(0, 0)] * 5
        for i in range(3):
            for j in range(3):
                s[i + j] = self.f2_add(s[i + j], self.f2_mul(a[i], b[j]))
        return [self.f2_add(s[0], self.f2_mul(xi, s[3])), self.f2_add(s[1], self.f2_mul(xi, s[4])), s[2]]

    def f6_inv(self, a):
        xi = self.C.xi
        a0, a1, a2 = a
        t0 = self.f2_sub(self.f2_sq(a0), self.f2_mul(xi, self.f2_mul(a1, a2)))
        t1 = self.f2_sub(self.f2_mul(xi, self.f2_sq(a2)), self.f2_mul(a0, a1))
        t2 = self.f2_sub(self.f2_sq(a1), self.f2_mul(a0, a2))
        den = self.f2_add(self.f2_mul(a0, t0), self.f2_mul(xi, self.f2_add(self.f2_mul(a2, t1), self.f2_mul(a1, t2))))
        d = _f2inv(den, self.p)
        return [self.f2_mul(t0, d), self.f2_mul(t1, d), self.f2_mul(t2, d)]

    def f12_inv(self, x):
        a0, a1 = x[0::2], x[1::2]                       # x = a0 + a1 w with a_k in Fp6 = Fp2[v], v = w^2
        a1sq = self.f6_mul(a1, a1)
        a1sq_v = [self.f2_mul(self.C.xi, a1sq[2]), a1sq[0], a1sq[1]]
        den = self.f6_inv([self.f2_sub(u, v) for u, v in zip(self.f6_mul(a0, a0), a1sq_v)])
        b0, b1 = self.f6_mul(a0, den), self.f6_mul(a1, den)
        out = [None] * 6
        out[0::2] = b0
        out[1::2] = [self.f2_neg(c) for c in b1]
        return out

    # twist points
    def jac(self, Q):
        return (Q[0], Q[1], (1, 0))

    def twist_frob(self, C, Q, k):
        """pi^k on the twist, D-type: x -> conj(x) gamma_2, y -> conj(y) gamma_3."""
        x, y = Q
        for _ in range(k):
            x = self.f2_mul(self.f2_conj(x), self.g[2])
            y = self.f2_mul(self.f2_conj(y), self.g[3])
        return (x, y)

    def line(self, C, a, b, c):
        """Place the line's three Fp2 coefficients: a at w^0, b at w^1 or w^2, c at w^3."""
        L = [(0, 0)] * 6
        if C.dtype == 'D':
            L[0], L[1], L[3] = a, b, c
        else:
            L[0], L[2], L[3] = c, b, a                  # times w^3 (an Fp4 factor): see the module note
        return L

    def dbl_line(self, C, T, P):
        X, Y, Z = T
        xP, yP = P
        Z2 = self.f2_sq(Z)
        Z3 = self.f2_mul(Z2, Z)
        X2 = self.f2_sq(X)
        Y2 = self.f2_sq(Y)
        a = self.f2_fp(self.f2_small(self.f2_mul(Y, Z3), 2), yP)               # 2 Y Z^3 yP
        b = self.f2_neg(self.f2_fp(self.f2_small(self.f2_mul(X2, Z2), 3), xP))  # -3 X^2 Z^2 xP
        c = self.f2_sub(self.f2_small(self.f2_mul(X2, X), 3), self.f2_small(Y2, 2))  # 3 X^3 - 2 Y^2
        return self.line(C, a, b, c), self.jdbl(T)

    def add_line(self, C, T, Q, P):
        X, Y, Z = T
        xQ, yQ = Q
        xP, yP = P
        Z2 = self.f2_sq(Z)
        Z3 = self.f2_mul(Z2, Z)
        N = self.f2_sub(self.f2_mul(yQ, Z3), Y)
        D = self.f2_sub(self.f2_mul(xQ, Z2), X)
        DZ = self.f2_mul(D, Z)
        a = self.f2_fp(DZ, yP)
        b = self.f2_neg(self.f2_fp(N, xP))
        c = self.f2_sub(self.f2_mul(N, xQ), self.f2_mul(DZ, yQ))
        return self.line(C, a, b, c), self.jadd(T, Q)

    def jdbl(self, T):
        X, Y, Z = T
        A, Bb = self.f2_sq(X), self.f2_sq(Y)
        Cc = self.f2_sq(Bb)
        t = self.f2_sub(self.f2_sub(self.f2_sq(self.f2_add(X, Bb)), A), Cc)
        D = self.f2_add(t, t)
        E = self.f2_small(A, 3)
        F = self.f2_sq(E)
        X3 = self.f2_sub(F, self.f2_add(D, D))
        Y3 = self.f2_sub(self.f2_mul(E, self.f2_sub(D, X3)), self.f2_small(Cc, 8))
        return (X3, Y3, self.f2_small(self.f2_mul(Y, Z), 2))

    def jadd(self, T, Q):
        """T Jacobian + Q affine; T != +-Q is guaranteed inside a Miller loop for points of order r."""
        X1, Y1, Z1 = T
        x2, y2 = Q
        Z1Z1 = self.f2_sq(Z1)
        U2 = self.f2_mul(x2, Z1Z1)
        S2 = self.f2_mul(self.f2_mul(y2, Z1), Z1Z1)
        H = self.f2_sub(U2, X1)
        Rr = self.f2_sub(S2, Y1)
        HH = self.f2_sq(H)
        HHH = self.f2_mul(H, HH)
        V = self.f2_mul(X1, HH)
        X3 = self.f2_sub(self.f2_sub(self.f2_sq(Rr), HHH), self.f2_add(V, V))
        Y3 = self.f2_sub(self.f2_mul(Rr, self.f2_sub(V, X3)), self.f2_mul(Y1, HHH))
        return (X3, Y3, self.f2_mul(Z1, H))
