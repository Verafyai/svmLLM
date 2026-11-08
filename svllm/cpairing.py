"""The pairing on the weights: pairing.py's algorithm with every field operation on fastbig's
table layers. An Fp element is a limb tensor [k], Fp2 is [2, k], Fp12 is [6, 2, k]; a whole Fp12
product (36 Fp2 products, 108 Fp multiplies) is one batched pass.
"""
import torch
from . import fastbig as F
from . import pairing as PR


class CircuitBackend(PR.IntBackend):
    def __init__(self, C):
        super().__init__(C)
        self.Fd = F.Field(C.p)
        self.k = self.Fd.k
        self.xi_t = self.c2(C.xi)
        self.g_t = torch.stack([self.c2(g) for g in self.g])

    # constants
    def c1(self, v):
        return self.Fd.n(v)

    def c2(self, v):
        return torch.stack([self.Fd.n(v[0]), self.Fd.n(v[1])])

    def zeros2(self, *shape):
        return F.zeros(*shape, 2, self.k)

    # Fp (batched over any leading dims)
    def fadd(self, a, b): return self.Fd.add(a, b)
    def fsub(self, a, b): return self.Fd.sub(a, b)
    def fneg(self, a): return self.Fd.sub(F.zeros(*a.shape), a)

    # Fp2 = [..., 2, k]
    def f2_add(self, a, b): return self.fadd(a, b)
    def f2_sub(self, a, b): return self.fsub(a, b)
    def f2_neg(self, a): return self.fneg(a)

    def f2_mul(self, a, b):
        a, b = torch.broadcast_tensors(a, b)
        s = self.fadd(a[..., 0, :], a[..., 1, :])
        t = self.fadd(b[..., 0, :], b[..., 1, :])
        m = self.Fd.mul(torch.stack([a[..., 0, :], a[..., 1, :], s], -2),
                        torch.stack([b[..., 0, :], b[..., 1, :], t], -2))
        t0, t1, t2 = m[..., 0, :], m[..., 1, :], m[..., 2, :]
        return torch.stack([self.fsub(t0, t1), self.fsub(self.fsub(t2, t0), t1)], -2)

    def f2_sq(self, a): return self.f2_mul(a, a)

    def f2_conj(self, a):
        return torch.stack([a[..., 0, :], self.fneg(a[..., 1, :])], -2)

    def f2_fp(self, a, s):
        return self.Fd.mul(a, s.expand(a.shape))

    def f2_small(self, a, k):
        return self.Fd.mul(a, self.c1(k).expand(a.shape))

    def f2_inv(self, a):
        n = self.Fd.inv(self.fadd(self.Fd.mul(a[0], a[0]), self.Fd.mul(a[1], a[1])))
        return torch.stack([self.Fd.mul(a[0], n), self.fneg(self.Fd.mul(a[1], n))])

    # Fp12 = [6, 2, k]
    def f12_one(self):
        o = self.zeros2(6)
        o[0, 0] = self.c1(1)
        return o

    def f12_mul(self, x, y):
        """Lazy reduction: the 108 Karatsuba-Fp2 products are formed unreduced, summed by degree,
        folded by w^6 = xi, and only the 12 output components are reduced (wide Barrett)."""
        k, W = self.k, 2 * self.k + 2
        def kar(z):                                        # [6, 2, k] -> [6, 3, k + 1]: z0, z1, z0 + z1 (no mod)
            z0, z1 = F.pad(z[:, 0], k + 1), F.pad(z[:, 1], k + 1)
            return torch.stack([z0, z1, F.add(z0, z1)[0]], 1)
        A, B = kar(x), kar(y)
        Pm = F.mul(A[:, None].expand(6, 6, 3, k + 1), B[None].expand(6, 6, 3, k + 1))   # [6, 6, 3, 2k + 2]
        G = F.zeros(11, 6, 3, W)
        for i in range(6):                                 # fixed wiring: product (i, j) to degree i + j
            for j in range(6):
                G[i + j, i] = Pm[i, j]
        S = F.add(G[:, 0:3], G[:, 3:6])[0]
        S = F.add(F.add(S[:, 0], S[:, 1])[0], S[:, 2])[0]          # [11, 3, W]: T0, T1, T2 per degree
        Z = F.zeros(3, W)
        T = torch.cat([S, Z[None].expand(1, 3, W)])                  # degree 11 is zero
        x0 = self.C.xi[0]
        sm = lambda v, c: F.mul(v, F.num(c, 1).expand(*v.shape[:-1], 1))[..., :W]
        T0, T1, T2 = T[:, 0], T[:, 1], T[:, 2]
        lo, hi = slice(0, 6), slice(6, 12)
        add = lambda *vs: _sum(vs)
        re_pos = add(T0[lo], sm(T0[hi], x0 + 1), T1[hi])
        re_neg = add(T1[lo], sm(T1[hi], x0), T2[hi])
        im_pos = add(T2[lo], T0[hi], sm(T2[hi], x0))
        im_neg = add(T0[lo], T1[lo], sm(T1[hi], x0 + 1), sm(T0[hi], x0))
        K = self._K(W)
        pos = torch.stack([re_pos, im_pos], 1)
        neg = torch.stack([re_neg, im_neg], 1)
        tot, _ = F.sub(F.add(pos, K.expand(pos.shape))[0], neg)
        return self.Fd.reduce_wide(tot)

    def _K(self, W):
        """2^10 p^2 rounded to a multiple of p: larger than any negative part, and 0 mod p."""
        if not hasattr(self, '_kk'):
            self._kk = F.num(2 ** 10 * self.C.p ** 2, W)
        return self._kk

    def f12_sq(self, x):
        return self.f12_mul(x, x)

    def f12_conj(self, x):
        return torch.stack([x[i] if i % 2 == 0 else self.fneg(x[i]) for i in range(6)])

    def f12_frob(self, x, k):
        for _ in range(k):
            x = self.f2_mul(self.f2_conj(x), self.g_t)
        return x

    def f6_inv(self, a):
        xi = self.xi_t
        a0, a1, a2 = a
        t0 = self.f2_sub(self.f2_sq(a0), self.f2_mul(xi, self.f2_mul(a1, a2)))
        t1 = self.f2_sub(self.f2_mul(xi, self.f2_sq(a2)), self.f2_mul(a0, a1))
        t2 = self.f2_sub(self.f2_sq(a1), self.f2_mul(a0, a2))
        den = self.f2_add(self.f2_mul(a0, t0), self.f2_mul(xi, self.f2_add(self.f2_mul(a2, t1), self.f2_mul(a1, t2))))
        d = self.f2_inv(den)
        return [self.f2_mul(t0, d), self.f2_mul(t1, d), self.f2_mul(t2, d)]

    def f6_mul(self, a, b):
        xi = self.xi_t
        s = [self.zeros2()] * 5
        for i in range(3):
            for j in range(3):
                s[i + j] = self.f2_add(s[i + j], self.f2_mul(a[i], b[j]))
        return [self.f2_add(s[0], self.f2_mul(xi, s[3])), self.f2_add(s[1], self.f2_mul(xi, s[4])), s[2]]

    def f12_inv(self, x):
        a0, a1 = [x[0], x[2], x[4]], [x[1], x[3], x[5]]
        a1sq = self.f6_mul(a1, a1)
        a1sq_v = [self.f2_mul(self.xi_t, a1sq[2]), a1sq[0], a1sq[1]]
        den = self.f6_inv([self.f2_sub(u, v) for u, v in zip(self.f6_mul(a0, a0), a1sq_v)])
        b0, b1 = self.f6_mul(a0, den), self.f6_mul(a1, den)
        return torch.stack([b0[0], self.f2_neg(b1[0]), b0[1], self.f2_neg(b1[1]), b0[2], self.f2_neg(b1[2])])

    def f12_is_one(self, x):
        return F.eq(x, self.f12_one())

    # twist points: coordinates are [2, k]
    def jac(self, Q):
        one = self.zeros2(); one[0] = self.c1(1)
        return (Q[0], Q[1], one)

    def twist_frob(self, C, Q, k):
        x, y = Q
        for _ in range(k):
            x = self.f2_mul(self.f2_conj(x), self.g_t[2])
            y = self.f2_mul(self.f2_conj(y), self.g_t[3])
        return (x, y)

    def line(self, C, a, b, c):
        L = self.zeros2(6)
        if C.dtype == 'D':
            L[0], L[1], L[3] = a, b, c
        else:
            L[0], L[2], L[3] = c, b, a
        return L


def _sum(vs):
    acc = vs[0]
    for v in vs[1:]:
        acc = F.add(acc, v)[0]
    return acc


_B = {}


def backend(C):
    if C.name not in _B:
        _B[C.name] = CircuitBackend(C)
    return _B[C.name]


def to_f2(v):
    """ccurves' Fp2 pair (c0 limbs, c1 limbs) -> [2, k]."""
    return torch.stack([v[0], v[1]])


def bn_pairing_is_one(pairs):
    """pairs: ccurves BN points (G1 affine limbs, G2 affine Fp2 pairs) or None."""
    B = backend(PR.BN)
    pp = [(P, (to_f2(Q[0]), to_f2(Q[1]))) for P, Q in pairs if P is not None and Q is not None]
    return B.f12_is_one(PR.pairing_product(B, PR.BN, pp))


def bls_gt_bytes(pairs):
    """-> 576 one-hot bytes, blst's Fp12 layout (little-endian Fp limbs). blst's value is
    py_ecc's (= this algorithm's) to the power -3, i.e. conj(f^3) in the cyclotomic subgroup."""
    B = backend(PR.BLS)
    pp = [(P, (to_f2(Q[0]), to_f2(Q[1]))) for P, Q in pairs if P is not None and Q is not None]
    f = PR.pairing_product(B, PR.BLS, pp)
    if pp:
        f = B.f12_conj(B.f12_mul(B.f12_sq(f), f))
    out = []
    for i in range(2):
        for j in range(3):
            c = f[2 * j + i]
            out += F.to_bytes(c[0], 48) + F.to_bytes(c[1], 48)
    return out
