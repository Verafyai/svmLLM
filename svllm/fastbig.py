"""Wide integer arithmetic for the curve syscalls, built from machine.py's lookup-table layers.

A 4-bit limb is a one-hot vector everywhere else in the machine. Here each limb is carried as the
position of its one (an integer 0..15), and every table layer is evaluated with LUT.ix, which reads
the output for that one-hot input straight out of the layer's own W2. The values are identical to
the dense layer's; the zero terms are skipped. `to_onehot` / `from_onehot` convert at the boundary
with the rest of the machine.

The arrangement is chosen to keep the number of sequential layers small:
  * add: a parallel-prefix (Kogge-Stone) adder. Every limb's sum is formed for carry-in 0 and 1,
    each limb's carry-out is a two-state function of its carry-in, and those functions are composed
    in log2(L) steps by the COMP table.
  * mul: all La x Lb nibble products in one batched L_MULN, the shifted rows summed as a tree of adds.
  * mod m: Barrett reduction (two more multiplies), then at most two borrow-selected subtractions.
"""
import torch
from .machine import L_ADD, L_MULN, LUT

COMP = LUT([2, 2, 2, 2], [2, 2], lambda a0, a1, b0, b1: (b1 if a0 else b0, b1 if a1 else b0))
SEL = LUT([2, 16, 16], [16], lambda c, s0, s1: s1 if c else s0)
APPLY = LUT([2, 2, 2], [2], lambda o0, o1, c: o1 if c else o0)
NOT4 = LUT([16], [16], lambda a: 15 - a)
SEL2 = LUT([2, 16, 16], [16], lambda c, a, b: a if c else b)
I = torch.long
OR8 = LUT([16], [16], lambda a: a | 8)        # set a nibble's top bit
AND7 = LUT([16], [16], lambda a: a & 7)       # clear it
AND3 = LUT([16], [16], lambda a: a & 3)       # clear the top two bits


def num(v, L):
    return torch.tensor([(v >> (4 * i)) & 15 for i in range(L)], dtype=I)


def val(t):
    return sum(int(x) << (4 * i) for i, x in enumerate(t.tolist()))


def to_onehot(t):
    return torch.nn.functional.one_hot(t, 16).float()


def from_onehot(t):
    assert bool((t.sum(-1) == 1).all())
    return t.argmax(-1)


def zeros(*shape):
    return torch.zeros(shape, dtype=I)


def add(A, B, cin=None):
    """A, B [..., L] -> (A + B + cin mod 16^L, carry-out [...])."""
    shp, L = A.shape[:-1], A.shape[-1]
    s0, g = L_ADD.ix(A, B, zeros(*shp, L))
    s1, p = L_ADD.ix(A, B, torch.ones(*shp, L, dtype=I))
    f0, f1 = g, p
    d = 1
    while d < L:
        n0, n1 = COMP.ix(f0[..., :-d], f1[..., :-d], f0[..., d:], f1[..., d:])
        f0 = torch.cat([f0[..., :d], n0], -1)
        f1 = torch.cat([f1[..., :d], n1], -1)
        d *= 2
    cin = zeros(*shp) if cin is None else cin
    c_all = APPLY.ix(f0, f1, cin.unsqueeze(-1).expand(*shp, L))
    c_in = torch.cat([cin.unsqueeze(-1), c_all[..., :-1]], -1)
    return SEL.ix(c_in, s0, s1), c_all[..., -1]


def sub(A, B):
    """(A - B mod 16^L, borrow in {0, 1}) with borrow = 1 iff A < B."""
    D, c = add(A, NOT4.ix(B), torch.ones(A.shape[:-1], dtype=I))
    return D, 1 - c


def pad(A, L):
    if A.shape[-1] >= L:
        return A[..., :L]
    return torch.cat([A, zeros(*A.shape[:-1], L - A.shape[-1])], -1)


def mul(A, B):
    La, Lb = A.shape[0], B.shape[0]
    W = La + Lb
    lo, hi = L_MULN.ix(A[:, None].expand(La, Lb), B[None].expand(La, Lb), zeros(La, Lb))
    rows = zeros(2 * La, W)
    for i in range(La):                              # fixed wiring: row i shifted by i limbs
        rows[2 * i, i:i + Lb] = lo[i]
        rows[2 * i + 1, i + 1:i + 1 + Lb] = hi[i]
    while rows.shape[0] > 1:
        if rows.shape[0] % 2:
            rows = torch.cat([rows, zeros(1, W)])
        rows, _ = add(rows[0::2], rows[1::2])
    return rows[0]


def select(bit, A, B):
    """bit ? A : B, limb by limb through the SEL2 table."""
    b = torch.full(A.shape, int(bit), dtype=I)
    return SEL2.ix(b, A, B)


def eq(A, B):
    return bool((A == B).all())


def is_zero(A):
    return bool((A == 0).all())


def bit_of(A, i):
    return (int(A[i // 4]) >> (i % 4)) & 1


class Field:
    """Integers mod m as k limbs; Barrett reduction with mu = 16^2k // m."""
    def __init__(self, m):
        self.m_int = m
        self.k = k = (m.bit_length() + 3) // 4
        self.m = num(m, k)
        self.mu = num(16 ** (2 * k) // m, k + 1)
        self.mk1 = num(m, k + 1)

    def n(self, v):
        return num(v % self.m_int, self.k)

    def reduce(self, X):
        k = self.k
        X = pad(X, 2 * k)
        q3 = mul(X[k - 1:], self.mu)[k + 1:2 * k + 2]
        r, _ = sub(X[:k + 1], mul(q3, self.mk1)[:k + 1])
        for _ in range(2):
            D, borrow = sub(r, self.mk1)
            r = select(borrow, r, D)
        return r[:k]

    def mul(self, A, B):
        return self.reduce(mul(A, B))

    def sq(self, A):
        return self.mul(A, A)

    def add(self, A, B):
        s, _ = add(pad(A, self.k + 1), pad(B, self.k + 1))
        D, borrow = sub(s, self.mk1)
        return select(borrow, s, D)[:self.k]

    def sub(self, A, B):
        D, borrow = sub(A, B)
        Dp, _ = add(D, self.m)
        return select(borrow, Dp, D)

    def neg(self, A):
        return self.sub(self.n(0), A)

    def pow(self, A, e):
        r = None
        for bit in bin(e)[2:]:
            if r is not None:
                r = self.sq(r)
            if bit == '1':
                r = A if r is None else self.mul(r, A)
        return r if r is not None else self.n(1)

    def inv(self, A):
        return self.pow(A, self.m_int - 2)

    def canonical(self, A):
        """A < 16^k possibly >= m -> A mod m (at most a few subtractions for our moduli)."""
        for _ in range(16):
            D, borrow = sub(pad(A, self.k + 1), self.mk1)
            if borrow:
                return pad(A, self.k)
            A = D
        raise AssertionError

    def lt(self, A, c):
        L = max(A.shape[0], (c.bit_length() + 3) // 4)
        return bool(sub(pad(A, L), num(c, L))[1])


def from_bytes(bs, big=False):
    """bytes as [2, 16] one-hot (low, high nibble) -> limbs, least significant first."""
    seq = list(reversed(bs)) if big else list(bs)
    return torch.stack([from_onehot(b) for b in seq]).flatten()


def to_bytes(A, nbytes, big=False):
    oh = to_onehot(pad(A, 2 * nbytes))
    out = [oh[2 * i:2 * i + 2] for i in range(nbytes)]
    return list(reversed(out)) if big else out
