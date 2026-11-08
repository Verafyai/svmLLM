"""Big-integer field arithmetic as circuits: numbers are vectors of one-hot 4-bit limbs, and every
operation is the lookup-table adders and nibble multipliers from machine.py, batched across rows
(partial products are formed for all 64 rows at once and summed as a tree), so a 256-bit multiply
is about a thousand batched table lookups rather than eight thousand sequential ones.
"""
import torch
from .machine import NIB, onehot, ZN, L_ADD, L_MULN, P_NOT, word, unword, LUT

P25519 = 2**255 - 19
D25519 = (-121665 * pow(121666, P25519 - 2, P25519)) % P25519


def num(v, L=64):
    return word(v, L)


def val(t):
    return unword(t)


def add_rows(A, B):
    """Batched ripple-carry add: A, B [batch, L, 16] -> sum [batch, L, 16], carry-out [batch, 2]."""
    out, c = [], onehot(0, 2).expand(A.shape[0], 2)
    for i in range(A.shape[1]):
        s, c = L_ADD(A[:, i], B[:, i], c)
        out.append(s)
    return torch.stack(out, 1), c


def mul_full(A, B):
    """A [La, 16] x B [Lb, 16] -> product [La + Lb, 16], exact."""
    La, Lb = A.shape[0], B.shape[0]
    W = La + Lb
    c = ZN.expand(Lb, NIB)
    rows = []
    for i in range(La):                                   # all Lb rows at once: A[i] * B[j] + carry
        lo, c = L_MULN(A[i].expand(Lb, NIB), B, c)
        rows.append(lo)
    rows.append(c)
    R = torch.stack(rows, 1)                              # [Lb, La + 1, 16]; row j is A * B[j]
    pad = torch.zeros(Lb, W, NIB)
    pad[:, :, 0] = 1.0
    for j in range(Lb):                                   # shift row j by j limbs (a fixed wiring)
        pad[j, j:j + La + 1] = R[j]
    rowsW = pad
    while rowsW.shape[0] > 1:                             # tree of batched adds
        if rowsW.shape[0] % 2:
            z = torch.zeros(1, W, NIB); z[:, :, 0] = 1.0
            rowsW = torch.cat([rowsW, z])
        rowsW, _ = add_rows(rowsW[0::2], rowsW[1::2])
    return rowsW[0]


def add(A, B):
    s, c = add_rows(A[None], B[None])
    return s[0], c[0]


def sub(A, B):
    s, c = add_rows(A[None], (B @ P_NOT)[None])
    # add with carry-in 1 = two's complement subtraction; redo with cin 1
    return _sub(A, B)


def _sub(A, B):
    out, c = [], onehot(1, 2)
    Bn = B @ P_NOT
    for i in range(A.shape[0]):
        s, c = L_ADD(A[i], Bn[i], c)
        out.append(s)
    return torch.stack(out), c[0]                         # borrow = 1 if A < B


MUL38 = None


def small_mul(A, k):
    """A [L, 16] times a small constant k < 16 -> [L + 1, 16]."""
    c, out = ZN, []
    kk = onehot(k, NIB)
    for i in range(A.shape[0]):
        lo, c = L_MULN(A[i], kk, c)
        out.append(lo)
    return torch.cat([torch.stack(out), c[None]])


def reduce25519(X):
    """X [L, 16] (any length) -> X mod 2^255 - 19, canonical, [64, 16]. Uses 2^256 = 38 (mod p)."""
    P = num(P25519)
    while X.shape[0] > 64:
        lo, hi = X[:64], X[64:]
        # 38 * hi = 32 * hi + 4 * hi + 2 * hi, as constant multiplies of hi (two nibble multiplies)
        h38 = mul_full(hi, num(38, 2))
        L = max(64, h38.shape[0]) + 1
        pad = lambda t: torch.cat([t, ZN.expand(L - t.shape[0], NIB)])
        X, _ = add(pad(lo), pad(h38))
        # drop high zero limbs is not allowed (data-dependent); keep folding until 64 limbs
        if X.shape[0] > 64 and X.shape[0] <= 66:
            lo, hi = X[:64], X[64:]
            h38 = mul_full(hi, num(38, 2))
            pad2 = lambda t: torch.cat([t, ZN.expand(65 - t.shape[0], NIB)])
            X, _ = add(pad2(lo), pad2(h38))
            # one more fold for the final carry limb (value <= 1)
            lo, hi = X[:64], X[64:]
            h38 = mul_full(hi, num(38, 2))[:64] if hi.shape[0] else ZN.expand(64, NIB)
            h38 = torch.cat([h38, ZN.expand(64 - h38.shape[0], NIB)]) if h38.shape[0] < 64 else h38[:64]
            X, _ = add(lo, h38)
    # now X < 2^256; subtract p while X >= p (at most twice)
    for _ in range(2):
        D, borrow = _sub(X, P)
        X = D * (1.0 - borrow) + X * borrow
    return X


def fmul(A, B):
    return reduce25519(mul_full(A, B))


def fsq(A):
    return fmul(A, A)


def fadd(A, B):
    s, c = add(A, B)
    return reduce25519(torch.cat([s, torch.stack([c[0] * ZN + c[1] * onehot(1, NIB)])]))


def fsub(A, B):
    """A - B mod p, for canonical A, B."""
    D, borrow = _sub(A, B)
    Dp, _ = add(D, num(P25519))
    return Dp * borrow + D * (1.0 - borrow)


def fpow_2k(A, k):
    for _ in range(k):
        A = fsq(A)
    return A


def legendre_ok(W):
    """True (1.0) iff W is a square mod p (W^((p-1)/2) is 0 or 1). Addition chain for 2^254 - 10."""
    z2 = fsq(W)                                   # 2
    z9 = fmul(fpow_2k(z2, 2), W)                  # 9
    z11 = fmul(z9, z2)                            # 11
    z_5_0 = fmul(fsq(z11), z9)                    # 2^5 - 1
    z_10_0 = fmul(fpow_2k(z_5_0, 5), z_5_0)
    z_20_0 = fmul(fpow_2k(z_10_0, 10), z_10_0)
    z_40_0 = fmul(fpow_2k(z_20_0, 20), z_20_0)
    z_50_0 = fmul(fpow_2k(z_40_0, 10), z_10_0)
    z_100_0 = fmul(fpow_2k(z_50_0, 50), z_50_0)
    z_200_0 = fmul(fpow_2k(z_100_0, 100), z_100_0)
    z_250_0 = fmul(fpow_2k(z_200_0, 50), z_50_0)  # 2^250 - 1
    z6 = fmul(fsq(z2), z2)                        # 6
    e = fmul(fpow_2k(z_250_0, 4), z6)             # (2^250 - 1) * 16 + 6 = 2^254 - 10
    one = num(1)
    is_one = float((e * one).sum() == 64.0)
    is_zero = float(e[:, 0].sum() == 64.0)
    return torch.tensor(max(is_one, is_zero))


def is_curve_point(bs):
    """32 bytes ([lo, hi] limbs each) -> 1.0 if they decompress to an ed25519 point (dalek rules:
    the sign bit is ignored for validity and y need not be reduced)."""
    limbs = []
    for k, b in enumerate(bs):
        lo, hi = b[0], b[1]
        if k == 31:
            hi = hi @ torch.eye(16)[[n & 7 for n in range(16)]]          # clear the top bit
        limbs += [lo, hi]
    y = reduce25519(torch.stack(limbs))
    y2 = fsq(y)
    u = fsub(y2, num(1))
    v = fadd(fmul(num(D25519), y2), num(1))
    return legendre_ok(fmul(u, v))
