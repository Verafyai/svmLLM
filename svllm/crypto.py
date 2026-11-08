"""SHA-256, Keccak-256 and base58 as hand-set circuits over one-hot 4-bit limbs.

Words are little-endian limb vectors (limb 0 = lowest nibble). Bit rotations are a fixed
permutation of limbs (the multiple-of-4 part) followed by a two-nibble shift lookup; XOR, AND
and NOT are per-limb lookups; additions are the ripple-carry adder from machine.py.
"""
import torch
from .machine import NIB, LUT, onehot, ZN, word, unword, add, L_XOR, L_AND, P_NOT, const, eq, is_zero

L_SHR = LUT([16, 16, 4], [16], lambda cur, nxt, s: ((cur >> s) | (nxt << (4 - s))) & 15)
S4 = [onehot(s, 4) for s in range(4)]


def xor(*ws):
    out = ws[0]
    for w in ws[1:]:
        out = L_XOR(out, w)
    return out


def band(a, b):
    return L_AND(a, b)


def bnot(a):
    return a @ P_NOT


def rotr(w, r):
    """Rotate right by r bits (w: L limbs, rotation within 4L bits)."""
    L = w.shape[0]
    q, s = divmod(r % (4 * L), 4)
    p = torch.cat([w[q:], w[:q]])
    if s == 0:
        return p
    return L_SHR(p, torch.cat([p[1:], p[:1]]), S4[s].expand(L, 4))


def shr(w, r):
    L = w.shape[0]
    q, s = divmod(r, 4)
    p = torch.cat([w[q:], ZN.expand(q, NIB)])
    if s == 0:
        return p
    return L_SHR(p, torch.cat([p[1:], ZN[None]]), S4[s].expand(L, 4))


def add32(*ws):
    out = ws[0]
    for w in ws[1:]:
        out = add(out, w, onehot(0, 2))[0]
    return out


# ------------------------------------------------------------------ SHA-256
K256 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
KW = [word(k, 8) for k in K256]
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def be_word(bs):
    """Four bytes (each [lo, hi] limbs) big-endian -> a 32-bit word (8 limbs, little-endian)."""
    return torch.cat([bs[3][None].reshape(2, NIB), bs[2].reshape(2, NIB), bs[1].reshape(2, NIB), bs[0].reshape(2, NIB)])


def word_be_bytes(w):
    return [w[6:8], w[4:6], w[2:4], w[0:2]]


def sha256_compress(H, block):
    """H: 8 words; block: 64 bytes (each a [2, 16] tensor). Returns the new H."""
    W = [be_word(block[4 * t:4 * t + 4]) for t in range(16)]
    for t in range(16, 64):
        s0 = xor(rotr(W[t - 15], 7), rotr(W[t - 15], 18), shr(W[t - 15], 3))
        s1 = xor(rotr(W[t - 2], 17), rotr(W[t - 2], 19), shr(W[t - 2], 10))
        W.append(add32(W[t - 16], s0, W[t - 7], s1))
    a, b, c, d, e, f, g, h = H
    for t in range(64):
        S1 = xor(rotr(e, 6), rotr(e, 11), rotr(e, 25))
        ch = xor(band(e, f), band(bnot(e), g))
        t1 = add32(h, S1, ch, KW[t], W[t])
        S0 = xor(rotr(a, 2), rotr(a, 13), rotr(a, 22))
        maj = xor(band(a, b), band(a, c), band(b, c))
        t2 = add32(S0, maj)
        a, b, c, d, e, f, g, h = add32(t1, t2), a, b, c, add32(d, t1), e, f, g
    return [add32(x, y) for x, y in zip(H, [a, b, c, d, e, f, g, h])]


class Sha256:
    """Streaming: absorb one byte at a time; a full 64-byte block runs the 64-round compression."""

    def __init__(self):
        self.H = [word(v, 8) for v in H0]
        self.buf, self.n = [], word(0)                   # n: message length in bytes (circuit counter)

    def byte(self, b):
        self.buf.append(b)
        self.n = add(self.n, word(0), onehot(1, 2))[0]
        if len(self.buf) == 64:
            self.H, self.buf = sha256_compress(self.H, self.buf), []

    def digest(self):
        bits = add(self.n, self.n, onehot(0, 2))[0]       # n * 8, by three doublings
        bits = add(bits, bits, onehot(0, 2))[0]
        bits = add(bits, bits, onehot(0, 2))[0]
        H, buf = self.H, self.buf + [word(0x80, 2)]
        while len(buf) % 64 != 56:
            buf.append(word(0, 2))
        buf += [bits[2 * k:2 * k + 2] for k in range(7, -1, -1)]
        for i in range(0, len(buf), 64):
            H = sha256_compress(H, buf[i:i + 64])
        return [b for w in H for b in word_be_bytes(w)]


# ------------------------------------------------------------------ Keccak-256
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
      0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
      0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
RCW = [word(c) for c in RC]
ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]


def keccak_f(A):
    """A[x][y]: 64-bit lanes (16 limbs). The 24-round Keccak-f[1600] permutation."""
    rotl = lambda w, n: rotr(w, (64 - n) % 64)
    for rnd in range(24):
        C = [xor(A[x][0], A[x][1], A[x][2], A[x][3], A[x][4]) for x in range(5)]
        D = [xor(C[(x - 1) % 5], rotl(C[(x + 1) % 5], 1)) for x in range(5)]
        A = [[xor(A[x][y], D[x]) for y in range(5)] for x in range(5)]
        B = [[None] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = rotl(A[x][y], ROT[x][y])
        A = [[xor(B[x][y], band(bnot(B[(x + 1) % 5][y]), B[(x + 2) % 5][y])) for y in range(5)] for x in range(5)]
        A[0][0] = xor(A[0][0], RCW[rnd])
    return A


class Keccak256:
    RATE = 136

    def __init__(self):
        self.A = [[word(0) for _ in range(5)] for _ in range(5)]
        self.buf = []

    def _absorb(self, blk):
        for i in range(self.RATE // 8):
            lane = torch.cat([blk[8 * i + k].reshape(2, NIB) for k in range(8)])
            x, y = i % 5, i // 5
            self.A[x][y] = xor(self.A[x][y], lane)
        self.A = keccak_f(self.A)

    def byte(self, b):
        self.buf.append(b)
        if len(self.buf) == self.RATE:
            self._absorb(self.buf); self.buf = []

    def digest(self):
        buf = self.buf + [word(0x01, 2)]
        while len(buf) % self.RATE:
            buf.append(word(0, 2))
        buf[-1] = L_XOR(buf[-1], word(0x80, 2))           # the last byte gets 0x80 (it may also be 0x01)
        self._absorb(buf)
        return [self.A[i % 5][i // 5][2 * k:2 * k + 2] for i in range(4) for k in range(8)]


# ------------------------------------------------------------------ base58 (Solana pubkeys)
B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
# one step of long division by 58: (remainder, next nibble) -> (quotient nibble, new remainder)
L_DIV58 = LUT([58, 16], [16, 58], lambda r, n: ((r * 16 + n) // 58, (r * 16 + n) % 58))
L_B58CH = LUT([58], [256], lambda d: ord(B58[d]))


def base58(bs):
    """bs: 32 bytes ([lo, hi] limbs). Returns the characters (256-way one-hots), most significant first."""
    nib = []                                             # big-endian nibbles
    for b in bs:
        nib += [b[1], b[0]]
    zeros = 0
    for b in bs:                                         # leading zero bytes become '1'
        if float(is_zero(b)):
            zeros += 1
        else:
            break
    digits = []
    while not float(is_zero(torch.stack(nib))):
        r, out = onehot(0, 58), []
        for n in nib:
            q, r = L_DIV58(r, n)
            out.append(q)
        nib, digits = out, digits + [r]
    return [onehot(ord('1'), 256)] * zeros + [L_B58CH(d) for d in reversed(digits)]
