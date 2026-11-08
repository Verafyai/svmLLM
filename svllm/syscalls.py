"""Solana syscalls in the weights (Agave 4.3.2 rules), as microprograms.

Each syscall is a short microprogram. Its datapath is circuits: compute charges (max, divide
by 250, saturating sums), the budget check, address translation, alignment and overlap checks,
and every byte it moves, compares, hashes or formats (hex, decimal, base64, UTF-8 validation).
The sequencing (which micro-step comes next) is a Python generator that reads exact 0/1
signals out of those circuits, the same way the driver loop reads the halt flag: one
`next()` is one micro-step. Micro-steps cost no compute units; Agave charges up front.

Constants supplied by the runtime, not computed: the program id (32 bytes) and the fixed text
the log collector adds ("Program log: ", "Program data: ", ...).
"""
import torch
from .machine import (NIB, onehot, LUT, AND, NOT, OR, gate, mux, ZN, word, unword, add, sub, eq, is_zero,
                      divmod_, zx, const)

AL = 7
SYS = ['abort', 'sol_log_', 'sol_memcpy_', 'sol_memset_', 'sol_memmove_', 'sol_memcmp_', 'sol_log_64_',
       'sol_log_compute_units_', 'sol_panic_', 'sol_sha256', 'sol_keccak256', 'sol_log_pubkey', 'sol_log_data',
       'sol_set_return_data', 'sol_get_return_data', 'sol_get_stack_height']
HEXCH = LUT([16], [256], lambda n: ord('0123456789abcdef'[n]))
DECCH = LUT([16], [256], lambda n: ord('0') + n if n < 10 else ord('?'))
B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
# base64: a 6-bit index arrives as (high 2 bits, low 4 bits) one-hots
B64CH = LUT([4, 16], [256], lambda hi, lo: ord(B64[(hi << 4) | lo]))
SPLIT = LUT([16], [4, 4], lambda n: (n >> 2, n & 3))           # nibble -> (top 2 bits, low 2 bits)
JOIN = LUT([4, 4], [16], lambda hi, lo: (hi << 2) | lo)          # two 2-bit pieces -> nibble
ALIGN = {a: const([1.0 if i % a == 0 else 0.0 for i in range(16)]) for a in (1, 2, 4, 8)}


def lt(a, b):
    return sub(a, b)[1]


def maxw(a, b):
    return mux(lt(a, b), b, a)


def chars(s):
    return [onehot(c, 256) for c in s.encode()]


class SysFail(Exception):
    def __init__(self, kind):
        super().__init__(kind); self.kind = kind


class Ctx:
    """Circuit helpers over the machine's memory and meter."""

    def __init__(self, m):
        self.m = m

    def consume(self, c):                          # consume_checked: fails if c > remaining, draining it
        m = self.m
        remaining, _ = sub(zx(m.limit), zx(m.cu))
        if float(lt(remaining, c)):
            m.cu = m.limit
            raise SysFail('ComputationalBudgetExceeded')
        m.cu = add(zx(m.cu), c, onehot(0, 2))[0][:m.cu.shape[0]]

    def xlate(self, addr, n, store, align=1):
        """translate_slice / translate_type: (physical address, region offset). Zero length skips."""
        if float(is_zero(n)):
            return None
        ok, phys, roff = self.m.translate(addr, n, torch.tensor(float(store)), with_offset=True)
        if not float(ok):
            raise SysFail('AccessViolation')
        if align > 1 and not float(roff[0] @ ALIGN[align]):
            raise SysFail('UnalignedPointer')
        return phys

    def at(self, phys, k):
        return add(phys, word(k, AL), onehot(0, 2))[0] if not isinstance(k, torch.Tensor) else add(phys, k[:AL], onehot(0, 2))[0]

    def rbyte(self, phys):
        b = self.m.mem.read(phys).view(2, NIB)
        return b + ZN * (1.0 - b.sum(-1, keepdim=True))

    def wbyte(self, phys, b):
        self.m.mem.write(phys, b.flatten(), torch.tensor(1.0))

    def rword(self, phys, nbytes=8):
        return zx(torch.cat([self.rbyte(self.at(phys, k)) for k in range(nbytes)]))

    def emit(self, ch):                            # the log output port: one byte, one-hot
        self.m.line.append(int(ch.argmax()))

    def emit_hex(self, v):                         # {:#x}: 0x then digits without leading zeros
        for c in chars('0x'):
            self.emit(c)
        started = torch.tensor(0.0)
        for j in range(15, -1, -1):
            started = OR(started, OR(NOT(v[j][0]), torch.tensor(float(j == 0))))
            if float(started):
                self.emit(HEXCH(v[j]))

    def emit_dec(self, v):
        """v < 2^24 (the meter's width): eight digits by 24-stage long division, leading zeros dropped."""
        digits, x = [], v[:6]
        for _ in range(8):
            x, r = divmod_n(x, word(10, 6))
            digits.append(r[0])
        started = torch.tensor(0.0)
        for j in range(7, -1, -1):
            started = OR(started, OR(NOT(digits[j][0]), torch.tensor(float(j == 0))))
            if float(started):
                self.emit(DECCH(digits[j]))

    def b64_group(self, bs):
        """1 to 3 bytes (each 2 one-hot nibbles) -> 4 base64 characters with '=' padding."""
        n = len(bs)
        bs = bs + [word(0, 2)] * (3 - n)
        (a_lo, a_hi), (b_lo, b_hi), (c_lo, c_hi) = [(b[0], b[1]) for b in bs]
        ah2, al2 = SPLIT(a_hi); al_h, al_l = SPLIT(a_lo)
        bh2, bl2 = SPLIT(b_hi); bl_h, bl_l = SPLIT(b_lo)
        ch2, cl2 = SPLIT(c_hi)
        idx = [(ah2, JOIN(al2, al_h)), (al_l, b_hi), (bl_h, JOIN(bl_l, ch2)), (cl2, c_lo)]
        for i, (hi, lo) in enumerate(idx):
            self.emit(B64CH(hi, lo) if i <= n else onehot(ord('='), 256))

    def flush(self, prefix):
        self.m.logs.append(prefix + bytes(self.m.line).decode()); self.m.line = []


def regs(m, *rs):
    return [m.regs.read(onehot(r, 16)).view(16, NIB) for r in rs]


def program(m, name):
    """The microprogram for one syscall. Yields once per micro-step; returns r0 (a word)."""
    x = Ctx(m)
    R1, R2, R3, R4, R5 = regs(m, 1, 2, 3, 4, 5)
    if name == 'abort':
        raise SysFail('Abort')

    if name in ('sol_log_', 'sol_panic_'):
        x.consume(maxw(word(100), R2) if name == 'sol_log_' else R2)
        p = x.xlate(R1, R2, False)
        dfa, i = onehot(0, 9), word(0)
        from .svm import L_UTF8
        while p is not None and not float(eq(i, R2)):
            b = x.rbyte(x.at(p, i))
            dfa = L_UTF8(dfa, b[0], b[1])
            if name == 'sol_log_':
                x.emit(onehot(unword(b), 256))
            i = add(i, word(0), onehot(1, 2))[0]
            yield
        if not float(dfa[0]):
            m.line = []
            raise SysFail('InvalidString')
        if name == 'sol_panic_':
            raise SysFail('Panic')
        x.flush('Program log: ')
        return word(0)

    if name == 'sol_log_64_':
        x.consume(word(100))
        for k, v in enumerate((R1, R2, R3, R4, R5)):
            if k:
                for c in chars(', '):
                    x.emit(c)
            x.emit_hex(v)
            yield
        x.flush('Program log: ')
        return word(0)

    if name == 'sol_log_compute_units_':
        x.consume(word(100))
        rem, _ = sub(zx(m.limit), zx(m.cu))
        x.emit_dec(rem)
        x.flush('Program consumption: ')
        m.logs[-1] += ' units remaining'
        return word(0)

    if name == 'sol_get_stack_height':
        x.consume(word(100))
        return word(1)

    if name in ('sol_memcpy_', 'sol_memmove_', 'sol_memset_', 'sol_memcmp_'):
        x.consume(maxw(word(10), divmod_(R3, word(250))[0]))
        n = R3
        if name == 'sol_memcpy_':
            d1, b1 = sub(R2, R1)
            d2, _ = sub(R1, R2)
            far = mux(AND(NOT(b1), NOT(eq(R1, R2))), NOT(lt(d1, n)), NOT(lt(d2, n)))
            if not float(far):
                raise SysFail('CopyOverlapping')
        if name == 'sol_memset_':
            pd = x.xlate(R1, n, True)
        elif name == 'sol_memcmp_':
            pa, pb = x.xlate(R1, n, False), x.xlate(R2, n, False)
            pr = x.xlate(R4, word(4), True, align=4)
        else:
            pd, ps = x.xlate(R1, n, True), x.xlate(R2, n, False)
        # memmove copies backwards when the destination is above the source
        back = lt(ps, pd) if name == 'sol_memmove_' and pd is not None else torch.tensor(0.0)
        i, diff = word(0), word(0)
        while not float(eq(i, n)):
            k = sub(sub(n, i)[0], word(1))[0] if float(back) else i
            if name == 'sol_memset_':
                x.wbyte(x.at(pd, k), R2[:2])
            elif name == 'sol_memcmp_':
                a, b = x.rbyte(x.at(pa, k)), x.rbyte(x.at(pb, k))
                if not float(eq(a, b)):
                    diff = sub(zx(a), zx(b))[0]
                    break
            else:
                x.wbyte(x.at(pd, k), x.rbyte(x.at(ps, k)))
            i = add(i, word(0), onehot(1, 2))[0]
            yield
        if name == 'sol_memcmp_':
            for k in range(4):
                x.wbyte(x.at(pr, k), diff[2 * k:2 * k + 2])
        return word(0)

    if name in ('sol_sha256', 'sol_keccak256'):
        from .crypto import Sha256, Keccak256, shr
        if float(lt(word(20000), R2)):
            raise SysFail('TooManySlices')
        x.consume(word(85))
        x.xlate(R3, word(32), True)                        # checked before hashing, as Agave does
        h = Sha256() if name == 'sol_sha256' else Keccak256()
        pv = x.xlate(R1, _times16(R2), False, align=8)
        k = word(0)
        while pv is not None and not float(eq(k, R2)):
            base = x.at(pv, _times16(k))
            ptr, ln = x.rword(base), x.rword(x.at(base, 8))
            p = x.xlate(ptr, ln, False)
            x.consume(maxw(word(10), shr(ln, 1)))
            i = word(0)
            while p is not None and not float(eq(i, ln)):
                h.byte(x.rbyte(x.at(p, i)))
                i = add(i, word(0), onehot(1, 2))[0]
                yield
            k = add(k, word(0), onehot(1, 2))[0]
        out = h.digest()
        pr = x.xlate(R3, word(32), True)
        for j in range(32):
            x.wbyte(x.at(pr, j), out[j])
        return word(0)

    if name == 'sol_log_pubkey':
        from .crypto import base58
        x.consume(word(100))
        p = x.xlate(R1, word(32), False)
        for c in base58([x.rbyte(x.at(p, j)) for j in range(32)]):
            x.emit(c)
        x.flush('Program log: ')
        return word(0)

    if name == 'sol_log_data':
        x.consume(word(100))
        if not float(lt(R2, word(1 << 59))):
            raise SysFail('InvalidLength')
        pv = x.xlate(R1, _times16(R2), False, align=8)
        x.consume(_sat_mul100(R2))
        descs, total, k = [], word(0), word(0)
        while pv is not None and not float(eq(k, R2)):     # read the (ptr, len) descriptors
            base = x.at(pv, _times16(k))
            ptr, ln = x.rword(base), x.rword(x.at(base, 8))
            descs.append((ptr, ln))
            s, c = add(total, ln, onehot(0, 2))
            total = mux(c[1], word(2**64 - 1), s)               # saturating
            k = add(k, word(0), onehot(1, 2))[0]
            yield
        x.consume(total)
        fields = [(x.xlate(p, ln, False), ln) for p, ln in descs]
        for fi, (p, ln) in enumerate(fields):
            if fi:
                x.emit(onehot(ord(' '), 256))
            i = word(0)
            while p is not None and not float(eq(i, ln)):
                grp, j = [], i
                for _ in range(3):
                    if float(eq(j, ln)):
                        break
                    grp.append(x.rbyte(x.at(p, j)))
                    j = add(j, word(0), onehot(1, 2))[0]
                x.b64_group(grp)
                i = j
                yield
        x.flush('Program data: ')
        return word(0)

    if name == 'sol_set_return_data':
        x.consume(add(divmod_(R2, word(250))[0], word(100), onehot(0, 2))[0])
        if not float(NOT(lt(word(1024), R2))):
            raise SysFail('ReturnDataTooLarge')
        p = x.xlate(R1, R2, False)
        i = word(0)
        while p is not None and not float(eq(i, R2)):
            m.rdata.write(i[:AL], x.rbyte(x.at(p, i)).flatten(), torch.tensor(1.0))
            i = add(i, word(0), onehot(1, 2))[0]
            yield
        m.rdlen = R2
        return word(0)

    if name == 'sol_get_return_data':
        x.consume(word(100))
        n = mux(lt(R2, m.rdlen), R2, m.rdlen)
        if not float(is_zero(n)):
            x.consume(divmod_(add(n, word(32), onehot(0, 2))[0], word(250))[0])
            pd = x.xlate(R1, n, True)
            pp = x.xlate(R3, word(32), True)
            # translate_mut! refuses two overlapping mutable borrows
            far = mux(lt(pp, pd), NOT(lt(sub(pd, pp)[0], word(32, AL))), NOT(lt(sub(pp, pd)[0], n[:AL])))
            if not float(far):
                raise SysFail('CopyOverlapping')
            i = word(0)
            while not float(eq(i, n)):
                b = m.rdata.read(i[:AL]).view(2, NIB)
                x.wbyte(x.at(pd, i), b + ZN * (1.0 - b.sum(-1, keepdim=True)))
                i = add(i, word(0), onehot(1, 2))[0]
                yield
            for k in range(32):
                x.wbyte(x.at(pp, k), m.PROGRAM_ID[k])
        return m.rdlen

    raise SysFail('Unsupported:' + name)


def divmod_n(A, B):
    """Restoring division on L-limb words, 4L bit-serial stages."""
    from .machine import shl1
    L = A.shape[0]
    Q, R = word(0, L), word(0, L)
    for _ in range(4 * L):
        A, abit = shl1(A, onehot(0, 2))
        R, _ = shl1(R, abit)
        T, borrow = sub(R, B)
        q = NOT(borrow)
        R = mux(q, T, R)
        Q, _ = shl1(Q, torch.stack([borrow, q]))
    return Q, R


def _times16(v):
    return zx(torch.cat([ZN[None], v[:15]]))                # shift left by one nibble


def _sat_mul100(v):
    """100 * v, saturating at 2^64 - 1."""
    hi = NOT(lt(v, word(184467440737095517)))              # v * 100 would overflow
    return mux(hi, word(2**64 - 1), _mulc(v, 100))


def _mulc(v, c):
    from .machine import mul
    return mul(v, word(c))
