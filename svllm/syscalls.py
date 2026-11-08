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
       'sol_set_return_data', 'sol_get_return_data', 'sol_get_stack_height', 'sol_get_clock_sysvar',
       'sol_get_epoch_schedule_sysvar', 'sol_get_rent_sysvar', 'sol_get_epoch_rewards_sysvar', 'sol_get_last_restart_slot',
       'sol_get_sysvar', 'sol_get_epoch_stake', 'sol_alloc_free_', 'sol_get_processed_sibling_instruction',
       'sol_create_program_address', 'sol_try_find_program_address', 'sol_invoke_signed_c', 'sol_invoke_signed_rust',
       'sol_curve_validate_point', 'sol_curve_group_op', 'sol_curve_multiscalar_mul', 'sol_secp256k1_recover',
       'sol_alt_bn128_group_op', 'sol_alt_bn128_compression', 'sol_poseidon', 'sol_curve_decompress',
       'sol_curve_pairing_map']
PDA_MARKER = b'ProgramDerivedAddress'
MM_INPUT_W = word(4 << 32)
HEAP_W = word(32 * 1024)
HEXCH = LUT([16], [256], lambda n: ord('0123456789abcdef'[n]))
DECCH = LUT([16], [256], lambda n: ord('0') + n if n < 10 else ord('?'))
B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
# base64: a 6-bit index arrives as (high 2 bits, low 4 bits) one-hots
B64CH = LUT([4, 16], [256], lambda hi, lo: ord(B64[(hi << 4) | lo]))
SPLIT = LUT([16], [4, 4], lambda n: (n >> 2, n & 3))           # nibble -> (top 2 bits, low 2 bits)
JOIN = LUT([4, 4], [16], lambda hi, lo: (hi << 2) | lo)          # two 2-bit pieces -> nibble
MOD8 = const(torch.eye(16)[[n & 7 for n in range(16)]])     # one-hot nibble -> one-hot (nibble & 7)
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
        return word(len(m.ctx['tx'].stack) if m.ctx else 1)

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
        m.rd_dirty = True
        if m.ctx:
            m.PROGRAM_ID = [word(b, 2) for b in m.ctx['frame'].program]
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

    from .sref import SYSVAR_CALLS, sysvar_struct, SYSVARS
    if name in SYSVAR_CALLS:
        data, align = sysvar_struct(SYSVAR_CALLS[name])          # runtime-supplied bytes
        x.consume(word(100 + len(data)))
        if not float(lt(R1, MM_INPUT_W)):
            raise SysFail('InvalidPointer')
        p = x.xlate(R1, word(len(data)), True, align=align)
        for j, b in enumerate(data):
            x.wbyte(x.at(p, j), word(b, 2))
            if j % 8 == 7:
                yield
        return word(0)

    if name == 'sol_get_sysvar':
        x.consume(add(word(100), maxw(divmod_(R4, word(250))[0], word(10)), onehot(0, 2))[0])
        if not float(lt(R2, MM_INPUT_W)):
            raise SysFail('InvalidPointer')
        x.xlate(R2, R4, True)
        pid = x.xlate(R1, word(32), False)
        sid = [x.rbyte(x.at(pid, j)) for j in range(32)]
        end, c1 = add(R3, R4, onehot(0, 2))
        _, c2 = add(R2, R4, onehot(0, 2))
        if float(OR(c1[1], c2[1])):
            raise SysFail('ArithmeticOverflow')
        # exact-match attention over the sysvar cache: key = the 32-byte id, value = the buffer
        hit = None
        for v in SYSVARS.values():
            if float(sum(eq(a, word(b, 2)) for a, b in zip(sid, v['id'])) == 32.0):
                hit = v['data']
        if hit is None:
            return word(2)
        if float(lt(word(len(hit)), end)):
            return word(1)
        p = x.xlate(R2, R4, True)
        i = word(0)
        while p is not None and not float(eq(i, R4)):
            k = add(R3, i, onehot(0, 2))[0]
            x.wbyte(x.at(p, i), word(hit[unword(k)], 2))       # the buffer is a constant table indexed by k
            i = add(i, word(0), onehot(1, 2))[0]
            yield
        return word(0)

    if name == 'sol_get_epoch_stake':
        if float(is_zero(R1)):
            x.consume(word(100)); return word(0)               # the runtime's epoch stake: 0 in this cluster
        x.consume(word(110)); x.xlate(R1, word(32), False)
        return word(0)

    if name == 'sol_alloc_free_':
        if float(OR(NOT(lt(R1, word(2**63 - 7))), NOT(is_zero(R2)))):
            return word(0)
        pos = m.heap_pos
        pad = sub(word(0), pos)[0][:1]                           # (-pos) mod 8 from the low nibble
        pad = zx(torch.stack([pad[0] @ MOD8]))
        start = add(pos, pad, onehot(0, 2))[0]
        end, c = add(start, R1, onehot(0, 2))
        if float(OR(c[1], lt(HEAP_W, end))):
            return word(0)
        m.heap_pos = end
        return add(word(3 << 32), start, onehot(0, 2))[0]

    if name == 'sol_get_processed_sibling_instruction':
        x.consume(word(100))
        return word(0)                                          # one top-level instruction: no siblings

    if name in ('sol_create_program_address', 'sol_try_find_program_address'):
        from .crypto import Sha256
        from .bignum import is_curve_point
        x.consume(word(1500))
        pv = x.xlate(R1, _times16(R2), False, align=8)
        if float(lt(word(16), R2)):
            raise SysFail('BadSeeds')
        seeds, k = [], word(0)
        while pv is not None and not float(eq(k, R2)):
            base = x.at(pv, _times16(k))
            ptr, ln = x.rword(base), x.rword(x.at(base, 8))
            if float(lt(word(32), ln)):
                raise SysFail('BadSeeds')
            p = x.xlate(ptr, ln, False)
            seed, i = [], word(0)
            while p is not None and not float(eq(i, ln)):
                seed.append(x.rbyte(x.at(p, i)))
                i = add(i, word(0), onehot(1, 2))[0]
            seeds.append(seed)
            k = add(k, word(0), onehot(1, 2))[0]
            yield
        pp = x.xlate(R3, word(32), False)
        pid = [x.rbyte(x.at(pp, j)) for j in range(32)]

        def derive(extra):
            if len(seeds) + len(extra) > 16:
                return None
            h = Sha256()
            for s in seeds + extra:
                for b in s:
                    h.byte(b)
            for b in pid + [word(c, 2) for c in PDA_MARKER]:
                h.byte(b)
            out = h.digest()
            return None if float(is_curve_point(out)) else out

        if name == 'sol_create_program_address':
            addr = derive([])
            yield
            if addr is None:
                return word(1)
            pa = x.xlate(R4, word(32), True)
            for j in range(32):
                x.wbyte(x.at(pa, j), addr[j])
            return word(0)
        bump = word(255, 2)
        for _ in range(255):
            addr = derive([[bump]])
            yield
            if addr is not None:
                pb, pa = x.xlate(R5, word(1), True), x.xlate(R4, word(32), True)
                far = mux(lt(pa, pb), NOT(lt(sub(pb, pa)[0], word(32, AL))), NOT(lt(sub(pa, pb)[0], word(1, AL))))
                if not float(far):
                    raise SysFail('CopyOverlapping')
                x.wbyte(pb, bump)
                for j in range(32):
                    x.wbyte(x.at(pa, j), addr[j])
                return word(0)
            nb, borrow = sub(bump, word(1, 2))
            bump = mux(borrow, word(0, 2), nb)
            x.consume(word(1500))
        return word(1)

    if name in ('sol_invoke_signed_c', 'sol_invoke_signed_rust'):
        from .runtime import _invoke_c, IxError
        from .sref import VMError
        tx = m.ctx['tx']
        export_rd(m)
        cu0 = unword(m.cu)
        base = tx.cu + cu0
        try:
            from . import cpi_w
            cu1 = yield from cpi_w.invoke(x, [None, R1, R2, R3, R4, R5], 'c' if name.endswith('_c') else 'rust')
        except (VMError, IxError) as e:
            m.cu = word(cu0 + (tx.cu - base), m.cu.shape[0])
            tx.cu = base - cu0
            if isinstance(e, IxError):
                m.ixerr = e.name
                raise SysFail('Ix')
            raise SysFail(e.kind)
        m.cu = word(cu1, m.cu.shape[0])
        import_rd(m)
        return word(0)

    if name in CURVE_W:
        r = yield from curve_program(x, name, R1, R2, R3, R4, R5)
        return r

    raise SysFail('Unsupported:' + name)


CURVE_W = ('sol_curve_validate_point', 'sol_curve_group_op', 'sol_curve_multiscalar_mul', 'sol_secp256k1_recover',
           'sol_alt_bn128_group_op', 'sol_alt_bn128_compression', 'sol_poseidon', 'sol_curve_decompress',
           'sol_curve_pairing_map')
BN_OPS_W = {0: (334, 64), 4: (535, 128), 2: (3840, 64), 6: (15670, 128), 3: (None, 32)}
BN_COMP_W = {0: (130, 32), 1: (498, 64), 2: (186, 64), 3: (13710, 128)}
C25519_COST_W = {(0, 'v'): 159, (0, 0): 473, (0, 1): 475, (0, 2): 2177, (0, 'msm'): (2273, 758),
                 (1, 'v'): 169, (1, 0): 521, (1, 1): 519, (1, 2): 2208, (1, 'msm'): (2303, 788)}


def curve_program(x, name, R1, R2, R3, R4, R5):
    """Curve syscalls: Agave's order (cost, input translation, compute, result translation on
    success). The arithmetic is ccurves.py's circuits."""
    from . import ccurves as Q
    isv = lambda R, k: bool(float(eq(R, word(k))))
    which = lambda R, ks: next((k for k in ks if isv(R, k)), None)

    def rd(addr, n):
        p = x.xlate(addr, word(n), False)
        return [x.rbyte(x.at(p, j)) for j in range(n)]

    def wr(addr, bs):
        p = x.xlate(addr, word(len(bs)), True)
        for j, b in enumerate(bs):
            x.wbyte(x.at(p, j), b)

    if name in ('sol_alt_bn128_group_op', 'sol_alt_bn128_compression'):
        table = BN_OPS_W if name == 'sol_alt_bn128_group_op' else BN_COMP_W
        op = unword(R1)
        if op & ~0x80 not in table:
            raise SysFail('InvalidAttribute')
        cost, size = table[op & ~0x80]
        n = unword(R3)
        if cost is None:
            cost = min(36364 + 12121 * max(n // 192 - 1, 0) + 85 + n + 32, 2**64 - 1)
        x.consume(word(cost))
        x.xlate(R4, word(size), True)
        p = x.xlate(R2, R3, False)
        inp = [x.rbyte(x.at(p, j)) for j in range(n)] if p is not None else []
        yield
        code, out = (Q.bn_group_op if name == 'sol_alt_bn128_group_op' else Q.bn_compression)(op, inp)
        yield
        if code:
            return word(1)
        wr(R4, out)
        return word(0)
    if name == 'sol_poseidon':
        if not isv(R1, 0) or which(R2, (0, 1)) is None:
            raise SysFail('PoseidonError')
        if float(lt(word(12), R4)):
            raise SysFail('InvalidLength')
        n = unword(R4)
        x.consume(word(61 * n * n + 542))
        x.xlate(R5, word(32), True)
        vals = []
        if n:
            dp = x.xlate(R3, word(16 * n), False, align=8)
            descs = [(x.rword(x.at(dp, 16 * i)), x.rword(x.at(dp, 16 * i + 8))) for i in range(n)]
            for ptr, ln in descs:
                q = x.xlate(ptr, ln, False)
                vals.append([x.rbyte(x.at(q, j)) for j in range(unword(ln))] if q is not None else [])
        yield
        code, out = Q.poseidon_hashv(unword(R2), vals)
        yield
        if code:
            return word(1)
        wr(R5, out)
        return word(0)
    bls = which(R1, (5, 6, 0x85, 0x86, 4, 0x84))
    bk = None if bls is None else bls & 0x7f
    if name == 'sol_curve_validate_point' and bk in (5, 6):
        n = 96 if bk == 5 else 192
        x.consume(word(1565 if bk == 5 else 1968))
        pt = rd(R2, n)
        yield
        return word(Q.bls_syscall('validate', bls, [pt])[0])
    if name == 'sol_curve_group_op' and bk in (5, 6):
        op = which(R2, (0, 1, 2))
        if op is None:
            raise SysFail('InvalidAttribute')
        n = 96 if bk == 5 else 192
        x.consume(word({(5, 0): 128, (5, 1): 129, (5, 2): 4627, (6, 0): 203, (6, 1): 204, (6, 2): 8255}[(bk, op)]))
        left, right = rd(R3, 32 if op == 2 else n), rd(R4, n)
        yield
        code, out = Q.bls_syscall({0: 'add', 1: 'sub', 2: 'mul'}[op], bls, [left, right])
        yield
        if code:
            return word(1)
        wr(R5, out)
        return word(0)
    if name == 'sol_curve_decompress':
        if bk not in (5, 6):
            raise SysFail('InvalidAttribute')
        n = 48 if bk == 5 else 96
        x.consume(word(2100 if bk == 5 else 3050))
        inp = rd(R2, n)
        yield
        code, out = Q.bls_syscall('decompress', bls, [inp])
        if code:
            return word(1)
        wr(R3, out)
        return word(0)
    if name == 'sol_curve_pairing_map':
        if bk != 4:
            raise SysFail('InvalidAttribute')
        n = unword(R2)
        x.consume(word(min(25445 + 13023 * max(n - 1, 0), 2**64 - 1)))
        if n * 96 >= 2**63:
            raise SysFail('InvalidLength')
        gp, hp = x.xlate(R3, word(96 * n), False), x.xlate(R4, word(192 * n), False)
        g1s = [[x.rbyte(x.at(gp, 96 * i + j)) for j in range(96)] for i in range(n)]
        g2s = [[x.rbyte(x.at(hp, 192 * i + j)) for j in range(192)] for i in range(n)]
        yield
        code, out = Q.bls_syscall('pairing', bls, [g1s, g2s])
        yield
        if code:
            return word(1)
        wr(R5, out)
        return word(0)
    if name == 'sol_secp256k1_recover':
        x.consume(word(25000))
        x.xlate(R4, word(64), True)
        h, sig = rd(R1, 32), rd(R3, 64)
        yield
        recid = unword(R2)                     # compared only against 4 and 256 (two borrow circuits)
        code, pub = Q.secp256k1_recover(h, recid, sig)
        yield
        if code:
            return word(code)
        wr(R4, pub)
        return word(0)
    cid = which(R1, (0, 1))
    if name == 'sol_curve_validate_point':
        if cid is None:
            raise SysFail('InvalidAttribute')
        x.consume(word(C25519_COST_W[(cid, 'v')]))
        pt = rd(R2, 32)
        yield
        return word(0 if Q.c25519_validate(cid, pt) else 1)
    if name == 'sol_curve_group_op':
        op = which(R2, (0, 1, 2))
        if cid is None or op is None:
            raise SysFail('InvalidAttribute')
        x.consume(word(C25519_COST_W[(cid, op)]))
        left, right = rd(R3, 32), rd(R4, 32)
        yield
        out = Q.c25519_group(cid, op, left, right)
        yield
        if out is None:
            return word(1)
        wr(R5, out)
        return word(0)
    # sol_curve_multiscalar_mul(curve, scalars, points, n, result)
    if float(lt(word(512), R4)):
        raise SysFail('InvalidLength')
    if cid is None:
        raise SysFail('InvalidAttribute')
    base, inc = C25519_COST_W[(cid, 'msm')]
    n = unword(R4)
    x.consume(word(base + inc * max(n - 1, 0)))
    sp = x.xlate(R2, word(32 * n), False)
    pp = x.xlate(R3, word(32 * n), False)
    ss = [[x.rbyte(x.at(sp, 32 * i + j)) for j in range(32)] for i in range(n)]
    ps = [[x.rbyte(x.at(pp, 32 * i + j)) for j in range(32)] for i in range(n)]
    yield
    out = Q.c25519_msm(cid, ss, ps)
    yield
    if out is None:
        return word(1)
    wr(R5, out)
    return word(0)


class CircuitPort:
    """The caller's memory as the CPI translation sees it: every access is the translate circuit
    (bounds, permissions, alignment on the region offset) and DeltaNet byte reads and writes."""
    def __init__(self, x):
        self.x = x

    def read(self, addr, n, align=1, store=False):
        from .sref import VMError
        if n == 0:
            return b''
        try:
            p = self.x.xlate(word(addr % 2**64), word(min(n, 2**64 - 1)), store, align=align)
        except SysFail as f:
            raise VMError(f.kind)
        return bytes(unword(self.x.rbyte(self.x.at(p, k))) for k in range(n))

    def write(self, addr, data, align=1):
        from .sref import VMError
        if not data:
            return
        try:
            p = self.x.xlate(word(addr % 2**64), word(len(data)), True, align=align)
        except SysFail as f:
            raise VMError(f.kind)
        for k, b in enumerate(data):
            self.x.wbyte(self.x.at(p, k), word(b, 2))


def export_rd(m):
    """Machine return data -> the transaction (Agave keeps one return-data slot per transaction)."""
    if m.ctx and m.rd_dirty:
        tx = m.ctx['tx']
        n = unword(m.rdlen)
        tx.rdd['data'] = bytes(unword(_rdbyte(m, k)) for k in range(n))
        tx.rdd['pid'] = m.ctx['frame'].program
        m.rd_dirty = False


def import_rd(m):
    if m.ctx:
        tx = m.ctx['tx']
        m.rdata, m.rdlen = __import__('svllm.svm', fromlist=['PagedRAM']).PagedRAM(32), word(len(tx.rdd['data']))
        for k, b in enumerate(tx.rdd['data']):
            m.rdata.write(word(k, AL), word(b, 2).flatten(), torch.tensor(1.0))
        m.PROGRAM_ID = [word(b, 2) for b in tx.rdd['pid']]


def _rdbyte(m, k):
    b = m.rdata.read(word(k, AL)).view(2, NIB)
    return b + ZN * (1.0 - b.sum(-1, keepdim=True))


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
