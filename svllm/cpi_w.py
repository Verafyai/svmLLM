"""sol_invoke_signed_c and sol_invoke_signed_rust in the weights.

The same steps as runtime._invoke_c (Agave 4.3.2's cpi_common), in the same order, with every
value one-hot and every decision a circuit output: struct fields are read from the caller's memory
through the translate circuit and DeltaNet RAM; sizes, costs and bounds are word adders,
multipliers, dividers and comparators; keys are compared with 32-byte equality circuits; PDA
signers are SHA-256 circuits plus the Ed25519 decompression circuits; BorrowedAccount's checks
(ownership, writability, resize limits) are circuits over the account rows.

The transaction context (the account table, the instruction stack, the meter total) is the host's
state, as it is in Agave. It is read into one-hot rows when the syscall starts and written back
when an account changes; the decisions in between are circuits. Running the callee is the
driver's job (Tx.execute), the same as running the caller.
"""
import torch
from .machine import onehot, word, unword, add, sub, eq, is_zero, NOT, OR, AND, mux, divmod_, mul, ZN, NIB
from .syscalls import SysFail, lt, AL
from . import runtime as RT
from .sref import VMError

M64W = word(2**64 - 1)
MM_INPUT = word(RT.MM_INPUT)
ONE2 = onehot(1, 2)
ZERO2 = onehot(0, 2)


def kb(bs):
    """Host bytes -> one-hot bytes (reading a row of the context table)."""
    return [word(b, 2) for b in bs]


def flat(bs):
    return torch.stack(bs).reshape(-1, NIB)


def keq(a, b):
    return bool(float(eq(flat(a), flat(b))))


def unbytes(bs):
    return bytes(unword(b) for b in bs)


def inc(w):
    return add(w, word(0, w.shape[0]), ONE2)[0]


def wadd(a, b):
    return add(a, b, ZERO2)[0]


def wlt(a, c):
    return bool(float(lt(a, c if isinstance(c, torch.Tensor) else word(c))))


def weq(a, c):
    return bool(float(eq(a, c if isinstance(c, torch.Tensor) else word(c))))


def div250(a):
    return divmod_(a, word(RT.CPI_BPU))[0]


class Meter:
    """The transaction's compute meter, read and written as one-hot words (the host keeps the total)."""
    def __init__(self, tx):
        self.tx = tx

    def consume(self, c):
        used, lim = word(self.tx.cu), word(self.tx.limit)
        rem, _ = sub(lim, used)
        if wlt(rem, c):
            self.tx.cu = self.tx.limit
            raise RT.IxError('ComputationalBudgetExceeded')
        self.tx.cu = unword(wadd(used, c))


class Mem:
    """The caller's memory through the translate circuit and DeltaNet RAM."""
    def __init__(self, x):
        self.x = x

    def tr(self, addr, n, store=False, align=1):
        try:
            return self.x.xlate(addr, n, store, align=align)
        except SysFail as f:
            raise VMError(f.kind)

    def bytes_at(self, addr, n_int, store=False, align=1):
        if n_int == 0:
            return []
        p = self.tr(addr, word(n_int), store, align)
        return [self.x.rbyte(self.x.at(p, k)) for k in range(n_int)]

    def word_at(self, addr, align=8, store=False):
        p = self.tr(addr, word(8), store, align)
        return self.x.rword(p)

    def write(self, addr, bs, align=1):
        if not bs:
            return
        p = self.tr(addr, word(len(bs)), True, align)
        for k, b in enumerate(bs):
            self.x.wbyte(self.x.at(p, k), b)

    def slice(self, addr, n, esz, align):
        """translate_slice(addr, n elements of esz bytes): length check, then translation."""
        if weq(n, 0):
            return None
        if not wlt(n, (2**63 + esz - 1) // esz):               # n * esz >= 2^63
            raise VMError('InvalidLength')
        return self.tr(addr, mul(n, word(esz)), False, align)


def field(x, p, off):
    return x.rword(x.at(p, off))


def invoke(x, R, abi):
    """Generator: the CPI syscall. R = [None, R1..R5] one-hot words. Returns compute units used
    by this syscall and its callee (host int, as the caller's meter delta), like _invoke_c."""
    m = x.m
    ctx = m.ctx
    tx, frame, meta = ctx['tx'], ctx['frame'], ctx['meta']
    cu0 = unword(m.cu)
    tx.cu += cu0
    base = tx.cu
    meter, mem = Meter(tx), Mem(x)
    R1, R2, R3, R4, R5 = R[1:]

    meter.consume(word(RT.INVOKE_UNITS))
    # ---- the instruction
    if abi == 'c':
        p = mem.tr(R1, word(40), False, 8)
        pid_addr, accs_addr, accs_len, data_addr, data_len = [field(x, p, 8 * i) for i in range(5)]
        program_id = mem.bytes_at(pid_addr, 32)
        mb = mem.slice(accs_addr, accs_len, 16, 8)
    else:
        p = mem.tr(R1, word(80), False, 8)
        accs_addr, accs_len, data_addr, data_len = field(x, p, 0), field(x, p, 16), field(x, p, 24), field(x, p, 40)
        program_id = [x.rbyte(x.at(p, 48 + j)) for j in range(32)]
        mb = mem.slice(accs_addr, accs_len, 34, 1)
    dp = mem.slice(data_addr, data_len, 1, 1)
    if wlt(word(255), accs_len):
        raise VMError('MaxInstructionAccountsExceeded')
    if wlt(word(10240), data_len):
        raise VMError('MaxInstructionDataLenExceeded')
    meter.consume(wadd(div250(data_len), div250(mul(accs_len, word(34)))))
    yield
    metas = []
    i = word(0)
    while not weq(i, accs_len):
        if abi == 'c':
            e = x.at(mb, mul(i, word(16)))
            pk_addr = x.rword(e)
            w, s = x.rbyte(x.at(e, 8)), x.rbyte(x.at(e, 9))
        else:
            e = x.at(mb, mul(i, word(34)))
            s, w = x.rbyte(x.at(e, 32)), x.rbyte(x.at(e, 33))
        if wlt(word(1, 2), s) or wlt(word(1, 2), w):
            raise RT.IxError('InvalidArgument')
        k = mem.bytes_at(pk_addr, 32) if abi == 'c' else [x.rbyte(x.at(e, j)) for j in range(32)]
        metas.append([k, float(s[0][1]), float(w[0][1])])     # flag bytes are 0 or 1: their low nibble's bit
        i = inc(i)
    data = [x.rbyte(x.at(dp, j)) for j in range(unword(data_len))] if dp is not None else []
    yield
    # ---- PDA signers
    signers = []
    if not weq(R5, 0):
        sb = mem.slice(R4, R5, 16, 8)
        if wlt(word(16), R5):
            raise VMError('TooManySigners')
        caller_pid = kb(frame.program)
        i = word(0)
        while not weq(i, R5):
            e = x.at(sb, mul(i, word(16)))
            sp, sn = x.rword(e), x.rword(x.at(e, 8))
            seeds_p = mem.slice(sp, sn, 16, 8)
            if wlt(word(16), sn):
                raise RT.IxError('MaxSeedLengthExceeded')
            seeds, j = [], word(0)
            while not weq(j, sn):
                f = x.at(seeds_p, mul(j, word(16)))
                q, ln = x.rword(f), x.rword(x.at(f, 8))
                qp = mem.slice(q, ln, 1, 1)
                seeds.append((ln, [x.rbyte(x.at(qp, t)) for t in range(unword(ln))] if qp is not None else []))
                j = inc(j)
            addr = _pda(seeds, caller_pid)
            yield
            if addr is None:
                raise VMError('BadSeeds')
            signers.append(addr)
            i = inc(i)
    # ---- the program and the privilege checks
    for k in RT.LOADERS | RT.PRECOMPILES:
        if keq(program_id, kb(k)):
            raise VMError('ProgramNotSupported')
    for a in range(len(metas)):                        # duplicate metas share the OR of their flags
        for b in range(len(metas)):
            if b != a and keq(metas[a][0], metas[b][0]):
                metas[a][1] = max(metas[a][1], metas[b][1]); metas[a][2] = max(metas[a][2], metas[b][2])
    first = [next(b for b in range(len(metas)) if keq(metas[a][0], metas[b][0])) for a in range(len(metas))]
    table = [(kb(k), a) for k, a in tx.acc.items()]   # the context's account rows
    caller = [(kb(k), float(s), float(w)) for k, s, w in frame.accounts]

    def row(key):
        return next((a for k, a in table if keq(k, key)), None)

    def caller_flags(key):
        return next(((s, w) for k, s, w in caller if keq(k, key)), None)

    for a, (k, s, w) in enumerate(metas):
        if row(k) is None or caller_flags(k) is None:
            raise RT.IxError('MissingAccount')
        if first[a] != a:
            continue
        cs, cw = caller_flags(k)
        if w and not cw:
            raise RT.IxError('PrivilegeEscalation')
        if s and not (cs or any(keq(k, g) for g in signers)):
            raise RT.IxError('PrivilegeEscalation')
    if caller_flags(program_id) is None:
        raise RT.IxError('MissingAccount')
    yield
    # ---- the account infos
    isz = 56 if abi == 'c' else 48
    if not wlt(R3, (RT.MM_INPUT + isz - 1) // isz):
        raise VMError('InvalidPointer')
    end, carry = add(R2, mul(R3, word(isz)), ZERO2)
    if float(carry[1]) or not wlt(end, MM_INPUT):
        raise VMError('InvalidPointer')
    ib = mem.slice(R2, R3, isz, 8)
    if wlt(word(255), R3):
        raise VMError('MaxInstructionAccountInfosExceeded')
    meter.consume(div250(mul(R3, word(80))))
    info_keys, j = [], word(0)
    while not weq(j, R3):
        info_keys.append(mem.bytes_at(x.rword(x.at(ib, mul(j, word(isz)))), 32))
        j = inc(j)
    def meta_of(key):                                  # the caller's first account with this key
        return next(meta[ci] for ci, (k2, _, _) in enumerate(caller) if keq(k2, key))

    translated = []
    for a, (k, s, w) in enumerate(metas):
        if first[a] != a:
            continue
        acct = row(k)
        if acct.executable:
            meter.consume(div250(word(len(acct.data))))
            continue
        jj = next((t for t, ik in enumerate(info_keys) if keq(ik, k)), None)
        if jj is None:
            raise RT.IxError('MissingAccount')
        mm = meta_of(k)
        e = x.at(ib, word(isz * jj))
        if abi == 'c':
            key_addr, lam_addr, data_addr, owner_addr = field(x, e, 0), field(x, e, 8), field(x, e, 24), field(x, e, 32)
            for got, want in ((key_addr, 'vm_key'), (owner_addr, 'vm_owner'), (lam_addr, 'vm_lamports'), (data_addr, 'vm_data')):
                if not weq(got, mm[want]):
                    raise VMError('InvalidPointer')
            mem.tr(lam_addr, word(8), True, 8); mem.tr(owner_addr, word(32), True, 1)
            len_addr = wadd(R2, word(56 * jj + 16))
        else:
            key_addr, rc_lam, rc_dat, owner_addr = field(x, e, 0), field(x, e, 8), field(x, e, 16), field(x, e, 24)
            if not weq(key_addr, mm['vm_key']) or not weq(owner_addr, mm['vm_owner']):
                raise VMError('InvalidPointer')
            cell = wadd(rc_lam, word(24))
            lam_addr = mem.word_at(cell)
            if not wlt(cell, MM_INPUT) or not weq(lam_addr, mm['vm_lamports']):
                raise VMError('InvalidPointer')
            mem.tr(lam_addr, word(8), True, 8); mem.tr(owner_addr, word(32), True, 1)
            dcell = wadd(rc_dat, word(24))
            if not wlt(dcell, MM_INPUT):
                raise VMError('InvalidPointer')
            data_addr = mem.word_at(dcell)
            if not weq(data_addr, mm['vm_data']):
                raise VMError('InvalidPointer')
            len_addr = wadd(dcell, word(8))
        if not wlt(len_addr, MM_INPUT):
            raise VMError('InvalidPointer')
        ref_len = mem.word_at(len_addr, align=1, store=True)
        if wlt(word(mm['len'] + 10240), ref_len):
            raise RT.IxError('InvalidRealloc')
        if not weq(ref_len, 0):
            mem.tr(data_addr, ref_len, True, 1)
        meter.consume(div250(ref_len))
        translated.append({'key': k, 'acct': acct, 'meta': mm, 'lam': lam_addr, 'owner': owner_addr, 'data': data_addr,
                           'len_addr': len_addr, 'ref_len': ref_len, 'writable': w})
    yield
    # ---- update_callee_account (BorrowedAccount's checks as circuits)
    cur = kb(tx.stack[-1].program)
    for t in translated:
        acct = t['acct']
        cw = caller_flags(t['key'])[1]
        owned = keq(kb(acct.owner), cur)
        lam = mem.word_at(t['lam'])
        old = word(acct.lamports)
        if not weq(lam, old):
            if not owned and wlt(lam, old):
                raise RT.IxError('ExternalAccountLamportSpend')
            if not cw:
                raise RT.IxError('ReadonlyLamportChange')
            acct.lamports = unword(lam)
        n = unword(t['ref_len'])
        d = mem.bytes_at(t['data'], n)
        _set_data(tx, acct, d, t['ref_len'], cw, owned)
        owner = mem.bytes_at(t['owner'], 32)
        if not keq(owner, kb(acct.owner)):
            if not owned or not cw or any(not bool(float(b[0][0])) or not bool(float(b[1][0])) for b in kb(acct.data)):
                raise RT.IxError('ModifiedProgramId')
            acct.owner = unbytes(owner)
    yield
    # ---- process_instruction's checks, then the callee runs
    callee = RT.Frame(unbytes(program_id), [(unbytes(k), bool(s), bool(w)) for k, s, w in metas], unbytes(data))
    progs = [kb(f.program) for f in tx.stack]
    if any(keq(program_id, pg) for pg in progs) and not keq(progs[-1], program_id):
        raise RT.IxError('ReentrancyNotAllowed')
    if not _balanced(tx):
        raise RT.IxError('UnbalancedInstruction')
    if not wlt(word(len(tx.stack)), word(RT.MAX_DEPTH)):
        raise RT.IxError('CallDepth')

    def post():
        if not _balanced(tx):
            raise RT.IxError('UnbalancedInstruction')
    tx.execute(callee, post)
    yield
    # ---- update_caller_account (writable accounts)
    for t in translated:
        if not t['writable']:
            continue
        acct = t['acct']
        mem.write(t['lam'], list(word(acct.lamports)[:16].view(8, 2, NIB)), 8)
        mem.write(t['owner'], kb(acct.owner))
        prev = mem.word_at(t['len_addr'], align=1)
        post_len = word(len(acct.data))
        if wlt(word(t['meta']['len'] + 10240), post_len):
            raise RT.IxError('InvalidRealloc')
        if not weq(prev, post_len):
            if wlt(post_len, prev):
                gap = unword(sub(prev, post_len)[0])
                mem.write(wadd(t['data'], post_len), [word(0, 2)] * gap)
            if not weq(post_len, 0):
                mem.tr(t['data'], post_len, True, 1)
            mem.write(t['len_addr'], list(post_len[:16].view(8, 2, NIB)))
            mem.write(sub(t['data'], word(8))[0], list(post_len[:16].view(8, 2, NIB)), 8)
        mem.write(t['data'], kb(acct.data))
    used = tx.cu - base
    tx.cu = base - cu0
    return cu0 + used


def _set_data(tx, acct, d, n_w, writable, owned):
    """BorrowedAccount::set_data in the writeback pattern: resize-and-set if can_resize passes,
    otherwise only unchanged contents pass."""
    err = None
    if wlt(word(10 * 1024 * 1024), n_w):
        err = 'InvalidRealloc'
    elif tx.resize + unword(n_w) - len(acct.data) > 0 and wlt(word(20 * 1024 * 1024), word(tx.resize + unword(n_w) - len(acct.data))):
        err = 'MaxAccountsDataAllocationsExceeded'
    elif not writable:
        err = 'ReadonlyDataModified'
    elif not owned:
        err = 'ExternalAccountDataModified'
    if err:
        same = len(d) == len(acct.data) and (not d or keq(d, kb(acct.data)))
        if not same:
            raise RT.IxError(err)
        return
    tx.resize += len(d) - len(acct.data)
    acct.data = unbytes(d)


def _balanced(tx):
    """The sum of all lamports equals the transaction's starting sum: 128-bit word adders."""
    tot = word(0, 32)
    for a in tx.acc.values():
        tot = add(tot, word(a.lamports, 32), ZERO2)[0]
    return bool(float(eq(tot, word(tx.lamports0, 32))))


def _pda(seeds, pid):
    """create_program_address with SHA-256 circuits and the Ed25519 decompression circuit."""
    from .crypto import Sha256
    from . import ccurves
    if len(seeds) > 16 or any(wlt(word(32), ln) for ln, _ in seeds):
        return None
    h = Sha256()
    for _, s in seeds:
        for b in s:
            h.byte(b)
    for b in pid + kb(b'ProgramDerivedAddress'):
        h.byte(b)
    out = h.digest()
    return None if ccurves.ed_decompress(out) is not None else out
