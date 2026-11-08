"""A transaction runtime around the VM, following Agave 4.3.2's program-runtime: the transaction
context (accounts, instruction stack, return data, lamport balance), BPF program execution
(serialize, run, deserialize with BorrowedAccount's rules), cross-program invocation through
sol_invoke_signed_c (translate, privilege checks, account sync both ways), and the System
Program builtin. Mainnet feature set as LiteSVM 0.18.2 ships it: syscall parameter address
restrictions on, virtual address space adjustments and direct mapping off, depth limit 5.

`vm(img, input, cu_limit, ctx)` runs one program and returns the VM result; it is either the
reference interpreter (sref) or the hand-set weights (svm). This module is the layer Agave keeps
outside the VM; the VM itself (instructions, memory, syscalls) is what runs in the weights.
"""
import struct
from . import solana
from .sref import MM_INPUT, MM_STACK, MM_HEAP, b58, VMError, pda

M64 = 2**64 - 1
SYSTEM = bytes(32)


def b58decode(s):
    A = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    n = 0
    for c in s:
        n = n * 58 + A.index(c)
    return n.to_bytes(32, 'big')


UPGRADEABLE = b58decode('BPFLoaderUpgradeab1e11111111111111111111111')
NATIVE_LOADER = b58decode('NativeLoader1111111111111111111111111111111')
LOADERS = {b58decode('BPFLoader1111111111111111111111111111111111'), b58decode('BPFLoader2111111111111111111111111111111111'),
           UPGRADEABLE, NATIVE_LOADER, b58decode('LoaderV411111111111111111111111111111111111')}
PRECOMPILES = {b58decode('Ed25519SigVerify111111111111111111111111111'), b58decode('KeccakSecp256k11111111111111111111111111111'),
               b58decode('Secp256r1SigVerify1111111111111111111111111')}
INVOKE_UNITS, CPI_BPU, MAX_DEPTH = 946, 250, 5
RELAX = True                       # SIMD-0392 is active in LiteSVM's mainnet set (measured)


class IxError(Exception):
    """An InstructionError (surfaces as itself) or, with fail=True, ProgramFailedToComplete."""
    def __init__(self, name, cu=None):
        super().__init__(name); self.name = name


class Acct:
    def __init__(self, key, owner, lamports, data, executable=False):
        self.key, self.owner, self.lamports, self.data, self.executable = key, owner, lamports, bytes(data), executable


class Frame:
    def __init__(self, program, metas, data):
        self.program, self.data = program, data
        # merge duplicates: later entries share the first one's flags (OR of all)
        self.accounts, first = [], {}
        for k, s, w in metas:
            if k in first:
                i = first[k]; self.accounts[i][1] |= s; self.accounts[i][2] |= w
            else:
                first[k] = len(self.accounts)
            self.accounts.append([k, s, w])
        for e in self.accounts:
            i = first[e[0]]; e[1], e[2] = self.accounts[i][1], self.accounts[i][2]
        self.first = first


class Tx:
    def __init__(self, accounts, programs, vm, cu_limit):
        self.acc = {a.key: a for a in accounts}
        self.programs, self.vm, self.limit = programs, vm, cu_limit
        self.cu, self.stack, self.logs = 0, [], []
        self.rdd = {'pid': bytes(32), 'data': b''}
        self.lamports0 = sum(a.lamports for a in accounts)
        self.resize = 0

    def delta(self):
        return sum(a.lamports for a in self.acc.values()) - self.lamports0

    # ------------------------------------------------------------ BorrowedAccount rules
    def owned(self, a):
        return a.owner == self.stack[-1].program

    def set_lamports(self, a, v, writable):
        """BorrowedAccount::set_lamports: checks run before the equality test."""
        if not self.owned(a) and v < a.lamports:
            raise IxError('ExternalAccountLamportSpend')
        if not writable:
            raise IxError('ReadonlyLamportChange')
        a.lamports = v

    def can_resize(self, a, n, writable):
        if n > 10 * 1024 * 1024:
            raise IxError('InvalidRealloc')
        if self.resize + n - len(a.data) > 20 * 1024 * 1024:
            raise IxError('MaxAccountsDataAllocationsExceeded')
        if not writable:
            raise IxError('ReadonlyDataModified')
        if not self.owned(a):
            raise IxError('ExternalAccountDataModified')

    def set_data(self, a, d, writable):
        """The writeback pattern: resize-and-set if allowed, else only an unchanged slice passes."""
        try:
            self.can_resize(a, len(d), writable)
        except IxError:
            if d != a.data:
                raise
            return
        self.resize += len(d) - len(a.data)
        a.data = bytes(d)

    def set_data_length(self, a, n, writable):
        self.can_resize(a, n, writable)
        self.resize += n - len(a.data)
        a.data = a.data[:n] + bytes(max(0, n - len(a.data)))

    def set_owner(self, a, owner, writable):
        if not self.owned(a) or not writable or any(a.data):
            raise IxError('ModifiedProgramId')
        a.owner = owner

    # ------------------------------------------------------------ instruction processing
    def process(self, frame):
        progs = [f.program for f in self.stack]
        if self.stack:
            if frame.program in progs and progs[-1] != frame.program:
                raise IxError('ReentrancyNotAllowed')
            if self.delta() != 0:
                raise IxError('UnbalancedInstruction')
        if len(self.stack) >= MAX_DEPTH:
            raise IxError('CallDepth')
        self.stack.append(frame)
        try:
            self.rdd['pid'], self.rdd['data'] = frame.program, b''
            pa = self.acc.get(frame.program)
            if pa is None:
                raise IxError('UnsupportedProgramId')
            if pa.owner == NATIVE_LOADER:
                self.builtin(frame)
            elif pa.owner == UPGRADEABLE:
                self.run_bpf(frame)
            else:
                raise IxError('UnsupportedProgramId')
            if self.delta() != 0:
                raise IxError('UnbalancedInstruction')
        finally:
            self.stack.pop()

    def serialize(self, frame):
        out, meta = bytearray(struct.pack('<Q', len(frame.accounts))), []
        for i, (k, s, w) in enumerate(frame.accounts):
            if frame.first[k] != i:
                out += bytes([frame.first[k]]) + bytes(7)
                meta.append(meta[frame.first[k]])
                continue
            a = self.acc[k]
            hdr = len(out)
            out += bytes([0xff, int(s), int(w), int(a.executable)]) + bytes(4) + k + a.owner
            out += struct.pack('<QQ', a.lamports, len(a.data))
            off = len(out)
            out += a.data + bytes(10240 + (-len(a.data)) % 8) + struct.pack('<Q', M64)
            meta.append({'key': k, 'hdr': hdr, 'off': off, 'len': len(a.data), 'writable': w, 'signer': s,
                         'vm_key': MM_INPUT + hdr + 8, 'vm_owner': MM_INPUT + hdr + 40,
                         'vm_lamports': MM_INPUT + hdr + 72, 'vm_data': MM_INPUT + off})
        out += struct.pack('<Q', len(frame.data)) + frame.data + frame.program
        return bytearray(out), meta

    def deserialize(self, buf, frame, meta):
        for i, (k, s, w) in enumerate(frame.accounts):
            if frame.first[k] != i:
                continue
            m, a = meta[i], self.acc[k]
            owner = bytes(buf[m['off'] - 48:m['off'] - 16])
            lam = struct.unpack_from('<Q', buf, m['off'] - 16)[0]
            if lam != a.lamports:
                self.set_lamports(a, lam, w)
            post = struct.unpack_from('<Q', buf, m['off'] - 8)[0]
            if max(post - m['len'], 0) > 10240 or post > 10 * 1024 * 1024:
                raise IxError('InvalidRealloc')
            self.set_data(a, bytes(buf[m['off']:m['off'] + post]), w)
            if owner != a.owner:
                self.set_owner(a, owner, w)

    def run_bpf(self, frame):
        img = self.programs[frame.program]
        buf, meta = self.serialize(frame)
        ctx = {'tx': self, 'frame': frame, 'meta': meta}
        r = self.vm(img, bytes(buf), self.limit - self.cu, ctx)
        self.cu += r['cu']
        self.logs += r['logs']
        if self.rdd['data']:
            self.logs.append('Program return: ' + b58(frame.program) + ' ' + __import__('base64').b64encode(self.rdd['data']).decode())
        st = r['status']
        if st == 'ok' and r['r0'] != 0:
            raise IxError(solana.outcome('ok', r['r0'], 0, []).get('err')[len('InstructionError(0, '):-1])
        if st != 'ok':
            if r.get('ixerr'):
                raise IxError(r['ixerr'])
            if st.startswith('Sys'):
                raise IxError('ComputationalBudgetExceeded' if 'Budget' in st else
                              'ArithmeticOverflow' if st == 'SysArithmeticOverflow' else 'ProgramFailedToComplete')
            self.cu = self.limit                                  # a VM fault depletes the meter
            raise IxError('ProgramFailedToComplete')
        self.deserialize(r['input'], frame, meta)

    # ------------------------------------------------------------ System Program (builtin)
    def builtin(self, frame):
        if frame.program != SYSTEM:
            raise IxError('UnsupportedProgramId')
        self.consume(150)
        d = frame.data
        if len(d) < 4:
            raise IxError('InvalidInstructionData')
        tag = struct.unpack_from('<I', d)[0]
        accs = frame.accounts

        def acc(i):
            if i >= len(accs):
                raise IxError('MissingAccount')
            k, s, w = accs[i]
            return self.acc[k], s, w

        if tag == 2:                                              # transfer(lamports)
            if len(d) < 12:
                raise IxError('InvalidInstructionData')
            n = struct.unpack_from('<Q', d, 4)[0]
            (fa, fs, fw), (ta, ts, tw) = acc(0), acc(1)
            if not fs:
                raise IxError('MissingRequiredSignature')
            if fa.data:
                raise IxError('InvalidArgument')
            if n > fa.lamports:
                raise IxError('Custom(1)')                         # SystemError::ResultWithNegativeLamports
            self.set_lamports(fa, fa.lamports - n, fw)
            self.set_lamports(ta, ta.lamports + n, tw)
        elif tag == 1:                                            # assign(owner)
            if len(d) < 36:
                raise IxError('InvalidInstructionData')
            a, s, w = acc(0)
            owner = d[4:36]
            if a.owner == owner:
                return
            if not s:
                raise IxError('MissingRequiredSignature')
            self.set_owner(a, owner, w)
        elif tag == 8:                                            # allocate(space)
            if len(d) < 12:
                raise IxError('InvalidInstructionData')
            space = struct.unpack_from('<Q', d, 4)[0]
            a, s, w = acc(0)
            if not s:
                raise IxError('MissingRequiredSignature')
            if a.data or a.owner != SYSTEM:
                raise IxError('Custom(0)')                         # SystemError::AccountAlreadyInUse
            if space > 10 * 1024 * 1024:
                raise IxError('Custom(3)')                         # SystemError::InvalidAccountDataLength
            self.set_data_length(a, space, w)
        else:
            raise IxError('InvalidInstructionData')

    def consume(self, n):
        if n > self.limit - self.cu:
            self.cu = self.limit
            raise IxError('ComputationalBudgetExceeded')
        self.cu += n


# ------------------------------------------------------------------ the CPI syscall (C ABI)
def invoke_c(ctx, r, mem, cu, limit):
    tx = ctx['tx']
    base = tx.cu + cu
    try:
        return _invoke_c(ctx, r, mem, cu, limit)
    except (VMError, IxError) as e:
        e.cu_local = cu + (tx.cu - base)
        tx.cu = base - cu
        raise


class RefPort:
    """Caller memory for the reference VM (sref.Mem)."""
    def __init__(self, mem):
        self.mem = mem

    def read(self, addr, n, align=1, store=False):
        if n == 0:
            return b''
        b, o = self.mem.at(addr, n, store)
        if o % align:
            raise VMError('UnalignedPointer')
        return bytes(b[o:o + n])

    def write(self, addr, data, align=1):
        if not data:
            return
        b, o = self.mem.at(addr, len(data), True)
        if o % align:
            raise VMError('UnalignedPointer')
        b[o:o + len(data)] = data


def _invoke_c(ctx, r, port, cu, limit):
    """sol_invoke_signed_c, following cpi_common / translate_*_c / update_*_account in Agave 4.3.2."""
    if not hasattr(port, 'read'):
        port = RefPort(port)
    tx, frame, meta = ctx['tx'], ctx['frame'], ctx['meta']
    tx.cu += cu
    base = tx.cu
    q = lambda addr, align=8: struct.unpack('<Q', port.read(addr, 8, align))[0]
    key = lambda addr: port.read(addr, 32)

    def sl(addr, n, esz, align):
        if n == 0:
            return b''
        if n * esz >= 2**63:
            raise VMError('InvalidLength')
        return port.read(addr, n * esz, align)

    a1, a2, a3, a4, a5 = r[1], r[2], r[3], r[4], r[5]
    tx.consume(INVOKE_UNITS)
    pid_addr, accs_addr, accs_len, data_addr, data_len = struct.unpack('<5Q', port.read(a1, 40, 8))
    program_id = key(pid_addr)
    mb = sl(accs_addr, accs_len, 16, 8)
    data = sl(data_addr, data_len, 1, 1)
    if accs_len > 255:
        raise VMError('MaxInstructionAccountsExceeded')
    if data_len > 10240:
        raise VMError('MaxInstructionDataLenExceeded')
    tx.consume(min(data_len // CPI_BPU + (accs_len * 34) // CPI_BPU, M64))
    metas = []
    for i in range(accs_len):
        pk_addr = struct.unpack_from('<Q', mb, 16 * i)[0]
        w, s = mb[16 * i + 8], mb[16 * i + 9]
        if s > 1 or w > 1:
            raise IxError('InvalidArgument')
        metas.append((key(pk_addr), bool(s), bool(w)))
    signers = []
    if a5:
        sb = sl(a4, a5, 16, 8)
        if a5 > 16:
            raise VMError('TooManySigners')
        for i in range(a5):
            p, n = struct.unpack_from('<QQ', sb, 16 * i)
            seeds_b = sl(p, n, 16, 8)
            if n > 16:
                raise IxError('MaxSeedLengthExceeded')
            seeds = [sl(*struct.unpack_from('<QQ', seeds_b, 16 * j), 1, 1) for j in range(n)]
            addr = pda(seeds, frame.program)
            if addr is None:
                raise VMError('BadSeeds')
            signers.append(addr)
    if program_id in LOADERS or program_id in PRECOMPILES:
        raise VMError('ProgramNotSupported')
    callee = Frame(program_id, metas, data)
    caller_flags = {k: (s, w) for k, s, w in frame.accounts}
    for i, (k, s, w) in enumerate(callee.accounts):
        if k not in tx.acc or k not in caller_flags:
            raise IxError('MissingAccount')
        if callee.first[k] != i:
            continue
        cs, cw = caller_flags[k]
        if w and not cw:
            raise IxError('PrivilegeEscalation')
        if s and not (cs or k in signers):
            raise IxError('PrivilegeEscalation')
    if program_id not in caller_flags:
        raise IxError('MissingAccount')
    if a2 + a3 * 56 >= MM_INPUT:
        raise VMError('InvalidPointer')
    ib = sl(a2, a3, 56, 8)
    if a3 > 255:
        raise VMError('MaxInstructionAccountInfosExceeded')
    tx.consume((a3 * 80) // CPI_BPU)
    info_keys = [key(struct.unpack_from('<Q', ib, 56 * i)[0]) for i in range(a3)]
    caller_index = {k: i for i, (k, _, _) in reversed(list(enumerate(frame.accounts)))}
    translated = []
    for i, (k, s, w) in enumerate(callee.accounts):
        if callee.first[k] != i:
            continue
        a = tx.acc[k]
        if a.executable:
            tx.consume(len(a.data) // CPI_BPU)
            continue
        if k not in info_keys:
            raise IxError('MissingAccount')
        j = info_keys.index(k)
        m = meta[caller_index[k]]
        f = struct.unpack_from('<6Q', ib, 56 * j)
        key_addr, lam_addr, dlen, data_addr, owner_addr = f[0], f[1], f[2], f[3], f[4]
        if (key_addr, owner_addr, lam_addr, data_addr) != (m['vm_key'], m['vm_owner'], m['vm_lamports'], m['vm_data']):
            raise VMError('InvalidPointer')
        port.read(lam_addr, 8, 8, True); port.read(owner_addr, 32, 1, True)
        len_addr = a2 + 56 * j + 16
        if len_addr >= MM_INPUT:
            raise VMError('InvalidPointer')
        ref_len = struct.unpack('<Q', port.read(len_addr, 8, 1, True))[0]
        if ref_len > m['len'] + 10240:
            raise IxError('InvalidRealloc')
        port.read(data_addr, ref_len, 1, True)
        tx.consume(ref_len // CPI_BPU)
        translated.append({'key': k, 'meta': m, 'lam': lam_addr, 'owner': owner_addr, 'data': data_addr,
                           'len_addr': len_addr, 'ref_len': ref_len, 'writable': w})
    for t in translated:                       # update_callee_account
        a, cw = tx.acc[t['key']], caller_flags[t['key']][1]
        lam = q(t['lam'])
        if lam != a.lamports:
            tx.set_lamports(a, lam, cw)
        tx.set_data(a, port.read(t['data'], t['ref_len']), cw)
        owner = key(t['owner'])
        if owner != a.owner:
            tx.set_owner(a, owner, cw)
    tx.process(callee)
    for t in translated:                       # update_caller_account (writable accounts)
        if not t['writable']:
            continue
        a = tx.acc[t['key']]
        port.write(t['lam'], struct.pack('<Q', a.lamports), 8)
        port.write(t['owner'], a.owner)
        prev, post = struct.unpack('<Q', port.read(t['len_addr'], 8, 1))[0], len(a.data)
        if post > t['meta']['len'] + 10240:
            raise IxError('InvalidRealloc')
        if prev != post:
            if post < prev:
                port.write(t['data'] + post, bytes(prev - post))
            port.read(t['data'], post, 1, True)
            port.write(t['len_addr'], struct.pack('<Q', post))
            port.write(t['data'] - 8, struct.pack('<Q', post), 8)
        port.write(t['data'], a.data)
    used = tx.cu - base
    tx.cu = base - cu
    return cu + used, 0


# ------------------------------------------------------------------ one whole transaction
def run_tx(req, truth, vm):
    """Replay a svmtx request from the exact pre-state LiteSVM reported. Returns an outcome dict
    comparable with the truth (ok, cu, program log lines, post-state or the error)."""
    import json
    accts = [Acct(bytes.fromhex(a['key']), bytes.fromhex(a['owner']), a['lamports'], bytes.fromhex(a['data']), a['executable'])
             for a in truth['pre'] if not a.get('missing')]
    progs = {bytes.fromhex(p['id']): solana.image(p['elf']) for p in req['programs']}
    tx = Tx(accts, progs, vm, req['cu'])
    ix = req['ix']
    frame = Frame(bytes.fromhex(ix['program']), [(bytes.fromhex(m['key']), m['signer'], m['writable']) for m in ix['metas']],
                  bytes.fromhex(ix['data']))
    try:
        tx.process(frame)
    except IxError as e:
        return {'ok': False, 'cu': tx.cu, 'logs': tx.logs, 'err': f'InstructionError(0, {e.name})'}
    # rent state transitions (transaction level): no writable account may become rent-paying.
    # SIMD-0392 (relax_post_exec_min_balance_check) grandfathers existing below-minimum balances.
    order = message_order(frame)
    pre = {a['key']: (a['lamports'], len(a['data']) // 2, a['owner']) for a in truth['pre'] if not a.get('missing')}
    for i, (k, w) in enumerate(order):
        if not w or k not in tx.acc:
            continue
        a = tx.acc[k]
        def state(lam, n):
            return 'none' if lam == 0 else 'exempt' if lam >= (128 + n) * 5080 else ('paying', n, lam)
        lam0, n0, own0 = pre[k.hex()]
        s0, s1 = state(lam0, n0), state(a.lamports, len(a.data))
        if RELAX and s0 not in ('none', 'exempt'):
            s0 = 'exempt'
        if RELAX and s0 == 'exempt' and s1 not in ('none', 'exempt') and own0 == a.owner.hex() and n0 >= len(a.data) and a.lamports >= lam0:
            s1 = 'exempt'
        if s1 in ('none', 'exempt'):
            continue
        if s0 in ('none', 'exempt') or s1[1] != s0[1] or s1[2] > s0[2]:
            return {'ok': False, 'cu': tx.cu, 'logs': tx.logs, 'err': f'InsufficientFundsForRent {{ account_index: {i} }}'}
    post = {a.key.hex(): (a.owner.hex(), a.lamports, a.data.hex()) for a in tx.acc.values()}
    return {'ok': True, 'cu': tx.cu, 'logs': tx.logs, 'post': post}


def truth_tx(reqs):
    import json, subprocess, os
    exe = os.path.join(solana.ROOT, 'truth', 'target', 'release', 'svmtx')
    out = subprocess.run([exe], input=''.join(json.dumps(r) + '\n' for r in reqs), capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.splitlines():
        t = json.loads(line)
        t['logs'] = [l for l in t['logs'] if l.startswith(solana.PROGRAM_LINES)]
        if t['ok']:
            t['post'] = {a['key']: (a['owner'], a['lamports'], a['data']) for a in t['post'] if not a.get('missing')}
        res.append(t)
    return res


def same_tx(a, b):
    if a['ok'] != b['ok'] or a['cu'] != b['cu'] or a['logs'] != b['logs']:
        return False
    return a['post'] == b['post'] if a['ok'] else a['err'] == b['err']


def message_order(frame):
    """Message::new key order: the fee payer, then writable signers, readonly signers, writable
    non-signers, readonly non-signers (the program id among them), each group sorted by key."""
    flags = {}
    for k, s, w in frame.accounts:
        fs, fw = flags.get(k, (False, False))
        flags[k] = (fs or bool(s), fw or bool(w))
    flags.setdefault(frame.program, (False, False))
    groups = [sorted(k for k, (s, w) in flags.items() if s and w), sorted(k for k, (s, w) in flags.items() if s and not w),
              sorted(k for k, (s, w) in flags.items() if not s and w), sorted(k for k, (s, w) in flags.items() if not s and not w)]
    order = [(None, True)]
    for g, keys in enumerate(groups):
        order += [(k, g in (0, 2)) for k in keys]
    return order
