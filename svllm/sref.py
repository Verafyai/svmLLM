"""Plain-Python reference for the full-layout machine (SBPF v0, solana-sbpf 0.12 default config).

Memory map, as Agave's program runtime sets it up:
  0x1_0000_0000  program (read-only; text bytes in text mode, the ELF ro section later)
  0x2_0000_0000  stack: 64 frames of 4 KB. With gaps (the default) frame i lives at
                 0x2_0000_0000 + i * 8 KB and the 4 KB after it is unmapped.
  0x3_0000_0000  heap, 32 KB
  0x4_0000_0000  input (the serialized accounts)
Calls: BPF-to-BPF through a function registry {key: pc}; frames save r6..r9, r10 and the
return pc. 1 CU per instruction.
"""
from .isa import dec, OPCODES
from .ref import M64, M32, sx32, sx64, s64, VMError

MM_PROGRAM, MM_STACK, MM_HEAP, MM_INPUT = 1 << 32, 2 << 32, 3 << 32, 4 << 32
FRAME, DEPTH, HEAP = 4096, 64, 32 * 1024


def fnv(b):
    h = 0xcbf29ce484222325
    for x in b:
        h = ((h ^ x) * 0x100000001b3) & M64
    return h


class Mem:
    def __init__(self, prog, inp, gaps, ro=None, ro_off=0):
        self.ro, self.gaps, self.ro_off = bytes(prog if ro is None else ro), gaps, ro_off
        self.stack, self.heap, self.input = bytearray(FRAME * DEPTH), bytearray(HEAP), bytearray(inp)

    def at(self, a, n, store):
        """Translate like solana-sbpf's aligned mapping. Returns (buffer, offset)."""
        reg, off = a >> 32, a & M32
        if a >> 35 == 0 and 1 <= reg <= 4:
            buf = [None, self.ro, self.stack, self.heap, self.input][reg]
            if reg == 1 and store:
                raise VMError('AccessViolation')
            if reg == 1:
                if off < self.ro_off:
                    raise VMError('AccessViolation')
                off -= self.ro_off
            if reg == 2 and self.gaps:
                if (off >> 12) & 1:
                    raise VMError('AccessViolation')
                off = ((off >> 13) << 12) | (off & 0xfff)
            if off + n <= len(buf):
                return buf, off
        raise VMError('AccessViolation')

    def slice(self, a, n, store):
        if n == 0:
            return None, 0
        return self.at(a, n, store)

    def load(self, a, n):
        b, o = self.at(a, n, False)
        return int.from_bytes(b[o:o + n], 'little')

    def store(self, a, n, v):
        b, o = self.at(a, n, True)
        b[o:o + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, 'little')


import base64, hashlib
PROGRAM_ID = hashlib.sha256(b'svmtruth:program').digest()
B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def b58(b):
    n, s = int.from_bytes(b, 'big'), ''
    while n:
        n, m = divmod(n, 58); s = B58[m] + s
    return '1' * (len(b) - len(b.lstrip(b'\0'))) + s


def keccak256(data):
    """Keccak-256 (the original padding, as Solana's sol_keccak256)."""
    RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
          0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
          0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
          0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
          0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
    ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
    rot = lambda x, n: ((x << n) | (x >> (64 - n))) & M64 if n else x
    rate = 136
    m = bytearray(data) + b'\x01'
    m += bytes((-len(m)) % rate); m[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for blk in range(0, len(m), rate):
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(m[blk + 8 * i:blk + 8 * i + 8], 'little')
        for rnd in range(24):
            C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
            D = [C[(x - 1) % 5] ^ rot(C[(x + 1) % 5], 1) for x in range(5)]
            A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
            B = [[0] * 5 for _ in range(5)]
            for x in range(5):
                for y in range(5):
                    B[y][(2 * x + 3 * y) % 5] = rot(A[x][y], ROT[x][y])
            A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
            A[0][0] ^= RC[rnd]
    return b''.join(A[i % 5][i // 5].to_bytes(8, 'little') for i in range(4))


import json as _json, os as _os
SYSVARS = {k: {'id': bytes.fromhex(v['id']), 'data': bytes.fromhex(v['data'])} for k, v in
           _json.load(open(_os.path.join(_os.path.dirname(__file__), 'sysvars.json'))).items()}


def sysvar_struct(name):
    """The in-memory (repr(Rust)) layout a get_*_sysvar syscall writes, from the bincode buffer."""
    b = SYSVARS[name]['data']
    if name == 'epoch_schedule':                     # u64, u64, bool (+7 padding), u64, u64
        return b[:16] + b[16:17] + bytes(7) + b[17:33], 8
    if name == 'rent':                               # u64, f64, u8 (+7 padding)
        return b[:17] + bytes(7), 8
    if name == 'epoch_rewards':                      # u64, u64, [32], (u128 at 48), u64, u64, bool, padding to 96
        return b[:48] + b[48:64] + b[64:80] + b[80:81] + bytes(15), 16
    return b, 8                                      # clock (5 x u64), last_restart_slot (u64)


P25519 = 2**255 - 19
D25519 = (-121665 * pow(121666, P25519 - 2, P25519)) % P25519


def on_curve(b):
    """curve25519-dalek's CompressedEdwardsY::decompress().is_some(): (y^2-1)(d y^2+1) is a square."""
    y = int.from_bytes(b, 'little') & (2**255 - 1)
    y2 = y * y % P25519
    w = (y2 - 1) * (D25519 * y2 + 1) % P25519
    return pow(w, (P25519 - 1) // 2, P25519) in (0, 1)


def pda(seeds, pid):
    """Pubkey::create_program_address, or None."""
    if len(seeds) > 16 or any(len(s) > 32 for s in seeds):
        return None
    h = hashlib.sha256(b''.join(seeds) + pid + b'ProgramDerivedAddress').digest()
    return None if on_curve(h) else h


SYSVAR_CALLS = {'sol_get_clock_sysvar': 'clock', 'sol_get_epoch_schedule_sysvar': 'epoch_schedule',
                'sol_get_rent_sysvar': 'rent', 'sol_get_epoch_rewards_sysvar': 'epoch_rewards',
                'sol_get_last_restart_slot': 'last_restart_slot'}


def syscall(name, r, mem, cu, limit, logs, rd):
    """Agave 4.3.2's syscalls (the ones Solana programs import most). Returns (cu, r0)."""
    a1, a2, a3, a4 = r[1], r[2], r[3], r[4]

    def consume(c):
        nonlocal cu
        if c > limit - cu:
            cu = limit
            raise SysError('ComputationalBudgetExceeded', cu)
        cu += c

    def sl(addr, n, store=False):                  # translate_slice::<u8>
        return (None, 0) if n == 0 else mem.at(addr, n, store)

    def typ(addr, n, align, store=False):          # translate_type (alignment on the host pointer)
        b, o = mem.at(addr, n, store)
        if o % align:
            raise VMError('UnalignedPointer')
        return b, o

    def vmslices(addr, n):                         # translate_slice::<VmSlice<u8>> then each slice
        if n == 0:
            return []
        if n * 16 >= 2**63:
            raise VMError('InvalidLength')
        b, o = mem.at(addr, n * 16, False)
        if o % 8:
            raise VMError('UnalignedPointer')
        return [(int.from_bytes(b[o + 16 * i:o + 16 * i + 8], 'little'), int.from_bytes(b[o + 16 * i + 8:o + 16 * i + 16], 'little')) for i in range(n)]

    def text(addr, n):
        b, o = sl(addr, n)
        s = bytes(b[o:o + n]) if b is not None else b''
        if not utf8_ok(s):
            raise VMError('InvalidString')
        return s.decode()

    try:
        if name == 'abort':
            raise VMError('Abort')
        if name == 'sol_panic_':
            consume(a2); text(a1, a2)
            raise VMError('Panic')
        if name == 'sol_log_':
            consume(max(100, a2)); logs.append('Program log: ' + text(a1, a2))
        elif name == 'sol_log_64_':
            consume(100); logs.append('Program log: ' + ', '.join(f'{x:#x}' for x in r[1:6]))
        elif name == 'sol_log_compute_units_':
            consume(100); logs.append(f'Program consumption: {limit - cu} units remaining')
        elif name == 'sol_log_pubkey':
            consume(100); b, o = mem.at(a1, 32, False); logs.append('Program log: ' + b58(bytes(b[o:o + 32])))
        elif name == 'sol_log_data':
            consume(100)
            vs = vmslices(a1, a2)
            consume(100 * len(vs)); consume(sum(n for _, n in vs) & M64 if sum(n for _, n in vs) < 2**64 else M64)
            fields = []
            for p, n in vs:
                b, o = sl(p, n); fields.append(bytes(b[o:o + n]) if b is not None else b'')
            logs.append('Program data: ' + ' '.join(base64.b64encode(f).decode() for f in fields))
        elif name in ('sol_memcpy_', 'sol_memmove_', 'sol_memset_', 'sol_memcmp_'):
            consume(max(10, a3 // 250))
            if name == 'sol_memcpy_' and not (a2 - a1 >= a3 if a2 > a1 else a1 - a2 >= a3):
                raise VMError('CopyOverlapping')
            if name == 'sol_memset_':
                db, do = sl(a1, a3, True)
                if a3: db[do:do + a3] = bytes([a2 & 0xff]) * a3
            elif name == 'sol_memcmp_':
                b1, o1 = sl(a1, a3); b2, o2 = sl(a2, a3)
                s1 = bytes(b1[o1:o1 + a3]) if a3 else b''; s2 = bytes(b2[o2:o2 + a3]) if a3 else b''
                res = next((x - y for x, y in zip(s1, s2) if x != y), 0)
                rb, ro = typ(a4, 4, 4, True)
                rb[ro:ro + 4] = (res & M32).to_bytes(4, 'little')
            else:
                db, do = sl(a1, a3, True); sb, so = sl(a2, a3)
                if a3: db[do:do + a3] = bytes(sb[so:so + a3])
        elif name in ('sol_sha256', 'sol_keccak256'):
            if a2 > 20000:
                raise VMError('TooManySlices')
            consume(85)
            mem.at(a3, 32, True)
            hsh = hashlib.sha256() if name == 'sol_sha256' else None
            buf = b''
            for p, n in vmslices(a1, a2):
                b, o = sl(p, n); chunk = bytes(b[o:o + n]) if b is not None else b''
                consume(max(10, n // 2))
                buf += chunk
            out = hashlib.sha256(buf).digest() if name == 'sol_sha256' else keccak256(buf)
            b, o = mem.at(a3, 32, True); b[o:o + 32] = out
        elif name == 'sol_set_return_data':
            consume(a2 // 250 + 100)
            if a2 > 1024:
                raise VMError('ReturnDataTooLarge')
            b, o = sl(a1, a2); rd['data'] = bytes(b[o:o + a2]) if a2 else b''
            rd['pid'] = rd['ctx']['frame'].program if rd.get('ctx') else PROGRAM_ID
        elif name == 'sol_get_return_data':
            consume(100)
            data = rd['data']; n = min(a2, len(data))
            if n:
                consume((n + 32) // 250)
                db, do = sl(a1, n, True); pb, po = typ(a3, 32, 1, True)
                if db is pb and not (do - po >= 32 if do > po else po - do >= n):
                    raise VMError('CopyOverlapping')        # translate_mut! refuses overlapping borrows
                db[do:do + n] = data[:n]; pb[po:po + 32] = rd.get('pid', PROGRAM_ID)
            return cu, len(data)
        elif name in SYSVAR_CALLS:
            data, align = sysvar_struct(SYSVAR_CALLS[name])
            consume(100 + len(data))
            if a1 >= MM_INPUT:
                raise VMError('InvalidPointer')      # syscall_parameter_address_restrictions
            b, o = typ(a1, len(data), align, True)
            b[o:o + len(data)] = data
        elif name == 'sol_get_sysvar':
            consume(100 + max(a4 // 250, 10))
            if a2 >= MM_INPUT:
                raise VMError('InvalidPointer')
            sl(a2, a4, True)
            ib, io = typ(a1, 32, 1)
            sid = bytes(ib[io:io + 32])
            if a3 + a4 >= 2**64 or a2 + a4 >= 2**64:
                raise VMError('ArithmeticOverflow')
            buf = next((v['data'] for v in SYSVARS.values() if v['id'] == sid), None)
            if buf is None:
                return cu, 2
            if a3 + a4 > len(buf):
                return cu, 1
            b, o = sl(a2, a4, True)
            if a4: b[o:o + a4] = buf[a3:a3 + a4]
        elif name in ('sol_create_program_address', 'sol_try_find_program_address'):
            consume(1500)
            vs = vmslices(a1, a2)
            if len(vs) > 16:
                raise VMError('BadSeeds')
            seeds = []
            for p, n in vs:
                if n > 32:
                    raise VMError('BadSeeds')
                b, o = sl(p, n); seeds.append(bytes(b[o:o + n]) if n else b'')
            pb, po = typ(a3, 32, 1)
            pid = bytes(pb[po:po + 32])
            if name == 'sol_create_program_address':
                addr = pda(seeds, pid)
                if addr is None:
                    return cu, 1
                b, o = mem.at(a4, 32, True); b[o:o + 32] = addr
                return cu, 0
            bump = 255
            for _ in range(255):
                addr = pda(seeds + [bytes([bump])], pid)
                if addr is not None:
                    bb, bo = mem.at(r[5], 1, True); ab, ao = mem.at(a4, 32, True)
                    if bb is ab and not (bo - ao >= 32 if bo > ao else ao - bo >= 1):
                        raise VMError('CopyOverlapping')
                    bb[bo] = bump; ab[ao:ao + 32] = addr
                    return cu, 0
                bump = max(bump - 1, 0)
                consume(1500)
            return cu, 1
        elif name == 'sol_get_epoch_stake':
            if a1 == 0:
                consume(100); return cu, 0
            consume(110); typ(a1, 32, 1); return cu, 0
        elif name == 'sol_alloc_free_':
            if a1 > 2**63 - 8 or a2 != 0:
                return cu, 0
            pos = rd.get('heap', 0); pos += (-pos) % 8
            if pos + a1 > HEAP:
                return cu, 0
            rd['heap'] = pos + a1
            return cu, MM_HEAP + pos
        elif name == 'sol_get_processed_sibling_instruction':
            consume(100); return cu, 0                 # one top-level instruction: no siblings
        elif name == 'sol_get_stack_height':
            consume(100); return cu, (len(rd['ctx']['tx'].stack) if rd.get('ctx') else 1)
        elif name == 'sol_remaining_compute_units':
            consume(100); return cu, limit - cu
        return cu, 0
    except SysError:
        raise
    except VMError as e:
        raise SysError(e.kind, cu)


class SysError(VMError):
    def __init__(self, kind, cu):
        super().__init__(kind); self.cu = cu


def utf8_ok(b):
    try:
        bytes(b).decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def run(prog, inp=b'', cu_limit=10000, gaps=True, funcs=None, trace=None, ro=None, ro_off=0, text_off=0,
        entry=0, syscalls=None, ctx=None):
    """syscalls: {key: name} for abort, sol_log_, sol_memcpy_, sol_memset_ (Agave 4.3.2 rules)."""
    funcs, syscalls, logs = funcs or {}, syscalls or {}, []
    rd = ctx['tx'].rdd if ctx else {'data': b'', 'pid': PROGRAM_ID}
    rd['ctx'] = ctx
    n = len(prog) // 8
    mem = Mem(prog, inp, gaps, ro, ro_off)
    r = [0] * 11
    r[1], r[10] = MM_INPUT, MM_STACK + FRAME
    frames, pc, cu, status, ixerr = [], entry, 0, None, None
    step_fp = FRAME * (2 if gaps else 1)
    try:
        while True:
            if cu >= cu_limit:
                raise VMError('ExceededMaxInstructions')
            cu += 1
            if pc >= n:
                raise VMError('ExecutionOverrun')
            op, d, s, off, imm = dec(prog, pc)
            if trace is not None:
                trace.append({'pc': pc, 'cu': cu, 'regs': list(r)})
            if op not in OPCODES:
                raise VMError('InvalidInstruction')
            kind, name, isreg = OPCODES[op]
            nxt = pc + 1
            if kind == 'alu':
                a = r[d]; b = r[s] if isreg else (imm & M64)
                if name == 'add': v = (a + b) & M64
                elif name == 'sub': v = (a - b) & M64
                elif name == 'mul': v = (a * b) & M64
                elif name in ('div', 'mod'):
                    if b == 0: raise VMError('DivideByZero')
                    v = a // b if name == 'div' else a % b
                elif name == 'or': v = a | b
                elif name == 'and': v = a & b
                elif name == 'xor': v = a ^ b
                elif name == 'lsh': v = (a << (b & 63)) & M64
                elif name == 'rsh': v = a >> (b & 63)
                elif name == 'arsh': v = (s64(a) >> (b & 63)) & M64
                elif name == 'neg': v = (-a) & M64
                elif name == 'mov': v = b
                r[d] = v
            elif kind == 'alu32':
                a = r[d] & M32; b = (r[s] if isreg else imm) & M32
                if name == 'add': v = sx64(a + b)
                elif name == 'sub': v = sx64(a - b)
                elif name == 'mul': v = sx64(a * b)
                elif name in ('div', 'mod'):
                    if b == 0: raise VMError('DivideByZero')
                    v = a // b if name == 'div' else a % b
                elif name == 'or': v = a | b
                elif name == 'and': v = a & b
                elif name == 'xor': v = a ^ b
                elif name == 'lsh': v = (a << (b & 31)) & M32
                elif name == 'rsh': v = a >> (b & 31)
                elif name == 'arsh': v = (sx32(a) >> (b & 31)) & M32
                elif name == 'neg': v = (-a) & M32
                elif name == 'mov': v = b
                r[d] = v
            elif kind == 'end':
                if imm not in (16, 32, 64): raise VMError('InvalidInstruction')
                v = r[d] & ((1 << imm) - 1)
                r[d] = int.from_bytes(v.to_bytes(imm // 8, 'little'), 'big') if name == 'be' else v
            elif kind == 'jmp':
                a = r[d]; b = r[s] if isreg else (imm & M64)
                t = {'ja': True, 'jeq': a == b, 'jne': a != b, 'jgt': a > b, 'jge': a >= b,
                     'jlt': a < b, 'jle': a <= b, 'jset': (a & b) != 0,
                     'jsgt': s64(a) > s64(b), 'jsge': s64(a) >= s64(b),
                     'jslt': s64(a) < s64(b), 'jsle': s64(a) <= s64(b)}[name]
                if t:
                    nxt = (pc + 1 + off) & M64
            elif kind in ('ldx', 'stx', 'st'):
                size = {'b': 1, 'h': 2, 'w': 4, 'dw': 8}[name]
                base = r[s] if kind == 'ldx' else r[d]
                addr = (base + off) & M64
                if kind == 'ldx':
                    r[d] = mem.load(addr, size)
                else:
                    mem.store(addr, size, r[s] if kind == 'stx' else (imm & M64))
            elif kind == 'lddw':
                hi = dec(prog, pc + 1)[4] if pc + 1 < n else 0
                r[d] = ((hi & M32) << 32) | (imm & M32)
                nxt = pc + 2
            elif kind == 'call' and (imm & M32) in syscalls and syscalls[imm & M32] in ('sol_invoke_signed_c', 'sol_invoke_signed_rust') and ctx:
                from .runtime import invoke_c, IxError
                try:
                    cu, r[0] = invoke_c(ctx, r, mem, cu, cu_limit, 'c' if syscalls[imm & M32].endswith('_c') else 'rust')
                except IxError as e:
                    status, cu, ixerr = 'SysIx', e.cu_local, e.name
                    break
                except VMError as e:
                    cu = e.cu_local
                    raise SysError(e.kind, cu)
            elif kind == 'call' and (imm & M32) in syscalls:
                cu, r[0] = syscall(syscalls[imm & M32], r, mem, cu, cu_limit, logs, rd)
            elif kind in ('call', 'callx'):
                if kind == 'call':
                    if (imm & M32) not in funcs:
                        raise VMError('InvalidInstruction')     # UnsupportedInstruction
                    target = funcs[imm & M32]
                frames.append((r[6:10], r[10], pc + 1))
                if len(frames) == DEPTH:
                    raise VMError('CallDepthExceeded')
                r[10] = (r[10] + step_fp) & M64
                if kind == 'callx':
                    target = ((r[imm] if 0 <= imm <= 10 else 0) - MM_PROGRAM - text_off) & M64
                    target //= 8
                if target >= n:
                    raise VMError('CallOutsideTextSegment')
                nxt = target
            elif kind == 'exit':
                if not frames:
                    status = 'ok'
                    break
                saved, fp, ret = frames.pop()
                r[6:10], r[10] = saved, fp
                if ret >= n:
                    raise VMError('CallOutsideTextSegment')
                nxt = ret
            pc = nxt
    except SysError as e:
        status, cu = 'Sys' + e.kind, e.cu
    except VMError as e:
        status = e.kind
    if rd['data'] and not ctx:
        logs.append('Program return: ' + b58(PROGRAM_ID) + ' ' + base64.b64encode(rd['data']).decode())
    return {'status': status, 'r0': r[0], 'cu': cu, 'regs': list(r), 'logs': logs, 'input': bytes(mem.input),
            'ixerr': ixerr if status == 'SysIx' else None,
            'hashes': (fnv(mem.stack), fnv(mem.heap), fnv(mem.input))}


def run_program(img, inp, cu_limit, ctx=None):
    """Run a loaded ELF image (solana.image) on a serialized input."""
    return run(img['text'], inp, cu_limit, gaps=img['config']['gaps'], funcs=img['funcs'], ro=img['ro'],
               ro_off=img['ro_vaddr'] - MM_PROGRAM, text_off=img['text_vaddr'] - MM_PROGRAM,
               entry=img['entry'], syscalls=img['syscalls'], ctx=ctx)
