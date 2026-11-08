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
    def __init__(self, prog, inp, gaps):
        self.ro, self.gaps = bytes(prog), gaps
        self.stack, self.heap, self.input = bytearray(FRAME * DEPTH), bytearray(HEAP), bytearray(inp)

    def at(self, a, n, store):
        """Translate like solana-sbpf's aligned mapping. Returns (buffer, offset)."""
        reg, off = a >> 32, a & M32
        if a >> 35 == 0 and 1 <= reg <= 4:
            buf = [None, self.ro, self.stack, self.heap, self.input][reg]
            if reg == 1 and store:
                raise VMError('AccessViolation')
            if reg == 2 and self.gaps:
                if (off >> 12) & 1:
                    raise VMError('AccessViolation')
                off = ((off >> 13) << 12) | (off & 0xfff)
            if off + n <= len(buf):
                return buf, off
        raise VMError('AccessViolation')

    def load(self, a, n):
        b, o = self.at(a, n, False)
        return int.from_bytes(b[o:o + n], 'little')

    def store(self, a, n, v):
        b, o = self.at(a, n, True)
        b[o:o + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, 'little')


def run(prog, inp=b'', cu_limit=10000, gaps=True, funcs=None, trace=None):
    funcs = funcs or {}
    n = len(prog) // 8
    mem = Mem(prog, inp, gaps)
    r = [0] * 11
    r[1], r[10] = MM_INPUT, MM_STACK + FRAME
    frames, pc, cu, status = [], 0, 0, None
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
                    target = ((r[imm] if 0 <= imm <= 10 else 0) - MM_PROGRAM) & M64
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
    except VMError as e:
        status = e.kind
    return {'status': status, 'r0': r[0], 'cu': cu, 'regs': list(r),
            'hashes': (fnv(mem.stack), fnv(mem.heap), fnv(mem.input))}
