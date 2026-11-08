"""Plain-Python reference interpreter for the reduced sBPF subset (SBPF v0 semantics).

Memory model (toy): two windows of `win` bytes (128 by default), mirroring Solana's layout:
  input  0x400000000 .. +win   (r1 points here at entry)
  stack  0x200001000 - win .. 0x200001000  (r10 = 0x200001000, the frame top)
The settlement core (svllm/settle.py) uses win = 1024.
Compute units: 1 per executed instruction (lddw counts once). The meter is checked
before each instruction, like solana-sbpf's interpreter.
"""
from .isa import dec, OPCODES, MM_INPUT, MM_STACK_TOP, SIZE_BYTES

M64 = 2**64 - 1
M32 = 2**32 - 1
WIN = 128


def sx32(v):
    return v - (1 << 32) if v & 0x80000000 else v


def sx64(v):                     # low 32 bits, sign-extended to 64
    return sx32(v & M32) & M64


def s64(v):
    return v - (1 << 64) if v >> 63 else v


class VMError(Exception):
    def __init__(self, kind):
        super().__init__(kind); self.kind = kind


def _addr(a, size, win=WIN):
    if MM_INPUT <= a and a + size <= MM_INPUT + win:
        return a - MM_INPUT
    if MM_STACK_TOP - win <= a and a + size <= MM_STACK_TOP:
        return win + (a - (MM_STACK_TOP - win))
    raise VMError('AccessViolation')


def run(prog, mem_in=b'', cu_limit=10000, trace=None, win=WIN):
    """Returns dict(status, r0, cu, regs, mem). status 'ok' or an error kind."""
    n = len(prog) // 8
    mem = bytearray(2 * win)
    mem[:len(mem_in)] = mem_in
    r = [0] * 11
    r[1], r[10] = MM_INPUT, MM_STACK_TOP
    pc, cu = 0, 0
    status = None
    try:
        while True:
            if cu >= cu_limit:
                raise VMError('ExceededMaxInstructions')
            cu += 1
            if pc >= n:
                raise VMError('ExecutionOverrun')
            op, d, s, off, imm = dec(prog, pc)
            if op not in OPCODES:
                raise VMError('InvalidInstruction')
            kind, name, isreg = OPCODES[op]
            nxt = pc + 1
            if trace is not None:
                trace.append({'pc': pc, 'cu': cu})
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
                # SBPF v0: add/sub/mul sign-extend the 32-bit result; the rest zero-extend.
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
                nb = imm // 8; v = r[d] & ((1 << imm) - 1)
                r[d] = int.from_bytes(v.to_bytes(nb, 'little'), 'big') if name == 'be' else v
            elif kind == 'jmp':
                a = r[d]; b = r[s] if isreg else (imm & M64)
                t = {'ja': True, 'jeq': a == b, 'jne': a != b, 'jgt': a > b, 'jge': a >= b,
                     'jlt': a < b, 'jle': a <= b, 'jset': (a & b) != 0,
                     'jsgt': s64(a) > s64(b), 'jsge': s64(a) >= s64(b),
                     'jslt': s64(a) < s64(b), 'jsle': s64(a) <= s64(b)}[name]
                if t:
                    nxt = (pc + 1 + off) & 0xffff
            elif kind in ('ldx', 'stx', 'st'):
                size = SIZE_BYTES[name]
                base = r[s] if kind == 'ldx' else r[d]
                i = _addr((base + off) & M64, size, win)
                if kind == 'ldx':
                    r[d] = int.from_bytes(mem[i:i + size], 'little')
                else:
                    v = r[s] if kind == 'stx' else (imm & M64)
                    mem[i:i + size] = (v & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')
            elif kind == 'lddw':
                hi = dec(prog, pc + 1)[4] if pc + 1 < n else 0
                r[d] = ((hi & 0xffffffff) << 32) | (imm & 0xffffffff)
                nxt = pc + 2
            elif kind == 'call':
                r[0] = 0  # syscall stub
            elif kind == 'exit':
                status = 'ok'
                break
            pc = nxt
    except VMError as e:
        status = e.kind
    return {'status': status, 'r0': r[0], 'cu': cu, 'regs': list(r), 'mem': bytes(mem)}
