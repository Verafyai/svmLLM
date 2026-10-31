"""Reduced sBPF (SBPF v0 encoding) opcodes, a small assembler, and a disassembler.

Instruction: 8 bytes, little endian: opcode u8 | dst:4 src:4 | off i16 | imm i32.
lddw takes two slots (imm of slot 2 is the high 32 bits).
"""
import re, struct

ALU_OPS = {  # name -> high nibble of ALU64 opcode
    'add': 0x00, 'sub': 0x10, 'mul': 0x20, 'div': 0x30, 'or': 0x40, 'and': 0x50,
    'lsh': 0x60, 'rsh': 0x70, 'neg': 0x80, 'mod': 0x90, 'xor': 0xa0, 'mov': 0xb0, 'arsh': 0xc0,
}
JMP_OPS = {
    'ja': 0x00, 'jeq': 0x10, 'jgt': 0x20, 'jge': 0x30, 'jset': 0x40, 'jne': 0x50, 'jsgt': 0x60,
    'jsge': 0x70, 'jlt': 0xa0, 'jle': 0xb0, 'jslt': 0xc0, 'jsle': 0xd0,
}
SIZES = {'b': 0x10, 'h': 0x08, 'w': 0x00, 'dw': 0x18}
SIZE_BYTES = {'b': 1, 'h': 2, 'w': 4, 'dw': 8}
CALL, EXIT, LDDW = 0x85, 0x95, 0x18

# opcode -> (kind, detail)
OPCODES = {}
for n, h in ALU_OPS.items():
    OPCODES[0x07 | h] = ('alu', n, False)
    if n != 'neg':
        OPCODES[0x0f | h] = ('alu', n, True)
for n, h in JMP_OPS.items():
    if n == 'ja':
        OPCODES[0x05] = ('jmp', n, False)
    else:
        OPCODES[0x05 | h] = ('jmp', n, False)
        OPCODES[0x0d | h] = ('jmp', n, True)
for s, m in SIZES.items():
    OPCODES[0x61 | m] = ('ldx', s, True)
    OPCODES[0x63 | m] = ('stx', s, True)
    OPCODES[0x62 | m] = ('st', s, False)
OPCODES[CALL] = ('call', None, False)
OPCODES[EXIT] = ('exit', None, False)
OPCODES[LDDW] = ('lddw', None, False)

MM_INPUT = 0x400000000
MM_STACK_TOP = 0x200001000


def enc(op, dst=0, src=0, off=0, imm=0):
    imm &= 0xffffffff
    imm = imm - (1 << 32) if imm >> 31 else imm
    return struct.pack('<BBhi', op, (src << 4) | dst, off, imm)


def dec(b, i):
    op, regs, off, imm = struct.unpack_from('<BBhi', b, i * 8)
    return op, regs & 0xf, regs >> 4, off, imm


def _int(s):
    return int(s, 0)


def _reg(s):
    m = re.fullmatch(r'r(\d+)', s.strip())
    if not m or int(m.group(1)) > 10:
        raise ValueError(f'bad register {s!r}')
    return int(m.group(1))


def assemble(text):
    """Assemble rbpf-style text. Returns bytes. Supports labels ('name:')."""
    lines = []
    for raw in text.splitlines():
        line = re.split(r'#|//|;', raw)[0].strip()
        if not line:
            continue
        while ':' in line.split()[0] if line else False:
            lab, _, line = line.partition(':')
            lines.append(('label', lab.strip()))
            line = line.strip()
            if not line:
                break
        if line:
            lines.append(('insn', line))
    labels, pc = {}, 0
    for kind, v in lines:
        if kind == 'label':
            labels[v] = pc
        else:
            pc += 2 if v.split()[0] == 'lddw' else 1
    out, pc = b'', 0
    for kind, v in lines:
        if kind == 'label':
            continue
        mn, _, rest = v.partition(' ')
        args = [a.strip() for a in rest.split(',')] if rest.strip() else []

        def target(a):
            if a in labels:
                return labels[a] - pc - 1
            return _int(a)
        m = re.fullmatch(r'([a-z]+)(64|32)?', mn)
        base = m.group(1) if m else mn
        if mn == 'exit':
            out += enc(EXIT)
        elif mn == 'call':
            out += enc(CALL, imm=_int(args[0]))
        elif mn == 'lddw':
            v64 = _int(args[1]) & (2**64 - 1)
            lo, hi = v64 & 0xffffffff, v64 >> 32
            out += enc(LDDW, _reg(args[0]), imm=struct.unpack('<i', struct.pack('<I', lo))[0])
            out += enc(0, imm=struct.unpack('<i', struct.pack('<I', hi))[0])
            pc += 1
        elif base in ALU_OPS and m.group(2) == '64':
            h = ALU_OPS[base]
            if base == 'neg':
                out += enc(0x87, _reg(args[0]))
            elif args[1].startswith('r'):
                out += enc(0x0f | h, _reg(args[0]), _reg(args[1]))
            else:
                out += enc(0x07 | h, _reg(args[0]), imm=_int(args[1]))
        elif base in JMP_OPS:
            h = JMP_OPS[base]
            if base == 'ja':
                out += enc(0x05, off=target(args[0]))
            elif args[1].startswith('r'):
                out += enc(0x0d | h, _reg(args[0]), _reg(args[1]), off=target(args[2]))
            else:
                out += enc(0x05 | h, _reg(args[0]), imm=_int(args[1]), off=target(args[2]))
        elif re.fullmatch(r'(ldx|stx|st)(b|h|w|dw)', mn):
            mm = re.fullmatch(r'(ldx|stx|st)(b|h|w|dw)', mn)
            k, s = mm.groups()
            def memref(a):
                mr = re.fullmatch(r'\[\s*(r\d+)\s*([+-]\s*\w+)?\s*\]', a)
                if not mr:
                    raise ValueError(f'bad memory operand {a!r}')
                return _reg(mr.group(1)), _int(mr.group(2).replace(' ', '')) if mr.group(2) else 0
            if k == 'ldx':
                r, o = memref(args[1]); out += enc(0x61 | SIZES[s], _reg(args[0]), r, o)
            elif k == 'stx':
                r, o = memref(args[0]); out += enc(0x63 | SIZES[s], r, _reg(args[1]), o)
            else:
                r, o = memref(args[0]); out += enc(0x62 | SIZES[s], r, 0, o, _int(args[1]))
        else:
            raise ValueError(f'unsupported instruction {v!r}')
        pc += 1
    return out


def disasm_one(b, i):
    op, d, s, off, imm = dec(b, i)
    if op not in OPCODES:
        return f'.invalid 0x{op:02x}'
    kind, n, reg = OPCODES[op]
    if kind == 'alu':
        if n == 'neg':
            return f'neg64 r{d}'
        return f'{n}64 r{d}, ' + (f'r{s}' if reg else f'{imm}')
    if kind == 'jmp':
        t = f'{off:+d}'
        if n == 'ja':
            return f'ja {t}'
        return f'{n} r{d}, ' + (f'r{s}' if reg else f'{imm}') + f', {t}'
    if kind == 'ldx':
        return f'ldx{n} r{d}, [r{s}{off:+d}]'
    if kind == 'stx':
        return f'stx{n} [r{d}{off:+d}], r{s}'
    if kind == 'st':
        return f'st{n} [r{d}{off:+d}], {imm}'
    if kind == 'lddw':
        hi = dec(b, i + 1)[4] if len(b) >= (i + 2) * 8 else 0
        return f'lddw r{d}, 0x{((hi & 0xffffffff) << 32) | (imm & 0xffffffff):x}'
    return {'call': f'call {imm}', 'exit': 'exit'}[kind]
