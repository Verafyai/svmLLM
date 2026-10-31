"""svmLLM: a reduced sBPF CPU whose every operation is a hand-constructed (not trained)
transformer block, in PyTorch.

What a "step" (one sBPF instruction, one forward pass) is made of:
  * DeltaNetRAM  - Gated DeltaNet state used as exact addressable memory, exactly the
                   Qwenputer trick: gate = 1, beta in {0,1}, one-hot keys.
                   Three of them: program memory, register file, data memory.
  * LUT          - a 2-layer ReLU FFN with constructed weights that implements an exact
                   lookup table over one-hot inputs (one hidden unit per input combination).
                   The ALU (adders, multipliers, shifters, comparators) is built from these.
  * Linear maps  - constant matrices: instruction decode, sign extension, permutations.
  * ReLU gating  - relu(x + g - 1) to AND / mux 0-1 signals.

64-bit values are 16 little-endian 4-bit limbs, each limb a one-hot vector of size 16,
so a register is a 16x16 = 256-dim activation. All activations are exactly 0 or 1 in
float32, so the computation is exact (verified against two reference interpreters).

The Python side is only a driver loop: it calls step() until the halt flag is set, and the
fixed-depth ripple circuits are written as Python loops over unrolled layers. In lazy mode
(default) the driver skips ALU branches whose mux gate is 0; full mode evaluates every
branch every step and gives identical results (tested), it is just slower.
"""
import torch
from .isa import OPCODES, MM_INPUT, MM_STACK_TOP

torch.set_grad_enabled(False)
F = torch.float32
NIB = 16
PARAMS = {'count': 0}


def relu(x):
    return torch.clamp_min(x, 0.0)


def onehot(i, n):
    v = torch.zeros(n, dtype=F); v[i] = 1.0; return v


def const(m):
    t = torch.as_tensor(m, dtype=F)
    PARAMS['count'] += t.numel()
    return t


class LUT:
    """Exact lookup table as a ReLU FFN. Inputs: one-hot groups with sizes `ins`.
    Hidden unit h <-> one input combination: w=1 on each member, bias=-(k-1).
    Output layer maps each hidden unit to the one-hot outputs of fn(*combo)."""

    def __init__(self, ins, outs, fn):
        import itertools
        combos = list(itertools.product(*[range(s) for s in ins]))
        H, Din, Dout = len(combos), sum(ins), sum(outs)
        W1 = torch.zeros(H, Din); W2 = torch.zeros(Dout, H)
        offs = [sum(ins[:i]) for i in range(len(ins))]
        oofs = [sum(outs[:i]) for i in range(len(outs))]
        for h, c in enumerate(combos):
            for g, v in enumerate(c):
                W1[h, offs[g] + v] = 1.0
            r = fn(*c)
            r = r if isinstance(r, tuple) else (r,)
            for g, v in enumerate(r):
                W2[oofs[g] + v, h] = 1.0
        self.W1, self.W2 = const(W1), const(W2)
        self.b1 = const(torch.full((H,), -(len(ins) - 1.0)))
        self.outs = outs

    def __call__(self, *xs):
        x = torch.cat(xs, dim=-1)
        y = relu(x @ self.W1.T + self.b1) @ self.W2.T
        return torch.split(y, self.outs, dim=-1) if len(self.outs) > 1 else y


class DeltaNetRAM:
    """Gated DeltaNet head with gate=1 and beta in {0,1}: S <- S + beta * k (v - S^T k)^T.
    With one-hot k this overwrites exactly one row; read is S^T q."""

    def __init__(self, n, dv):
        self.S = torch.zeros(n, dv, dtype=F)

    def read(self, q):
        return q @ self.S

    def write(self, k, v, beta):
        self.S = self.S + beta * torch.outer(k, v - k @ self.S)


def AND(*xs):
    return relu(sum(xs) - (len(xs) - 1.0))


def NOT(x):
    return 1.0 - x


def OR(a, b):
    return a + b - AND(a, b)


def gate(v, g):                 # v: 0/1 tensor, g: 0/1 scalar  -> v if g else 0
    return relu(v + g - 1.0)


def mux(g, a, b):               # g ? a : b
    return gate(a, g) + gate(b, NOT(g))


# ---------------------------------------------------------------- constant tables
ZN = onehot(0, NIB)
L_ADD = LUT([16, 16, 2], [16, 2], lambda a, b, c: ((a + b + c) & 15, (a + b + c) >> 4))
L_AND = LUT([16, 16], [16], lambda a, b: a & b)
L_OR = LUT([16, 16], [16], lambda a, b: a | b)
L_XOR = LUT([16, 16], [16], lambda a, b: a ^ b)
L_MULN = LUT([16, 16, 16], [16, 16], lambda a, b, c: ((a * b + c) & 15, (a * b + c) >> 4))
L_SHL1 = LUT([16, 2], [16, 2], lambda a, i: (((a << 1) & 15) | i, a >> 3))
L_BSL = LUT([16, 16, 4], [16], lambda cur, prev, s: ((cur << s) | (prev >> (4 - s))) & 15)
L_BSR = LUT([16, 16, 4], [16], lambda cur, nxt, s: ((cur >> s) | (nxt << (4 - s))) & 15)
L_SHAMT = LUT([16, 16], [4, 16], lambda n0, n1: (n0 & 3, (n0 >> 2) | ((n1 & 3) << 2)))
L_KRON = LUT([16, 16], [256], lambda a, b: a | (b << 4))
P_NOT = const(torch.eye(16).flip(0))                     # x -> 15 - x
P_FLIP8 = const(torch.eye(16)[[i ^ 8 for i in range(16)]])  # x -> x ^ 8 (as row perm)
SIGN_IND = const([1.0 if i >= 8 else 0.0 for i in range(16)])  # top bit of a limb
SHIFT256 = [const(torch.roll(torch.eye(256), k, dims=1)) for k in range(8)]  # one-hot addr + k


def word(v, n=16):
    t = torch.zeros(n, NIB, dtype=F)
    for i in range(n):
        t[i, (v >> (4 * i)) & 15] = 1.0
    return t


def unword(t):
    return sum(int(t[i].argmax()) << (4 * i) for i in range(t.shape[0]))


# ---------------------------------------------------------------- arithmetic circuits
def add(A, B, cin):
    out, c = [], cin
    for i in range(A.shape[0]):
        s, c = L_ADD(A[i], B[i], c)
        out.append(s)
    return torch.stack(out), c


def sub(A, B):      # returns A - B and borrow flag (scalar, 1 if A < B unsigned)
    S, c = add(A, B @ P_NOT, onehot(1, 2))
    return S, c[0]


def bitwise(L, A, B):
    return L(A, B)          # batched over the 16 limbs, no carries


def eq(A, B):
    per = relu(A + B - 1.0).sum(-1)       # 1 per limb if equal
    return relu(per.sum() - (A.shape[0] - 1.0))


def is_zero(A):
    return relu(A[:, 0].sum() - (A.shape[0] - 1.0))


def mul(A, B):
    acc = word(0)
    for j in range(16):
        c, part = ZN, []
        for i in range(16 - j):               # only limbs that survive the shift
            lo, c = L_MULN(A[i], B[j], c)
            part.append(lo)
        P = torch.stack([ZN] * j + part)
        acc, _ = add(acc, P, onehot(0, 2))
    return acc


def shl1(A, inbit):
    out, c = [], inbit
    for i in range(A.shape[0]):
        o, c = L_SHL1(A[i], c)
        out.append(o)
    return torch.stack(out), c


def divmod_(A, B):   # restoring division, 64 bit-serial stages
    Q, R = word(0), word(0)
    for _ in range(64):
        A, abit = shl1(A, onehot(0, 2))
        R, _ = shl1(R, abit)
        T, borrow = sub(R, B)
        q = NOT(borrow)
        R = mux(q, T, R)
        Q, _ = shl1(Q, torch.stack([borrow, q]))
    return Q, R


def shift(A, B, kind):
    """lsh / rsh / arsh by B & 63: nibble-granular permutation mux, then bit-shift LUT."""
    s0, s1 = L_SHAMT(B[0], B[1])
    sign = (A[15] @ SIGN_IND) if kind == 'arsh' else torch.tensor(0.0)
    fill = ZN * NOT(sign) + onehot(15, NIB) * sign
    out = torch.zeros_like(A)
    for k in range(16):
        if kind == 'lsh':
            P = torch.cat([fill.expand(k, NIB), A[:16 - k]])
        else:
            P = torch.cat([A[k:], fill.expand(k, NIB)])
        out = out + gate(P, s1[k])
    if kind == 'lsh':
        prev = torch.cat([ZN[None], out[:-1]])
        return L_BSL(out, prev, s0.expand(16, 4))
    nxt = torch.cat([out[1:], fill[None]])
    return L_BSR(out, nxt, s0.expand(16, 4))


def sext(limbs, n=16):           # sign-extend a limb vector to n limbs (linear in top limb)
    s = limbs[-1] @ SIGN_IND
    fill = ZN * NOT(s) + onehot(15, NIB) * s
    return torch.cat([limbs, fill.expand(n - limbs.shape[0], NIB)])


# ---------------------------------------------------------------- decode table
ALU_NAMES = ['add', 'sub', 'mul', 'div', 'or', 'and', 'lsh', 'rsh', 'neg', 'mod', 'xor', 'mov', 'arsh']
JMP_NAMES = ['ja', 'jeq', 'jgt', 'jge', 'jset', 'jne', 'jsgt', 'jsge', 'jlt', 'jle', 'jslt', 'jsle']
SZ = {'b': 0, 'h': 1, 'w': 2, 'dw': 3}
CTRL = (['valid', 'use_reg', 'is_alu', 'is_jmp', 'is_ldx', 'is_stx', 'is_st', 'is_lddw', 'is_call', 'is_exit']
        + ['alu_' + n for n in ALU_NAMES] + ['j_' + n for n in JMP_NAMES] + ['sz' + str(i) for i in range(4)])
CI = {n: i for i, n in enumerate(CTRL)}
_dec = torch.zeros(256, len(CTRL))
for op, (kind, name, reg) in OPCODES.items():
    _dec[op, CI['valid']] = 1; _dec[op, CI['use_reg']] = float(reg)
    _dec[op, CI['is_' + kind]] = 1
    if kind == 'alu': _dec[op, CI['alu_' + name]] = 1
    if kind == 'jmp': _dec[op, CI['j_' + name]] = 1
    if kind in ('ldx', 'stx', 'st'): _dec[op, CI['sz%d' % SZ[name]]] = 1
DECODE = const(_dec)

# program-memory row layout: [present(1) | opcode(256) | dst(16) | src(16) | off 4 limbs | imm 8 limbs]
PDIM = 1 + 256 + 16 + 16 + 64 + 128
# data-address checks over the 14 high limbs (limbs 2..15)
PAT_IN = word(MM_INPUT)[2:]
PAT_ST = word(MM_STACK_TOP - 128)[2:]
SIZE_B = [1, 2, 4, 8]
V_IN = const([[1.0 if x + s <= 128 else 0.0 for x in range(256)] for s in SIZE_B])
V_ST = const([[1.0 if 128 <= x and x + s <= 256 else 0.0 for x in range(256)] for s in SIZE_B])
ERRORS = ['ExceededMaxInstructions', 'ExecutionOverrun', 'InvalidInstruction', 'DivideByZero', 'AccessViolation']


def encode_row(prog, i):
    from .isa import dec
    op, d, s, off, imm = dec(prog, i)
    v = torch.zeros(PDIM)
    v[0] = 1; v[1 + op] = 1; v[257 + d] = 1; v[273 + s] = 1
    v[289:353] = word(off & 0xffff, 4).flatten()
    v[353:481] = word(imm & 0xffffffff, 8).flatten()
    return v


class Machine:
    def __init__(self, prog, mem_in=b'', cu_limit=10000, lazy=True):
        assert len(prog) // 8 <= 256, 'program memory is 256 slots'
        self.lazy = lazy
        self.prog = DeltaNetRAM(256, PDIM)               # firmware load = delta-rule writes
        for i in range(len(prog) // 8):
            self.prog.write(onehot(i, 256), encode_row(prog, i), 1.0)
        self.regs = DeltaNetRAM(16, 256)
        for r in range(16):
            v = MM_INPUT if r == 1 else MM_STACK_TOP if r == 10 else 0
            self.regs.write(onehot(r, 16), word(v).flatten(), 1.0)
        self.mem = DeltaNetRAM(256, 32)
        mem = bytes(mem_in) + bytes(256 - len(mem_in))
        for a in range(256):
            self.mem.write(onehot(a, 256), word(mem[a], 2).flatten(), 1.0)
        self.pc = word(0, 4)
        self.cu = word(0, 6)
        self.limit = word(cu_limit, 6)
        self.halted, self.status = False, None

    def reg(self, r):
        return unword(self.regs.read(onehot(r, 16)).view(16, NIB))

    # ------------------------------------------------------------ one instruction
    def step(self):
        # CU meter: exceeded if cu == limit before executing
        cu_exc = eq(self.cu, self.limit)
        self.cu, _ = add(self.cu, word(0, 6), torch.stack([cu_exc, NOT(cu_exc)]))
        # fetch: pc -> one-hot key (pc < 256 requires limbs 2,3 == 0)
        in_range = AND(self.pc[2, 0], self.pc[3, 0])
        key = gate(L_KRON(self.pc[0], self.pc[1]), in_range)
        row = self.prog.read(key)
        pc1, _ = add(self.pc, word(0, 4), onehot(1, 2))
        key2 = gate(L_KRON(pc1[0], pc1[1]), AND(pc1[2, 0], pc1[3, 0]))
        row2 = self.prog.read(key2)
        present = row[0]
        c = row[1:257] @ DECODE                         # decode = one linear layer
        g = lambda n: c[CI[n]]
        dst, src = row[257:273], row[273:289]
        off = row[289:353].view(4, NIB)
        imm = sext(row[353:481].view(8, NIB))
        # register reads (DeltaNet queries)
        A = self.regs.read(dst).view(16, NIB)
        RB = self.regs.read(src).view(16, NIB)
        B = mux(g('use_reg'), RB, imm)
        # ---------------- ALU
        branches = {
            'add': lambda: add(A, B, onehot(0, 2))[0], 'sub': lambda: sub(A, B)[0],
            'mul': lambda: mul(A, B), 'div': lambda: divmod_(A, B)[0], 'mod': lambda: divmod_(A, B)[1],
            'or': lambda: bitwise(L_OR, A, B), 'and': lambda: bitwise(L_AND, A, B),
            'xor': lambda: bitwise(L_XOR, A, B), 'lsh': lambda: shift(A, B, 'lsh'),
            'rsh': lambda: shift(A, B, 'rsh'), 'arsh': lambda: shift(A, B, 'arsh'),
            'neg': lambda: sub(word(0), A)[0], 'mov': lambda: B,
        }
        alu = torch.zeros(16, NIB)
        for n, f in branches.items():
            gn = g('alu_' + n)
            if self.lazy and float(gn) == 0.0:
                continue                                 # gate is 0: contribution is exactly 0
            alu = alu + gate(f(), gn)
        div0 = AND(OR(g('alu_div'), g('alu_mod')), is_zero(B))
        # ---------------- branch condition
        e = eq(A, B)
        _, ult = sub(A, B)
        Af = torch.cat([A[:15], (A[15] @ P_FLIP8.T)[None]])
        Bf = torch.cat([B[:15], (B[15] @ P_FLIP8.T)[None]])
        _, slt = sub(Af, Bf)
        jset = NOT(is_zero(bitwise(L_AND, A, B)))
        truth = {'ja': torch.tensor(1.0), 'jeq': e, 'jne': NOT(e), 'jgt': AND(NOT(ult), NOT(e)),
                 'jge': NOT(ult), 'jlt': ult, 'jle': OR(ult, e), 'jset': jset,
                 'jsgt': AND(NOT(slt), NOT(e)), 'jsge': NOT(slt), 'jslt': slt, 'jsle': OR(slt, e)}
        taken = sum(AND(g('j_' + n), t) for n, t in truth.items())
        target, _ = add(pc1, off, onehot(0, 2))
        pc2, _ = add(self.pc, word(2, 4), onehot(0, 2))
        seq = mux(g('is_lddw'), pc2, pc1)
        next_pc = mux(taken, target, seq)
        # ---------------- memory
        mem_op = g('is_ldx') + g('is_stx') + g('is_st')
        base = mux(g('is_ldx'), RB, A)
        addr, _ = add(base, sext(off), onehot(0, 2))
        m_in = relu(relu(addr[2:] + PAT_IN - 1.0).sum() - 13.0)
        m_st = relu(relu(addr[2:] + PAT_ST - 1.0).sum() - 13.0)
        abyte = L_KRON(addr[0], addr[1])
        sz = torch.stack([g('sz%d' % i) for i in range(4)])
        ok_in = (sz * (V_IN @ abyte)).sum()
        ok_st = (sz * (V_ST @ abyte)).sum()
        mem_ok = OR(AND(m_in, ok_in), AND(m_st, ok_st))
        memviol = AND(mem_op, NOT(mem_ok))
        live = [sz[i:].sum() for i in [0, 1, 2, 2, 3, 3, 3, 3]]   # byte k used for size > k
        ld = []
        for k in range(8):
            kk = abyte @ SHIFT256[k]
            byte = self.mem.read(kk).view(2, NIB)
            ld.append(gate(byte, live[k]) + ZN * NOT(live[k]))
        loaded = torch.cat(ld)
        # ---------------- errors and halt
        overrun = AND(NOT(cu_exc), NOT(present))
        invalid = AND(NOT(cu_exc), present, NOT(g('valid')))
        div0, memviol = AND(div0, NOT(cu_exc)), AND(memviol, NOT(cu_exc))
        err = torch.stack([cu_exc, overrun, invalid, div0, memviol])
        ok = NOT(err.sum().clamp(max=1.0))
        # ---------------- writes (DeltaNet with input-dependent beta)
        sval = mux(g('is_stx'), RB, imm)
        st_on = AND(OR(g('is_stx'), g('is_st')), mem_ok, ok)
        for k in range(8):
            self.mem.write(abyte @ SHIFT256[k], sval[2 * k:2 * k + 2].flatten(), AND(st_on, live[k]))
        lddw_v = torch.cat([row[353:481].view(8, NIB), row2[353:481].view(8, NIB)])
        wval = (gate(alu, g('is_alu')) + gate(loaded, g('is_ldx')) + gate(lddw_v, g('is_lddw'))
                + gate(word(0), g('is_call')))
        wkey = mux(g('is_call'), onehot(0, 16), dst)
        wr = AND(OR(OR(g('is_alu'), g('is_ldx')), OR(g('is_lddw'), g('is_call'))), ok)
        self.regs.write(wkey, wval.flatten(), wr)
        self.pc = mux(ok, next_pc, self.pc)
        exit_ = AND(g('is_exit'), ok)
        if float(err.sum()) > 0:
            self.halted, self.status = True, ERRORS[int(err.argmax())]
        elif float(exit_) > 0:
            self.halted, self.status = True, 'ok'


def run(prog, mem_in=b'', cu_limit=10000, lazy=True, trace=None, max_steps=100000):
    m = Machine(prog, mem_in, cu_limit, lazy)
    steps = 0
    while not m.halted and steps < max_steps:          # the Python driver loop
        pc = unword(m.pc)
        m.step(); steps += 1
        if trace is not None:
            trace.append({'pc': pc, 'cu': unword(m.cu), 'regs': [m.reg(r) for r in range(11)]})
    mem = bytes(unword(m.mem.read(onehot(a, 256)).view(2, NIB)) for a in range(256))
    return {'status': m.status, 'r0': m.reg(0), 'cu': unword(m.cu),
            'regs': [m.reg(r) for r in range(11)], 'mem': mem}
