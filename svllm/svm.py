"""The full-layout sBPF machine in hand-set weights: Solana's real memory map and BPF-to-BPF calls.

Builds on the circuits in machine.py (LUT FFNs, ripple adders, DeltaNet memory, ReLU gating).
New here:
  * Address translation in the circuits: region select from the high 32 bits, the stack
    frame-gap rule, per-region bounds (offset + size <= length, by a subtractor's borrow)
    and the read-only program region. Region lengths and bases are firmware constants.
  * PagedRAM: DeltaNet memory over a 2^24 (program) or 2^28 (data) row address space.
    Keys are one-hot per 4-bit limb, so the key is the Kronecker product of the limbs. A dense
    read is S^T q with q one-hot; PagedRAM computes the identical result by indexing the one
    row q selects, and stores only rows that were written (all others are exactly zero).
    Small RAMs (registers, frames) stay dense.
  * Function table as exact-match attention: the call immediate is the query, the
    registered 32-bit keys are the keys, the target pcs are the values. A match is an
    equality circuit over 8 limbs; at most one key matches.
  * A call-frame stack (DeltaNet, 64 rows) saving r6..r9, r10 and the return pc.
"""
import torch
from .machine import (F, NIB, relu, onehot, const, LUT, DeltaNetRAM, AND, NOT, OR, gate, mux, ZN,
                      L_KRON, L_OR, L_AND, L_XOR, P_AND1, word, unword, add, sub, bitwise, eq, is_zero, mul,
                      divmod_, shift, sext, zx, byteswap, DECODE, CI, PDIM, P_FLIP8, encode_row as _enc_row)
from .sref import MM_PROGRAM, MM_STACK, MM_HEAP, MM_INPUT, FRAME, DEPTH, HEAP, fnv
from . import syscalls

PCL, AL = 6, 7                     # pc limbs (2^24 instructions), physical address limbs (2^28 bytes)
ERRORS = ['ExceededMaxInstructions', 'ExecutionOverrun', 'InvalidInstruction', 'DivideByZero',
          'AccessViolation', 'CallDepthExceeded', 'CallOutsideTextSegment', 'ComputationalBudgetExceeded',
          'Abort', 'CopyOverlapping', 'InvalidString']
SYS = ['abort', 'sol_log_', 'sol_memcpy_', 'sol_memset_']


def _utf8_dfa():
    """Strict UTF-8 (Rust's from_utf8) as a 9-state automaton: 0 accept, 1-3 that many continuation
    bytes due, 4 after E0, 5 after ED, 6 after F0, 7 after F4, 8 reject."""
    def nxt(s, b):
        if s == 0:
            if b < 0x80: return 0
            if 0xc2 <= b <= 0xdf: return 1
            if b == 0xe0: return 4
            if 0xe1 <= b <= 0xec or 0xee <= b <= 0xef: return 2
            if b == 0xed: return 5
            if b == 0xf0: return 6
            if 0xf1 <= b <= 0xf3: return 3
            if b == 0xf4: return 7
            return 8
        lo, hi, to = {1: (0x80, 0xbf, 0), 2: (0x80, 0xbf, 1), 3: (0x80, 0xbf, 2), 4: (0xa0, 0xbf, 1),
                      5: (0x80, 0x9f, 1), 6: (0x90, 0xbf, 2), 7: (0x80, 0x8f, 2)}.get(s, (1, 0, 8))
        return to if lo <= b <= hi else 8
    return LUT([9, 16, 16], [9], lambda s, lo, hi: nxt(s, lo | (hi << 4)))


L_UTF8 = _utf8_dfa()
L_SR1 = LUT([16, 16], [16], lambda cur, nxt: ((cur >> 1) | (nxt << 3)) & 15)
ODD = const([float(i & 1) for i in range(16)])
SIZE_W = [word(s) for s in (1, 2, 4, 8)]


def limbs(t, n):                   # truncate or zero-extend a limb vector to n limbs
    return t[:n] if t.shape[0] >= n else torch.cat([t, ZN.expand(n - t.shape[0], NIB)])


class PagedRAM:
    """DeltaNet state S (rows x dv) addressed by limb one-hot keys. read = S^T kron(key limbs);
    write = delta rule with beta in {0, 1}. Exact: only the selected row can change."""

    def __init__(self, dv):
        self.dv, self.rows = dv, {}

    @staticmethod
    def index(K):
        s = K.sum(-1)
        if bool((s == 0).any()):
            return None                                   # a zero key reads zero, writes nothing
        assert bool((s == 1).all()) and bool(((K == 0) | (K == 1)).all()), 'key is not one-hot'
        return sum(int(K[i].argmax()) << (4 * i) for i in range(K.shape[0]))

    def read(self, K):
        i = self.index(K)
        return self.rows.get(i, torch.zeros(self.dv, dtype=F)) if i is not None else torch.zeros(self.dv, dtype=F)

    def write(self, K, v, beta):
        if float(beta) == 0.0:
            return
        i = self.index(K)
        if i is None:
            return
        old = self.rows.get(i, torch.zeros(self.dv, dtype=F))
        self.rows[i] = old + beta * (v - old)


class Machine:
    def __init__(self, prog, inp=b'', cu_limit=10000, gaps=True, funcs=None, lazy=True, ro=None, ro_off=0,
                 text_off=0, entry=0, syscalls=None, program_id=None):
        self.lazy, self.n = lazy, len(prog) // 8
        ro = bytes(prog if ro is None else ro)
        self.prog = PagedRAM(PDIM)
        for i in range(self.n):
            self.prog.write(word(i, PCL), _enc_row(prog, i), 1.0)
        self.regs = DeltaNetRAM(16, 256)
        for r in range(16):
            v = MM_INPUT if r == 1 else MM_STACK + FRAME if r == 10 else 0
            self.regs.write(onehot(r, 16), word(v).flatten(), 1.0)
        # physical layout: [program | stack | heap | input]
        R = len(ro)
        self.base = {1: 0, 2: R, 3: R + FRAME * DEPTH, 4: R + FRAME * DEPTH + HEAP}
        self.rlen = {1: R, 2: FRAME * DEPTH, 3: HEAP, 4: len(inp)}
        self.RO_OFF = word(ro_off)
        self.TEXT_BASE = word(MM_PROGRAM + text_off)
        self.BASE = {k: word(v, AL) for k, v in self.base.items()}
        self.RLEN = {k: word(v) for k, v in self.rlen.items()}
        self.GAPS = torch.tensor(float(gaps))
        self.FP_STEP = word(FRAME * (2 if gaps else 1))
        self.mem = PagedRAM(32)
        blob = ro + bytes(FRAME * DEPTH + HEAP) + bytes(inp)
        for a, b in enumerate(blob):
            if b:
                self.mem.write(word(a, AL), word(b, 2).flatten(), 1.0)
        # function table: exact-match attention, keys 8 limbs (32 bits), values pc
        funcs = funcs or {}
        self.fkeys = [word(k, 8) for k in funcs]
        self.fvals = [word(v, PCL) for v in funcs.values()]
        # syscall table: the same exact-match attention, values are the syscall one-hot
        syscalls = syscalls or {}
        self.skeys = [word(k, 8) for k in syscalls]
        self.snames = list(syscalls.values())
        self.svals = [onehot(i, max(1, len(self.snames))) for i in range(len(self.snames))]
        self.sys_gen = None
        self.rdata, self.rdlen = PagedRAM(32), word(0)
        self.heap_pos = word(0)                          # the runtime's bump allocator position
        self.ctx, self.ixerr, self.rd_dirty = None, None, False
        self.PROGRAM_ID = [word(b, 2) for b in (program_id or bytes(32))]
        # syscall microcode state: mode one-hot (log, memcpy, memset), byte counter, length,
        # source and destination physical addresses, fill byte, UTF-8 automaton state
        self.mode = torch.zeros(3)
        self.i, self.cnt = word(0), word(0)
        self.src_p, self.dst_p = torch.zeros(AL, NIB), torch.zeros(AL, NIB)
        self.cbyte, self.dfa = word(0, 2), onehot(0, 9)
        self.logs, self.line = [], []
        self.frames = DeltaNetRAM(256, 5 * 256 + PCL * NIB)
        self.depth = word(0, 2)
        self.pc = word(entry, PCL)
        self.cu = word(0, PCL)
        self.limit = word(cu_limit, PCL)
        self.halted, self.status = False, None

    def reg(self, r):
        return unword(self.regs.read(onehot(r, 16)).view(16, NIB))

    def translate(self, addr, size, store, with_offset=False):
        """addr: 16 limbs, size: 16 limbs (bytes). Returns (valid scalar, physical address AL limbs)."""
        off = addr[:8]
        # stack frame gaps: bit 12 set is a gap; otherwise drop bit 12 and shift the rest down
        hi = torch.stack([L_SR1(off[i], off[i + 1] if i < 7 else ZN) for i in range(3, 8)])
        gapped = torch.cat([off[:3], hi])
        gapbit = off[3] @ ODD
        soff = mux(self.GAPS, gapped, off)
        in_gap = AND(self.GAPS, gapbit)
        valid, phys, roff = torch.tensor(0.0), torch.zeros(AL, NIB), torch.zeros(8, NIB)
        for k in (1, 2, 3, 4):
            sel = eq(addr[8:], word(k << 32)[8:])
            o = soff if k == 2 else off
            if k == 1:                                    # the read-only region may start past MM_PROGRAM
                o16, below = sub(zx(o), self.RO_OFF)
                o = o16[:8]
            end, carry = add(zx(o), size, onehot(0, 2))
            _, borrow = sub(self.RLEN[k], end)            # borrow = length < end
            ok = AND(sel, NOT(borrow), carry[0])          # carry[0]: offset + size did not overflow
            if k == 2:
                ok = AND(ok, NOT(in_gap))
            if k == 1:
                ok = AND(ok, NOT(store), NOT(below))
            p, _ = add(self.BASE[k], limbs(o, AL), onehot(0, 2))
            valid = valid + ok
            phys = phys + gate(p, ok)
            roff = roff + gate(o, ok)
        return (valid, phys, roff) if with_offset else (valid, phys)

    def sys_step(self):
        """One micro-step of the running syscall."""
        try:
            next(self.sys_gen)
        except StopIteration as done:
            self.regs.write(onehot(0, 16), done.value.flatten(), torch.tensor(1.0))
            self.sys_gen = None
        except syscalls.SysFail as f:
            self.halted, self.status, self.sys_gen = True, 'Sys' + f.kind, None

    def step(self):
        if self.sys_gen is not None:                      # a syscall is mid-flight
            return self.sys_step()
        lazy = self.lazy
        cu_exc = eq(self.cu, self.limit)
        self.cu, _ = add(self.cu, word(0, PCL), torch.stack([cu_exc, NOT(cu_exc)]))
        row = self.prog.read(self.pc)
        pc1, _ = add(self.pc, word(0, PCL), onehot(1, 2))
        row2 = self.prog.read(pc1)
        present = row[0]
        c = row[1:257] @ DECODE
        g = lambda n: c[CI[n]]
        dst, src = row[257:273], row[273:289]
        off = row[289:353].view(4, NIB)
        imm = sext(row[353:481].view(8, NIB))
        A = self.regs.read(dst).view(16, NIB)
        RB = self.regs.read(src).view(16, NIB)
        B = mux(g('use_reg'), RB, imm)
        # ---------------- ALU (64- and 32-bit), byte order
        branches = {
            'add': lambda: add(A, B, onehot(0, 2))[0], 'sub': lambda: sub(A, B)[0],
            'mul': lambda: mul(A, B), 'div': lambda: divmod_(A, B)[0], 'mod': lambda: divmod_(A, B)[1],
            'or': lambda: bitwise(L_OR, A, B), 'and': lambda: bitwise(L_AND, A, B),
            'xor': lambda: bitwise(L_XOR, A, B), 'lsh': lambda: shift(A, B, 'lsh'),
            'rsh': lambda: shift(A, B, 'rsh'), 'arsh': lambda: shift(A, B, 'arsh'),
            'neg': lambda: sub(word(0), A)[0], 'mov': lambda: B,
        }
        Az, Bz = zx(A[:8]), zx(B[:8])
        Bs = zx(torch.stack([B[0], B[1] @ P_AND1]))
        branches32 = {
            'add': lambda: sext(add(A, B, onehot(0, 2))[0][:8]), 'sub': lambda: sext(sub(A, B)[0][:8]),
            'mul': lambda: sext(mul(A, B)[:8]), 'div': lambda: divmod_(Az, Bz)[0], 'mod': lambda: divmod_(Az, Bz)[1],
            'or': lambda: zx(bitwise(L_OR, A, B)[:8]), 'and': lambda: zx(bitwise(L_AND, A, B)[:8]),
            'xor': lambda: zx(bitwise(L_XOR, A, B)[:8]), 'lsh': lambda: zx(shift(Az, Bs, 'lsh')[:8]),
            'rsh': lambda: shift(Az, Bs, 'rsh'), 'arsh': lambda: zx(shift(sext(A[:8]), Bs, 'arsh')[:8]),
            'neg': lambda: zx(sub(word(0), A)[0][:8]), 'mov': lambda: Bz,
        }
        w32 = g('w32')
        alu = torch.zeros(16, NIB)
        for n, f in branches.items():
            gn = g('alu_' + n)
            if lazy and float(gn) == 0.0:
                continue
            v = (branches32[n]() if float(w32) else f()) if lazy else mux(w32, branches32[n](), f())
            alu = alu + gate(v, gn)
        div0 = AND(OR(g('alu_div'), g('alu_mod')), mux(w32, is_zero(B[:8]), is_zero(B)))
        if lazy and float(g('is_end')) == 0.0:          # gate is 0: exactly zero contribution
            end_v, end_bad = word(0), torch.tensor(0.0)
        else:
            wid = [eq(imm, word(16)), eq(imm, word(32)), eq(imm, word(64))]
            le_v = gate(zx(A[:4]), wid[0]) + gate(zx(A[:8]), wid[1]) + gate(A, wid[2])
            be_v = gate(byteswap(A, 2), wid[0]) + gate(byteswap(A, 4), wid[1]) + gate(byteswap(A, 8), wid[2])
            end_v = mux(g('end_be'), be_v, le_v)
            end_bad = AND(g('is_end'), NOT(wid[0] + wid[1] + wid[2]))
        # ---------------- branches
        if lazy and float(g('is_jmp')) == 0.0:
            taken = torch.tensor(0.0)
        else:
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
        target, _ = add(pc1, limbs(sext(off), PCL), onehot(0, 2))
        pc2, _ = add(self.pc, word(2, PCL), onehot(0, 2))
        seq = mux(g('is_lddw'), pc2, pc1)
        next_pc = mux(taken, target, seq)
        # ---------------- memory
        mem_op = g('is_ldx') + g('is_stx') + g('is_st')
        store = g('is_stx') + g('is_st')
        sz = torch.stack([g('sz%d' % i) for i in range(4)])
        live = [sz[i:].sum() for i in [0, 1, 2, 2, 3, 3, 3, 3]]
        if lazy and float(mem_op) == 0.0:
            mem_ok, phys, loaded = torch.tensor(0.0), torch.zeros(AL, NIB), word(0)
        else:
            base = mux(g('is_ldx'), RB, A)
            addr, _ = add(base, sext(off), onehot(0, 2))
            mem_ok, phys = self.translate(addr, sum(gate(SIZE_W[i], sz[i]) for i in range(4)), store)
            keys = [add(phys, word(k, AL), onehot(0, 2))[0] for k in range(8)]
            ld = []
            for k in range(8):
                byte = self.mem.read(gate(keys[k], mem_ok)).view(2, NIB)
                byte = byte + ZN * (1.0 - byte.sum(-1, keepdim=True))   # unwritten row = byte 0
                ld.append(gate(byte, live[k]) + ZN * NOT(live[k]))
            loaded = torch.cat(ld)
        memviol = AND(mem_op, NOT(mem_ok))
        # ---------------- calls and returns
        is_callx = row[1 + 0x8d]
        ik = row[353:481].view(8, NIB)
        # syscall table lookup first (SBPF v0 checks the loader's registry before the program's)
        shits = [eq(ik, k) for k in self.skeys] if float(g('is_call')) or not lazy else []
        sid = sum(gate(v, h) for v, h in zip(self.svals, shits)) if shits else torch.zeros(1)
        is_sys = AND(g('is_call'), NOT(is_callx), sid.sum())
        is_call = AND(g('is_call'), NOT(is_callx), NOT(is_sys))
        callish = OR(is_call, is_callx)
        at_top = is_zero(self.depth)
        popping = AND(g('is_exit'), NOT(at_top))
        dep1, _ = add(self.depth, word(0, 2), onehot(1, 2))
        depm, _ = sub(self.depth, word(1, 2))
        if lazy and float(callish) == 0.0 and float(popping) == 0.0:
            unsupported = depth_exc = outside = torch.tensor(0.0)
            call_pc, frame_row, fr = word(0, PCL), torch.zeros(5 * 256 + PCL * NIB), None
        else:
            # exact-match attention over the function table
            hits = [eq(ik, k) for k in self.fkeys]
            found = sum(hits) if hits else torch.tensor(0.0)
            fpc = sum(gate(v, h) for v, h in zip(self.fvals, hits)) if hits else word(0, PCL)
            unsupported = AND(is_call, NOT(found))
            # callx: target = r[imm]; pc = (target - MM_PROGRAM) / 8
            tx = self.regs.read(imm[0]).view(16, NIB)
            t, _ = sub(tx, self.TEXT_BASE)
            t3 = shift(t, word(3), 'rsh')
            xhigh = NOT(is_zero(t3[PCL:]))
            call_pc = mux(is_callx, limbs(t3, PCL), fpc)
            depth_exc = AND(callish, NOT(unsupported), eq(dep1, word(64, 2)))
            tgt_present = self.prog.read(call_pc)[0]
            outside = AND(callish, NOT(unsupported), NOT(depth_exc), OR(NOT(tgt_present), AND(is_callx, xhigh)))
            saved = torch.cat([self.regs.read(onehot(r, 16)) for r in (6, 7, 8, 9, 10)])
            frame_row = torch.cat([saved, pc1.flatten()])
            fr = self.frames.read(L_KRON(depm[0], depm[1]))
            ret_present = self.prog.read(fr[1280:].view(PCL, NIB))[0]
            outside = outside + AND(popping, NOT(ret_present))
        # ---------------- errors and halt
        overrun = AND(NOT(cu_exc), NOT(present))
        invalid = AND(NOT(cu_exc), present, OR(OR(NOT(g('valid')), end_bad), unsupported))
        div0, memviol = AND(div0, NOT(cu_exc)), AND(memviol, NOT(cu_exc))
        depth_exc, outside = AND(depth_exc, NOT(cu_exc)), AND(outside, NOT(cu_exc))
        err = torch.stack([cu_exc, overrun, invalid, div0, memviol, depth_exc, outside])
        ok = NOT(err.sum().clamp(max=1.0))
        # ---------------- writes
        sval = mux(g('is_stx'), RB, imm)
        st_on = AND(store, mem_ok, ok)
        if not (lazy and float(st_on) == 0.0):
            for k in range(8):
                self.mem.write(gate(keys[k], mem_ok), sval[2 * k:2 * k + 2].flatten(), AND(st_on, live[k]))
        lddw_v = torch.cat([row[353:481].view(8, NIB), row2[353:481].view(8, NIB)])
        wval = (gate(alu, g('is_alu')) + gate(loaded, g('is_ldx')) + gate(lddw_v, g('is_lddw'))
                + gate(end_v, g('is_end')))
        wr = AND(OR(OR(g('is_alu'), g('is_ldx')), OR(g('is_lddw'), g('is_end'))), ok)
        self.regs.write(dst, wval.flatten(), wr)
        if float(AND(is_sys, ok)):                        # start the syscall's microprogram
            self.sys_gen = syscalls.program(self, self.snames[int(sid.argmax())])
            self.sys_step()
        # push a frame on call: save, depth + 1, r10 += frame step (pushes even when the depth
        # check then fails, as solana-sbpf does; the error halts the run either way)
        push = AND(callish, NOT(unsupported), ok)
        if not (lazy and float(push) == 0.0):
            self.frames.write(L_KRON(self.depth[0], self.depth[1]), frame_row, push)
            r10 = self.regs.read(onehot(10, 16)).view(16, NIB)
            self.regs.write(onehot(10, 16), add(r10, self.FP_STEP, onehot(0, 2))[0].flatten(), push)
        pop = AND(popping, ok)
        if not (lazy and float(pop) == 0.0):
            for i, r in enumerate((6, 7, 8, 9, 10)):
                self.regs.write(onehot(r, 16), fr[256 * i:256 * (i + 1)], pop)
        ret_pc = fr[1280:].view(PCL, NIB) if fr is not None else word(0, PCL)
        self.depth = mux(push, dep1, mux(pop, depm, self.depth))
        new_pc = mux(callish, call_pc, mux(popping, ret_pc, next_pc))
        self.pc = mux(ok, new_pc, self.pc)
        exit_ = AND(g('is_exit'), at_top, ok)
        if float(err.sum()) > 0:
            self.halted, self.status = True, ERRORS[int(err.argmax())]
        elif float(exit_) > 0:
            self.halted, self.status = True, 'ok'

    def region_bytes(self, k):
        b, n = self.base[k], self.rlen[k]
        out = bytearray(n)
        for i, v in self.mem.rows.items():
            if b <= i < b + n:
                out[i - b] = unword(v.view(2, NIB))
        return bytes(out)


def run_program(img, inp, cu_limit, lazy=True, program_id=None, program_id_b58='', ctx=None):
    """Run a loaded ELF image (solana.image) on a serialized input, in the weights."""
    m = Machine(img['text'], inp, cu_limit, img['config']['gaps'], img['funcs'], lazy, ro=img['ro'],
                ro_off=img['ro_vaddr'] - MM_PROGRAM, text_off=img['text_vaddr'] - MM_PROGRAM,
                entry=img['entry'], syscalls=img['syscalls'], program_id=program_id)
    m.ctx = ctx
    if ctx:
        syscalls.import_rd(m)
    steps = 0
    while not m.halted:
        m.step(); steps += 1
    if ctx:
        syscalls.export_rd(m)
    if float(NOT(is_zero(m.rdlen))) and not ctx:       # Agave logs return data, success or not
        x, n, i = syscalls.Ctx(m), m.rdlen, word(0)
        while not float(eq(i, n)):
            grp = []
            for _ in range(3):
                if float(eq(i, n)):
                    break
                b = m.rdata.read(i[:AL]).view(2, NIB)
                grp.append(b + ZN * (1.0 - b.sum(-1, keepdim=True)))
                i = add(i, word(0), onehot(1, 2))[0]
            x.b64_group(grp)
        x.flush('Program return: ' + program_id_b58 + ' ')
    return {'status': 'SysIx' if m.status == 'SysIx' else m.status, 'r0': m.reg(0), 'cu': unword(m.cu), 'logs': m.logs,
            'input': m.region_bytes(4), 'steps': steps, 'ixerr': m.ixerr}


def run(prog, inp=b'', cu_limit=10000, gaps=True, funcs=None, lazy=True, max_steps=10**7):
    m = Machine(prog, inp, cu_limit, gaps, funcs, lazy)
    steps = 0
    while not m.halted and steps < max_steps:
        m.step(); steps += 1
    return {'status': m.status, 'r0': m.reg(0), 'cu': unword(m.cu), 'regs': [m.reg(r) for r in range(11)],
            'hashes': (fnv(m.region_bytes(2)), fnv(m.region_bytes(3)), fnv(m.region_bytes(4)))}
