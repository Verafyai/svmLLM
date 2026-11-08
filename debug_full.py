"""Find the first step where the weights machine diverges from the reference."""
import random, sys
from svllm import sref, svm
from svllm.srandprog import gen
from svllm.machine import unword
from svllm.isa import disasm_one


def first_divergence(c):
    tr = []
    sref.run(*c, trace=tr)
    m = svm.Machine(*c)
    for i, t in enumerate(tr):
        if m.halted:
            return f'weights halted early at step {i}: {m.status}'
        pc = unword(m.pc)
        regs = [m.reg(r) for r in range(11)]
        if pc != t['pc'] or regs != t['regs']:
            prev = disasm_one(c[0], tr[i - 1]['pc']) if i else '-'
            diff = [(r, hex(regs[r]), hex(t['regs'][r])) for r in range(11) if regs[r] != t['regs'][r]]
            return f'step {i}: after "{prev}" pc weights {pc} ref {t["pc"]} regs {diff}'
        m.step()
    return f'no divergence over {len(tr)} steps; weights status {m.status}'


if __name__ == '__main__':
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
    cases = []
    for _ in range(600):
        p, mm, f = gen(rng)
        cases.append((p, mm, 600, rng.random() < 0.7, f))
    for i in map(int, sys.argv[2].split(',')):
        print(i, first_divergence(cases[i]))
