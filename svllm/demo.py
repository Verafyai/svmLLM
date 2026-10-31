"""Run one demo on the weights machine and the reference interpreter, print the trace.

  python -m svllm.demo fibonacci 10          # input words, comma separated
  python -m svllm.demo voting votes:0:1,2:0  # voting uses id:choice pairs
  python -m svllm.demo gcd                   # no input: the demo's first preset
"""
import sys, time
from .programs import DEMOS, preset_mem
from .isa import assemble, disasm_one
from . import machine, ref


def main(argv):
    if not argv or argv[0] not in {d['id'] for d in DEMOS}:
        print('usage: python -m svllm.demo <id> [words]\nids:', ', '.join(d['id'] for d in DEMOS)); return 2
    d = next(d for d in DEMOS if d['id'] == argv[0])
    spec = argv[1] if len(argv) > 1 else d['presets'][0][1]
    if isinstance(spec, str) and not spec.startswith('votes:'):
        spec = [int(x, 0) for x in spec.split(',') if x.strip()]
    prog, mem = assemble(d['asm']), preset_mem(spec)
    tr = []; t = time.time(); w = machine.run(prog, mem, 10000, trace=tr); dt = time.time() - t
    r = ref.run(prog, mem, 10000)
    for i, s in enumerate(tr):
        print(f"{i + 1:4d}  pc={s['pc']:3d}  cu={s['cu']:4d}  {disasm_one(prog, s['pc']):28s} r0={s['regs'][0]:#x}")
    print(f"\n{d['title']}: weights -> {w['status']} r0={w['r0']} cu={w['cu']}  ({dt:.2f}s on CPU)")
    print(f"{' ' * len(d['title'])}  reference -> {r['status']} r0={r['r0']} cu={r['cu']}")
    same = (w['status'], w['r0'], w['cu'], w['regs'], w['mem']) == (r['status'], r['r0'], r['cu'], r['regs'], r['mem'])
    print('match: registers, memory, status and CU identical' if same else 'MISMATCH'); return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
