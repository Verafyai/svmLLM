"""H1: the instruction subset the TinySVM settlement programs actually execute.

Runs the compiled programs (the .so files Verafy Harness builds with cargo-build-sbf and
pins in packages/tinysvm/programs.lock.json) on Anza's solana-sbpf interpreter with
instruction tracing (xcheck/src/bin/sotrace.rs), over inputs encoded exactly as the
TinySVM runtime encodes them, and counts every executed opcode and syscall. Compares
that dynamic subset with the hand-written settlement core in svllm/settle.py.

    python -m svllm.subset <deploy dir> [cases per program]

writes results/settlement_subset.json.
"""
import hashlib, json, os, struct, subprocess, sys
from . import settle
from .isa import OPCODES, ALU_OPS, JMP_OPS
from .settle_cases import random_cases, named_cases

BIN = os.path.join(os.path.dirname(__file__), '..', 'xcheck', 'target', 'release', 'sotrace')
PROGRAM_ID = bytes(range(32))
MAX_PERMITTED_DATA_INCREASE = 10 * 1024

_ALU = {v: k for k, v in ALU_OPS.items()}
_ALU.update({0xd0: 'end'})
_JMP = {v: k for k, v in JMP_OPS.items()}
_SZ = {0x10: 'b', 0x08: 'h', 0x00: 'w', 0x18: 'dw'}


def opname(key):
    """sBPF v0 mnemonic for a tracer key (opcode byte; calls carry src in bit 8)."""
    if key & 0xff == 0x85:
        return 'call (internal)' if key >> 8 else 'call (syscall)'
    op = key
    cls = op & 7
    if op == 0x18:
        return 'lddw'
    if op == 0x95:
        return 'exit'
    if op == 0x8d:
        return 'callx'
    if cls in (4, 7):
        name = _ALU.get(op & 0xf0, f'alu?{op:02x}')
        if name == 'end':
            return 'be' if op & 8 else 'le'
        bits = '64' if cls == 7 else '32'
        src = '' if name == 'neg' else (' reg' if op & 8 else ' imm')
        return f'{name}{bits}{src}'
    if cls == 5:
        name = _JMP.get(op & 0xf0, f'jmp?{op:02x}')
        return name if name == 'ja' else f'{name} ' + ('reg' if op & 8 else 'imm')
    if cls == 1:
        return 'ldx' + _SZ[op & 0x18]
    if cls == 3:
        return 'stx' + _SZ[op & 0x18]
    if cls == 2:
        return 'st' + _SZ[op & 0x18]
    return f'?{op:02x}'


def in_svmllm(name):
    """Is this mnemonic in svmLLM's reduced subset (OPCODES)?"""
    known = set()
    for op, (kind, n, reg) in OPCODES.items():
        if kind == 'alu':
            known.add(f'{n}64' if n == 'neg' else f'{n}64 ' + ('reg' if reg else 'imm'))
        elif kind == 'jmp':
            known.add(n if n == 'ja' else f'{n} ' + ('reg' if reg else 'imm'))
        elif kind in ('ldx', 'stx', 'st'):
            known.add(kind + n)
        else:
            known.add(kind)
    return name in known


# ---------------------------------------------------------------- loader input (aligned)
def serialize(accounts, data):
    """The BPF loader's aligned input buffer. accounts: [(key, owner, writable, bytes)]."""
    b = bytearray(struct.pack('<Q', len(accounts)))
    for key, owner, writable, d in accounts:
        b += bytes([0xff, 0, 1 if writable else 0, 0]) + bytes(4) + key + owner
        b += struct.pack('<QQ', 1_000_000, len(d)) + d + bytes(MAX_PERMITTED_DATA_INCREASE)
        b += bytes((-len(b)) % 8)
        b += struct.pack('<Q', 0)
    b += struct.pack('<Q', len(data)) + data + PROGRAM_ID
    return bytes(b)


def _k(h):
    return bytes.fromhex(h)


def acct(name, d):
    return (hashlib.sha256(name.encode()).digest(), PROGRAM_ID, True, d)


def tinysvm_input(program, call):
    """(.so name, loader input) for one call, encoded as packages/tinysvm/src/index.js does."""
    u8, u32, u64 = (lambda x: struct.pack('<B', x)), (lambda x: struct.pack('<I', x)), (lambda x: struct.pack('<Q', x & (2**64 - 1)))
    out = acct('out', bytes(4 + 16 * 40))
    if program == 'graded_payout':
        s, i = call['state'], call['input']
        st = u64(s['reserve']) + u64(s['paid']) + u64(s['claims'])
        data = u8(0) + u64(i['percent']) + u64(i['coverage']) + _k(i['pot']) + _k(i['beneficiary'])
        return program, serialize([acct('state', st), out], data)
    if program == 'ranked_split':
        i = call['input']
        data = u8(0) + u64(i['prize']) + u8(len(i['shares'])) + b''.join(map(u32, i['shares'])) + u8(len(i['entrants']))
        data += b''.join(u8(r) for r in i['ranking']) + _k(i['pot']) + b''.join(_k(e) for e in i['entrants'])
        return program, serialize([out], data)
    if program == 'milestone_escrow':
        s, i = call['state'], call['input']
        mask = 0
        for q in s['released']:
            mask |= 1 << q
        data = u8(0) + u8(i['milestone']) + u8(len(i['tranches'])) + b''.join(map(u64, i['tranches'])) + _k(i['pot']) + _k(i['contractor'])
        return program, serialize([acct('state', u64(mask) + u64(s['paid'])), out], data)
    if program == 'bond':
        s = call['state']
        tag = {'lock': 0, 'release': 1, 'slash': 2}.get(call['op'], 3)
        st = b''.join(u64(s[k]) for k in ('deposited', 'locked', 'slashed', 'released'))
        return program, serialize([acct('bond', st)], u8(tag) + u64(call['amount']))
    if program == 'ledger':
        cap = call['capacity']
        t = bytearray(4 + cap * 40)
        struct.pack_into('<I', t, 0, len(call['table']))
        for j, row in enumerate(call['table']):
            t[4 + j * 40:36 + j * 40] = _k(row['key'])
            struct.pack_into('<Q', t, 36 + j * 40, row['balance'])
        lst = u32(len(call['transfers'])) + b''.join(_k(x['key']) + struct.pack('<q', x['delta']) for x in call['transfers'])
        return program, serialize([acct('table', bytes(t))], u8(0) + lst)
    if program == 'escrow':                       # the jury pay rule, IEEE-754 doubles
        f64 = lambda x: struct.pack('<d', x)
        i = call
        data = u8(0) + f64(i['escrow']) + f64(i['jurors']) + f64(i['auditors']) + f64(i['storage']) + u32(i['n']) + u8(1) + f64(i['pay'])
        return program, serialize([acct('out', bytes(64))], data)
    raise ValueError(program)


def jury_cases(n, seed=20261106):
    import random
    r = random.Random(seed)
    out = []
    for _ in range(n):
        j = r.randint(100_000, 950_000); a = r.randint(0, 1_000_000 - j); s = r.randint(0, 1_000_000 - j - a)
        out.append(('escrow', {'escrow': r.randint(1, 50_000_000) / 1e6, 'jurors': j / 1e6, 'auditors': a / 1e6, 'storage': s / 1e6,
                               'n': r.randint(1, 15), 'pay': r.randint(1, 10**9) / 1e6 if r.random() < 0.3 else 0.0}))
    return out


def trace(deploy, calls):
    lines = []
    for program, call in calls:
        so, inp = tinysvm_input(program, call)
        lines.append(f'{os.path.join(deploy, so + ".so")} {inp.hex()} 200000\n')
    out = subprocess.run([BIN], input=''.join(lines), capture_output=True, text=True, check=True).stdout.splitlines()
    res = []
    for line in out:
        f = line.split()
        ops = {int(k, 16): int(v) for k, v in (x.split(':') for x in f[3].split(',') if x)}
        sys_ = {} if f[4] == '-' else {k: int(v) for k, v in (x.split(':') for x in f[4].split(','))}
        res.append({'status': f[0], 'r0': int(f[1]) if f[0] == 'ok' else f[1], 'count': int(f[2]), 'ops': ops, 'syscalls': sys_})
    return res


def main():
    deploy = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    calls = random_cases(n) + [c for c in named_cases().values()]
    calls += jury_cases(n)

    def encodable(c):                 # the TinySVM encoder refuses these too (RangeError)
        try:
            tinysvm_input(*c)
            return True
        except (struct.error, OverflowError, ValueError):
            return False
    calls = [c for c in calls if encodable(c)]
    runs = trace(deploy, calls)
    per = {}
    for (program, call), r in zip(calls, runs):
        p = per.setdefault(program, {'calls': 0, 'instructions': 0, 'ops': {}, 'syscalls': {}, 'codes': {}})
        p['calls'] += 1
        p['instructions'] += r['count']
        for k, v in r['ops'].items():
            p['ops'][opname(k)] = p['ops'].get(opname(k), 0) + v
        for k, v in r['syscalls'].items():
            p['syscalls'][k] = p['syscalls'].get(k, 0) + v
        code = str(r['r0'])
        p['codes'][code] = p['codes'].get(code, 0) + 1
        # The core must agree with the compiled program on the error code (0 = ok).
        if program != 'escrow':
            mine = settle.run_program(program, call, 'ref')
            assert (0 if mine['ok'] else mine['code']) == r['r0'], (program, call, mine, r)
    integer = [p for p in per if p != 'escrow']
    union = {}
    for p in integer:
        for k, v in per[p]['ops'].items():
            union[k] = union.get(k, 0) + v
    core = {}
    from .isa import disasm_one
    for name, prog in settle.PROGRAMS.items():
        for i in range(len(prog) // 8):
            ins = disasm_one(prog, i)
            if ins.startswith('.invalid'):
                continue                                   # the second slot of lddw
            mn = ins.split()[0]
            if mn in ('lddw', 'exit', 'ja'):
                core[mn] = 1
            elif mn.startswith(('ldx', 'stx', 'st')):
                core[mn] = 1
            elif mn == 'neg64':
                core['neg64'] = 1
            elif mn.endswith('64'):
                core[mn + (' reg' if ins.split(',')[-1].strip().startswith('r') else ' imm')] = 1
            else:
                core[mn + (' reg' if ins.split(',')[1].strip().startswith('r') else ' imm')] = 1
    report = {
        'what': 'Executed sBPF instructions of the compiled TinySVM programs (cargo-build-sbf 1.18.8, platform tools v1.41), traced on solana-sbpf 0.12.2, over inputs encoded as packages/tinysvm/src/index.js encodes them.',
        'cases_per_program': {p: per[p]['calls'] for p in per},
        'programs': {p: {**per[p], 'ops': dict(sorted(per[p]['ops'].items(), key=lambda kv: -kv[1])),
                         'distinct_ops': len(per[p]['ops']),
                         'outside_svmllm_subset': sorted(k for k in per[p]['ops'] if not in_svmllm(k))} for p in per},
        'integer_programs_union': {'distinct_ops': len(union), 'ops': dict(sorted(union.items(), key=lambda kv: -kv[1])),
                                   'outside_svmllm_subset': sorted(k for k in union if not in_svmllm(k))},
        'settlement_core_ops': sorted(core),
        'settlement_core_distinct_ops': len(core),
    }
    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/settlement_subset.json', 'w'), indent=1)
    print(json.dumps({p: {'calls': v['calls'], 'mean_instructions': round(v['instructions'] / v['calls'], 1), 'distinct_ops': len(v['ops']),
                          'outside_subset': report['programs'][p]['outside_svmllm_subset'], 'syscalls': v['syscalls']} for p, v in per.items()}, indent=1))
    print('integer union:', len(union), 'distinct; outside svmLLM subset:', report['integer_programs_union']['outside_svmllm_subset'])
    print('settlement core:', len(core), 'distinct:', sorted(core))


if __name__ == '__main__':
    main()
