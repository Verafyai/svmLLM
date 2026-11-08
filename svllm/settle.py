"""The settlement core: Verafy Harness's integer exit programs as reduced-sBPF programs,
run on the hand-built weights machine.

Verafy Harness settles a jury's ruling with exact integer programs (TinySVM, Part E of
its spec): `graded_payout`, `ranked_split`, `milestone_escrow`, `bond` and `ledger`.
runtime.js (`packages/core/src/settle.js`) is the reference; TinySVM runs the same
programs as compiled sBPF in LiteSVM. This module is a third runtime for those five:
the same settlement math, written by hand in svmLLM's reduced sBPF subset, executed on
the weights machine (`machine.run`, every instruction a forward pass through constant
tensors) with 1024-byte memory windows.

What runs in the weights and what does not:

* In the weights: every number and every decision. Bounds checks, overflow checks,
  conservation checks, the 128-bit-safe multiply-then-divide, the ranking's duplicate
  check, the ledger's row search and append, and the error code, which is r0 at exit.
* Outside the weights (this module, plain Python): layout. Account keys are interned to
  small integer ids before the run (equality is all the programs ever ask of a key),
  and the transfer list is assembled from the numbers the run wrote: which key each
  delta belongs to, in the order the reference writes them. Structural limits that the
  TinySVM encoder enforces (a u8 that does not fit, a key that is not 32 bytes) raise
  here too, before anything runs.

Not covered: the escrow program (the jury pay rule splits escrow in IEEE-754 doubles,
reproduced bit for bit by TinySVM's soft-float; this core is integer only) and
escrow hold and release. See SETTLEMENT.md.

Every result has the shape runtime.js returns: {ok, code?, transfers?, state?, table?},
with integers as Python ints. `run_program(..., engine='ref')` runs the same programs on
the plain-Python reference interpreter instead, for fast checks.
"""
import struct
from . import ref, machine
from .isa import assemble

WIN = 1024            # bytes per memory window (input and stack)
CU_LIMIT = 20000
U64 = (1 << 64) - 1
I64_MAX = (1 << 63) - 1

# Error codes shared with packages/core/src/settle.js and programs/abi.
BAD_INPUT, BAD_ACCOUNTS, SMALL_ACCOUNT, OVERFLOW, NOT_CONSERVED, OUT_OF_BOUNDS = 1, 2, 3, 4, 5, 6

# Ledger layout limits (the window is 1024 bytes): rows and transfers per request.
LEDGER_ROWS, LEDGER_MOVES = 32, 16
LEDGER_TABLE, LEDGER_LIST = 24, 24 + 16 * LEDGER_ROWS

# -------------------------------------------------------------------------------------
# The programs. Memory is u64 slots at r1 (the input window). r0 at exit is the code.
# -------------------------------------------------------------------------------------

GRADED = '''# graded_payout.claim
# in:  [0] percent  [8] coverage  [16] reserve  [24] paid  [32] claims
# out: [16] reserve  [24] paid  [32] claims  [40] pay
ldxdw r2, [r1+0]
jgt r2, 100, out_of_bounds
ldxdw r3, [r1+8]
mov64 r4, r3              # owed = coverage * percent / 100, without a 128-bit product:
rsh64 r4, 32              #   hi = coverage >> 32, lo = coverage & 0xffffffff
mov64 r5, r3
lsh64 r5, 32
rsh64 r5, 32
mul64 r4, r2              # hi * percent  (< 2^39)
mov64 r6, r4
div64 r6, 100             # q1
mov64 r7, r6
mul64 r7, 100
sub64 r4, r7              # rem1 = hi * percent - 100 * q1  (< 100)
lsh64 r4, 32
mul64 r5, r2              # lo * percent  (< 2^39)
add64 r4, r5
div64 r4, 100             # q2 = (rem1 << 32 + lo * percent) / 100
lsh64 r6, 32
add64 r6, r4              # owed = q1 << 32 + q2
ldxdw r7, [r1+16]         # reserve
mov64 r8, r6              # pay = min(owed, reserve)
jlt r6, r7, capped
mov64 r8, r7
capped:
lddw r9, 0x7fffffffffffffff
jgt r8, r9, overflow
ldxdw r3, [r1+24]
mov64 r4, r3
add64 r4, r8              # paid + pay
jlt r4, r3, overflow
ldxdw r5, [r1+32]
mov64 r6, r5
add64 r6, 1               # claims + 1
jeq r6, 0, overflow
sub64 r7, r8
stxdw [r1+16], r7
stxdw [r1+24], r4
stxdw [r1+32], r6
stxdw [r1+40], r8
mov64 r0, 0
exit
out_of_bounds:
mov64 r0, 6
exit
overflow:
mov64 r0, 4
exit
'''

RANKED = '''# ranked_split.split
# in:  [0] prize  [8] k  [16] n  [24] ranking length  [32 + 8i] share i  [152 + 8i] ranking i
# out: [272 + 8i] pay for place i (dust included in place 0)
ldxdw r2, [r1+8]
jeq r2, 0, bad_input
jgt r2, 15, bad_input
ldxdw r3, [r1+16]
jgt r3, 32, bad_input
ldxdw r4, [r1+24]
jne r4, r2, bad_input
jgt r2, r3, bad_input
mov64 r5, 0               # the shares must sum to exactly 1,000,000
mov64 r6, r1
add64 r6, 32
mov64 r7, r2
sum_shares:
jeq r7, 0, shares_done
ldxdw r8, [r6+0]
add64 r5, r8
add64 r6, 8
sub64 r7, 1
ja sum_shares
shares_done:
jne r5, 1000000, bad_input
mov64 r5, 0               # seen: one bit per entrant
mov64 r6, r1
add64 r6, 152
mov64 r7, r2
check_rank:
jeq r7, 0, ranks_done
ldxdw r8, [r6+0]
jge r8, r3, out_of_bounds
mov64 r9, 1
lsh64 r9, r8
mov64 r0, r5
and64 r0, r9
jne r0, 0, out_of_bounds
or64 r5, r9
add64 r6, 8
sub64 r7, 1
ja check_rank
ranks_done:
ldxdw r3, [r1+0]          # prize
lddw r9, 0x7fffffffffffffff
jgt r3, r9, overflow
mov64 r4, r3
rsh64 r4, 32              # hi
mov64 r5, r3
lsh64 r5, 32
rsh64 r5, 32              # lo
mov64 r6, r1
add64 r6, 32              # share pointer; the pay slot is at +240
mov64 r7, 0               # total paid
mov64 r8, r2
place:
jeq r8, 0, places_done
ldxdw r9, [r6+0]          # share s (<= 1,000,000)
mov64 r0, r4
mul64 r0, r9              # hi * s  (< 2^51)
mov64 r2, r0
div64 r2, 1000000         # q1
stxdw [r10-8], r2
mul64 r2, 1000000
sub64 r0, r2              # rem1 (< 1,000,000)
lsh64 r0, 32
mul64 r9, r5              # lo * s  (< 2^52)
add64 r0, r9
div64 r0, 1000000         # q2
ldxdw r2, [r10-8]
lsh64 r2, 32
add64 r2, r0              # pay = floor(prize * s / 1e6)
stxdw [r6+240], r2
add64 r7, r2
add64 r6, 8
sub64 r8, 1
ja place
places_done:
mov64 r9, r3
sub64 r9, r7              # rounding dust goes to first place
ldxdw r0, [r1+272]
add64 r0, r9
stxdw [r1+272], r0
mov64 r0, 0
exit
bad_input:
mov64 r0, 1
exit
out_of_bounds:
mov64 r0, 6
exit
overflow:
mov64 r0, 4
exit
'''

MILESTONE = '''# milestone_escrow.release
# in:  [0] milestone  [8] m  [16] released mask  [24] paid  [32 + 8i] tranche i
# out: [16] mask  [24] paid  [544] tranche paid
ldxdw r2, [r1+8]
jeq r2, 0, bad_input
jgt r2, 64, bad_input
ldxdw r3, [r1+0]
jge r3, r2, out_of_bounds
ldxdw r4, [r1+16]
mov64 r5, 1
lsh64 r5, r3              # this milestone's bit
mov64 r6, r4
and64 r6, r5
jne r6, 0, already_released
mov64 r7, r3
lsh64 r7, 3
add64 r7, r1
ldxdw r8, [r7+32]         # tranches[milestone]
lddw r9, 0x7fffffffffffffff
jgt r8, r9, overflow
ldxdw r6, [r1+24]
mov64 r0, r6
add64 r0, r8
jlt r0, r6, overflow
or64 r4, r5
stxdw [r1+16], r4
stxdw [r1+24], r0
stxdw [r1+544], r8
mov64 r0, 0
exit
bad_input:
mov64 r0, 1
exit
out_of_bounds:
mov64 r0, 6
exit
already_released:
mov64 r0, 100
exit
overflow:
mov64 r0, 4
exit
'''

BOND = '''# bond: lock, release, slash. There is no spend.
# in:  [0] op (0 lock, 1 release, 2 slash)  [8] amount  [16] deposited  [24] locked  [32] slashed  [40] released
# out: the four state words in place
ldxdw r2, [r1+0]
ldxdw r3, [r1+8]
ldxdw r4, [r1+16]
ldxdw r5, [r1+24]
ldxdw r6, [r1+32]
ldxdw r7, [r1+40]
jeq r2, 0, lock
jeq r2, 1, release
jeq r2, 2, slash
mov64 r0, 1
exit
lock:
mov64 r8, r4
add64 r4, r3
jlt r4, r8, overflow
mov64 r8, r5
add64 r5, r3
jlt r5, r8, overflow
ja check
release:
jgt r3, r5, release_exceeds
sub64 r5, r3
mov64 r8, r7
add64 r7, r3
jlt r7, r8, overflow
ja check
slash:
jgt r3, r5, slash_exceeds
sub64 r5, r3
mov64 r8, r6
add64 r6, r3
jlt r6, r8, overflow
check:
mov64 r8, r5              # locked + slashed + released, as 128 bits, must equal deposited
add64 r8, r6
mov64 r9, 0
jge r8, r5, no_carry1
add64 r9, 1
no_carry1:
mov64 r2, r8
add64 r8, r7
jge r8, r2, no_carry2
add64 r9, 1
no_carry2:
jne r9, 0, not_conserved
jne r8, r4, not_conserved
stxdw [r1+16], r4
stxdw [r1+24], r5
stxdw [r1+32], r6
stxdw [r1+40], r7
mov64 r0, 0
exit
release_exceeds:
mov64 r0, 101
exit
slash_exceeds:
mov64 r0, 100
exit
overflow:
mov64 r0, 4
exit
not_conserved:
mov64 r0, 5
exit
'''

LEDGER = f'''# ledger.apply
# in:  [0] count  [8] capacity  [16] m  [{LEDGER_TABLE} + 16i] row i (key id, balance)  [{LEDGER_LIST} + 16j] transfer j (key id, delta)
# out: [0] count and the rows in place
ldxdw r2, [r1+0]
ldxdw r3, [r1+8]
jgt r2, r3, bad_accounts
ldxdw r4, [r1+16]
mov64 r5, 0               # the deltas must sum to exactly zero, as a 128-bit signed sum
mov64 r6, 0
mov64 r7, r1
add64 r7, {LEDGER_LIST}
mov64 r8, r4
sum:
jeq r8, 0, sum_done
ldxdw r9, [r7+8]
mov64 r0, r5
add64 r5, r9
jge r5, r0, no_carry
add64 r6, 1
no_carry:
jsge r9, 0, non_negative
sub64 r6, 1
non_negative:
add64 r7, 16
sub64 r8, 1
ja sum
sum_done:
jne r5, 0, not_conserved
jne r6, 0, not_conserved
mov64 r7, r1
add64 r7, {LEDGER_LIST}
mov64 r8, r4
apply:
jeq r8, 0, done
ldxdw r9, [r7+8]          # delta
jeq r9, 0, next
ldxdw r0, [r7+0]          # key id
mov64 r5, r1
add64 r5, {LEDGER_TABLE}
mov64 r6, 0
search:
jeq r6, r2, not_found
ldxdw r4, [r5+0]
jeq r4, r0, found
add64 r5, 16
add64 r6, 1
ja search
not_found:
jslt r9, 0, insufficient
jeq r2, r3, small_account
stxdw [r5+0], r0          # append the key with balance 0
stdw [r5+8], 0
add64 r2, 1
found:
ldxdw r4, [r5+8]
jslt r9, 0, debit
mov64 r6, r4
add64 r4, r9
jlt r4, r6, overflow
ja store
debit:
mov64 r6, r9
neg64 r6
jgt r6, r4, insufficient
sub64 r4, r6
store:
stxdw [r5+8], r4
next:
add64 r7, 16
sub64 r8, 1
ja apply
done:
stxdw [r1+0], r2
mov64 r0, 0
exit
bad_accounts:
mov64 r0, 2
exit
small_account:
mov64 r0, 3
exit
overflow:
mov64 r0, 4
exit
not_conserved:
mov64 r0, 5
exit
insufficient:
mov64 r0, 100
exit
'''

SOURCES = {'graded_payout': GRADED, 'ranked_split': RANKED, 'milestone_escrow': MILESTONE,
           'bond': BOND, 'ledger': LEDGER}
PROGRAMS = {name: assemble(src) for name, src in SOURCES.items()}


def _ops(prog):
    from .isa import dec
    out, i = set(), 0
    while i < len(prog) // 8:
        op = dec(prog, i)[0]
        out.add(op)
        i += 2 if op == 0x18 else 1
    return out


# The opcodes the settlement core uses: the machine decodes only these.
OPS = frozenset().union(*map(_ops, PROGRAMS.values()))


def weights_inventory():
    """Every constant table in the weights, its size, and whether the settlement machine
    (ops = OPS, 1024-byte windows) carries it. All counts are exact element counts."""
    m = machine
    lut = lambda t: t.W1.numel() + t.W2.numel() + t.b1.numel()
    from .isa import OPCODES as OPC
    names = {OPC[o][1] for o in OPS}
    rows = [
        ('L_ADD 16-stage adder cell', lut(m.L_ADD), True),
        ('L_AND', lut(m.L_AND), 'and' in names),
        ('L_OR', lut(m.L_OR), 'or' in names),
        ('L_XOR', lut(m.L_XOR), 'xor' in names),
        ('L_MULN nibble multiplier', lut(m.L_MULN), 'mul' in names),
        ('L_SHL1 (division)', lut(m.L_SHL1), bool(names & {'div', 'mod'})),
        ('L_BSL (lsh)', lut(m.L_BSL), 'lsh' in names),
        ('L_BSR (rsh, arsh)', lut(m.L_BSR), bool(names & {'rsh', 'arsh'})),
        ('L_SHAMT (shift amount)', lut(m.L_SHAMT), bool(names & {'lsh', 'rsh', 'arsh'})),
        ('L_KRON (fetch key)', lut(m.L_KRON), True),
        ('P_NOT (subtract)', m.P_NOT.numel(), True),
        ('P_FLIP8 (signed compare)', m.P_FLIP8.numel(), True),
        ('SIGN_IND (sign extension)', m.SIGN_IND.numel(), True),
        ('P_AND1 (ALU32 shift amount)', m.P_AND1.numel(), False),
        ('DECODE', m.DECODE.numel(), True),
        ('SHIFT256 (128-byte memory: byte k keys)', sum(t.numel() for t in m.SHIFT256), False),
        ('V_IN, V_ST (128-byte memory: bounds)', m.V_IN.numel() + m.V_ST.numel(), False),
    ]
    wide = m._WIDE.get(WIN) or m.WideMem(WIN)
    m._WIDE[WIN] = wide
    rows.append((f'WideMem({WIN}) memory unit', wide.params, True))
    base = sum(n for name, n, _ in rows if not name.startswith('WideMem'))
    assert base == m.PARAMS['count'], (base, m.PARAMS['count'])
    used = sum(n for _, n, u in rows if u)
    return {'tables': [{'table': name, 'params': n, 'in_settlement_machine': bool(u)} for name, n, u in rows],
            'svmllm_machine_params': base, 'settlement_machine_params': used,
            'settlement_machine_bytes_float32': used * 4}

# -------------------------------------------------------------------------------------
# Encoding: the same structural limits as the TinySVM encoder (packages/tinysvm/src).
# -------------------------------------------------------------------------------------


class Unsupported(ValueError):
    """An input this runtime does not encode: the same refusals as the TinySVM encoder,
    plus the ledger's layout limits. Raised before anything runs; never a disagreement."""


def _u(x, bits, what):
    x = int(x)
    if not 0 <= x < (1 << bits):
        raise Unsupported(f'{what}: {x} does not fit in a u{bits}')
    return x


def _key(k):
    if not (isinstance(k, str) and len(k) == 64 and all(c in '0123456789abcdef' for c in k.lower())):
        raise Unsupported('a key is 32 bytes of hex')
    return k.lower()


def _mem(slots):
    b = bytearray(WIN)
    for off, v in slots:
        struct.pack_into('<Q', b, off, v & U64)
    return bytes(b)


def _rd(mem, off, signed=False):
    return struct.unpack_from('<q' if signed else '<Q', mem, off)[0]


def _err(code):
    return {'ok': False, 'code': code}


def _transfers(pairs):
    return [{'key': k, 'delta': d} for k, d in pairs]


def encode(program, call):
    """(memory image, decode function) for one call in jsRuntime's input shape."""
    if program == 'graded_payout':
        st, inp = call['state'], call['input']
        pct, cov = _u(inp['percent'], 64, 'percent'), _u(inp['coverage'], 64, 'coverage')
        pot, ben = _key(inp['pot']), _key(inp['beneficiary'])
        mem = _mem([(0, pct), (8, cov), (16, _u(st['reserve'], 64, 'reserve')),
                    (24, _u(st['paid'], 64, 'paid')), (32, _u(st['claims'], 64, 'claims'))])

        def dec(out):
            pay = _rd(out, 40)
            return {'ok': True, 'transfers': _transfers([(pot, -pay), (ben, pay)] if pay else []),
                    'state': {'reserve': _rd(out, 16), 'paid': _rd(out, 24), 'claims': _rd(out, 32)}}
        return mem, dec
    if program == 'ranked_split':
        inp = call['input']
        shares, ranking, entrants = list(inp['shares']), list(inp['ranking']), list(inp['entrants'])
        k, n = _u(len(shares), 8, 'places'), _u(len(entrants), 8, 'entrants')
        sh = [_u(s, 32, 'share') for s in shares]
        rk = [_u(r, 8, 'rank') for r in ranking]
        pot, ents = _key(inp['pot']), [_key(e) for e in entrants]
        prize = _u(inp['prize'], 64, 'prize')
        slots = [(0, prize), (8, k), (16, n), (24, len(rk))]
        slots += [(32 + 8 * i, s) for i, s in enumerate(sh[:15])]
        slots += [(152 + 8 * i, r) for i, r in enumerate(rk[:15])]

        def dec(out):
            pays = [_rd(out, 272 + 8 * i, signed=True) for i in range(k)]
            return {'ok': True, 'transfers': _transfers([(pot, -prize)] + [(ents[rk[i]], pays[i]) for i in range(k)]),
                    'state': {}}
        return _mem(slots), dec
    if program == 'milestone_escrow':
        st, inp = call['state'], call['input']
        tr = [_u(t, 64, 'tranche') for t in inp['tranches']]
        m = _u(len(tr), 8, 'tranche count')
        ms = _u(inp['milestone'], 8, 'milestone')
        mask = 0
        for i in st['released']:
            mask |= 1 << int(i)
        mask = _u(mask, 64, 'released mask')
        pot, to = _key(inp['pot']), _key(inp['contractor'])
        slots = [(0, ms), (8, m), (16, mask), (24, _u(st['paid'], 64, 'paid'))]
        slots += [(32 + 8 * i, t) for i, t in enumerate(tr[:64])]

        def dec(out):
            t, mk = _rd(out, 544), _rd(out, 16)
            return {'ok': True, 'transfers': _transfers([(pot, -t), (to, t)]),
                    'state': {'released': [i for i in range(64) if mk >> i & 1], 'paid': _rd(out, 24)}}
        return _mem(slots), dec
    if program == 'bond':
        st = call['state']
        tag = {'lock': 0, 'release': 1, 'slash': 2}.get(call['op'], 3)
        mem = _mem([(0, tag), (8, _u(call['amount'], 64, 'amount')),
                    (16, _u(st['deposited'], 64, 'deposited')), (24, _u(st['locked'], 64, 'locked')),
                    (32, _u(st['slashed'], 64, 'slashed')), (40, _u(st['released'], 64, 'released'))])

        def dec(out):
            return {'ok': True, 'state': {'deposited': _rd(out, 16), 'locked': _rd(out, 24),
                                          'slashed': _rd(out, 32), 'released': _rd(out, 40)}}
        return mem, dec
    if program == 'ledger':
        table, moves, cap = list(call['table']), list(call['transfers']), int(call['capacity'])
        if cap > LEDGER_ROWS or len(table) > LEDGER_ROWS or len(moves) > LEDGER_MOVES:
            raise Unsupported(f'the settlement core holds at most {LEDGER_ROWS} ledger rows and {LEDGER_MOVES} transfers')
        ids, keys = {}, [None]
        def kid(k):
            k = _key(k)
            if k not in ids:
                ids[k] = len(keys); keys.append(k)
            return ids[k]
        slots = [(0, len(table)), (8, cap), (16, len(moves))]
        for i, row in enumerate(table):
            slots += [(LEDGER_TABLE + 16 * i, kid(row['key'])), (LEDGER_TABLE + 16 * i + 8, _u(row['balance'], 64, 'balance'))]
        for j, t in enumerate(moves):
            d = int(t['delta'])
            if not -(1 << 63) <= d <= I64_MAX:
                raise Unsupported(f'delta: {d} does not fit in an i64')
            slots += [(LEDGER_LIST + 16 * j, kid(t['key'])), (LEDGER_LIST + 16 * j + 8, d)]

        def dec(out):
            n = _rd(out, 0)
            return {'ok': True, 'table': [{'key': keys[_rd(out, LEDGER_TABLE + 16 * i)], 'balance': _rd(out, LEDGER_TABLE + 16 * i + 8)} for i in range(n)]}
        return _mem(slots), dec
    raise Unsupported(f'the settlement core does not cover {program!r}')


def weights_digest():
    """SHA-256 over every constant tensor the settlement machine carries (float32 bytes,
    in a fixed order) and the five programs' bytecode: what a prover would commit to."""
    import hashlib
    m = machine
    wide = m._WIDE.get(WIN) or m.WideMem(WIN)
    m._WIDE[WIN] = wide
    h = hashlib.sha256()
    luts = [m.L_ADD, m.L_AND, m.L_OR, m.L_MULN, m.L_SHL1, m.L_BSL, m.L_BSR, m.L_SHAMT, m.L_KRON]
    tensors = [t for L in luts for t in (L.W1, L.b1, L.W2)]
    tensors += [m.P_NOT, m.P_FLIP8, m.SIGN_IND, machine.subset_decode(OPS)]
    tensors += [wide.L2LO, wide.KRON, wide.HI_IN, wide.HI_ST, wide.V_IN, wide.V_ST]
    for t in tensors:
        h.update(t.contiguous().numpy().tobytes())
    for name in sorted(PROGRAMS):
        h.update(name.encode() + PROGRAMS[name])
    return h.hexdigest()


def run_program(program, call, engine='weights', lazy=True, trace=None):
    """Run one settlement call. engine: 'weights' (the hand-built weights machine) or 'ref'
    (the reference interpreter). Returns the runtime.js-shaped result plus `cu`."""
    mem, dec = encode(program, call)
    prog = PROGRAMS[program]
    if engine == 'weights':
        r = machine.run(prog, mem, CU_LIMIT, lazy=lazy, trace=trace, win=WIN, ops=OPS)
    elif engine == 'ref':
        r = ref.run(prog, mem, CU_LIMIT, trace=trace, win=WIN)
    else:
        raise ValueError(engine)
    if r['status'] != 'ok':
        raise RuntimeError(f'{program}: the VM stopped with {r["status"]}')
    out = _err(r['r0']) if r['r0'] else dec(r['mem'][:WIN])
    out['cu'] = r['cu']
    return out


def call_of(method, req):
    """Map a jsRuntime method call ({method, input}) to (program, call)."""
    if method == 'exit':
        prog, entry = req['program'], req['entry']
        want = {'graded_payout': 'claim', 'ranked_split': 'split', 'milestone_escrow': 'release'}
        if want.get(prog) != entry:
            return None, None
        return prog, {'state': req.get('state'), 'input': req['input']}
    if method in ('bond', 'ledger'):
        return method, req
    return None, None
