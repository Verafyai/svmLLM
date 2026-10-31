"""Demo programs (the 10 on the site) and the named conformance suite."""
import struct
from .isa import assemble, ALU_OPS, JMP_OPS


def u64s(*vals):
    b = b''.join(struct.pack('<Q', v & (2**64 - 1)) for v in vals)
    return b + bytes(128 - len(b))


DEMOS = [
 {'id': 'spend-cap', 'title': 'Spend cap', 'blurb': 'Rejects any transfer above a $1B limit (amount in whole dollars at input+0). Exit code 1 = over the cap.',
  'asm': '''# r1 = input. amount (u64) at [r1+0]
ldxdw r2, [r1+0]
lddw r3, 1000000000      # the $1B cap
jgt r2, r3, over_cap
mov64 r0, 0              # success
exit
over_cap:
mov64 r0, 1              # custom error: amount over cap
exit''',
  'presets': [('$250M transfer', [250_000_000]), ('exactly $1B', [1_000_000_000]), ('$1B + $1', [1_000_000_001])]},
 {'id': 'counter', 'title': 'Counter', 'blurb': 'Reads a counter from memory, increments it N times with a load and a store each pass, returns the new value.',
  'asm': '''ldxdw r3, [r1+8]         # N
loop:
jeq r3, 0, done
ldxdw r2, [r1+0]         # read counter
add64 r2, 1
stxdw [r1+0], r2         # write it back
sub64 r3, 1
ja loop
done:
ldxdw r0, [r1+0]
exit''',
  'presets': [('start 0, +5', [0, 5]), ('start 41, +1', [41, 1]), ('start 1000, +12', [1000, 12])]},
 {'id': 'fibonacci', 'title': 'Fibonacci', 'blurb': 'Iterative fib(n) for n at input+0. fib(93) is the largest that fits in a u64.',
  'asm': '''ldxdw r3, [r1+0]         # n
mov64 r0, 0
mov64 r2, 1
loop:
jeq r3, 0, done
mov64 r4, r0
add64 r4, r2
mov64 r0, r2
mov64 r2, r4
sub64 r3, 1
ja loop
done:
exit''',
  'presets': [('n = 10', [10]), ('n = 50', [50]), ('n = 90', [90])]},
 {'id': 'factorial', 'title': 'Factorial with overflow check', 'blurb': 'n! for n at input+0. Each multiply is checked by dividing back; exit code 2 on u64 overflow.',
  'asm': '''ldxdw r3, [r1+0]         # n
mov64 r0, 1
mov64 r2, 1              # i
loop:
jgt r2, r3, done
mov64 r4, r0
mul64 r4, r2
mov64 r5, r4
div64 r5, r2
jne r5, r0, overflow     # (r0*i)/i != r0 means it wrapped
mov64 r0, r4
add64 r2, 1
ja loop
done:
exit
overflow:
mov64 r0, 2              # custom error: overflow
exit''',
  'presets': [('5!', [5]), ('20!', [20]), ('21! (overflows)', [21])]},
 {'id': 'gcd', 'title': 'GCD', 'blurb': "Euclid's algorithm on two u64s at input+0 and input+8.",
  'asm': '''ldxdw r2, [r1+0]
ldxdw r3, [r1+8]
loop:
jeq r3, 0, done
mov64 r4, r2
mod64 r4, r3
mov64 r2, r3
mov64 r3, r4
ja loop
done:
mov64 r0, r2
exit''',
  'presets': [('gcd(48, 18)', [48, 18]), ('gcd(1071, 462)', [1071, 462]), ('gcd(2^40, 6^20)', [2**40, 6**20])]},
 {'id': 'prime', 'title': 'Prime check', 'blurb': 'Trial division up to sqrt(n). Returns 1 if n at input+0 is prime, else 0.',
  'asm': '''ldxdw r2, [r1+0]         # n
mov64 r0, 0
jlt r2, 2, done
mov64 r0, 1
jlt r2, 4, done
mov64 r4, r2
and64 r4, 1
jeq r4, 0, composite
mov64 r3, 3              # divisor
loop:
mov64 r4, r3
mul64 r4, r3
jgt r4, r2, done         # d*d > n: prime
mov64 r4, r2
mod64 r4, r3
jeq r4, 0, composite
add64 r3, 2
ja loop
composite:
mov64 r0, 0
done:
exit''',
  'presets': [('97', [97]), ('7919', [7919]), ('8633 = 89 x 97', [8633])]},
 {'id': 'transfer', 'title': 'Balance transfer', 'blurb': 'Moves amount (input+16) from slot A (input+0) to slot B (input+8). Exit 1 if A is short, 2 if B would overflow.',
  'asm': '''ldxdw r2, [r1+0]         # from balance
ldxdw r3, [r1+8]         # to balance
ldxdw r4, [r1+16]        # amount
jgt r4, r2, insufficient
mov64 r5, r3
add64 r5, r4
jlt r5, r3, overflow
sub64 r2, r4
stxdw [r1+0], r2
stxdw [r1+8], r5
mov64 r0, 0
exit
insufficient:
mov64 r0, 1
exit
overflow:
mov64 r0, 2
exit''',
  'presets': [('100 -> B, A has 500', [500, 20, 100]), ('A short', [50, 0, 100]), ('B overflows', [10, 2**64 - 5, 10])]},
 {'id': 'vault', 'title': 'Vault with withdrawal limit', 'blurb': 'balance, withdrawn today, daily limit, request at input+0/8/16/24. Exit 1 if over balance, 2 if over the daily limit.',
  'asm': '''ldxdw r2, [r1+0]         # balance
ldxdw r3, [r1+8]         # withdrawn today
ldxdw r4, [r1+16]        # daily limit
ldxdw r5, [r1+24]        # request
jgt r5, r2, over_balance
mov64 r6, r3
add64 r6, r5
jgt r6, r4, over_limit
sub64 r2, r5
stxdw [r1+0], r2
stxdw [r1+8], r6
mov64 r0, 0
exit
over_balance:
mov64 r0, 1
exit
over_limit:
mov64 r0, 2
exit''',
  'presets': [('withdraw 300 of 1000/day', [5000, 600, 1000, 300]), ('over daily limit', [5000, 900, 1000, 300]), ('over balance', [200, 0, 1000, 300])]},
 {'id': 'voting', 'title': 'Voting tally', 'blurb': 'Votes are (voter id 0-63, choice 0/1) byte pairs from input+16, count at input+8. A 64-bit bitmap rejects double votes with exit 3. Tallies land at input+64 and input+72.',
  'asm': '''ldxdw r6, [r1+8]         # number of votes
mov64 r7, r1
add64 r7, 16             # cursor
mov64 r8, 0              # voted bitmap
loop:
jeq r6, 0, done
ldxb r2, [r7+0]          # voter id
ldxb r3, [r7+1]          # choice
mov64 r4, 1
lsh64 r4, r2
jset r8, r4, double_vote
or64 r8, r4
lsh64 r3, 3              # choice * 8
add64 r3, r1
ldxdw r5, [r3+64]
add64 r5, 1
stxdw [r3+64], r5
add64 r7, 2
sub64 r6, 1
ja loop
done:
mov64 r0, 0
exit
double_vote:
mov64 r0, 3
exit''',
  'presets': [('5 voters', 'votes:0:1,1:0,2:1,3:1,63:0'), ('voter 2 votes twice', 'votes:0:1,2:0,2:1'), ('one vote', 'votes:7:1')]},
 {'id': 'escrow', 'title': 'Escrow release', 'blurb': 'amount, deadline slot, current slot, delivered flag at input+0/8/16/24. Releases to the seller if delivered, refunds the buyer after the deadline, otherwise exit 4.',
  'asm': '''ldxdw r2, [r1+0]         # escrowed amount
ldxdw r3, [r1+8]         # deadline slot
ldxdw r4, [r1+16]        # current slot
ldxdw r5, [r1+24]        # delivered flag
jeq r2, 0, empty
jeq r5, 1, release
jgt r4, r3, refund
mov64 r0, 4              # condition not met yet
exit
release:
stxdw [r1+32], r2        # seller receives
ja clear
refund:
stxdw [r1+40], r2        # buyer refunded
clear:
stdw [r1+0], 0
mov64 r0, 0
exit
empty:
mov64 r0, 5
exit''',
  'presets': [('delivered: release', [1000, 500, 100, 1]), ('past deadline: refund', [1000, 500, 600, 0]), ('not yet', [1000, 500, 100, 0])]},
]


def preset_mem(p):
    if isinstance(p, str) and p.startswith('votes:'):
        pairs = [tuple(int(x) for x in v.split(':')) for v in p[6:].split(',')]
        b = struct.pack('<QQ', 0, len(pairs)) + b''.join(bytes(pr) for pr in pairs)
        return b + bytes(128 - len(b))
    return u64s(*p)


def demo_cases():
    out = []
    for d in DEMOS:
        prog = assemble(d['asm'])
        for label, p in d['presets']:
            out.append((f"demo/{d['id']}/{label}", prog, preset_mem(p), 10000))
    return out


EDGE = [0, 1, 7, 0x7fffffff, 0x80000000, 0xffffffff, 2**63 - 1, 2**63, 2**64 - 1, 0x123456789abcdef0, 1000000007]


def conformance_cases():
    cases = []
    def add(name, asm, mem=b'', lim=10000):
        cases.append((name, assemble(asm), mem or bytes(128), lim))
    # ALU, register form, on edge pairs (a few pairs per op)
    pairs = [(EDGE[i], EDGE[(i * 3 + 5) % len(EDGE)]) for i in range(len(EDGE))]
    for op in ALU_OPS:
        for a, b in pairs[:4]:
            if op in ('div', 'mod') and b == 0:
                b = 3
            if op == 'neg':
                add(f'alu/neg/{a:x}', f'lddw r2, {a}\nneg64 r2\nmov64 r0, r2\nexit')
            else:
                add(f'alu/{op}/reg/{a:x}/{b:x}', f'lddw r2, {a}\nlddw r3, {b}\n{op}64 r2, r3\nmov64 r0, r2\nexit')
        imm = {'lsh': 13, 'rsh': 63, 'arsh': 7, 'div': -3, 'mod': 1000}.get(op, -12345)
        if op != 'neg':
            add(f'alu/{op}/imm', f'lddw r2, {2**63 + 99991}\n{op}64 r2, {imm}\nmov64 r0, r2\nexit')
    for op in JMP_OPS:
        for a, b in [(5, 5), (2**63, 1), (1, 2**64 - 1)]:
            if op == 'ja':
                add('jmp/ja', 'ja +1\nmov64 r0, 9\nexit'); break
            add(f'jmp/{op}/{a:x}/{b:x}', f'lddw r2, {a}\nlddw r3, {b}\nmov64 r0, 1\n{op} r2, r3, +1\nmov64 r0, 0\nexit')
        if op != 'ja':
            add(f'jmp/{op}/imm', f'mov64 r2, -1\nmov64 r0, 1\n{op} r2, -1, +1\nmov64 r0, 0\nexit')
    for s, nb in [('b', 1), ('h', 2), ('w', 4), ('dw', 8)]:
        add(f'mem/stx{s}+ldx{s}/stack', f'lddw r2, 0x8877665544332211\nstx{s} [r10-16], r2\nldx{s} r0, [r10-16]\nexit')
        add(f'mem/st{s}/input', f'st{s} [r1+3], -2\nldxdw r0, [r1+0]\nexit')
        add(f'mem/oob/{s}', f'ldx{s} r0, [r1+{129 - nb}]\nexit')
    add('mem/stack-oob-low', 'ldxdw r0, [r1-8]\nexit')
    add('err/div0', 'mov64 r2, 0\nmov64 r0, 5\ndiv64 r0, r2\nexit')
    add('err/mod0', 'mov64 r2, 0\nmov64 r0, 5\nmod64 r0, r2\nexit')
    add('err/overrun', 'mov64 r0, 1')
    add('err/jump-past-end', 'ja +3\nexit')
    add('err/cu-limit', 'loop:\nja loop', lim=50)
    cases.append(('err/invalid-opcode', bytes([0xff] + [0] * 7) + assemble('exit'), bytes(128), 100))
    # loops
    add('loop/sum-1..100', 'mov64 r0, 0\nmov64 r2, 100\nl:\nadd64 r0, r2\nsub64 r2, 1\njne r2, 0, l\nexit')
    add('loop/gcd', 'mov64 r2, 1071\nmov64 r3, 462\nl:\njeq r3, 0, d\nmov64 r4, r2\nmod64 r4, r3\nmov64 r2, r3\nmov64 r3, r4\nja l\nd:\nmov64 r0, r2\nexit')
    add('loop/popcount', 'lddw r2, 0xf0f0f0f0deadbeef\nmov64 r0, 0\nl:\njeq r2, 0, d\nmov64 r3, r2\nand64 r3, 1\nadd64 r0, r3\nrsh64 r2, 1\nja l\nd:\nexit')
    add('loop/signed-countdown', 'mov64 r2, 5\nmov64 r0, 0\nl:\njslt r2, -3, d\nadd64 r0, 1\nsub64 r2, 1\nja l\nd:\nexit')
    add('loop/memcpy-16', 'lddw r2, 0x0102030405060708\nstxdw [r1+0], r2\nstxdw [r1+8], r2\nmov64 r3, 0\nl:\njeq r3, 16, d\nmov64 r4, r1\nadd64 r4, r3\nldxb r5, [r4+0]\nstxb [r4+32], r5\nadd64 r3, 1\nja l\nd:\nldxdw r0, [r1+40]\nexit')
    add('misc/lddw', 'lddw r0, 0xdeadbeefcafef00d\nexit')
    add('misc/call-stub', 'mov64 r0, 77\ncall 0\nexit')  # syscall stub: r0 <- 0 (not cross-checked vs rbpf)
    return cases
