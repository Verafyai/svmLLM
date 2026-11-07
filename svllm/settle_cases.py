"""Settlement test cases: named edge cases, plus random calls shaped like the property
test in Verafy Harness (packages/tinysvm/test/tinysvm.test.js), same ranges."""
import hashlib
import random

U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


def key(name):
    """Stand-in for accountKey(): SHA-256 of the name, hex."""
    return hashlib.sha256(name.encode()).hexdigest()


def graded(state, percent, coverage, i=0):
    return ('graded_payout', {'state': state, 'input': {'percent': percent, 'coverage': coverage, 'pot': key(f'pot:{i}'), 'beneficiary': key(f'b:{i}')}})


def ranked(prize, shares, ranking, n, i=0):
    return ('ranked_split', {'state': {}, 'input': {'prize': prize, 'shares': shares, 'ranking': ranking, 'pot': key(f'pot:{i}'),
                                                     'entrants': [key(f'e:{i}:{q}') for q in range(n)]}})


def milestone(released, paid, ms, tranches):
    return ('milestone_escrow', {'state': {'released': released, 'paid': paid},
                                 'input': {'milestone': ms, 'tranches': tranches, 'pot': key('pot'), 'contractor': key('c')}})


def bond(state, op, amount):
    return ('bond', {'state': state, 'op': op, 'amount': amount})


def ledger(table, transfers, capacity):
    return ('ledger', {'table': table, 'transfers': transfers, 'capacity': capacity})


def named_cases():
    a, b, c = key('a'), key('b'), key('c')
    st = lambda r, p=0, n=0: {'reserve': r, 'paid': p, 'claims': n}
    bs = lambda d, l, s, r: {'deposited': d, 'locked': l, 'slashed': s, 'released': r}
    return {
        'graded/43pct': graded(st(50_000_000_000), 43, 5_000_000_000),
        'graded/capped-at-reserve': graded(st(1_000), 100, 2_000),
        'graded/zero-pay': graded(st(0), 50, 2_000),
        'graded/101pct': graded(st(1_000), 101, 1_000),
        'graded/u64-coverage': graded(st(U64_MAX), 100, U64_MAX),
        'graded/u64-coverage-99pct': graded(st(U64_MAX), 99, U64_MAX - 12345),
        'graded/pay-over-i64': graded(st(U64_MAX), 100, I64_MAX + 1),
        'graded/paid-overflow': graded(st(10, U64_MAX - 5), 100, 10),
        'graded/claims-overflow': graded(st(10, 0, U64_MAX), 100, 10),
        'ranked/three-of-four': ranked(1_000_001, [500_000, 300_000, 200_000], [2, 0, 3], 4),
        'ranked/duplicate-rank': ranked(10, [500_000, 500_000], [1, 1], 4),
        'ranked/rank-out-of-range': ranked(10, [500_000, 500_000], [0, 4], 4),
        'ranked/shares-not-1e6': ranked(10, [500_000, 400_000], [0, 1], 4),
        'ranked/no-places': ranked(10, [], [], 4),
        'ranked/16-places': ranked(10, [62_500] * 16, list(range(16)), 20),
        'ranked/33-entrants': ranked(10, [1_000_000], [0], 33),
        'ranked/k-over-n': ranked(10, [500_000, 500_000], [0, 1], 1),
        'ranked/ranking-length': ranked(10, [500_000, 500_000], [0], 4),
        'ranked/i64-prize': ranked(I64_MAX, [333_333, 333_333, 333_334], [2, 1, 0], 3),
        'ranked/prize-over-i64': ranked(I64_MAX + 1, [1_000_000], [0], 1),
        'ranked/15-places': ranked(2**52 + 7, [66_666] * 14 + [66_676], list(range(14, -1, -1)), 32),
        'milestone/first': milestone([], 0, 1, [100, 200, 300]),
        'milestone/again': milestone([1], 200, 1, [100, 200, 300]),
        'milestone/out-of-range': milestone([], 0, 3, [100, 200, 300]),
        'milestone/none': milestone([], 0, 0, []),
        'milestone/65': milestone([], 0, 0, [1] * 65),
        'milestone/64th': milestone(list(range(63)), 5, 63, [7] * 64),
        'milestone/tranche-over-i64': milestone([], 0, 0, [I64_MAX + 1]),
        'milestone/paid-overflow': milestone([], U64_MAX, 0, [1]),
        'bond/lock': bond(bs(0, 0, 0, 0), 'lock', 500_000_000),
        'bond/slash-over': bond(bs(500_000_000, 500_000_000, 0, 0), 'slash', 500_000_001),
        'bond/slash': bond(bs(500_000_000, 500_000_000, 0, 0), 'slash', 499_999_999),
        'bond/release-over': bond(bs(10, 5, 5, 0), 'release', 6),
        'bond/release': bond(bs(10, 5, 5, 0), 'release', 5),
        'bond/spend': bond(bs(10, 5, 5, 0), 'spend', 1),
        'bond/lock-overflow': bond(bs(U64_MAX, U64_MAX, 0, 0), 'lock', 1),
        'bond/not-conserved': bond(bs(10, 5, 0, 0), 'slash', 1),
        'bond/128-bit-sum': bond(bs(U64_MAX, U64_MAX, U64_MAX, 2), 'release', 0),
        'ledger/insufficient-new': ledger([], [{'key': a, 'delta': -5}, {'key': b, 'delta': 5}], 4),
        'ledger/move': ledger([{'key': a, 'balance': 10}], [{'key': a, 'delta': -5}, {'key': b, 'delta': 5}], 4),
        'ledger/not-conserved': ledger([], [{'key': a, 'delta': 7}], 4),
        'ledger/full': ledger([{'key': a, 'balance': 10}], [{'key': a, 'delta': -5}, {'key': b, 'delta': 5}], 1),
        'ledger/over-capacity': ledger([{'key': a, 'balance': 1}, {'key': b, 'balance': 1}], [], 1),
        'ledger/overflow': ledger([{'key': a, 'balance': U64_MAX}, {'key': b, 'balance': 10}], [{'key': b, 'delta': -1}, {'key': a, 'delta': 1}], 4),
        'ledger/zero-deltas-skipped': ledger([], [{'key': a, 'delta': 0}, {'key': c, 'delta': 0}], 1),
        'ledger/i64-sum-wraps': ledger([], [{'key': a, 'delta': I64_MAX}, {'key': b, 'delta': I64_MAX}, {'key': c, 'delta': 2}], 4),
        'ledger/i64-min': ledger([{'key': a, 'balance': 1 << 63}], [{'key': a, 'delta': -(1 << 63)}, {'key': b, 'delta': I64_MAX}, {'key': b, 'delta': 1}], 4),
    }


def random_cases(n, seed=20261106):
    """n calls per program, in the property test's ranges."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        out.append(graded({'reserve': r.randint(0, 2**50), 'paid': r.randint(0, 2**40), 'claims': r.randint(0, 1000)},
                          r.randint(0, 100), r.randint(0, 2**50), i))
        nn = r.randint(1, 12); k = r.randint(1, min(nn, 15))
        cuts = sorted(r.randint(0, 1_000_000) for _ in range(k - 1))
        shares = [c - (cuts[j - 1] if j else 0) for j, c in enumerate(cuts + [1_000_000])]
        ranking = r.sample(range(nn), k)
        out.append(ranked(r.randint(0, 2**52), shares, ranking, nn, i))
        m = r.randint(1, 64)
        out.append(milestone([q for q in range(m) if r.random() < 0.3], r.randint(0, 2**40), r.randint(0, m - 1),
                             [r.randint(0, 2**45) for _ in range(m)]))
        locked = r.randint(0, 2**45)
        out.append(bond({'deposited': locked + 10, 'locked': locked, 'slashed': 6, 'released': 4},
                        r.choice(['lock', 'release', 'slash', 'slash']), r.randint(0, locked * 2 + 1)))
        keys = [key(f'acct:{i}:{q}') for q in range(r.randint(1, 6))]
        table = [{'key': kk, 'balance': r.randint(0, 10**9)} for kk in keys[:r.randint(0, len(keys))]]
        moves = []
        for _ in range(r.randint(1, 4)):
            amt = r.randint(0, 10**9)
            moves += [{'key': r.choice(keys), 'delta': -amt}, {'key': r.choice(keys), 'delta': amt}]
        out.append(ledger(table, moves, 6))
    return out
