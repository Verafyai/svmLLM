"""A line-by-line Python transcription of the five integer settlement functions in
Verafy Harness `packages/core/src/settle.js` (runtime.js side), used as the oracle in
this repo's tests. The cross-runtime test in Verafy Harness compares the weights machine
against runtime.js itself and TinySVM; this file lets the suites here run without Node.
"""
U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


def err(code):
    return {'ok': False, 'code': code}


def _transfers_out(lst):
    if len(lst) > 16:
        return err(3)
    if sum(d for _, d in lst) != 0:
        return err(5)
    return {'ok': True, 'transfers': [{'key': k, 'delta': d} for k, d in lst]}


def graded_claim(state, inp):
    pct = int(inp['percent'])
    if pct > 100:
        return err(6)
    reserve, paid, claims = int(state['reserve']), int(state['paid']), int(state['claims'])
    owed = int(inp['coverage']) * pct // 100
    pay = owed if owed < reserve else reserve
    if pay > I64_MAX:
        return err(4)
    out = _transfers_out([[inp['pot'], -pay], [inp['beneficiary'], pay]] if pay > 0 else [])
    if not out['ok']:
        return out
    if paid + pay > U64_MAX or claims + 1 > U64_MAX:
        return err(4)
    return {**out, 'state': {'reserve': reserve - pay, 'paid': paid + pay, 'claims': claims + 1}}


def ranked_split(_state, inp):
    ranking, shares, entrants = inp['ranking'], inp['shares'], inp['entrants']
    k, n = len(shares), len(entrants)
    if k == 0 or k + 1 > 16 or n > 32:
        return err(1)
    if k != len(ranking) or k > n:
        return err(1)
    if sum(int(x) for x in shares) != 1_000_000:
        return err(1)
    seen = set()
    for r in ranking:
        if r >= n or r in seen:
            return err(6)
        seen.add(r)
    P = int(inp['prize'])
    if P > I64_MAX:
        return err(4)
    lst = [[inp['pot'], -P]]
    total = 0
    for i in range(k):
        pay = P * int(shares[i]) // 1_000_000
        total += pay
        lst.append([entrants[ranking[i]], pay])
    lst[1][1] += P - total
    out = _transfers_out(lst)
    return {**out, 'state': {}} if out['ok'] else out


def milestone_release(state, inp):
    tranches, milestone = inp['tranches'], inp['milestone']
    m = len(tranches)
    if m == 0 or m > 64:
        return err(1)
    if milestone >= m:
        return err(6)
    released = set(state['released'])
    if milestone in released:
        return err(100)
    t = int(tranches[milestone])
    if t > I64_MAX:
        return err(4)
    out = _transfers_out([[inp['pot'], -t], [inp['contractor'], t]])
    if not out['ok']:
        return out
    paid = int(state['paid']) + t
    if paid > U64_MAX:
        return err(4)
    return {**out, 'state': {'released': sorted(released | {milestone}), 'paid': paid}}


def bond_apply(state, op, amount):
    dep, locked, slashed, released = (int(state[k]) for k in ('deposited', 'locked', 'slashed', 'released'))
    a = int(amount)
    if op == 'lock':
        dep += a; locked += a
        if dep > U64_MAX or locked > U64_MAX:
            return err(4)
    elif op == 'release':
        if a > locked:
            return err(101)
        locked -= a; released += a
        if released > U64_MAX:
            return err(4)
    elif op == 'slash':
        if a > locked:
            return err(100)
        locked -= a; slashed += a
        if slashed > U64_MAX:
            return err(4)
    else:
        return err(1)
    if locked + slashed + released != dep:
        return err(5)
    return {'ok': True, 'state': {'deposited': dep, 'locked': locked, 'slashed': slashed, 'released': released}}


def ledger_apply(table, transfers, capacity):
    if len(table) > capacity:
        return err(2)
    if sum(int(t['delta']) for t in transfers) != 0:
        return err(5)
    rows = [{'key': r['key'], 'balance': int(r['balance'])} for r in table]
    for t in transfers:
        d = int(t['delta'])
        if d == 0:
            continue
        row = next((r for r in rows if r['key'] == t['key']), None)
        if row is None:
            if d < 0:
                return err(100)
            if len(rows) == capacity:
                return err(3)
            row = {'key': t['key'], 'balance': 0}
            rows.append(row)
        nb = row['balance'] + d
        if nb < 0:
            return err(100)
        if nb > U64_MAX:
            return err(4)
        row['balance'] = nb
    return {'ok': True, 'table': rows}


def initial_state(program):
    if program == 'graded_payout':
        return {'reserve': 0, 'paid': 0, 'claims': 0}
    if program == 'milestone_escrow':
        return {'released': [], 'paid': 0}
    return {}


def run_program(program, call):
    if program == 'graded_payout':
        return graded_claim(call['state'], call['input'])
    if program == 'ranked_split':
        return ranked_split(call.get('state'), call['input'])
    if program == 'milestone_escrow':
        return milestone_release(call['state'], call['input'])
    if program == 'bond':
        return bond_apply(call['state'], call['op'], call['amount'])
    if program == 'ledger':
        return ledger_apply(call['table'], call['transfers'], call['capacity'])
    raise ValueError(program)
