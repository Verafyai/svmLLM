"""Judgement filter on real transactions: the CPI program asks the System Program to move lamports,
the transaction runs on LiteSVM, the reference runtime and the weights, and the rule judge reviews
the actual outcome (who lost and gained lamports, the logs, the CU, the exact error).
Writes results/judge_real.json. `python3 judge_real.py [weights]`."""
import json, struct, sys, time
import torch
torch.set_num_threads(1)
import check_cpi as cp
from svllm import runtime, judge
from svllm.sref import b58

KEYS = [cp.A, cp.B, cp.C, cp.EX, cp.SYS, cp.CPI, cp.D, cp.E, cp.F]
NAMES = {cp.A: 'savings (known)', cp.B: 'wallet (signer)', cp.D: 'fresh address'}


def request(dest, lamports, signer=True):
    ix = struct.pack('<IQ', 2, lamports)
    inner = [(KEYS.index(cp.B), 1, 1), (KEYS.index(dest), 0, 1)]
    data = (bytes([0, KEYS.index(cp.SYS), len(inner)]) + b''.join(bytes(x) for x in inner) + bytes([len(KEYS)]) + bytes(range(len(KEYS)))
            + struct.pack('<H', len(ix)) + ix + b'\x00')
    accounts = [{'key': cp.A.hex(), 'owner': cp.SYS.hex(), 'lamports': 10**9, 'data': ''},
                {'key': cp.B.hex(), 'owner': cp.SYS.hex(), 'lamports': 10 * 10**9, 'data': ''},
                {'key': cp.C.hex(), 'owner': cp.EX.hex(), 'lamports': 10**9, 'data': '00' * 1024},
                {'key': cp.D.hex(), 'owner': cp.SYS.hex(), 'lamports': 0, 'data': ''},
                {'key': cp.E.hex(), 'owner': cp.SYS.hex(), 'lamports': 0, 'data': ''},
                {'key': cp.F.hex(), 'owner': cp.SYS.hex(), 'lamports': 10**9, 'data': ''}]
    flags = {k: (False, False) for k in KEYS}
    flags.update({cp.A: (False, True), cp.B: (signer, True), cp.C: (False, True), cp.D: (False, True), cp.E: (False, True), cp.F: (False, True)})
    metas = [{'key': k.hex(), 'signer': flags[k][0], 'writable': flags[k][1]} for k in KEYS]
    return {'cu': 200000, 'programs': [{'id': cp.CPI.hex(), 'elf': cp.CPI_ELF}, {'id': cp.EX.hex(), 'elf': cp.EX_ELF}],
            'accounts': accounts, 'ix': {'program': cp.CPI.hex(), 'metas': metas, 'data': data.hex()}}


SOL = 10**9
HIST = [(2 * SOL // 100, cp.A), (3 * SOL // 100, cp.A), (2 * SOL // 100, cp.A), (25 * SOL // 1000, cp.A)]   # 0.02-0.03 SOL to savings
EXAMPLES = [
    ('routine', 'Routine transfer to savings', 'The wallet sends 0.025 SOL to its own savings account, as it has four times before.', cp.A, 25 * SOL // 1000, True),
    ('drain', 'Valid by code, 9 SOL to a fresh address', 'The same program, the same signature, 9 SOL: 400x the usual amount, to an address this wallet has never paid.', cp.D, 9 * SOL, True),
    ('unsigned', 'The wallet did not sign', 'The program asks the System Program to move the wallet\'s lamports, but the wallet is not a signer. The runtime refuses before any judgement.', cp.A, SOL // 100, False),
]


def review(outcome, pre, dest, amount):
    """Turn the transaction outcome into the judge's inputs. The amount and destination are read
    from the post-state: the lamports the wallet actually lost and the account that gained them."""
    vm = {'status': 'ok' if outcome['ok'] else 'error', 'r0': 0, 'cu': outcome['cu']}
    if outcome['ok']:
        before = {a['key']: a.get('lamports', 0) for a in pre}
        after = {k: v[1] for k, v in outcome['post'].items()}
        moved = before[cp.B.hex()] - after[cp.B.hex()]
        gained = [k for k in after if after[k] > before.get(k, 0)]
        dest_name = NAMES.get(bytes.fromhex(gained[0]), gained[0][:8]) if gained else 'none'
    else:
        moved, dest_name = 0, NAMES[dest]
    t = judge.Tx(amount=moved or amount, destination=dest_name, time=10 * 86400, balance_before=10 * SOL)
    ctx = judge.Context(history=[judge.Tx(a, NAMES[d], i * 86400, 10 * SOL) for i, (a, d) in enumerate(HIST)])
    v = judge.RuleJudge().review(vm, t, ctx)
    return {'moved_lamports': moved, 'destination': dest_name, 'decision': v.decision, 'reason': v.reason, 'signals': v.signals}


def main(weights):
    reqs = [request(d, amt, s) for _, _, _, d, amt, s in EXAMPLES]
    truth = runtime.truth_tx(reqs)
    out = []
    for (eid, title, story, dest, amt, signer), q, t in zip(EXAMPLES, reqs, truth):
        ref = runtime.run_tx(q, t, cp.ref_vm)
        rec = {'id': eid, 'title': title, 'story': story, 'truth': {'ok': t['ok'], 'err': t.get('err'), 'cu': t['cu'], 'logs': t['logs']},
               'ref_same': runtime.same_tx(ref, t), 'review': review(t, t['pre'], dest, amt)}
        if weights:
            t0 = time.time()
            w = runtime.run_tx(q, t, cp.weights_vm)
            rec['weights_same'] = runtime.same_tx(w, t)
            rec['weights_seconds'] = round(time.time() - t0, 1)
        out.append(rec)
        print(eid, t['ok'], t.get('err'), t['cu'], rec['review']['decision'], rec.get('weights_same'), flush=True)
        json.dump(out, open('results/judge_real.json', 'w'), indent=1)


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'weights')
