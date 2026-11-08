"""Reference interpreter vs LiteSVM on real programs: TinySVM's six and the syscall exerciser."""
import random, collections, sys, os
from svllm import solana, sref

DEPLOY = '/home/claude/verafyharness/packages/tinysvm/programs/target/deploy'


def judge(r, meta, acc, cu):
    accts, err = (solana.deserialize(r['input'], meta, acc) if r['status'] == 'ok' and r['r0'] == 0 else (None, None))
    return solana.outcome(r['status'], r['r0'], r['cu'], r['logs'], accts, err, limit=cu)


def cases(n, seed, kind):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = solana.gen_exerciser(rng) if kind == 'ex' else solana.gen_case(rng, DEPLOY)
        cu = 200000 if rng.random() < 0.8 else rng.choice([50, 120, 200, 300, 400, 700, 1000])
        out.append(c + (cu,))
    return out


if __name__ == '__main__':
    kind, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cs = cases(n, seed, kind)
    tr = solana.truth_many([(e, a, d, cu) for _, e, a, d, cu in cs])
    ok, bad, ex = 0, collections.Counter(), []
    for (p, e, a, d, cu), t in zip(cs, tr):
        inp, meta = solana.serialize(a, d)
        r = sref.run_program(solana.image(e), inp, cu)
        o = judge(r, meta, a, cu)
        s = solana.same(o, t); ok += s
        if not s:
            bad[(p, r['status'], t.get('err'), o['cu'] == t['cu'], o['logs'] == t['logs'], o['ok'] == t['ok'])] += 1
            if len(ex) < 6:
                ex.append((d.hex()[:84], r['status'], o['cu'], t['cu'], o['logs'][-2:], t['logs'][-2:], t.get('err'), o.get('err')))
    print('ref vs litesvm', ok, '/', len(cs))
    print(collections.Counter('ok' if t['ok'] else t['err'] for t in tr).most_common(8))
    for k, v in bad.most_common(12): print(v, k)
    for x in ex: print(x)
