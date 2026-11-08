import pickle, time, sys
from svllm import solana, sref, svm
import check_ref
cases = pickle.load(open('/tmp/claude-0/-home-claude/32786c52-b5fd-5ba7-b1a9-8871f5ca6177/scratchpad/pda_cases.pkl', 'rb'))
tr = solana.truth_many([(e, a, d, cu) for _, e, a, d, cu in cases])
pick = [i for i, t in enumerate(tr) if t['ok']][:int(sys.argv[1])] + [i for i, t in enumerate(tr) if not t['ok']][:int(sys.argv[2])]
ok = 0
for i in pick:
    p, e, a, d, cu = cases[i]; t = tr[i]
    inp, meta = solana.serialize(a, d); t0 = time.time()
    r = svm.run_program(solana.image(e), inp, cu, program_id=solana.PROGRAM_ID, program_id_b58=sref.b58(solana.PROGRAM_ID))
    o = check_ref.judge(r, meta, a, cu); s = solana.same(o, t); ok += s
    print(i, s, r['status'], o['cu'], t['cu'], round(time.time() - t0, 1), 's', flush=True)
print('weights PDA', ok, '/', len(pick))
