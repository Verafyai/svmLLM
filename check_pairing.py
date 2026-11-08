"""Both pairings on the weights against the reference: `python3 check_pairing.py bn` (about 10 min), `python3 check_pairing.py bls` (about 35 min)."""
import time, torch, sys
torch.set_num_threads(1)
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from svllm import pairing as PR, cpairing as CP, fastbig as F, curves as R
from py_ecc import optimized_bn128 as n, optimized_bls12_381 as b
which = sys.argv[1]
if which == 'bn':
    I = PR.IntBackend(PR.BN)
    P = n.normalize(n.multiply(n.G1, 5)); Q = n.normalize(n.multiply(n.G2, 7))
    P2 = n.normalize(n.neg(n.multiply(n.G1, 35))); Q2 = n.normalize(n.G2)
    g1 = lambda P: (F.num(int(P[0].n), 64), F.num(int(P[1].n), 64))
    g2 = lambda Q: (tuple(F.num(int(c) % PR.BN.p, 64) for c in Q[0].coeffs), tuple(F.num(int(c) % PR.BN.p, 64) for c in Q[1].coeffs))
    t = time.time()
    one = CP.bn_pairing_is_one([(g1(P), g2(Q)), (g1(P2), g2(Q2))])
    print('bn product e(5G,7H) e(-35G,H) is one:', one, round(time.time() - t), 's', flush=True)
    t = time.time()
    one = CP.bn_pairing_is_one([(g1(P), g2(Q))])
    print('bn single pairing is one (expect False):', one, round(time.time() - t), 's', flush=True)
else:
    P = b.normalize(b.multiply(b.G1, 5)); Q = b.normalize(b.multiply(b.G2, 7))
    g1 = lambda P: (F.num(int(P[0]) % PR.BLS.p, 96), F.num(int(P[1]) % PR.BLS.p, 96))
    g2 = lambda Q: (tuple(F.num(int(c) % PR.BLS.p, 96) for c in Q[0].coeffs), tuple(F.num(int(c) % PR.BLS.p, 96) for c in Q[1].coeffs))
    t = time.time()
    out = CP.bls_gt_bytes([(g1(P), g2(Q))])
    got = bytes(int(x[0].argmax()) | (int(x[1].argmax()) << 4) for x in out)
    ref = R.bls_gt([((int(P[0]) % PR.BLS.p, int(P[1]) % PR.BLS.p), tuple(tuple(int(c) % PR.BLS.p for c in v.coeffs) for v in Q))])
    print('bls Gt bytes match reference:', got == ref, round(time.time() - t), 's', flush=True)
