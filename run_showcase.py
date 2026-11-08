"""Curated runs for the microsite: real compiled Solana programs calling real syscalls, each run on
LiteSVM (Agave 4.3.2), the reference VM and the hand-set weights, with the weights' wall time.
Writes results/showcase.json. Slow: some cases take minutes on the weights."""
import json, time, struct, random, hashlib, sys
import torch
torch.set_num_threads(1)
from svllm import solana, sref, curves as C, runtime
import check_curves as cc
import check_cpi as cp
from py_ecc import optimized_bn128 as PB

O = solana.OUT_VM
op = cc.op
TAG = {4: 'sol_log_', 5: 'sol_log_64_', 6: 'sol_log_compute_units_', 8: 'sol_sha256', 9: 'sol_keccak256',
       11: 'sol_log_data', 12: 'sol_set_return_data', 17: 'sol_get_clock_sysvar', 26: 'sol_create_program_address',
       27: 'sol_try_find_program_address', 28: 'sol_curve_validate_point', 29: 'sol_curve_group_op',
       33: 'sol_secp256k1_recover', 34: 'sol_poseidon', 35: 'sol_alt_bn128_group_op', 32: 'sol_curve_pairing_map'}


class Case:
    def __init__(self):
        self.data = bytearray(4096)
        self.n = 0
        self.ops = []

    def put(self, bs):
        o = 1024 + self.n
        self.data[o:o + len(bs)] = bs
        self.n += (len(bs) + 31) // 32 * 32
        return O + o

    def call(self, tag, *args):
        self.ops.append(op(tag, *args))

    def build(self):
        return b''.join(self.ops), [{'data': bytes(self.data), 'owner': 'program', 'writable': True}]


def cases():
    out = []

    def add(cid, title, story, c, cu=200000):
        data, acc = c.build()
        out.append({'id': cid, 'title': title, 'story': story, 'calls': [TAG.get(b[0], str(b[0])) for b in c.ops],
                    'case': (cc.EX_ELF, acc, data, cu)})

    c = Case()
    msg = c.put(b'hello from the weights')
    c.call(4, msg, 22); c.call(5, 1, 2, 3, 4, 5); c.call(6)
    add('logs', 'Logging', 'Log a message, five numbers and the remaining compute units.', c)

    c = Case()
    m = c.put(b'svmLLM'); v = c.put(struct.pack('<QQ', m, 6))
    c.call(8, v, 1, O + 3072); c.call(9, v, 1, O + 3136)
    add('hash', 'SHA-256 and Keccak-256', 'Hash the bytes "svmLLM" both ways; the digests land in account memory.', c)

    c = Case()
    s = c.put(b'vault'); sv = c.put(struct.pack('<QQ', s, 5)); pid = c.put(hashlib.sha256(b'some-program').digest())
    c.call(27, sv, 1, pid, O + 3072, O + 3200)
    add('pda', 'Find a program-derived address', 'Search bump seeds from 255 down until SHA-256(seeds, bump, program, marker) lands off the Ed25519 curve.', c)

    c = Case()
    B = C.ed_decompress(bytes.fromhex('58' + '66' * 31))
    P1, P2 = C.ed_compress(C.ed_mul(7, B)), C.ed_compress(C.ed_mul(11, B))
    p1, p2 = c.put(P1), c.put(P2)
    c.call(28, 0, p1); c.call(29, 0, 0, p1, p2, O + 3072)
    add('ed25519', 'Ed25519: validate and add', 'Check that 7B is on the curve, then add 7B + 11B. The result is 18B.', c)

    c = Case()
    k = c.put((123456789).to_bytes(32, 'little')); p = c.put(P1)
    c.call(29, 1, 2, k, c.put(C.ristretto_encode(C.ed_mul(7, B))), O + 3072)
    add('ristretto', 'Ristretto: scalar multiply', 'Multiply a Ristretto point by 123,456,789.', c)

    c = Case()
    h, sig, recid = None, None, None
    for seed in range(50):
        h, sig, recid = cc.secp_case(random.Random(seed))
        if C.secp256k1_recover(h, recid, sig)[0] == 0:
            break
    c.call(33, c.put(h), recid, c.put(sig), O + 3072)
    add('secp256k1', 'secp256k1: recover a public key', 'Ethereum-style signature recovery: the 64-byte public key that signed this hash.', c)

    c = Case()
    a, b2 = c.put((1).to_bytes(32, 'big')), c.put((2).to_bytes(32, 'big'))
    d = c.put(struct.pack('<QQQQ', a, 32, b2, 32))
    c.call(34, 0, 0, d, 2, O + 3072)
    add('poseidon', 'Poseidon hash', 'Poseidon (circom BN254 parameters) of the field elements 1 and 2.', c)

    c = Case()
    g = lambda P: C.be_to_le(C.BN_G1.out((int(PB.normalize(P)[0].n), int(PB.normalize(P)[1].n))), 32)
    inp = g(PB.multiply(PB.G1, 3)) + g(PB.multiply(PB.G1, 4))
    c.call(35, 0, c.put(inp), 128, O + 3072)
    add('bn254', 'BN254: add two points', 'alt_bn128 G1 addition, the operation Ethereum-compatible verifiers lean on: 3G + 4G.', c)

    c = Case()
    import check_curves
    pair = check_curves.bn_pairing_identity(random.Random(3), 0)
    c.call(35, 3, c.put(pair), len(pair), O + 3072)
    add('bn254-pairing', 'BN254: pairing check', 'e(aG, bH) * e(-abG, H) = 1: the check at the heart of a Groth16 proof verifier.', c, cu=200000)

    c = Case()
    from py_ecc import optimized_bls12_381 as PL
    ka = 0x1234567
    g1 = check_curves.bls_enc(PL.multiply(PL.G1, ka), False, 0x80) + check_curves.bls_enc(PL.neg(PL.G1), False, 0x80)
    g2 = check_curves.bls_enc(PL.G2, True, 0x80) + check_curves.bls_enc(PL.multiply(PL.G2, ka), True, 0x80)
    c.call(32, 0x84, 2, c.put(g1), c.put(g2), O + 3072)
    add('bls-pairing', 'BLS12-381: pairing product', 'e(aG1, G2) * e(-G1, aG2): the check behind BLS signature aggregation. The 576-byte result is the identity of the target group.', c, cu=200000)

    c = Case()
    c.call(4, 0x100000000 + 1, 5000)
    add('fault', 'A bad pointer', 'Log 5,000 bytes from inside the program region: the translation circuit refuses it.', c)

    c = Case()
    for _ in range(3):
        c.call(29, 0, 2, c.put((5).to_bytes(32, 'little')), c.put(P1), O + 3072)
    add('budget', 'Out of compute units', 'Three Ed25519 multiplies at 2,177 CU each under a 5,000 CU budget.', c, cu=5000)
    return out


def main(which):
    cs = cases()
    if which != 'all':
        cs = [c for c in cs if c['id'] in which.split(',')]
    truth = solana.truth_many([c['case'] for c in cs])
    img = solana.image(cc.EX_ELF)
    res = []
    try:
        res = json.load(open('results/showcase.json'))
    except Exception:
        pass
    done = {r['id'] for r in res}
    for c, t in zip(cs, truth):
        e, acc, data, cu = c['case']
        ref = cc.ref_run(img, acc, data, cu)
        t0 = time.time()
        w = cc.weights_run(img, acc, data, cu)
        wt = time.time() - t0
        rec = {'id': c['id'], 'title': c['title'], 'story': c['story'], 'calls': c['calls'], 'cu_limit': cu,
               'truth': {'ok': t['ok'], 'err': t.get('err'), 'cu': t['cu'], 'logs': t['logs']},
               'ref_same': solana.same(ref, t), 'weights_same': solana.same(w, t), 'weights_seconds': round(wt, 1),
               'returns': [], 'output': None}
        if t['ok']:
            accd = t['accounts'][0]
            rec['returns'] = [int.from_bytes(accd[512 + 8 * k:520 + 8 * k], 'little') for k in range(len(c['calls']))]
            rec['output'] = accd[3072:3136].hex()
        res = [r for r in res if r['id'] != c['id']] + [rec]
        json.dump(res, open('results/showcase.json', 'w'), indent=1)
        print(c['id'], 'weights matches LiteSVM:', rec['weights_same'], f'{wt:.0f}s', flush=True)




def cpi_cases():
    """A few generated CPI transactions with distinct outcomes, described from their own bytes."""
    import random as _r
    rng = _r.Random(77)
    reqs = [cp.gen(rng) for _ in range(120)]
    truth = runtime.truth_tx(reqs)
    names = {cp.EX: 'the syscall exerciser', cp.SYS: 'the System Program', cp.CPI: 'itself'}
    keys = [cp.A, cp.B, cp.C, cp.EX, cp.SYS, cp.CPI, cp.D, cp.E, cp.F]
    want = ['ok-rust', 'ok-c', 'ok-self', 'PrivilegeEscalation', 'ExternalAccountLamportSpend', 'ReadonlyLamportChange', 'UnbalancedInstruction']
    picked = {}
    for q, t in zip(reqs, truth):
        d = bytes.fromhex(q['ix']['data'])
        mode, prog = d[0], keys[d[1]] if d[1] < len(keys) else None
        abi = 'Rust' if mode & 8 else 'C'
        if t['ok']:
            key = 'ok-self' if prog == cp.CPI else ('ok-rust' if abi == 'Rust' else 'ok-c')
        else:
            key = t['err'].split(', ')[-1].rstrip(')')
        if key in want and key not in picked and q['cu'] == 200000:
            what = f"calls {names.get(prog, 'an account that is not a program')} through the {abi} ABI"
            if mode & 1: what += ', then writes 0xAB into an account it owns'
            if mode & 2: what += ', moving a lamport between accounts'
            if mode & 4: what += ', and aborts afterwards'
            picked[key] = (q, t, what)
    return [(k,) + picked[k] for k in want if k in picked]


def run_cpi():
    out = []
    for key, q, t, what in cpi_cases():
        ref = runtime.run_tx(q, t, cp.ref_vm)
        t0 = time.time()
        w = runtime.run_tx(q, t, cp.weights_vm)
        rec = {'id': key, 'what': 'The CPI program ' + what + '.', 'truth': {'ok': t['ok'], 'err': t.get('err'), 'cu': t['cu'], 'logs': t['logs']},
               'ref_same': runtime.same_tx(ref, t), 'weights_same': runtime.same_tx(w, t), 'weights_seconds': round(time.time() - t0, 1)}
        out.append(rec)
        json.dump(out, open('results/showcase_cpi.json', 'w'), indent=1)
        print(key, 'weights matches LiteSVM:', rec['weights_same'], rec['weights_seconds'], flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'cpi':
        run_cpi()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else 'all')
