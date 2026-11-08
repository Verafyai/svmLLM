"""Cross-program invocation: random transactions (the CPI program calling the exerciser, the System
Program or itself), replayed by the runtime + reference VM (or the weights) against LiteSVM."""
import hashlib, random, struct, sys, collections, os
from svllm import solana, sref, runtime

H = lambda s: hashlib.sha256(s.encode()).digest()
ROOT = os.path.dirname(os.path.abspath(__file__))
CPI_ELF = os.path.join(ROOT, 'programs/cpi/target/deploy/cpi.so')
EX_ELF = os.path.join(ROOT, 'programs/exerciser/target/deploy/exerciser.so')
CPI, EX, SYS = H('cpi'), H('exerciser'), bytes(32)
A, B, C = H('acct-a'), H('acct-b'), H('acct-c')


def ex_ops(rng):
    op = lambda t, *a: bytes([t]) + b''.join((x % 2**64).to_bytes(8, 'little') for x in (list(a) + [0] * 5)[:5])
    o = solana.OUT_VM
    choices = [op(5, *[rng.getrandbits(32) for _ in range(5)]), op(3, o + 512, rng.randrange(256), 40),
               op(12, o + 300, 8), op(6), op(14), op(4, o + 128, 5)]
    return b''.join(rng.choice(choices) for _ in range(rng.randint(0, 2)))


def ix_of(keys, k):
    return keys.index(k) if k in keys else 0


def cpi_data(rng, keys, depth):
    """keys: the caller's account list (index -> key). Returns the CPI program's instruction data."""
    mode = rng.choice([0, 0, 0, 1, 2, 4])
    target = rng.choice(['ex', 'ex', 'sys', 'self'] if depth < 6 else ['ex', 'sys'])
    pid = {'ex': EX, 'sys': SYS, 'self': CPI}[target]
    prog = keys.index(pid) if pid in keys and rng.random() < 0.95 else rng.randrange(len(keys))
    if target == 'sys':
        tag = rng.choice([2, 2, 1, 8])
        if tag == 2:
            inner = [(ix_of(keys, B), 1, 1), (ix_of(keys, A), 0, 1)]
            data = struct.pack('<IQ', 2, rng.choice([0, 1, 5, 10**6, 10**12]))
        elif tag == 1:
            inner = [(ix_of(keys, B), 1, 1)]
            data = struct.pack('<I', 1) + rng.choice([CPI, EX, SYS])
        else:
            inner = [(ix_of(keys, B), 1, 1)]
            data = struct.pack('<IQ', 8, rng.choice([0, 8, 100]))
    elif target == 'ex':
        inner = [(ix_of(keys, rng.choice([A, C])), 0, 1)]
        data = ex_ops(rng)
    else:
        inner = [(i, 0, 1) for i in range(len(keys)) if keys[i] in (A, B, C) and rng.random() < 0.7] or [(0, 0, 1)]
        sub_keys = [keys[i] for i, _, _ in inner] + [EX, SYS, CPI]
        inner += [(ix_of(keys, EX), 0, 0), (ix_of(keys, SYS), 0, 0), (ix_of(keys, CPI), 0, 0)]
        data = cpi_data(rng, sub_keys, depth + 1)
    for _ in range(rng.choice([0, 0, 0, 1])):                       # flip a privilege bit
        if inner:
            i = rng.randrange(len(inner)); a, s, w = inner[i]
            inner[i] = (a, s ^ (rng.random() < 0.5), w ^ (rng.random() < 0.5))
    infos = list(range(len(keys)))
    if rng.random() < 0.1:
        infos = infos[:-1]
    seeds = b'\x00' if rng.random() < 0.85 else b'\x01\x01\x03abc'
    return (bytes([mode, prog, len(inner)]) + b''.join(bytes(x) for x in inner) + bytes([len(infos)]) + bytes(infos)
            + struct.pack('<H', len(data)) + data + seeds)


def gen(rng):
    owners = {A: rng.choice([CPI, CPI, EX, SYS]), B: SYS, C: rng.choice([EX, CPI])}
    sizes = {A: rng.choice([1024, 1024, 0, 64]), B: 0, C: 1024}
    accounts = [{'key': k.hex(), 'owner': owners[k].hex(), 'lamports': rng.choice([10**9, 10**9, 900000, 1]),
                 'data': ('00' * sizes[k])} for k in (A, B, C)]
    keys = [A, B, C, EX, SYS, CPI]
    flags = {A: (0, 1), B: (rng.random() < 0.8, 1), C: (0, rng.random() < 0.9), EX: (0, 0), SYS: (0, 0), CPI: (0, 0)}
    metas = [{'key': k.hex(), 'signer': bool(flags[k][0]), 'writable': bool(flags[k][1])} for k in keys]
    cu = 200000 if rng.random() < 0.85 else rng.choice([1500, 3000, 6000])
    return {'cu': cu, 'programs': [{'id': CPI.hex(), 'elf': CPI_ELF}, {'id': EX.hex(), 'elf': EX_ELF}],
            'accounts': accounts, 'ix': {'program': CPI.hex(), 'metas': metas, 'data': cpi_data(rng, keys, 0).hex()}}


def ref_vm(img, inp, limit, ctx):
    return sref.run_program(img, inp, limit, ctx=ctx)


def weights_vm(img, inp, limit, ctx):
    from svllm import svm
    return svm.run_program(img, inp, limit, program_id=ctx['frame'].program, ctx=ctx)


if __name__ == '__main__':
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    vm = weights_vm if len(sys.argv) > 3 and sys.argv[3] == 'weights' else ref_vm
    rng = random.Random(seed)
    reqs = [gen(rng) for _ in range(n)]
    truth = runtime.truth_tx(reqs)
    ok, bad, ex = 0, collections.Counter(), []
    for q, t in zip(reqs, truth):
        o = runtime.run_tx(q, t, vm)
        s = runtime.same_tx(o, t); ok += s
        if not s:
            bad[(o.get('err'), t.get('err'), o['cu'] == t['cu'], o['logs'] == t['logs'])] += 1
            if len(ex) < 5:
                ex.append((q['ix']['data'][:60], o['ok'], o['cu'], t['cu'], o.get('err'), t.get('err'), o['logs'][:2], t['logs'][:2]))
    print(('runtime + weights' if vm is weights_vm else 'runtime + reference') + ' vs litesvm', ok, '/', n)
    print(collections.Counter('ok' if t['ok'] else t['err'] for t in truth).most_common(10))
    for k, v in bad.most_common(10): print(v, k)
    for x in ex: print(x)
