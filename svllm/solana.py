"""Run real Solana programs: the loaded ELF image (from Agave's own loader), the input buffer
(Agave's ABI v1 serializer, byte for byte), the post-run account checks, and the ground
truth (LiteSVM 0.18.2 = Agave 4.3.2 runtime, mainnet feature set).

Outside the weights, as in Agave: ELF loading and relocation (truth/src/bin/elfimage.rs runs
Agave's loader), serializing accounts into the input region, and copying them back. Inside the
weights: every instruction and syscall the program executes.
"""
import hashlib, json, os, random, struct, subprocess
from functools import lru_cache

ROOT = os.path.join(os.path.dirname(__file__), '..')
ELFIMAGE = os.path.join(ROOT, 'truth', 'target', 'release', 'elfimage')
TRUTH = os.path.join(ROOT, 'truth', 'target', 'release', 'svmtruth')
MAX_PERMITTED_DATA_INCREASE = 10 * 1024
SYSCALL_NAMES = ('abort', 'sol_log_', 'sol_memcpy_', 'sol_memset_', 'sol_memmove_', 'sol_memcmp_', 'sol_log_64_',
                 'sol_log_compute_units_', 'sol_panic_', 'sol_sha256', 'sol_keccak256', 'sol_log_pubkey',
                 'sol_log_data', 'sol_set_return_data', 'sol_get_return_data', 'sol_get_stack_height',
                 'sol_remaining_compute_units', 'sol_get_clock_sysvar', 'sol_get_epoch_schedule_sysvar',
                 'sol_get_rent_sysvar', 'sol_get_epoch_rewards_sysvar', 'sol_get_last_restart_slot', 'sol_get_sysvar',
                 'sol_get_epoch_stake', 'sol_alloc_free_', 'sol_get_processed_sibling_instruction',
                 'sol_create_program_address', 'sol_try_find_program_address', 'sol_curve_validate_point',
                 'sol_curve_group_op', 'sol_curve_multiscalar_mul', 'sol_curve_decompress', 'sol_curve_pairing_map',
                 'sol_secp256k1_recover', 'sol_poseidon', 'sol_alt_bn128_group_op', 'sol_alt_bn128_compression',
                 'sol_invoke_signed_c', 'sol_invoke_signed_rust')
# log lines a program produces (the runtime's own invoke / consumed / success lines are not compared)
PROGRAM_LINES = ('Program log: ', 'Program data: ', 'Program consumption: ', 'Program return: ')


def h(s):
    return hashlib.sha256(s.encode()).digest()


PROGRAM_ID = h('svmtruth:program')


@lru_cache(None)
def image(path):
    """Agave's loaded image: text, read-only region, entry, functions, syscalls."""
    d = json.loads(subprocess.run([ELFIMAGE, path], capture_output=True, text=True, check=True).stdout)
    d['text'], d['ro'] = bytes.fromhex(d['text']), bytes.fromhex(d['ro'])
    d['funcs'] = {k: pc for k, pc in d['funcs']}
    d['syscalls'] = {k: n for k, n in d['syscalls'] if n in SYSCALL_NAMES}
    d['all_syscalls'] = {k: n for k, n in json.loads(subprocess.run([ELFIMAGE, path], capture_output=True, text=True).stdout)['syscalls']}
    return d


def rent_exempt(n):
    return (128 + n) * 5080              # LiteSVM 0.18.2 rent: 128 bytes of overhead, checked against truth


def serialize(accounts, data, keys=None):
    """Agave ABI v1 (loader v2/v3), no duplicates. accounts: [{data, owner, writable}].
    Returns (buffer, meta) with each account's data offset."""
    out, meta = bytearray(struct.pack('<Q', len(accounts))), []
    for i, a in enumerate(accounts):
        key = keys[i] if keys else h(f'svmtruth:account:{i}')
        owner = PROGRAM_ID if a['owner'] == 'program' else bytes(32)
        d = a['data']
        out += bytes([0xff, 0, int(a['writable']), 0]) + bytes(4) + key + owner
        out += struct.pack('<QQ', max(rent_exempt(len(d)), 1), len(d))
        off = len(out)
        out += d + bytes(MAX_PERMITTED_DATA_INCREASE + (-len(d)) % 8)
        out += struct.pack('<Q', 2**64 - 1)
        meta.append({'off': off, 'len': len(d), 'hdr': off - 16 - 64 - 8, 'owner': owner, 'writable': a['writable'],
                     'lamports': max(rent_exempt(len(d)), 1)})
    out += struct.pack('<Q', len(data)) + data + PROGRAM_ID
    return bytes(out), meta


def deserialize(buf, meta, accounts):
    """Agave's ABI v1 writeback (deserialize_parameters_for_abiv1 and BorrowedAccount's rules),
    with virtual_address_space_adjustments off. Returns (new data list, error or None)."""
    new, lam_delta = [], 0
    for m, a in zip(meta, accounts):
        owned, writable, old = a['owner'] == 'program', a['writable'], a['data']
        owner = bytes(buf[m['off'] - 16 - 32:m['off'] - 16])
        lam = struct.unpack_from('<Q', buf, m['off'] - 16)[0]
        if lam != m['lamports']:
            if not owned and lam < m['lamports']:
                return None, 'ExternalAccountLamportSpend'
            if not writable:
                return None, 'ReadonlyLamportChange'
            lam_delta += lam - m['lamports']
        post = struct.unpack_from('<Q', buf, m['off'] - 8)[0]
        if max(post - len(old), 0) > MAX_PERMITTED_DATA_INCREASE or post > 10 * 1024 * 1024:
            return None, 'InvalidRealloc'
        if m['off'] + post > len(buf):
            return None, 'InvalidArgument'
        d = bytes(buf[m['off']:m['off'] + post])
        err = None if writable and owned else ('ReadonlyDataModified' if not writable else 'ExternalAccountDataModified')
        if err and d != old:
            return None, err
        if owner != m['owner']:
            if not owned or not writable or any(d):
                return None, 'ModifiedProgramId'
        new.append(d if not err else old)
    if lam_delta:
        return None, 'UnbalancedInstruction'
    return new, None


def truth_many(cases):
    """cases: (elf_path, accounts, data, cu). Ground truth from LiteSVM."""
    req = ''.join(json.dumps({'elf': e, 'cu': cu, 'data': d.hex(),
                              'accounts': [{'data': a['data'].hex(), 'owner': a['owner'], 'writable': a['writable']} for a in acc]}) + '\n'
                  for e, acc, d, cu in cases)
    lines = subprocess.run([TRUTH], input=req, capture_output=True, text=True, check=True).stdout.splitlines()
    out = []
    for line in lines[1:]:
        r = json.loads(line)
        logs = [l for l in r['logs'] if l.startswith(PROGRAM_LINES)]
        if r['ok']:
            out.append({'ok': True, 'cu': r['cu'], 'logs': logs, 'accounts': [bytes.fromhex(x) for x in r['accounts']]})
        else:
            out.append({'ok': False, 'cu': r['cu'], 'logs': logs, 'err': r['err'], 'raw_logs': r['logs']})
    return out


def outcome(status, r0, cu, logs, accounts=None, err=None, limit=None):
    """Map a VM result to what the runtime reports. A VM fault (not a syscall error) depletes the
    whole remaining budget (Agave's deplete_cu_meter_on_vm_failure); a failed syscall charge reports
    ComputationalBudgetExceeded; any other failure is ProgramFailedToComplete."""
    if status == 'ok' and r0 == 0 and err is None:
        return {'ok': True, 'cu': cu, 'logs': logs, 'accounts': accounts}
    if status == 'ok' and r0 != 0:
        e = f'InstructionError(0, Custom({r0 & 0xffffffff}))' if r0 < 2**32 else f'InstructionError(0, builtin {r0:#x})'
    elif err:
        e = f'InstructionError(0, {err})'
    elif status.startswith('Sys'):
        # a syscall that fails with an InstructionError surfaces that error itself
        e = ('InstructionError(0, ComputationalBudgetExceeded)' if 'Budget' in status else
             f'InstructionError(0, {status[3:]})' if status[3:] in ('ArithmeticOverflow',) else
             'InstructionError(0, ProgramFailedToComplete)')
    else:
        e = 'InstructionError(0, ProgramFailedToComplete)'
        cu = limit if limit is not None else cu
    return {'ok': False, 'cu': cu, 'logs': logs, 'err': e}


def same(a, b):
    if a['ok'] != b['ok'] or a['cu'] != b['cu'] or a['logs'] != b['logs']:
        return False
    return a['accounts'] == b['accounts'] if a['ok'] else a['err'] == b['err']


# ------------------------------------------------------------------ test cases
def u8(x): return struct.pack('<B', x & 0xff)
def u32(x): return struct.pack('<I', x & 0xffffffff)
def u64(x): return struct.pack('<Q', x & (2**64 - 1))
def i64(x): return struct.pack('<q', x)
def f64(x): return struct.pack('<d', x)


def transfers(rng, n, keys):
    return u32(n) + b''.join(rng.choice(keys) + i64(rng.choice([1, -1, 5, -5, 1000, -1000, 0, 2**62, -2**62])) for _ in range(n))


def gen_case(rng, deploy):
    """One instruction for one of the six TinySVM programs: mostly valid, some edge and bad."""
    prog = rng.choice(['bond', 'escrow', 'ledger', 'graded_payout', 'ranked_split', 'milestone_escrow'])
    elf = os.path.join(deploy, prog + '.so')
    big = [0, 1, 7, 1000, 10**6, 2**32, 2**63, 2**64 - 1]
    keys = [h(f"k{i}") for i in range(6)]
    out = lambda: {'data': bytes(4 + 16 * 40), 'owner': 'program', 'writable': True}
    bad = rng.random() < 0.15
    if prog == 'bond':
        st = b''.join(u64(rng.choice(big)) for _ in range(4))
        acc = [{'data': st if not bad else st[:rng.randrange(32)], 'owner': 'program', 'writable': rng.random() > 0.05}]
        data = u8(rng.choice([0, 1, 2, 0, 1, 2, 3])) + u64(rng.choice(big))
    elif prog == 'escrow':
        op = rng.choice([0, 1, 2, 9])
        if op == 0:
            acc = [{'data': bytes(64), 'owner': 'program', 'writable': True}]
            data = u8(0) + b''.join(f64(rng.choice([100.0, 1.0, 0.25, 0.1, 1e6, 3.3, 0.0])) for _ in range(4)) + u32(rng.choice([0, 1, 3, 5])) + u8(rng.choice([0, 1])) + f64(rng.choice([0.0, 50.0, 0.5]))
        else:
            pot = rng.choice(keys)
            st = u64(rng.choice(big[:5])) + u64(rng.choice([0, 1, 5])) + (pot if rng.random() < 0.7 else bytes(32))
            acc = [{'data': st, 'owner': 'program', 'writable': True}]
            if op == 1:
                data = u8(1) + rng.choice(keys) + u64(rng.choice(big))
            elif op == 2:
                acc.append({'data': transfers(rng, rng.randint(0, 4), keys), 'owner': 'system', 'writable': False})
                data = u8(2)
            else:
                data = u8(9)
    elif prog == 'ledger':
        cap = rng.choice([2, 4, 6])
        n = rng.randint(0, cap)
        t = u32(n) + b''.join(keys[i] + u64(rng.choice([0, 5, 1000, 2**63])) for i in range(n)) + bytes(40 * (cap - n))
        acc = [{'data': t, 'owner': 'program', 'writable': True}]
        data = u8(0) + transfers(rng, rng.randint(0, 4), keys)
    elif prog == 'graded_payout':
        st = u64(rng.choice([0, 1000, 10**6])) + u64(rng.choice([0, 10])) + u64(rng.choice([0, 1, 2]))
        acc = [{'data': st, 'owner': 'program', 'writable': True}, out()]
        data = u8(0) + u64(rng.choice([0, 1, 50, 100, 101])) + u64(rng.choice([0, 500, 10**6])) + rng.choice(keys) + rng.choice(keys)
    elif prog == 'ranked_split':
        k = rng.randint(1, 4); m = rng.randint(1, 4)
        shares = [rng.choice([0, 1, 50, 100, 5000]) for _ in range(k)]
        ranking = rng.sample(range(m), m) if rng.random() < 0.9 else [0] * m
        acc = [out()]
        data = u8(0) + u64(rng.choice([0, 100, 10**6, 2**63])) + u8(k) + b''.join(u32(s) for s in shares) + u8(m) + b''.join(u8(r) for r in ranking) + rng.choice(keys) + b''.join(rng.choice(keys) for _ in range(m))
    else:
        n = rng.randint(1, 5)
        st = u64(rng.choice([0, 1, 3, 2**n - 1])) + u64(rng.choice([0, 100]))
        acc = [{'data': st, 'owner': 'program', 'writable': True}, out()]
        data = u8(0) + u8(rng.choice([0, 1, 2, 4, 9])) + u8(n) + b''.join(u64(rng.choice([1, 100, 10**6])) for _ in range(n)) + rng.choice(keys) + rng.choice(keys)
    if bad:
        r = rng.random()
        if r < 0.4:
            data = data[:rng.randrange(max(1, len(data)))]
        elif r < 0.7:
            data = data + bytes(rng.randint(1, 3))
        else:
            data = bytes([rng.randrange(256)]) + data[1:]
    return prog, elf, acc, data


# ------------------------------------------------------------------ syscall exerciser cases
EXERCISER = os.path.join(ROOT, 'programs', 'exerciser', 'target', 'deploy', 'exerciser.so')
OUT_VM = 0x400000000 + 96                    # account 0's data, one account, no duplicates
TAGS = {'memmove': 0, 'memcmp': 1, 'memcpy': 2, 'memset': 3, 'log': 4, 'log64': 5, 'logcu': 6, 'panic': 7,
        'sha256': 8, 'keccak': 9, 'logpubkey': 10, 'logdata': 11, 'setret': 12, 'getret': 13, 'height': 14,
        'remaining': 15, 'abort': 16, 'clock': 17, 'epoch_schedule': 18, 'rent': 19, 'epoch_rewards': 20,
        'last_restart': 21, 'get_sysvar': 22, 'epoch_stake': 23, 'alloc': 24, 'sibling': 25, 'pda': 26,
        'find_pda': 27}


def gen_exerciser(rng):
    """Account 0 (1024 bytes): [0, 128) slice descriptors, [128, 512) text and bytes, [512, 1024) results."""
    data = bytearray(1024)
    texts = [b'hello', 'héllo ✓ \U0001f680'.encode(), b'\xff\xfe bad', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'ok', b'']
    pos = 128
    spans = []
    for s in texts:
        data[pos:pos + len(s)] = s; spans.append((OUT_VM + pos, len(s))); pos += len(s) + 3
    data[300:332] = bytes(range(32)); data[340:400] = bytes(rng.getrandbits(8) for _ in range(60))
    from .sref import SYSVARS
    for j, nm in enumerate(['clock', 'slot_hashes', 'rent']):
        data[400 + 32 * j:432 + 32 * j] = SYSVARS[nm]['id']
    data[460:492] = SYSVARS[rng.choice(list(SYSVARS))]['id'] if rng.random() < 0.8 else bytes(32)
    nsl = rng.randint(0, 7)
    for i in range(nsl):                         # VmSlice descriptors: ptr, len
        p, n = rng.choice([(OUT_VM + 300, 32), (OUT_VM + 340, 60), (OUT_VM + 128, 5), (OUT_VM + 340, 0),
                           (0x300000000, 100), (0x100000000, 64), (0x500000000, 4), (OUT_VM + 1000, 64), (OUT_VM + 900, 2**62)])
        data[16 * i:16 * i + 8] = p.to_bytes(8, 'little'); data[16 * i + 8:16 * i + 16] = n.to_bytes(8, 'little')

    def addr(kind='any'):
        c = rng.random()
        if kind == 'text' and c < 0.7:
            return rng.choice(spans)
        p = rng.choice([OUT_VM + rng.choice([0, 1, 2, 4, 512, 520, 600, 1000, 1020]), OUT_VM + 1024 + rng.choice([0, 100, 10239]),
                        0x200000000 + rng.choice([0, 8, 4000, 4092, 4096, 8192, 8200]), 0x300000000 + rng.choice([0, 3, 32760, 32768]),
                        0x100000000 + rng.choice([0, 8, 100]), 0, 0x500000000, OUT_VM - 96 + rng.choice([0, 20000, 11500])])
        return p, None

    ops = []
    for _ in range(rng.randint(1, 4)):
        name = rng.choice(list(TAGS))
        n = rng.choice([0, 1, 4, 8, 32, 100, 250, 600, 2**63])
        if name in ('memmove', 'memcpy'):
            d = rng.choice([OUT_VM + 512, OUT_VM + 600, OUT_VM + 640]) if rng.random() < 0.7 else addr()[0]
            s = rng.choice([OUT_VM + 300, OUT_VM + 340, OUT_VM + 520, 0x100000000]) if rng.random() < 0.7 else addr()[0]
            args = [d, s, rng.choice([0, 4, 32, 60, 100, 300]) if rng.random() < 0.85 else n]
        elif name == 'memcmp':
            args = [rng.choice([OUT_VM + 300, OUT_VM + 340, addr()[0]]), rng.choice([OUT_VM + 300, OUT_VM + 301, addr()[0]]),
                    rng.choice([0, 1, 16, 32, 60]), rng.choice([OUT_VM + 520, OUT_VM + 522, OUT_VM + 1020, OUT_VM + 1023, 0x100000000])]
        elif name == 'memset':
            args = [rng.choice([OUT_VM + 512, OUT_VM + 700]) if rng.random() < 0.7 else addr()[0], rng.randrange(512), rng.choice([0, 8, 100, 512, 513, n])]
        elif name in ('log', 'panic'):
            p, ln = addr('text') if rng.random() < 0.8 else (addr()[0], None)
            ln = rng.choice([0, 3, 100]) if ln is None else ln
            args = [p, ln, rng.randrange(1000), rng.randrange(80)]
        elif name == 'log64':
            args = [rng.getrandbits(64) for _ in range(5)]
        elif name in ('sha256', 'keccak'):
            args = [rng.choice([OUT_VM, OUT_VM + 4, OUT_VM + 16, addr()[0]]), rng.choice([0, 1, nsl, nsl, 20001]),
                    rng.choice([OUT_VM + 520, OUT_VM + 1000, OUT_VM + 992, 0x100000000, OUT_VM + 523])]
        elif name == 'logpubkey':
            args = [rng.choice([OUT_VM + 300, OUT_VM + 1, OUT_VM + 1000, addr()[0]])]
        elif name == 'logdata':
            args = [rng.choice([OUT_VM, OUT_VM + 8, addr()[0]]), rng.choice([0, 1, nsl, 3])]
        elif name == 'setret':
            args = [rng.choice([OUT_VM + 300, OUT_VM + 128, addr()[0]]), rng.choice([0, 5, 32, 1024, 1025, 2000])]
        elif name == 'getret':
            args = [rng.choice([OUT_VM + 600, OUT_VM + 1020, addr()[0]]), rng.choice([0, 3, 32, 100]), rng.choice([OUT_VM + 700, OUT_VM + 1010, addr()[0]])]
        elif name in ('clock', 'epoch_schedule', 'rent', 'epoch_rewards', 'last_restart'):
            args = [rng.choice([0x300000000, 0x300000000 + rng.choice([4, 8, 16, 32760]), 0x200000000 + rng.choice([8, 4000]),
                                OUT_VM + 512, 0x100000000, 0])]
        elif name == 'get_sysvar':
            args = [rng.choice([OUT_VM + 400, OUT_VM + 432, OUT_VM + 460]), rng.choice([0x300000000, 0x300000003, OUT_VM + 600, 0x100000000]),
                    rng.choice([0, 1, 8, 16, 40, 20000, 2**64 - 1]), rng.choice([0, 1, 8, 17, 33, 40, 100, 20488])]
        elif name == 'epoch_stake':
            args = [rng.choice([0, OUT_VM + 300, OUT_VM + 1020, 0x500000000])]
        elif name == 'alloc':
            args = [rng.choice([0, 1, 7, 100, 4096, 32768, 40000, 2**63]), rng.choice([0, 0, 0, 0x300000000])]
        elif name in ('pda', 'find_pda'):
            seeds_at = rng.choice([OUT_VM, OUT_VM, OUT_VM + 4, 0x500000000])
            args = [seeds_at, rng.choice([0, 1, 2, nsl, 17]), rng.choice([OUT_VM + 300, OUT_VM + 340, OUT_VM + 1020]),
                    rng.choice([OUT_VM + 600, OUT_VM + 1000, 0x100000000]), rng.choice([OUT_VM + 640, OUT_VM + 610, OUT_VM + 600])]
        elif name == 'sibling':
            args = [rng.choice([0, 1, 5]), OUT_VM + 600, OUT_VM + 700, OUT_VM + 800, OUT_VM + 900]
        else:
            args = []
        args = (args + [0] * 5)[:5]
        ops.append(bytes([TAGS[name]]) + b''.join((a & (2**64 - 1)).to_bytes(8, 'little') for a in args))
    acc = [{'data': bytes(data), 'owner': 'program', 'writable': True}]
    return 'exerciser', EXERCISER, acc, b''.join(ops)
