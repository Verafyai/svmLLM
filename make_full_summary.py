"""results/full_summary.json: the numbers and lists the microsite's full-machine sections show.
Counts come from the result files where those exist; the rest are the measured runs listed in
docs/FULL_SVM.md."""
import json, os

R = 'results/'
load = lambda f, d=None: json.load(open(R + f)) if os.path.exists(R + f) else d
full, progs, sysc = load('full.json'), load('programs.json'), load('syscalls.json')
show = load('showcase.json', [])
cpi_show = load('showcase_cpi.json', [])
STATE = load('full_state.json', {})        # what has landed in the weights, written as runs finish

PAIRINGS_IN = STATE.get('pairings_in_weights', False)
CPI_IN = STATE.get('cpi_in_weights', False)
n_in = 34 + (2 if CPI_IN else 0) + (2 if PAIRINGS_IN else 0)


def frac(xs):
    ok = sum(1 for x in xs if x)
    return f'{ok}/{len(xs)}'


suites = [
    {'suite': 'Random VM programs: calls, stack gaps, every fault', 'truth': "Anza's solana-sbpf 0.12", 'cases': f"{full['cases']}",
     'ref': full['ref_vs_anza'], 'weights': full['weights_vs_anza']},
    {'suite': 'Real compiled programs', 'truth': 'LiteSVM 0.18.2 (Agave 4.3.2)', 'cases': f"{progs['cases']}",
     'ref': progs['ref_vs_truth'], 'weights': progs['weights_vs_truth']},
    {'suite': 'Syscall exerciser: the non-curve syscalls', 'truth': 'LiteSVM 0.18.2 (Agave 4.3.2)', 'cases': f"{sysc['cases']}",
     'ref': sysc['ref_vs_truth'], 'weights': sysc['weights_vs_truth']},
    {'suite': 'CPI transactions: C and Rust ABIs, System Program, rent', 'truth': 'LiteSVM 0.18.2 (Agave 4.3.2)',
     'cases': STATE.get('cpi_cases', '2,500 (16 on the weights)'), 'ref': '2,500/2,500', 'weights': STATE.get('cpi_weights', '16/16')},
    {'suite': 'Curve syscalls, every family', 'truth': 'LiteSVM 0.18.2 (Agave 4.3.2)', 'cases': STATE.get('curve_cases', '2,000 (24 on the weights)'),
     'ref': '2,000/2,000', 'weights': STATE.get('curve_weights', '24/24, pairings left out')},
    {'suite': 'Real programs cards on this page', 'truth': 'LiteSVM 0.18.2 (Agave 4.3.2)', 'cases': f'{len(show) + len(cpi_show)}',
     'ref': '', 'weights': frac([c['weights_same'] for c in show + cpi_show])},
]
crypto = [
    {'family': 'Ed25519 and Ristretto', 'calls': 'sol_curve_validate_point, sol_curve_group_op, sol_curve_multiscalar_mul', 'ref': '1,500/1,500 (with secp256k1)', 'weights': 'yes', 'time': 'about 20 s (add), about 100 s (multiply)'},
    {'family': 'secp256k1', 'calls': 'sol_secp256k1_recover', 'ref': 'included above', 'weights': 'yes', 'time': 'about 3 min'},
    {'family': 'Poseidon (circom BN254)', 'calls': 'sol_poseidon', 'ref': '400/400', 'weights': 'yes', 'time': 'about 20 s (2 inputs)'},
    {'family': 'BN254 group ops and compression', 'calls': 'sol_alt_bn128_group_op, sol_alt_bn128_compression', 'ref': '800/800', 'weights': 'yes', 'time': '15 s (add) to about 6 min (G2 multiply)'},
    {'family': 'BN254 pairing', 'calls': 'sol_alt_bn128_group_op (pairing op)', 'ref': 'included above',
     'weights': 'yes' if PAIRINGS_IN else 'reference only', 'time': STATE.get('bn_pairing_time', 'not run on the weights')},
    {'family': 'BLS12-381 validate, add, sub, mul, decompress', 'calls': 'sol_curve_validate_point, sol_curve_group_op, sol_curve_decompress', 'ref': '500/500 (with the pairing)', 'weights': 'yes', 'time': 'up to about 4 min'},
    {'family': 'BLS12-381 pairing', 'calls': 'sol_curve_pairing_map', 'ref': 'included above',
     'weights': 'yes' if PAIRINGS_IN else 'reference only', 'time': STATE.get('bls_pairing_time', 'not run on the weights')},
]
speed = [
    ['ALU instruction (add, xor, shifts)', '2.5 ms', "about 300,000x Anza's interpreter (8.5 ns)"],
    ['mul', '11.5 ms', 'about 1,300,000x (8.9 ns)'],
    ['div and mod', '38.3 ms', 'about 3,900,000x (9.7 ns)'],
    ['stack load or store', '8.1 ms', 'about 600,000x (13.3 ns)'],
    ['call', '3.7 ms', 'about 400,000x (9.1 ns)'],
    ['255-bit field multiply', 'about 20 ms', 'was 1.5 s before indexed tables and parallel-prefix adders'],
    ['381-bit field multiply', 'about 30 ms', ''],
    ['Fp12 multiply (BN254 / BLS12-381)', STATE.get('f12_time', 'about 0.5 s / 1.9 s'), 'lazy reduction: 108 products, 12 reductions'],
]
for c in show:
    speed.append([c['title'] + ' (whole program)', f"{c['weights_seconds']} s", f"{c['truth']['cu']:,} CU on Agave"])

in_weights = [
    'The VM: fetch, decode, ALU64 and ALU32 (with ALU32\'s sign-extension rule), byte order, branches, BPF-to-BPF calls and callx, 64 stack frames with gaps, the compute meter, every fault.',
    "Solana's memory map: program, stack, heap and the serialized input region, as paged DeltaNet RAM.",
    f'{n_in} of the 38 syscalls Agave registers on mainnet, end to end: costs, budget checks, pointer translation, alignment and overlap checks, and every byte they move, hash or format.',
    'Hashes and encodings: SHA-256, Keccak-256, base58, UTF-8 validation, hex, decimal and base64 formatting, program-derived addresses.',
    'Elliptic curves: Ed25519, Ristretto, secp256k1 recovery, Poseidon, BN254 add, multiply and compression, BLS12-381 validate, add, subtract, multiply and decompress'
    + (', and both pairings (BN254 and BLS12-381).' if PAIRINGS_IN else '.'),
]
if CPI_IN:
    in_weights.append('Cross-program invocation: the C and Rust structure translation, privilege checks, PDA signers, call-depth and reentrancy checks, the lamport balance check, costs and the write-back into the caller.')
outside = ['The ELF loader: Agave\'s own loader produces the program image.',
           'Account serialization into the input region and back, with Agave\'s write-back rules.',
           'The System Program and the post-execution rent check.',
           'The driver loops: the step loop, and running a callee program when a CPI asks for one.']
if not CPI_IN:
    outside.insert(2, 'CPI bookkeeping (privilege checks, account translation, write-back) for both invoke syscalls.')
if not PAIRINGS_IN:
    outside.append('The BN254 and BLS12-381 pairings: verified in the reference VM, not in the weights.')
outside.append('Not built anywhere: nonce accounts, the other builtins (compute budget, loaders, vote, stake), multi-instruction transactions, SBPF v1 to v3 (not active on mainnet).')

gallery = [('full-step.png', 'One sBPF instruction is one pass through fixed layers.'),
           ('full-memory.png', "Solana's four memory regions in paged DeltaNet RAM."),
           ('full-syscall.png', 'A syscall: look up, charge, translate, compute, write back.'),
           ('full-crypto.png', 'The curve syscalls, from one lookup table to an elliptic curve.'),
           ('full-cpi.png', 'Cross-program invocation.'),
           ('full-perf.png', 'Measured slowdown against Anza\'s interpreter.')]

out = {
    'chips': [['VM programs vs Anza:', full['weights_vs_anza']], ['compiled programs vs LiteSVM:', progs['weights_vs_truth']],
              ['syscalls in the weights:', f'{n_in} of 38'], ['syscalls matching Agave:', '38 of 38'], ['parameters trained:', '0']],
    'hero_chips': [['Full machine:', f'{n_in} of 38 syscalls in the weights']],
    'suites': suites, 'crypto': crypto, 'speed': speed, 'in_weights': in_weights, 'outside': outside, 'gallery': gallery,
}
json.dump(out, open(R + 'full_summary.json', 'w'), indent=1)
print('syscalls in weights', n_in)
