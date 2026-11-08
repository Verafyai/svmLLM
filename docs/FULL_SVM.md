# The Solana VM in hand-set weights: results

This branch (`full-svm`) takes svmLLM from a reduced sBPF toy to Agave 4.3.2's program runtime:
SBPF v0 with ALU32 and byte-order instructions, the real memory map, BPF-to-BPF calls, the loader's
ELF image, account serialization, all 38 registered syscalls, cross-program invocation through both
ABIs, the System Program and the transaction-level rent check. Every piece is checked against
ground truth, not against itself.

Nothing here was trained. The weights are constant tensors written by hand: lookup-table ReLU
layers, DeltaNet RAM with beta in {0, 1}, exact-match attention for the function and syscall
tables, and ReLU gating.

## Ground truth

| What | Truth | How |
|---|---|---|
| VM rungs (instructions, memory map, calls) | Anza solana-sbpf 0.12 interpreter | `xcheck/` (svmcheck), registers captured through the trace hook |
| Programs, syscalls, CPI, System Program | LiteSVM 0.18.2 = Agave 4.3.2 runtime, mainnet feature set | `truth/` (svmtruth for one instruction, svmtx for whole transactions) |
| ELF loading and relocation | Agave 4.3.2 loader | `truth/src/bin/elfimage.rs` dumps the loaded image |

## Results

Pass counts are all measured on this machine (2 CPU cores, no GPU).

| Suite | Cases | Reference vs truth | Weights vs truth |
|---|---|---|---|
| VM: random programs with calls, gaps, every fault (seed 2026) | 600 | 600/600 (Anza) | 600/600 |
| Real compiled programs (TinySVM's six, seed 12) | 150 | 150/150 | 150/150 |
| Syscall exerciser, 29 non-curve syscalls (seed 101) | 400 | 400/400 | 400/400 |
| CPI, C and Rust ABIs, System Program, rent (seeds 21, 33; weights seed 7, CPI checks as circuits) | 2,500 (16 on the weights) | 2,500/2,500 | 16/16 |
| Curve syscalls, all families incl. pairings (seed 11) | 2,000 | 2,000/2,000 | see below |
| Curve syscalls in the weights, pairings left out (seed 41) | 24 | 24/24 | 24/24 |
| Showcase programs, one per syscall family, both pairings included (`run_showcase.py`) | 12 | 12/12 | 12/12 |
| Generated CPI transactions with distinct outcomes (`run_showcase.py cpi`) | 6 | 6/6 | 6/6 |
| Real transactions reviewed by the judgement filter (`judge_real.py weights`) | 3 | 3/3 | 3/3 |
| BN254 pairing-product check vs reference (`check_pairing.py bn`) | 1 | | match, 554 s |
| BLS12-381 pairing, blst's exact Gt bytes vs reference (`check_pairing.py bls`) | 1 | | match, 2,158 s |

Per family, reference vs LiteSVM: Ed25519 / Ristretto / secp256k1 1,500/1,500; BN254 800/800
(every op, both endiannesses, both return codes seen); Poseidon 400/400; BLS12-381 500/500.

Weights vs LiteSVM on curve syscalls (end to end, through the exerciser program): first run
11/12 (the miss was a BLS case started before BLS was in the weights), second run 10/12 (both misses
were pairing calls, before the pairings were in the weights). Both pairings are now in the weights
and each runs end to end inside a program in the showcase.

Each curve operation in the weights was also checked byte for byte against the reference on its
own: Ed25519 / Ristretto validate, add, sub, mul; secp256k1 recovery (valid and invalid); Poseidon
(BE and LE); BN254 G1/G2 add and mul and all four compression ops; BLS12-381 validate, add, sub,
mul and decompress for G1 and G2.

## What Agave does that this had to match

These came out of mismatches against LiteSVM, not from documentation:

- A VM fault drains the whole compute budget (`deplete_cu_meter_on_vm_failure`); a syscall error
  keeps the exact count. Syscall InstructionErrors surface as themselves, other syscall errors as
  ProgramFailedToComplete.
- `translate_mut!` refuses overlapping borrows (CopyOverlapping); alignment is checked on the host
  pointer; addresses at or above the input region are InvalidPointer for sysvar syscalls.
- `sol_remaining_compute_units` is not registered on mainnet: calling it is a VM fault.
- SIMD-0392 (relaxed post-execution rent check) is active.
- BorrowedAccount check order matters: `set_lamports` checks before the equality test, `set_owner`
  checks ownership and writability first, `can_resize` checks four things in a fixed order.
- System Program: bincode decoding before the account count, then AccountAlreadyInUse /
  AddressWithSeedMismatch / ResultWithNegativeLamports as Custom(0) / Custom(5) / Custom(1).
- An account LiteSVM reports as missing is a default account (0 lamports, System owner) to the
  runtime.
- BLS12-381 is enabled on mainnet. blst's pairing value is py_ecc's e(P, Q) to the power -3
  (blst's Miller loop runs over the negative BLS parameter and its hard part uses
  3 (p^4 - p^2 + 1) / r); found by matching e(G1, G2) against LiteSVM.
- arkworks details for BN254: uncompressed points carry SWFlags in the top two bits of y, the
  "negative" flag is ignored when uncompressed, G2 addition checks the curve but not the subgroup,
  G2 multiplication and the pairing check both, scalars are unreduced 256-bit integers.

## What is in the weights and what is not

| In the weights | Outside the weights (Python, as in Agave's host code) |
|---|---|
| Fetch, decode, ALU64 / ALU32, byte order, branches, calls and frames, address translation with stack gaps, CU meter | The driver loop and the syscall micro-step sequencing |
| Every syscall's datapath: charges, budget check, translation, alignment and overlap checks, every byte moved, compared, hashed, formatted | ELF loading and relocation (Agave's loader produces the image) |
| SHA-256, Keccak-256, base58, UTF-8 validation, hex / decimal / base64 formatting | Serializing accounts into the input region and deserializing them back |
| Ed25519, Ristretto, secp256k1, Poseidon, BN254 and BLS12-381 (group ops, compression, both pairings) | Starting the callee program when a CPI asks for one |
| CPI (`cpi_w.py`): C and Rust struct translation, privilege checks, PDA signers, reentrancy, depth and lamport-balance checks, costs, write-back | The System Program and the rent check |

The transaction context (account table, instruction stack, meter total) is host state, as in Agave:
the CPI circuits read it into one-hot rows and write back changed accounts.

### Pairings (`pairing.py`, `cpairing.py`)

One optimal-ate implementation over a backend. Fp12 is flat (six Fp2 coefficients of w, w^6 = xi),
which is py_ecc's field with the same w and blst's tower flattened. Lines are evaluated up to Fp4
factors (removed by the final exponentiation), the easy part uses conjugation, one inversion and
Frobenius, and the hard part evaluates the base-p digits of (p^4 - p^2 + 1) / r in one shared
square-and-multiply. The integer backend equals py_ecc exactly on both curves; the circuit backend
batches each Fp12 product (108 unreduced products, 12 wide Barrett reductions). Measured, one
thread: Fp12 multiply about 0.5 s (BN254) and 1.9 s (BLS12-381); a BN254 two-pair check 554 s;
a BLS12-381 pairing 2,158 s.

### How the curve arithmetic runs

`fastbig.py` keeps each 4-bit limb as the position of its one instead of the 16-wide one-hot vector,
and evaluates each table layer with `LUT.ix`, which reads the output for that one-hot input
straight out of the layer's own second-layer weights. That is the same value the dense ReLU layer
computes (its hidden layer is exactly the one matching unit), without the zero terms; the two were
compared on random one-hot inputs for every table used. Additions are parallel-prefix
(Kogge-Stone) adders built from three small tables; multiplies form all nibble products in one
batched lookup; reduction is Barrett's for any modulus.

Measured: a 255-bit field multiply takes about 30 ms on one thread, a 381-bit one about 70 ms.
One curve syscall on the weights takes from about 10 seconds (Ed25519 add) to several minutes
(BN254 G2 multiply, BLS12-381 G2 validation with its subgroup check).

## Speed of the VM itself

Measured on one thread, lazy mode, against Anza's interpreter on the same machine:

| Kernel | Weights, ms per instruction | Anza, ns per instruction | Slowdown |
|---|---|---|---|
| ALU (add, xor, shifts) | 2.5 | 8.5 | about 300,000x |
| mul | 11.5 | 8.9 | about 1,300,000x |
| div and mod | 38.3 | 9.7 | about 3,900,000x |
| stack load / store | 8.1 | 13.3 | about 600,000x |
| calls | 3.7 | 9.1 | about 400,000x |

## Not done

- **System Program nonce accounts** (instructions 4, 5, 6, 7, 12) and CreateAccountAllowPrefund:
  they raise NotImplementedError in the runtime.
- **Other builtins**: compute budget, the loaders, vote, stake. Only the System Program is a builtin here.
- **SBPF v1, v2, v3.** None of `enable_sbpf_v1/v2/v3_deployment_and_execution` is in LiteSVM 0.18.2's
  mainnet feature list, so mainnet executes v0 only and there is no mainnet behavior to check
  them against.
- **Transaction layer**: one instruction per transaction; no fee payer debit, signature
  verification, blockhash or account locking.
- The runtime pieces listed as outside the weights.
- Larger weights runs: weights batches are small because a CPI transaction takes minutes and a BLS
  pairing over half an hour on one thread.
