# The settlement core: Verafy Harness exits on svmLLM weights

Research track (Part H of the Verafy Harness spec). A demo, not a settlement engine.

[Verafy Harness](https://github.com/Verafyai/verafyHarness) settles a jury's ruling with exact integer programs. runtime.js (`packages/core/src/settle.js`) is the reference, and TinySVM runs the same programs as compiled sBPF in LiteSVM. This branch adds a third runtime: the same settlement math, written by hand in svmLLM's reduced sBPF, executed on the hand-built weights machine. Every instruction is one forward pass through constant tensors; every activation is exactly 0 or 1; nothing is trained, and nothing drifts.

| Part | Status |
|---|---|
| H1. The instruction subset the compiled exit programs execute | Built: traced on solana-sbpf (`svllm/subset.py`, `results/settlement_subset.json`) |
| H2. The settlement programs in svmLLM's subset, on the weights | Built: `svllm/settle.py`, 5 programs, 288 instructions, 31 opcodes |
| H3. A third runtime in Verafy Harness's cross-runtime test | Built: `packages/tinysvm/src/svmllm.js` in verafyHarness, optional, skipped with the reason when svmLLM is absent |
| H4. Size and latency | Measured, below |
| H5. A proof of one settlement | Mock only. Real-prover cost modeled; the success criterion is untested |
| The jury pay rule and escrow hold and release | Not covered (the pay rule is IEEE-754 doubles; this core is integer only) |

## What is covered

`graded_payout.claim`, `ranked_split.split`, `milestone_escrow.release`, `bond` (lock, release, slash) and `ledger` (apply a transfer list). Each returns exactly what runtime.js returns, including the integer error code, so a refusal compares like a success.

**Not covered:** the jury pay rule and escrow hold and release (the `escrow` program). runtime.js splits a ruling's escrow in doubles, rounding each share with `Math.round`, and TinySVM reproduces those doubles bit for bit with soft-float in sBPF. Doing the same here would mean an IEEE-754 unit in lookup tables; it was left out, so those calls stay two-runtime.

**In the weights:** every number and every decision. Bounds, overflow and conservation checks; the multiply-then-divide without a 128-bit product (split the 64-bit operand at 32 bits, divide twice); the ranking's duplicate check; the ledger's row search and append; the error code, which is r0 at exit.

**Outside the weights (plain Python, `svllm/settle.py`):** layout. Account keys are interned to small integers before the run, since equality is all the programs ask of a key, and the transfer list is assembled from the numbers the run wrote, in the order runtime.js writes them. Structural refusals that TinySVM's encoder makes (a value that does not fit a u8) are made here too, before anything runs. The ledger holds at most 32 rows and 16 transfers per call (`Unsupported` beyond that).

## H1: the subset

`python -m svllm.subset <verafyHarness>/packages/tinysvm/programs/target/deploy 300` runs the compiled programs (cargo-build-sbf 1.18.8, platform tools v1.41, hashes pinned in verafyHarness `programs.lock.json`) on Anza's solana-sbpf 0.12.2 with instruction tracing (`xcheck/src/bin/sotrace.rs`), over inputs encoded exactly as TinySVM encodes them, and asserts the error code matches this core's on every call. Measured Nov 6, 2026:

| Program | Calls | Mean instructions executed | Distinct opcodes | Outside svmLLM's subset |
|---|---|---|---|---|
| graded_payout | 309 | 1,343.5 | 34 | internal `call` |
| ranked_split | 312 | 2,621.2 | 36 | internal `call` |
| milestone_escrow | 308 | 2,096.4 | 33 | internal `call` |
| bond | 309 | 594.1 | 25 | internal `call` |
| ledger | 308 | 1,229.2 | 37 | `be`, internal `call` |
| escrow (jury split, doubles) | 300 | 11,225.4 | 37 | internal `call` |

Across the five integer programs: 40 distinct opcodes, all 64-bit (no ALU32 executed), plus `sol_memcpy_` and `sol_memset_` syscalls. The compiled code also needs the loader's input layout (10 KiB of slack per account) and 4 KiB stack frames.

So the compiled `.so` files cannot run on svmLLM: no calls, no syscalls, and memory windows far smaller than the loader's buffer. The honest route was to write the settlement functions again in svmLLM's subset. The result uses 31 opcodes, all 64-bit: `add sub mul div or and lsh rsh mov` (imm and reg forms as used), `neg`, `ja jeq jne jgt jge jlt jsge jslt`, `ldxdw stxdw stdw`, `lddw`, `exit`. No `mod`, `xor`, `arsh`, `call`, ALU32 or byte loads. The measured calls below run 27 to 156 instructions; the compiled programs average 594 to 2,621, because the compiled code spends most of its time parsing the loader buffer and copying 32-byte keys.

The trade-off: this is a second implementation of `settle.js`, kept in step by tests, not by construction. A change to an exit program in Verafy Harness must be made here too, and the cross-runtime test is what catches a miss.

## H2: the weights

- `machine.Machine(..., win=1024, ops=settle.OPS)`. `win` widens the two memory windows from 128 to 1,024 bytes (`WideMem`: one ReLU layer turns the low 11 address bits into a one-hot key, and bytes 1 to 7 of a word reuse the adder's lookup table instead of shift permutations). `ops` decodes only the 31 opcodes (anything else is InvalidInstruction) and builds only their circuits. The default machine is unchanged: same 1,130,672 parameters, and `run_tests.py` reproduces `results/results.json`.
- 690,256 constant parameters, 2,761,024 bytes as float32 (`settle.weights_inventory()`). Every weight is 0 or 1 and every bias a small negative integer, so an integer encoding loses nothing.
- `settle.weights_digest()`: SHA-256 over every tensor and the five programs, the commitment a prover would publish: `5c1385cb5f28e4c25b92f848e87423d80a931a1487f2c27ee9f937fbda9faa9b`.

## Checks

`python run_settlement.py` (measured Nov 6, 2026, Apple M4, 16 GB, torch 2.15.0): 1,547 calls, 47 named edge cases (overflow at 2^63 and 2^64, out-of-range grades and ranks, double release, 128-bit sums) and 300 random calls per program in the ranges of Verafy Harness's property test.

| Comparison | Result |
|---|---|
| Reference interpreter vs oracle (`svllm/settle_oracle.py`, a transcription of settle.js) | 1547/1547 |
| Anza's solana-sbpf 0.12 vs reference interpreter, same bytes and windows | 1547/1547 |
| Weights machine vs reference interpreter | 1547/1547 |
| Weights machine vs oracle | 1547/1547 |
| Full mode (every subset circuit, every step) vs lazy mode | 20/20 |

In Verafy Harness (`npm test` with `SVMLLM_HOME` set): runtime.js, TinySVM and the weights agree on the three exit examples and on 300 random calls per integer program, seed 20261106, faults 0.

`python -m unittest tests/test_settlement.py` runs the quick version.

## H4: size and latency

Measured Nov 6, 2026, Apple M4, 16 GB, one CPU thread, torch 2.15.0. Medians of 20 runs (lazy) and 4 (full). An earlier run the same day gave 169 ms lazy for graded_payout, so read them as plus or minus about 10%.

| Call | Steps | Weights, lazy | Weights, full | Reference interpreter | Dense FLOPs, full mode |
|---|---|---|---|---|---|
| graded_payout, 43 percent | 40 | 179 ms | 2,153 ms | 0.030 ms | 7.18 x 10^9 |
| ranked_split, 3 places of 4 | 156 | 645 ms | 7,808 ms | 0.107 ms | 2.80 x 10^10 |
| milestone_escrow release | 27 | 67 ms | 1,070 ms | 0.022 ms | 4.85 x 10^9 |
| bond slash | 29 | 81 ms | 1,328 ms | 0.019 ms | 5.21 x 10^9 |
| ledger, one move | 88 | 189 ms | 3,740 ms | 0.060 ms | 1.58 x 10^10 |

Steps over the 1,547-call suite: graded 4 to 41, ranked 4 to 624, milestone 4 to 27, bond 11 to 29, ledger 5 to 334. Full mode is about 1.8 x 10^8 dense FLOPs per step (torch FlopCounterMode), most of it the nibble multiplier and the 64-stage divider. TinySVM runs the same calls in 40 to 71 µs. The weights are thousands of times slower than either; that is the cost of every step being a network.

## H5: a proof

Run from Verafy Harness: `node packages/tinysvm/scripts/svmllm-proof.mjs --write` (results in its `results/svmllm-settlement-2026-11-06.json`).

- **Mock prover (packages/prove):** ran. One graded-insurance settlement (43 percent, 2,150,000,000 base units) proved under the commitment `svmllm-settlement/<weights digest>`, and the proof verifies; a tampered answer and another commitment fail. The mock binds commitment, input and answer. It proves nothing about the computation.
- **DeepProve:** not run. Its license is not accepted; that is Rex's decision.
- **FlashProve:** not run. It needs Linux and an NVIDIA GPU, and it targets a specific transformer architecture, not this network.

**Modeled** with Verafy Harness `packages/economics` (preset "L2, FlashProve, unpacked"), assuming proving cost scales with dense FLOPs. The judgment proof: 2 x 17B active parameters (Llama 4 Maverick, vendor-reported by Meta) x 1,024 tokens = 3.48 x 10^13 FLOPs, $0.0103 proving per request. A settlement: 0.014% to 0.080% of that for the measured calls, and at most 0.32% at the worst step count in the suite. Verification and storage per proof would not shrink with it. By parameters alone the gap looks even larger (690,256 against 17B), which understates the settlement: its lookup tables are reused hundreds of times a step.

What the model leaves out: a proof would need full mode (lazy mode skips circuits depending on the data) and a fixed step count (the halt check is in the Python driver), so a real circuit is sized for the worst case, not the typical one. The memory is a recurrent DeltaNet state carried across up to 624 sequential steps, which provers built for one transformer pass over a prompt may not handle as a plain cost multiple.

**Success criterion: untested, so not met.** A settlement proof cheaper than the judgment proof is plausible on this model, but the model is an assumption until a prover that takes this network has run. Until then, this stays a demo.
