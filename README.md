<p align="center"><img src="assets/svmllm-logo.png" alt="SVMLLM" width="360"></p>

<h1 align="center">SVMLLM</h1>
<p align="center"><b>A Solana VM compiled into transformer weights.</b><br>
Reduced sBPF programs run on hand-built attention, FFN and DeltaNet blocks: exact results, exact compute units, zero training.</p>
<p align="center"><a href="https://svmllm.rexstjohn.com">Live demo: svmllm.rexstjohn.com</a> | <a href="https://github.com/Verafyai/svmLLM">github.com/Verafyai/svmLLM</a></p>

---

## The pitch

Solana programs compile to sBPF bytecode. SVMLLM executes that bytecode inside a neural network whose weights were constructed by hand rather than learned. Instruction fetch is a one-hot attention-style lookup, the ALU is a stack of exact lookup-table feed-forward blocks, and registers and memory live in Gated DeltaNet state configured as exact RAM, the trick introduced by [Qwenputer](https://divagr.com/blog/qwenputer). Every run is checked against a reference interpreter, and the reference interpreter is checked against Anza's [solana-sbpf](https://github.com/anza-xyz/sbpf).

On top of the VM sits a **judgement filter**: after the program has run, a separate review step looks at the amount, the destination and the history, and returns allow, hold or block with a reason.

## Why: code that does exactly what it says

Exploits often happen because code did exactly what it was written to do, with an outcome a sensible human operator would have stopped. That gap motivated this project: fuse the best of smart contracts (trustless, deterministic and decentralized execution) with the best of human intelligence (judgement based on context).

A transformer that can execute contract bytecode exactly is a step toward a single model that both runs the program deterministically and reasons about whether the result makes sense.

## How it works

One sBPF instruction is one call to `Machine.step()` in [`svllm/machine.py`](svllm/machine.py):

```
sBPF program ──► fetch + decode ──► ALU (FFN) ──► registers / memory (DeltaNet RAM) ──► exit code + CU
  8-byte insns     one-hot PC query    LUT blocks     delta-rule writes, beta 0/1          24-bit counter
```

**Representation.** A 64-bit value is 16 little-endian 4-bit limbs, each a one-hot vector of 16, so a register is a 256-dim activation. Every activation is exactly 0 or 1 in float32, which is why the computation is exact.

**Exact RAM (DeltaNet).** A Gated DeltaNet head with gate = 1 and beta in {0, 1} updates `S <- S + beta * k (v - S^T k)^T`. With a one-hot key `k` this overwrites exactly one row; reading is `S^T q`. Three of these hold the program (256 slots x 481 dims: present bit, opcode one-hot, dst, src, offset limbs, imm limbs), the register file (16 x 256) and data memory (256 bytes, two limbs each).

```python
class DeltaNetRAM:
    def read(self, q):
        return q @ self.S
    def write(self, k, v, beta):
        self.S = self.S + beta * torch.outer(k, v - k @ self.S)
```

**Exact ALU (FFN).** `LUT` turns any function of one-hot inputs into a 2-layer ReLU FFN with one hidden unit per input combination (weights 1, bias -(k-1)). From these: a 16-stage ripple-carry adder over (a, b, carry), nibble-times-word multipliers with shifted adds, 64-stage restoring division for div and mod, nibble permutations plus bit-shift tables for lsh, rsh and arsh, and comparators (signed ones flip the top bit).

```python
L_ADD = LUT([16, 16, 2], [16, 2], lambda a, b, c: ((a + b + c) & 15, (a + b + c) >> 4))
def add(A, B, cin):
    out, c = [], cin
    for i in range(A.shape[0]):
        s, c = L_ADD(A[i], B[i], c)
        out.append(s)
    return torch.stack(out), c
```

**Decode and control.** Decode is a single constant linear layer from the opcode one-hot to control gates. Muxes are ReLU gating (`relu(v + g - 1)`). Branches choose PC + 1 + off or PC + 1; address checks match the high limbs against the input and stack windows; the CU meter is a 6-limb counter compared to the limit. Errors (ExceededMaxInstructions, ExecutionOverrun, InvalidInstruction, DivideByZero, AccessViolation) are flags that also gate every write off.

### What is in the weights, and what is not

| In the weights (constant tensors, about 1.13M parameters, none trained) | Not in the weights |
|---|---|
| Fetch, decode, ALU, branch logic, address checks, CU meter, register and memory reads and writes | The driver loop: Python calls `step()` until the halt flag is 1 |
| | Fixed-depth circuits are Python loops over unrolled blocks rather than one fused layer stack |
| | Lazy mode skips ALU branches whose gate is exactly 0 (full mode computes all of them and is tested equal) |
| | The judgement filter (rules plus an LLM stub) |

This is a standalone PyTorch machine in the style of Qwenputer, not yet embedded in a pretrained language model. See next steps.

## What it can do

Reduced sBPF (SBPF v0 encoding): `r0` to `r10`; all 64-bit ALU ops in imm and reg forms (add, sub, mul, div, mod, or, and, xor, lsh, rsh, arsh, neg, mov); `ldx`/`stx`/`st` in b, h, w and dw; all conditional and unconditional jumps; `lddw`; `call` (stub); `exit`; 1 CU per instruction. Memory is two 128-byte windows, mirroring Solana's layout: input at `0x400000000` (pointed to by `r1`) and stack below `r10 = 0x200001000`.

## Demo programs

All ten are in [`svllm/programs.py`](svllm/programs.py), each with three presets, and run on the site with a step trace and CU counts.

1. **Spend cap**: exit 1 if the amount is over a $1B limit.
2. **Counter**: memory read, increment, write back, N times.
3. **Fibonacci**: iterative, up to fib(93).
4. **Factorial with overflow check**: exit 2 when a multiply wraps.
5. **GCD**: Euclid with mod64.
6. **Prime check**: trial division to sqrt(n).
7. **Balance transfer**: between two memory slots; exit 1 if short, 2 on overflow.
8. **Vault with withdrawal limit**: exit 1 over balance, 2 over the daily limit.
9. **Voting tally**: 64-bit bitmap rejects double votes with exit 3.
10. **Escrow**: release when delivered, refund after the deadline, otherwise exit 4.

## Judgement filter

```
sBPF program ──► VM (exact result + CU, in the weights) ──► review step (rules, NOT in the weights) ──► allow / hold / block + reason
                                                                └─ optional LLMJudge (pluggable, stub by default)
```

[`svllm/judge.py`](svllm/judge.py). The VM answers "is this valid by the code?". The review step answers "would a sensible operator let this through?". Rules, in order:

1. Program error or nonzero exit code: **block** (never overrides a code-level rejection).
2. More than 3 withdrawals in 15 minutes moving at least 50% of the balance: **block** (drain pattern).
3. At least 50x the median past amount to a never-seen destination: **block**.
4. At least 10x the median past amount: **hold**.
5. New destination and at least 3x the median: **hold**.
6. Otherwise **allow**.

`LLMJudge(complete=fn)` builds a prompt with the VM result, the request, the history and the rule verdict, and expects JSON back. It may escalate (allow to hold or block) but never downgrades a rule block. With no backend it is a stub that returns the rule verdict, labeled as such.

Runnable examples (`python -m svllm.judge`; the VM step is run on the weights machine in `run_tests.py`):

| Example | Program | VM result (weights) | Verdict | Reason |
|---|---|---|---|---|
| Valid by code, 100x usual, never-seen address | `vault` | ok, exit 0, 13 CU | **BLOCK** | 100x the usual amount to a destination never seen before. Valid by code, but a sensible operator would stop it. |
| Normal transfer to a known address | `transfer` | ok, exit 0, 12 CU | **ALLOW** | Valid by code and consistent with history. |
| Rapid repeated drain | `vault` | ok, exit 0, 13 CU | **BLOCK** | Drain pattern: 4 withdrawals in 15 minutes moving 96% of the balance. Each one passed the code checks. |
| Spend cap: over $1B | `spend-cap` | ok, exit 1, 5 CU | **BLOCK** | The program itself rejected this (exit code 1). The judge never overrides a code-level rejection. |
| Moderate amount, new address | `transfer` | ok, exit 0, 12 CU | **HOLD** | New destination and 4.9x the usual amount. Hold for confirmation. |

## Quickstart

```bash
git clone https://github.com/Verafyai/svmLLM && cd svmLLM
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy
python -m svllm.demo fibonacci 10       # weights machine vs reference, with the trace
python -m svllm.judge                   # judgement-filter examples
(cd xcheck && cargo build --release)    # optional: solana-sbpf 0.12 cross-check binary
python run_tests.py                     # all suites; writes results/
python export_site.py                  # regenerate site/js/data.js from results/
node tests/js_parity.mjs               # browser interpreter and judge vs Python
```

`N_RANDOM` and `N_FULL` environment variables size the random and full-mode suites.

## Test results

| Suite | Cases | Reference vs solana-sbpf | Weights machine vs reference | Steps executed on weights |
|---|---|---|---|---|
| Conformance (named, per opcode and error) | 135 | 134/134 | 135/135 | 1,508 |
| Demo presets (10 programs x 3) | 30 | 30/30 | 30/30 | 2,764 |
| Random programs (seed 1337) | 1000 | 1000/1000 | 1000/1000 | 44,756 |

- Full mode (no lazy branch skipping) equals lazy mode: **32/32** (first 16 conformance + first 16 random).
- Random-program outcomes: ok 551, DivideByZero 201, ExceededMaxInstructions 159, AccessViolation 89.
- Browser JS interpreter vs Python reference: **565/565** cases, assembler 10/10, judge 5/5 (`tests/js_parity.mjs`).
- Weights: 1,130,672 constant parameters, none trained.

Cross-check details: [`xcheck/`](xcheck) drives Anza's `solana-sbpf` 0.12 interpreter with `SBPFVersion::V0`, `r1` at the input region and only the top 128 bytes of the first stack frame mapped, so its memory model matches the toy. Status, r0, CU and the input window are compared (its registers after a run are not exposed reliably). The weights machine is compared against the reference on status, r0, CU, all 11 registers and both memory windows. The one conformance case not run on solana-sbpf is `misc/call-stub`, because `call` is a local stub.

## The site

A static demo site (plain HTML, CSS and JS, no build step) lives in [`site/`](site) (serve it with `python -m http.server -d site`). It has the ten demos with presets, a Run button, step traces and CU counts, a paste-your-own box (256 instructions, 8 KB), the judgement-filter examples, and an animated walkthrough that steps through real recorded traces of the weights machine. Results are labeled honestly: **live** runs use a JavaScript port of the reference interpreter in your browser; **weights** results are precomputed traces from the PyTorch runs (`results/weights_traces.json`).

## Where this could run

This is a demo that exact computation can live in a model's weights. It is not a node replacement.

**Realistic today**

- **Sidecar next to the chain software.** A node operator runs the model beside their own client and uses it to review, flag or hold transactions in their own RPC endpoint, mempool view or wallet guard. It affects only that operator's infrastructure, not consensus. This is the realistic path today.
- **Off-chain checker and audit.** Re-run a transaction on the weights machine, compare it to the reference result, and attach the review to an audit trail.
- **Judgement layer in front of a wallet or multisig.** The exact VM result plus the review step decides allow, hold or block before a signer approves.

**Research direction: part of consensus**

Making the weights machine part of consensus would need:

- **Determinism.** GPU floating point can differ across hardware and software versions. The weights here use exact lookup-table style math with values that stay exactly 0 or 1, but a full Qwen-sized model is a much bigger problem.
- **Cost and speed.** A tiny program takes tens of thousands of tokens, and every validator would have to repeat it. Each sBPF instruction is one full pass through the circuits, roughly 10 ms on CPU for simple ops and about 200 ms for div and mod.
- **Compute limits.** The model run would need to be metered and bounded by compute units like any other execution.

The judgement filter is illustrative heuristics on synthetic histories, not a security product.

## Honest limitations

- Reduced subset. No ALU32 (SBPF v0 ALU32 sign extension quirks are not modeled), no syscalls (`call` zeroes r0), no accounts, no CPI, no internal function calls, no SBPF v1+ instructions.
- Memory is two 128-byte windows and programs are at most 256 slots.
- CU model is 1 per instruction. Real Solana pricing for syscalls and CPI is not modeled.
- It is slow: CPU PyTorch, roughly 10 ms per simple instruction and about 200 ms for div and mod. A research demo, not a validator.
- The driver loop and the circuit sequencing are Python. The judgement filter is heuristics, not learned and not in the weights.
- Not yet inside a pretrained LLM.

## Future possibilities

- **Accounts**: serialize Solana account data into the input window and enforce owner, signer and writable rules in the circuits.
- **Syscalls**: `sol_log`, `sol_memcpy`, hashing, each as additional FFN circuits.
- **CPI and internal calls**: a call stack of frames in DeltaNet memory.
- **A real CU model**: per instruction class and per syscall, matching the runtime's cost tables.
- **Judgement in the same model**: pair the exact VM layers with the language layers that do the review.

## Next steps: run inside the real Qwenputer weights (needs a GPU)

1. Get a single 24 GB GPU (the Qwenputer post uses an L4). The checkpoint is 18.8 GB, and the post reports about 3,200 CPU tokens per second for its batched executor on an L4.
2. Reproduce Qwenputer first, using the commands from the post:
   ```bash
   git clone https://github.com/divagr18/qwenputer && cd qwenputer/experimental/riscv && uv sync --extra hf
   hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-rv
   uv run python -m chipformer.rv.chat qwenputer-rv --firmware tests/data/rv/calc-bridge.elf \
       --ask "Use calc: what is 123456789 * 987654321?"
   ```
3. Two routes for sBPF. **(a) Firmware route, no new weights:** write an sBPF interpreter in C, build it for RV32IM, and run it as firmware on the CPU that already sits in Qwenputer's weights. Validate with this repo's conformance and random suites. **(b) Native route:** express SVMLLM's circuits in Qwenputer's circuit compiler (attention heads for fetch, FFN units for the ALU tables, DeltaNet heads for program, registers and memory), emitting ADDR, VAL and STEP tokens per instruction, as extra layers alongside the RISC-V ones.
4. Run `run_tests.py` against the embedded version and require the same pass rates. Then wire the judgement filter to Qwen itself through `LLMJudge(complete=...)`, so one model runs the program exactly and reviews the result.

## Credits

- [Qwenputer](https://divagr.com/blog/qwenputer) by divagr1925: a RISC-V CPU compiled into Qwen3.5-4B with Gated DeltaNet layers as exact RAM. [Announcement](https://x.com/divagr1925/status/2115394340357116169), [weights on Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer). SVMLLM borrows the central idea (gate 1, beta 1, one-hot keys) and applies it to sBPF.
- Anza's [solana-sbpf](https://github.com/anza-xyz/sbpf) for the reference VM used in cross-checking.
- Sibling project: [EVMLLM](https://github.com/Verafyai/evmLLM), an Ethereum VM executed by transformer weights ([evmllm.rexstjohn.com](https://evmllm.rexstjohn.com)).

## License

MIT. See [LICENSE](LICENSE).
