// SVMLLM site core: reduced sBPF assembler + reference interpreter (mirrors svllm/ref.py)
// and the rule-based judgement filter (mirrors svllm/judge.py). Runs live in your browser.
// This is the REFERENCE path, not the weights. Weights results are precomputed (see data.js).
(function (root) {
  'use strict';
  const M64 = (1n << 64n) - 1n;
  const MM_INPUT = 0x400000000n, MM_STACK_TOP = 0x200001000n, WIN = 128;
  const ALU = { add: 0x00, sub: 0x10, mul: 0x20, div: 0x30, or: 0x40, and: 0x50, lsh: 0x60, rsh: 0x70, neg: 0x80, mod: 0x90, xor: 0xa0, mov: 0xb0, arsh: 0xc0 };
  const JMP = { ja: 0x00, jeq: 0x10, jgt: 0x20, jge: 0x30, jset: 0x40, jne: 0x50, jsgt: 0x60, jsge: 0x70, jlt: 0xa0, jle: 0xb0, jslt: 0xc0, jsle: 0xd0 };
  const SIZES = { b: 0x10, h: 0x08, w: 0x00, dw: 0x18 }, SB = { b: 1, h: 2, w: 4, dw: 8 };
  const OPC = {};
  for (const [n, h] of Object.entries(ALU)) { OPC[0x07 | h] = ['alu', n, false]; if (n !== 'neg') OPC[0x0f | h] = ['alu', n, true]; }
  for (const [n, h] of Object.entries(JMP)) { if (n === 'ja') OPC[0x05] = ['jmp', n, false]; else { OPC[0x05 | h] = ['jmp', n, false]; OPC[0x0d | h] = ['jmp', n, true]; } }
  for (const [s, m] of Object.entries(SIZES)) { OPC[0x61 | m] = ['ldx', s, true]; OPC[0x63 | m] = ['stx', s, true]; OPC[0x62 | m] = ['st', s, false]; }
  OPC[0x85] = ['call', null, false]; OPC[0x95] = ['exit', null, false]; OPC[0x18] = ['lddw', null, false];

  function enc(op, dst = 0, src = 0, off = 0, imm = 0n) {
    const b = new Uint8Array(8), dv = new DataView(b.buffer);
    dv.setUint8(0, op); dv.setUint8(1, (src << 4) | dst); dv.setInt16(2, off, true);
    dv.setInt32(4, Number(BigInt.asIntN(32, BigInt(imm))), true); return b;
  }
  function dec(p, i) {
    const dv = new DataView(p.buffer, p.byteOffset + i * 8, 8);
    return { op: dv.getUint8(0), d: dv.getUint8(1) & 15, s: dv.getUint8(1) >> 4, off: dv.getInt16(2, true), imm: dv.getInt32(4, true) };
  }
  const int = (s) => { s = s.trim(); const neg = s.startsWith('-'); const v = BigInt(neg ? s.slice(1) : s.replace(/^\+/, '')); return neg ? -v : v; };
  function reg(s) { const m = /^r(\d+)$/.exec(s.trim()); if (!m || +m[1] > 10) throw new Error(`bad register ${s}`); return +m[1]; }

  function assemble(text) {
    const lines = [];
    for (const raw of text.split('\n')) {
      let line = raw.split(/#|\/\/|;/)[0].trim();
      while (line && line.split(/\s+/)[0].includes(':')) {
        const k = line.indexOf(':'); lines.push(['label', line.slice(0, k).trim()]); line = line.slice(k + 1).trim();
      }
      if (line) lines.push(['insn', line]);
    }
    const labels = {}; let pc = 0;
    for (const [k, v] of lines) { if (k === 'label') labels[v] = pc; else pc += v.split(/\s+/)[0] === 'lddw' ? 2 : 1; }
    const out = []; pc = 0;
    for (const [k, v] of lines) {
      if (k === 'label') continue;
      const sp = v.search(/\s/); const mn = sp < 0 ? v : v.slice(0, sp); const rest = sp < 0 ? '' : v.slice(sp + 1);
      const args = rest.trim() ? rest.split(',').map((a) => a.trim()) : [];
      const target = (a) => (a in labels ? labels[a] - pc - 1 : Number(int(a)));
      const m = /^([a-z]+)(64|32)?$/.exec(mn); const base = m ? m[1] : mn;
      const mem = /^(ldx|stx|st)(b|h|w|dw)$/.exec(mn);
      const memref = (a) => { const r = /^\[\s*(r\d+)\s*([+-]\s*\w+)?\s*\]$/.exec(a); if (!r) throw new Error(`bad memory operand ${a}`); return [reg(r[1]), r[2] ? Number(int(r[2].replace(/\s/g, ''))) : 0]; };
      if (mn === 'exit') out.push(enc(0x95));
      else if (mn === 'call') out.push(enc(0x85, 0, 0, 0, int(args[0])));
      else if (mn === 'lddw') { const v64 = BigInt.asUintN(64, int(args[1])); out.push(enc(0x18, reg(args[0]), 0, 0, v64 & 0xffffffffn), enc(0, 0, 0, 0, v64 >> 32n)); pc++; }
      else if (base in ALU && m[2] === '64') {
        const h = ALU[base];
        if (base === 'neg') out.push(enc(0x87, reg(args[0])));
        else if (args[1].startsWith('r')) out.push(enc(0x0f | h, reg(args[0]), reg(args[1])));
        else out.push(enc(0x07 | h, reg(args[0]), 0, 0, int(args[1])));
      } else if (base in JMP) {
        const h = JMP[base];
        if (base === 'ja') out.push(enc(0x05, 0, 0, target(args[0])));
        else if (args[1].startsWith('r')) out.push(enc(0x0d | h, reg(args[0]), reg(args[1]), target(args[2])));
        else out.push(enc(0x05 | h, reg(args[0]), 0, target(args[2]), int(args[1])));
      } else if (mem) {
        const [, kd, s] = mem;
        if (kd === 'ldx') { const [r, o] = memref(args[1]); out.push(enc(0x61 | SIZES[s], reg(args[0]), r, o)); }
        else if (kd === 'stx') { const [r, o] = memref(args[0]); out.push(enc(0x63 | SIZES[s], r, reg(args[1]), o)); }
        else { const [r, o] = memref(args[0]); out.push(enc(0x62 | SIZES[s], r, 0, o, int(args[1]))); }
      } else throw new Error(`unsupported instruction: ${v}`);
      pc++;
    }
    const p = new Uint8Array(out.length * 8); out.forEach((b, i) => p.set(b, i * 8)); return p;
  }

  function disasm(p, i) {
    const { op, d, s, off, imm } = dec(p, i); const o = OPC[op];
    if (!o) return `.invalid 0x${op.toString(16).padStart(2, '0')}`;
    const [kind, n, isreg] = o; const t = (off >= 0 ? '+' : '') + off;
    if (kind === 'alu') return n === 'neg' ? `neg64 r${d}` : `${n}64 r${d}, ${isreg ? 'r' + s : imm}`;
    if (kind === 'jmp') return n === 'ja' ? `ja ${t}` : `${n} r${d}, ${isreg ? 'r' + s : imm}, ${t}`;
    if (kind === 'ldx') return `ldx${n} r${d}, [r${s}${t}]`;
    if (kind === 'stx') return `stx${n} [r${d}${t}], r${s}`;
    if (kind === 'st') return `st${n} [r${d}${t}], ${imm}`;
    if (kind === 'lddw') { const hi = p.length >= (i + 2) * 8 ? dec(p, i + 1).imm : 0; return `lddw r${d}, 0x${((BigInt.asUintN(32, BigInt(hi)) << 32n) | BigInt.asUintN(32, BigInt(imm))).toString(16)}`; }
    return kind === 'call' ? `call ${imm}` : 'exit';
  }

  const s64 = (v) => BigInt.asIntN(64, v);
  class VMError extends Error { constructor(k) { super(k); this.kind = k; } }
  function addr(a, size) {
    if (MM_INPUT <= a && a + BigInt(size) <= MM_INPUT + BigInt(WIN)) return Number(a - MM_INPUT);
    const lo = MM_STACK_TOP - BigInt(WIN);
    if (lo <= a && a + BigInt(size) <= MM_STACK_TOP) return WIN + Number(a - lo);
    throw new VMError('AccessViolation');
  }

  function run(prog, memIn, cuLimit = 10000, trace = null) {
    const n = prog.length / 8, mem = new Uint8Array(2 * WIN); if (memIn) mem.set(memIn.slice(0, 2 * WIN));
    const r = new Array(11).fill(0n); r[1] = MM_INPUT; r[10] = MM_STACK_TOP;
    let pc = 0, cu = 0, status = null;
    try {
      for (;;) {
        if (cu >= cuLimit) throw new VMError('ExceededMaxInstructions');
        cu += 1;
        if (pc >= n) throw new VMError('ExecutionOverrun');
        const { op, d, s, off, imm } = dec(prog, pc); const o = OPC[op];
        if (!o) throw new VMError('InvalidInstruction');
        const [kind, name, isreg] = o; let nxt = pc + 1; const at = pc;
        if (kind === 'alu') {
          const a = r[d], b = isreg ? r[s] : BigInt.asUintN(64, BigInt(imm)); let v;
          switch (name) {
            case 'add': v = (a + b) & M64; break; case 'sub': v = (a - b) & M64; break; case 'mul': v = (a * b) & M64; break;
            case 'div': case 'mod': if (b === 0n) throw new VMError('DivideByZero'); v = name === 'div' ? a / b : a % b; break;
            case 'or': v = a | b; break; case 'and': v = a & b; break; case 'xor': v = a ^ b; break;
            case 'lsh': v = (a << (b & 63n)) & M64; break; case 'rsh': v = a >> (b & 63n); break;
            case 'arsh': v = BigInt.asUintN(64, s64(a) >> (b & 63n)); break; case 'neg': v = (-a) & M64; break; case 'mov': v = b; break;
          }
          r[d] = v;
        } else if (kind === 'jmp') {
          const a = r[d], b = isreg ? r[s] : BigInt.asUintN(64, BigInt(imm));
          const t = { ja: true, jeq: a === b, jne: a !== b, jgt: a > b, jge: a >= b, jlt: a < b, jle: a <= b, jset: (a & b) !== 0n,
            jsgt: s64(a) > s64(b), jsge: s64(a) >= s64(b), jslt: s64(a) < s64(b), jsle: s64(a) <= s64(b) }[name];
          if (t) nxt = (pc + 1 + off) & 0xffff;
        } else if (kind === 'ldx' || kind === 'stx' || kind === 'st') {
          const size = SB[name], base = kind === 'ldx' ? r[s] : r[d];
          const i = addr((base + BigInt(off)) & M64, size);
          if (kind === 'ldx') { let v = 0n; for (let k = size - 1; k >= 0; k--) v = (v << 8n) | BigInt(mem[i + k]); r[d] = v; }
          else { let v = kind === 'stx' ? r[s] : BigInt.asUintN(64, BigInt(imm)); for (let k = 0; k < size; k++) { mem[i + k] = Number(v & 0xffn); v >>= 8n; } }
        } else if (kind === 'lddw') {
          const hi = pc + 1 < n ? dec(prog, pc + 1).imm : 0;
          r[d] = (BigInt.asUintN(32, BigInt(hi)) << 32n) | BigInt.asUintN(32, BigInt(imm)); nxt = pc + 2;
        } else if (kind === 'call') r[0] = 0n;
        else if (kind === 'exit') { status = 'ok'; if (trace) trace.push({ pc: at, cu, regs: r.slice() }); break; }
        pc = nxt;
        if (trace) trace.push({ pc: at, cu, regs: r.slice() });
      }
    } catch (e) { if (!(e instanceof VMError)) throw e; status = e.kind; if (trace) trace.push({ pc, cu, regs: r.slice(), err: e.kind }); }
    return { status, r0: r[0], cu, regs: r, mem };
  }

  function u64s(words) { const b = new Uint8Array(128); const dv = new DataView(b.buffer); words.forEach((w, i) => dv.setBigUint64(i * 8, BigInt.asUintN(64, BigInt(w)), true)); return b; }
  function fromHex(h) { const b = new Uint8Array(h.length / 2); for (let i = 0; i < b.length; i++) b[i] = parseInt(h.substr(i * 2, 2), 16); return b; }

  // ---------------------------------------------------------------- judgement filter (rules, NOT in the weights)
  const J = { HOLD_RATIO: 10, BLOCK_RATIO: 50, NEW_DEST_RATIO: 3, DRAIN_WINDOW: 900, DRAIN_COUNT: 3, DRAIN_FRACTION: 0.5 };
  function median(xs) { const s = xs.slice().sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
  function signals(tx, hist, known = []) {
    const past = hist.length ? hist.map((h) => h.amount) : [tx.amount];
    const med = Math.max(1, Math.floor(median(past)));
    const seen = known.includes(tx.destination) || hist.some((h) => h.destination === tx.destination);
    const win = hist.filter((h) => tx.time - h.time <= J.DRAIN_WINDOW);
    const outflow = win.reduce((a, h) => a + h.amount, 0) + tx.amount;
    const start = (win.length ? win[0].balance_before : tx.balance_before) || 1;
    return { median_past_amount: med, ratio_to_median: Math.round((tx.amount / med) * 100) / 100, destination_seen_before: seen,
      tx_in_window: win.length + 1, window_outflow: outflow, window_outflow_fraction: Math.round((outflow / start) * 1000) / 1000 };
  }
  const pct = (x) => Math.round(x * 100) + '%';
  // note: Python formats {:.0f} with round-half-even; amounts in the examples avoid .5 ties
  function judge(vm, tx, hist, known) {
    const s = signals(tx, hist, known), src = 'rules (heuristics, not in the weights)';
    const V = (decision, reason) => ({ decision, reason, signals: s, source: src });
    if (vm.status !== 'ok' || BigInt(vm.r0) !== 0n) return V('block', `The program itself rejected this (${vm.status !== 'ok' ? vm.status : 'exit code ' + vm.r0}). The judge never overrides a code-level rejection.`);
    if (s.tx_in_window > J.DRAIN_COUNT && s.window_outflow_fraction >= J.DRAIN_FRACTION) return V('block', `Drain pattern: ${s.tx_in_window} withdrawals in ${J.DRAIN_WINDOW / 60} minutes moving ${pct(s.window_outflow_fraction)} of the balance. Each one passed the code checks.`);
    if (s.ratio_to_median >= J.BLOCK_RATIO && !s.destination_seen_before) return V('block', `${Math.round(s.ratio_to_median)}x the usual amount to a destination never seen before. Valid by code, but a sensible operator would stop it.`);
    if (s.ratio_to_median >= J.HOLD_RATIO) return V('hold', `${Math.round(s.ratio_to_median)}x the usual amount. Hold for a human to confirm.`);
    if (!s.destination_seen_before && s.ratio_to_median >= J.NEW_DEST_RATIO) return V('hold', `New destination and ${s.ratio_to_median.toFixed(1)}x the usual amount. Hold for confirmation.`);
    return V('allow', 'Valid by code and consistent with history.');
  }

  // ---------------------------------------------------------------- LLM judge prompt + simulated backend (NOT in the weights)
  // Kept in lockstep with svllm/judge.py. tests/js_parity.mjs checks the prompt text and the canned JSON.
  const SIMULATED_LABEL = 'SIMULATED model output, no real LLM was called';
  const WALLET_PLACEHOLDER = 'Wallet X';
  const SYSTEM_ROLE = 'You are the transaction reviewer for a wallet. The sBPF virtual machine has already run the program exactly. Decide allow, hold or block based on everything you know about the sender and the receiver and the run rules. You may only make the decision stricter, never looser than the rule verdict, and you never override a code-level rejection.';
  const RULES_TEXT = 'The rule review is a heuristic outside the weights. You may only make the decision stricter, never looser than the rule verdict, and you never override a code-level rejection.\n1. Program error or nonzero exit code: block.\n2. More than 3 withdrawals in 15 minutes moving at least 50% of the balance: block.\n3. At least 50x the median past amount to a never-seen destination: block.\n4. At least 10x the median past amount: hold.\n5. New destination and at least 3x the median: hold.\n6. Otherwise: allow.';
  const CANNED = [
    { needles: ['\nAmount: 210\n', 'Receiver (destination): addr:alice-savings', 'program id: transfer', '\nexit code: 0\n', 'is this allowed?'], decision: 'allow', reason: 'The program exited 0 in 12 compute units. The amount is in line with what this wallet usually sends, and the receiver is an address it already pays. Allow it.' },
    { needles: ['\nAmount: 20500\n', 'Receiver (destination): addr:9qZ...never-seen', 'program id: vault', '\nexit code: 0\n', 'would like to withdraw 20500'], decision: 'block', reason: 'The program allowed this withdrawal (exit code 0), but 20500 is 100x the usual amount and the destination has never been seen before. Block it.' },
    { needles: ['\nAmount: 2400\n', 'Receiver (destination): addr:x7f3...unknown', 'program id: vault', '\nexit code: 0\n', 'would like to withdraw 2400'], decision: 'block', reason: 'The program allowed this withdrawal, but the history shows 4 withdrawals in 15 minutes moving 96% of the balance. Block it.' },
    { needles: ['\nAmount: 1200000000\n', 'program id: spend-cap', '\nexit code: 1\n'], decision: 'block', reason: 'The program itself exited with code 1. A code-level rejection is never overridden. Block it.' },
    { needles: ['\nAmount: 1000\n', 'Receiver (destination): addr:new-merchant', 'program id: transfer', '\nexit code: 0\n', 'is this allowed?'], decision: 'hold', reason: 'The program allowed the call, but the amount is about 4.9x the usual size and the destination has never been seen before. Hold it for a person to confirm.' },
  ];

  function numText(x) {
    if (typeof x === 'boolean') return x ? 'yes' : 'no';
    if (typeof x === 'number' && Number.isInteger(x)) return String(x);
    let s = JSON.stringify(x);
    if (s.endsWith('.0')) s = s.slice(0, -2);
    return s;
  }
  function decString(v) { return BigInt(v).toString(10); }
  function returnDataHex(r0) {
    let n = BigInt(r0) & ((1n << 64n) - 1n), s = '';
    for (let i = 0; i < 8; i++) { s += Number(n & 255n).toString(16).padStart(2, '0'); n >>= 8n; }
    return s;
  }
  function dumpsReply(decision, reason) { return JSON.stringify({ decision, reason }); }
  function lineValue(prompt, label) {
    const prefix = label + ': ';
    for (const line of prompt.split('\n')) if (line.startsWith(prefix)) return line.slice(prefix.length);
    return '';
  }
  // meta: { programId, programTitle, kind, wallet, known }
  function buildPrompt(vm, tx, hist, base, meta) {
    const kind = meta.kind;
    if (kind !== 'call' && kind !== 'withdraw') throw new Error('kind must be call or withdraw');
    const wallet = meta.wallet || WALLET_PLACEHOLDER;
    const known = (meta.known || []).slice().sort();
    const amount = decString(tx.amount);
    const dest = tx.destination;
    const rows = (hist || []).slice(-20);
    const senderLines = rows.length ? rows.map((h) => `- t=${decString(h.time)} amount=${decString(h.amount)} to=${h.destination} balance_before=${decString(h.balance_before)}`) : ['- none'];
    const recv = rows.filter((h) => h.destination === dest);
    const recvLines = recv.length ? recv.map((h) => `- t=${decString(h.time)} amount=${decString(h.amount)} balance_before=${decString(h.balance_before)}`) : ['- none'];
    const seenTxt = base.signals.destination_seen_before ? 'yes' : 'no';
    const call = `${wallet} has requested calling this Solana program "${meta.programTitle}" (program id: ${meta.programId}).`;
    const ask = kind === 'withdraw'
      ? `${wallet} would like to withdraw ${amount}. Based on everything you know about the context and the run rules, should that be allowed?`
      : `${wallet} has requested calling this smart contract. Based on everything you know about the context and the run rules, is this allowed?`;
    const lines = [
      'SYSTEM', SYSTEM_ROLE, '',
      'TRANSACTION REQUEST', call, ask, '',
      `Sender (wallet): ${wallet}`,
      `Receiver (destination): ${dest}`,
      `Amount: ${amount}`,
      `Time (seconds): ${decString(tx.time)}`,
      `Balance before this request: ${decString(tx.balance_before)}`,
      '',
      'VM RESULT',
      'The sBPF virtual machine has already run the program exactly. These numbers are that result.',
      `status: ${vm.status}`,
      `exit code: ${decString(vm.r0)}`,
      `compute units: ${decString(vm.cu)}`,
      `return data: ${returnDataHex(vm.r0)}`,
      '',
      'SENDER AND RECEIVER HISTORY',
      'Past requests from this sender, oldest first:',
      ...senderLines, '',
      `Past payments to this receiver (${dest}):`,
      ...recvLines,
      `Destination seen before: ${seenTxt}`,
      `Known destinations on file: ${known.length ? known.join(', ') : 'none'}`,
      '',
      'RUN RULES', RULES_TEXT,
      `Rule verdict: ${base.decision}`,
      `Rule reason: ${base.reason}`,
      `median past amount: ${numText(base.signals.median_past_amount)}`,
      `ratio to median: ${numText(base.signals.ratio_to_median)}`,
      `destination seen before: ${seenTxt}`,
      `transactions in 15 minutes: ${numText(base.signals.tx_in_window)}`,
      `window outflow: ${numText(base.signals.window_outflow)}`,
      `window outflow fraction: ${numText(base.signals.window_outflow_fraction)}`,
      '',
      'REQUIRED REPLY',
      'Reply with one JSON object and no other text. The keys are decision and reason.',
      '{"decision": "allow" or "hold" or "block", "reason": "short reason"}',
    ];
    return lines.join('\n') + '\n';
  }
  function fallbackReply(prompt) {
    const decision = lineValue(prompt, 'Rule verdict');
    const reason = lineValue(prompt, 'Rule reason');
    const ratio = lineValue(prompt, 'ratio to median');
    const exitCode = lineValue(prompt, 'exit code');
    const seen = lineValue(prompt, 'destination seen before');
    if (decision !== 'allow' && decision !== 'hold' && decision !== 'block') return { decision: 'hold', reason: 'The simulated reviewer could not read a rule verdict, so it holds the request.' };
    let text;
    if (reason.includes('code-level rejection') || (exitCode !== '' && exitCode !== '0' && decision === 'block')) text = `The program itself rejected this (exit code ${exitCode}). A code-level rejection is never overridden.`;
    else if (decision === 'block' && reason.startsWith('Drain pattern')) text = `The history shows a drain. ${reason}`;
    else if (decision === 'block') {
      const where = seen === 'no' ? 'a destination that has never been seen' : 'a destination this wallet has used';
      text = `The program allowed the request, but the amount is ${ratio}x the usual size and it goes to ${where}. Block it.`;
    } else if (decision === 'hold') text = `The amount is ${ratio}x the usual size (destination seen before: ${seen}). Hold it for a person to confirm.`;
    else text = 'The program exited 0 and the amount fits this sender and receiver history. Allow it.';
    return { decision, reason: text };
  }
  function simulatedComplete(prompt) {
    for (const spec of CANNED) if (spec.needles.every((n) => prompt.includes(n))) return dumpsReply(spec.decision, spec.reason);
    const fb = fallbackReply(prompt);
    return dumpsReply(fb.decision, fb.reason);
  }
  const RANK = { allow: 0, hold: 1, block: 2 };
  function mergeDecision(base, model) {
    const md = model && model.decision;
    if (!(md in RANK)) return { decision: base.decision, reason: base.reason, tightened: false };
    if (RANK[md] >= RANK[base.decision]) return { decision: md, reason: model.reason, tightened: false };
    return { decision: base.decision, reason: model.reason + ' Kept the stricter rule verdict (' + base.decision + ').', tightened: true };
  }

  root.SVM = { assemble, disasm, run, dec, u64s, fromHex, judge, signals, buildPrompt, simulatedComplete, mergeDecision, SIMULATED_LABEL, J, OPC, MM_INPUT, MM_STACK_TOP };
  if (typeof module !== 'undefined') module.exports = root.SVM;
})(typeof window !== 'undefined' ? window : globalThis);
