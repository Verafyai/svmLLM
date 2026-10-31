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

  root.SVM = { assemble, disasm, run, dec, u64s, fromHex, judge, signals, J, OPC, MM_INPUT, MM_STACK_TOP };
  if (typeof module !== 'undefined') module.exports = root.SVM;
})(typeof window !== 'undefined' ? window : globalThis);
