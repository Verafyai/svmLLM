// SVMLLM site UI. Live runs use SVM (js/core.js, reference interpreter). Weights results are
// precomputed from real runs of svllm/machine.py and loaded from js/data.js.
(function () {
  'use strict';
  const D = window.SVM_DATA, S = window.SVM;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  const h64 = (v) => '0x' + BigInt(v).toString(16);
  const fmt = (v) => { v = BigInt(v); return v < 1000000000000n ? v.toLocaleString('en-US') : h64(v); };
  // Exit codes are always decimal, with the same hex in parentheses and in the tooltip.
  function exitHTML(v) {
    const n = BigInt(v), dec = n.toLocaleString('en-US'), hx = '0x' + n.toString(16);
    return `<span class="exitcode" title="${hx}">${dec} (${hx})</span>`;
  }
  function sourceHasInsn(src) {
    for (const raw of String(src).split('\n')) {
      let code = raw.split(/#|\/\/|;/)[0].trim();
      while (code && code.split(/\s+/)[0].includes(':')) code = code.slice(code.indexOf(':') + 1).trim();
      if (code) return true;
    }
    return false;
  }
  const EMPTY_PROG = 'Add a program first. This box has no instructions to run.';
  function reveal(el) {
    if (!el || !window.matchMedia('(max-width: 860px)').matches) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function clearRan(btn) {
    if (!btn) return;
    btn.classList.remove('ran');
    const f = btn.querySelector('.ran-flag');
    if (f) f.remove();
  }
  function markRan(btn) {
    if (!btn) return;
    if (btn.parentElement) [...btn.parentElement.children].forEach((b) => { if (b !== btn) clearRan(b); });
    if (btn.classList.contains('ran')) return;
    btn.classList.add('ran');
    const s = document.createElement('span');
    s.className = 'ran-flag';
    s.textContent = 'Ran';
    btn.appendChild(s);
  }

  // ---------------------------------------------------------------- input parsing
  function parseInput(spec) {
    spec = spec.trim();
    if (spec.startsWith('votes:')) {
      const pairs = spec.slice(6).split(',').map((p) => p.split(':').map(Number));
      const b = new Uint8Array(128); new DataView(b.buffer).setBigUint64(8, BigInt(pairs.length), true);
      pairs.forEach(([id, c], i) => { b[16 + 2 * i] = id & 255; b[17 + 2 * i] = c & 255; }); return b;
    }
    const words = spec ? spec.split(',').map((w) => w.trim()).filter(Boolean).map((w) => BigInt(w.replace(/_/g, ''))) : [];
    if (words.length > 16) throw new Error('at most 16 input words (128 bytes)');
    return S.u64s(words);
  }
  const words = (mem, n) => { const dv = new DataView(mem.buffer, mem.byteOffset); return Array.from({ length: n }, (_, i) => dv.getBigUint64(i * 8, true)); };

  // map pc -> source line for listings
  function listing(asm) {
    const out = []; let pc = 0;
    for (const raw of asm.split('\n')) {
      let code = raw.split(/#|\/\/|;/)[0].trim(); let isInsn = false;
      while (code && code.split(/\s+/)[0].includes(':')) code = code.slice(code.indexOf(':') + 1).trim();
      if (code) { isInsn = true; }
      out.push({ text: raw, pc: isInsn ? pc : null });
      if (isInsn) pc += code.split(/\s+/)[0] === 'lddw' ? 2 : 1;
    }
    return out;
  }

  // ---------------------------------------------------------------- hero chips + results
  const R = D.results;
  const pass = (s) => s;
  const totalCases = ['conformance', 'demo', 'random'].reduce((a, k) => a + R[k].cases, 0);
  $('hero-chips').innerHTML = [
    `<span class="chip">weights vs reference: <b>${['conformance', 'demo', 'random'].map((k) => R[k].weights_vs_ref.split('/')[0]).reduce((a, b) => a + +b, 0)}/${totalCases}</b></span>`,
    `<span class="chip">reference vs solana-sbpf: <b>${['conformance', 'demo', 'random'].map((k) => +R[k].ref_vs_solana_sbpf.split('/')[0]).reduce((a, b) => a + b, 0)}/${['conformance', 'demo', 'random'].map((k) => +R[k].ref_vs_solana_sbpf.split('/')[1]).reduce((a, b) => a + b, 0)}</b></span>`,
    `<span class="chip"><b>${(R.weights_param_count / 1e6).toFixed(2)}M</b> constant parameters, 0 trained</span>`,
    `<span class="chip"><b>10</b> demo programs</span>`].join('');
  const rows = [['conformance', 'Conformance suite (ALU edge cases, jumps, memory sizes, errors, loops)'], ['demo', 'The 10 demos x 3 presets'], ['random', 'Random programs (ALU, jumps, loads and stores, out-of-bounds, div by zero)']];
  $('res-table').innerHTML = '<tr><th>Suite</th><th>Cases</th><th>Instructions executed</th><th>Reference vs solana-sbpf</th><th>Weights vs reference</th></tr>' +
    rows.map(([k, label]) => `<tr><td>${label}</td><td>${R[k].cases}</td><td>${R[k].total_steps.toLocaleString('en-US')}</td><td class="pass">${pass(R[k].ref_vs_solana_sbpf)}</td><td class="pass">${pass(R[k].weights_vs_ref)}</td></tr>`).join('') +
    `<tr><td>Full mode (every ALU branch evaluated) equals lazy mode</td><td>${R.full_mode_equals_lazy.split('/')[1]}</td><td></td><td></td><td class="pass">${R.full_mode_equals_lazy}</td></tr>`;
  $('res-foot').textContent = `Failures recorded: ${D.failures}. Cases named call are skipped for solana-sbpf because call is a local stub. Statuses in the random suite: ${Object.entries(R.random.statuses).map(([k, v]) => k + ' ' + v).join(', ')}.`;

  // ---------------------------------------------------------------- explainer stepper
  const exSel = $('ex-prog');
  const exOpts = [];
  D.demos.forEach((d) => d.presets.forEach((p, i) => { if (p.weights) exOpts.push({ d, p }); }));
  exOpts.forEach((o, i) => { const opt = document.createElement('option'); opt.value = i; opt.textContent = `${o.d.title}: ${o.p.label} (${o.p.weights.steps.length} steps)`; exSel.appendChild(opt); });
  exSel.value = exOpts.findIndex((o) => o.d.id === 'fibonacci');
  let ex = null, exI = 0, playTimer = null, playing = false, speedId = '1';
  // Slowest rate that still finishes in about 20s. A 600-step trace lands on 64x, about 10s.
  function defaultSpeedId(n) {
    const budget = 20000;
    for (const id of ['1', '4', '16', '64']) if (n * (1000 / Number(id)) <= budget) return id;
    return '64';
  }
  function speedMs() { const n = Number(speedId); return n > 0 ? 1000 / n : 0; }
  function applySpeed(id) {
    speedId = id;
    document.querySelectorAll('#ex-speeds [data-speed]').forEach((b) => {
      const on = b.dataset.speed === id;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }
  const NOTES = {
    alu: (n) => `ALU: the decode layer turned the opcode one-hot into the ${n} gate. The ${n} circuit is built from lookup-table FFNs over 16 one-hot limbs; the result is written back with a DeltaNet delta-rule write (beta = 1, one-hot key for the destination register).`,
    jmp: () => 'Branch: comparators (eq, unsigned and signed less-than by subtraction borrow) set the taken flag; a ReLU mux picks PC + 1 + off or PC + 1. No register write (beta = 0).',
    ldx: () => 'Load: the address adder runs, address checks match the high limbs against the input or stack window, then up to 8 one-hot byte queries read DeltaNet data memory.',
    stx: () => 'Store: same address path, then up to 8 delta-rule writes into data memory with beta = 1 only for the live bytes.',
    st: () => 'Store immediate: the sign-extended imm limbs are written through the delta rule into data memory.',
    lddw: () => 'lddw: the fetch block reads two program slots (PC and PC + 1) and concatenates their imm limbs into one 64-bit value.',
    call: () => 'call: syscall stub. The weights write zero into r0. Real syscalls are future work.',
    exit: () => 'exit: the halt flag goes to 1 and the Python driver loop stops. r0 is the exit code; CU is the instruction count.',
  };
  function kindOf(insn) { const m = insn.split(' ')[0]; if (m === 'exit') return ['exit']; if (m === 'call') return ['call']; if (m === 'lddw') return ['lddw'];
    if (/^ldx/.test(m)) return ['ldx']; if (/^stx/.test(m)) return ['stx']; if (/^st/.test(m)) return ['st']; if (/64$/.test(m)) return ['alu', m.replace('64', '')]; return ['jmp']; }
  function loadEx() {
    const o = exOpts[+exSel.value]; ex = { o, lst: listing(o.d.asm), steps: o.p.weights.steps }; exI = 0; stop();
    $('ex-asm').dataset.asmId = '';
    applySpeed(defaultSpeedId(ex.steps.length));
    drawEx(); showCode();
  }
  function renderAsm(cur, on) {
    const host = $('ex-asm');
    const id = ex.o.d.id + '\n' + ex.o.p.label;
    if (host.dataset.asmId !== id) {
      host.dataset.asmId = id;
      host.innerHTML = ex.lst.map((l, i) => `<span class="ln" data-i="${i}"><i>${l.pc ?? ''}</i>${esc(l.text)}</span>`).join('');
    }
    host.querySelectorAll('.ln.cur').forEach((el) => el.classList.remove('cur'));
    if (!on) return;
    const line = ex.lst.findIndex((l) => l.pc === cur);
    const el = line >= 0 ? host.querySelector(`[data-i="${line}"]`) : null;
    if (!el) return;
    el.classList.add('cur');
    const er = el.getBoundingClientRect(), br = host.getBoundingClientRect();
    if (er.top < br.top) host.scrollTop -= br.top - er.top;
    else if (er.bottom > br.bottom) host.scrollTop += er.bottom - br.bottom;
  }
  function renderLimbs(r0) {
    const host = $('ex-limbs');
    if (host.childElementCount !== 16) {
      host.innerHTML = Array.from({ length: 16 }, () => `<div class="limb">${Array.from({ length: 16 }, () => '<span></span>').join('')}</div>`).join('');
    }
    for (let k = 0; k < 16; k++) {
      const nib = Number((r0 >> BigInt(4 * k)) & 15n);
      const spans = host.children[k].children;
      host.children[k].title = `limb ${k} = ${nib}`;
      for (let j = 0; j < 16; j++) spans[j].classList.toggle('hot', 15 - j === nib);
    }
  }
  function drawEx(animate) {
    const st = exI > 0 ? ex.steps[exI - 1] : null;
    const cur = st ? st.pc : 0;
    renderAsm(cur, !!st);
    const prev = exI > 1 ? ex.steps[exI - 2].regs : null;
    const regs = st ? st.regs : Array(11).fill('0x0').map((v, i) => (i === 1 ? '0x400000000' : i === 10 ? '0x200001000' : v));
    $('ex-regs').innerHTML = regs.map((v, i) => `<div class="${prev && prev[i] !== v ? 'chg' : ''}"><span>r${i}</span> ${v}</div>`).join('');
    $('ex-pc').textContent = st ? st.pc : 0; $('ex-cu').textContent = st ? st.cu : 0; $('ex-n').textContent = `${exI}/${ex.steps.length}`;
    const done = exI === ex.steps.length;
    $('ex-status').innerHTML = done ? (ex.o.p.weights.status === 'ok' ? `<span class="ok">exit ${exitHTML(ex.o.p.weights.r0)}</span>` : `<span class="err">${esc(ex.o.p.weights.status)}</span>`) : exI ? 'running' : 'ready';
    renderLimbs(BigInt(regs[0]));
    if (st) { const [k, n] = kindOf(st.insn); $('ex-note').innerHTML = `<b>${esc(st.insn)}</b>: ${NOTES[k](n)}`; }
    else $('ex-note').textContent = 'Press Step or Play. Each step is one call to Machine.step(), recorded from the PyTorch run.';
    if (st) pulse(kindOf(st.insn)[0], !!animate && speedMs() >= 1000);
  }
  let pulseToken = 0;
  function pulse(kind, animate) {
    const order = { alu: [0, 1, 2, 3, 4], jmp: [0, 1, 2, 4], ldx: [0, 1, 2, 3, 4], stx: [0, 1, 2, 3, 4], st: [0, 1, 3, 4], lddw: [0, 1, 3, 4], call: [0, 1, 3, 4], exit: [0, 1, 4] }[kind] || [0];
    const els = document.querySelectorAll('#pipeline .stage');
    const token = ++pulseToken;
    if (!animate) { els.forEach((e) => e.classList.remove('on')); els[order[order.length - 1]].classList.add('on'); return; }
    let i = 0;
    const tick = () => {
      if (token !== pulseToken) return;
      els.forEach((e) => e.classList.remove('on'));
      if (i < order.length) { els[order[i]].classList.add('on'); i++; setTimeout(tick, 110); }
      else els[4].classList.add('on');
    };
    tick();
  }
  function stepEx() { if (exI < ex.steps.length) { exI++; drawEx(true); return true; } stop(); return false; }
  function stop() { playing = false; clearTimeout(playTimer); playTimer = null; $('ex-play').textContent = 'Play'; }
  function runToEnd() { stop(); if (!ex) return; exI = ex.steps.length; drawEx(false); }
  function startPlay() {
    if (exI >= ex.steps.length) exI = 0;
    if (speedId === 'end') { runToEnd(); return; }
    playing = true;
    $('ex-play').textContent = 'Pause';
    stepEx();
    if (!playing || exI >= ex.steps.length) { stop(); return; }
    let acc = 0, last = performance.now();
    const loop = () => {
      if (!playing) return;
      if (speedId === 'end') { runToEnd(); return; }
      const now = performance.now(), interval = speedMs();
      acc += now - last; last = now;
      let stepped = 0;
      while (acc >= interval && exI < ex.steps.length && stepped < 64) { acc -= interval; exI++; stepped++; }
      if (stepped) drawEx(stepped === 1);
      if (exI >= ex.steps.length) { stop(); return; }
      playTimer = setTimeout(loop, Math.min(32, Math.max(0, interval - acc)));
    };
    playTimer = setTimeout(loop, speedMs());
  }
  $('ex-step').onclick = () => { stop(); stepEx(); };
  $('ex-back').onclick = () => { stop(); if (exI > 0) { exI--; drawEx(false); } };
  $('ex-reset').onclick = () => { stop(); exI = 0; drawEx(false); };
  $('ex-play').onclick = () => { if (playing) return stop(); startPlay(); };
  $('ex-speeds').onclick = (e) => {
    const b = e.target.closest('[data-speed]');
    if (!b) return;
    applySpeed(b.dataset.speed);
    if (b.dataset.speed === 'end') runToEnd();
  };
  exSel.onchange = loadEx;

  // ---------------------------------------------------------------- code tabs
  const RUST = {
    'spend-cap': `// Equivalent Rust logic (illustrative; the demo runs the hand-written sBPF on the left)
const CAP: u64 = 1_000_000_000; // $1B
pub fn check(input: &[u8]) -> u64 {
    let amount = u64::from_le_bytes(input[0..8].try_into().unwrap());
    if amount > CAP { 1 } else { 0 } // exit code 1 = over the cap
}`,
    vault: `// Equivalent Rust logic (illustrative)
pub fn withdraw(s: &mut [u64; 4]) -> u64 {
    let (balance, today, limit, req) = (s[0], s[1], s[2], s[3]);
    if req > balance { return 1; }
    if today + req > limit { return 2; }
    s[0] = balance - req; s[1] = today + req; 0
}`,
    transfer: `// Equivalent Rust logic (illustrative)
pub fn transfer(s: &mut [u64; 3]) -> u64 {
    let (a, b, amt) = (s[0], s[1], s[2]);
    if amt > a { return 1; }
    let Some(nb) = b.checked_add(amt) else { return 2 };
    s[0] = a - amt; s[1] = nb; 0
}`,
    fibonacci: `// Equivalent Rust logic (illustrative)
pub fn fib(mut n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    while n != 0 { let t = a.wrapping_add(b); a = b; b = t; n -= 1; }
    a
}`,
  };
  const RUN = `# reference interpreter and weights machine on one demo, with the trace
python -m svllm.demo {id} 10

# in Python
from svllm.isa import assemble
from svllm import machine, ref
prog = assemble(open('prog.s').read())
mem  = (10).to_bytes(8, 'little') + bytes(120)     # input words at r1
w = machine.run(prog, mem, cu_limit=10000)          # the weights
r = ref.run(prog, mem, cu_limit=10000)              # the reference
assert (w['status'], w['r0'], w['cu']) == (r['status'], r['r0'], r['cu'])

# full suite (reference vs solana-sbpf, weights vs reference)
python run_tests.py`;
  let tab = 'asm';
  function showCode() {
    const d = ex.o.d; let t;
    if (tab === 'asm') t = d.asm;
    else if (tab === 'rust') t = RUST[d.id] || `// Rust equivalent shown for spend-cap, vault, transfer and fibonacci.\n// Pick one of those in the dropdown above.\n\n${RUST['spend-cap']}`;
    else if (tab === 'run') t = RUN.replace('{id}', d.id).replace('{preset}', '0');
    else if (tab === 'lut') t = '# svllm/machine.py: an exact lookup table as a 2-layer ReLU FFN\n' + D.code.lut + '\n\n# the ALU tables built from it\n' + D.code.tables + '\n\n# a 64-bit adder = 16 chained LUT blocks over one-hot 4-bit limbs\n' + D.code.adder;
    else if (tab === 'deltanet') t = '# svllm/machine.py: Gated DeltaNet state as exact RAM (Qwenputer trick)\n' + D.code.deltanet + '\n\n# one instruction step: register reads are DeltaNet queries\n' + D.code.regs;
    else if (tab === 'fetch') t = '# svllm/machine.py, Machine.step(): fetch and decode\n' + D.code.fetch;
    else if (tab === 'alu') t = '# svllm/machine.py, Machine.step(): every ALU op is a gated circuit\n' + D.code.alu;
    else if (tab === 'write') t = '# svllm/machine.py, Machine.step(): writes are delta-rule updates with input-dependent beta\n' + D.code.write + '\n\n# the driver loop (Python, not weights)\n' + D.code.driver;
    $('code-out').textContent = t;
  }
  document.querySelectorAll('#codetabs .tabs button').forEach((b) => (b.onclick = () => {
    document.querySelectorAll('#codetabs .tabs button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); tab = b.dataset.t; showCode();
  }));
  loadEx();

  // ---------------------------------------------------------------- demos
  let cur = null, curPreset = 0;
  const tabsEl = $('demo-tabs');
  D.demos.forEach((d, i) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = `${i + 1}. ${d.title}`; b.onclick = () => pick(i, { scroll: true }); tabsEl.appendChild(b); });
  function pick(i, opts = {}) {
    cur = D.demos[i]; [...tabsEl.children].forEach((b, j) => b.classList.toggle('on', i === j));
    $('d-title').textContent = cur.title; $('d-blurb').textContent = cur.blurb; $('d-asm').value = cur.asm;
    $('d-votes-hint').textContent = cur.id === 'voting' ? 'or votes:id:choice,id:choice' : '';
    $('d-presets').innerHTML = ''; cur.presets.forEach((p, j) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = p.label; b.onclick = () => setPreset(j, { scroll: true }); $('d-presets').appendChild(b); });
    setPreset(0, opts);
  }
  function setPreset(j, opts = {}) {
    curPreset = j; [...$('d-presets').children].forEach((b, k) => b.classList.toggle('on', k === j)); $('d-words').value = cur.presets[j].spec;
    clearRan($('d-run'));
    runDemo();
    markRan($('d-presets').children[j]);
    if (opts.scroll) reveal($('d-out'));
  }
  function resultBox(title, tag, r, extra = '') {
    const ok = r.status === 'ok';
    return `<div class="res-box"><div class="k"><span class="tag ${tag}">${title}</span>${extra}</div>
      <div class="big ${ok ? 'ok' : 'err'}">${ok ? 'exit ' + exitHTML(r.r0) : esc(r.status)}</div><div class="muted small">compute units: <b>${r.cu}</b>${ok ? ' &middot; r0 = ' + h64(r.r0) : ''}</div></div>`;
  }
  function traceTable(prog, tr) {
    let prev = null; const rows = tr.map((s, i) => {
      const chg = prev ? s.regs.map((v, k) => (v !== prev[k] ? `r${k}=${fmt(v)}` : null)).filter(Boolean).join(' ') : '';
      prev = s.regs; return `<tr class="${s.err ? 'e' : ''}"><td>${i + 1}</td><td>${s.pc}</td><td>${esc(s.pc < prog.length / 8 ? S.disasm(prog, s.pc) : '(past end)')}</td><td class="cu">${s.cu}</td><td>${s.err ? esc(s.err) : chg}</td></tr>`;
    });
    return `<table><tr><th>#</th><th>pc</th><th>instruction</th><th>CU</th><th>register changes / error</th></tr>${rows.join('')}</table>`;
  }
  function runDemo() {
    const out = $('d-out');
    try {
      if (!sourceHasInsn($('d-asm').value)) throw new Error(EMPTY_PROG);
      const prog = S.assemble($('d-asm').value), mem = parseInput($('d-words').value), lim = Math.min(100000, Math.max(1, +$('d-lim').value || 10000));
      if (prog.length / 8 > 256) throw new Error('program is longer than 256 instructions');
      const tr = [], r = S.run(prog, mem, lim, tr);
      const p = cur.presets[curPreset], w = p.weights;
      const sameInput = hex(prog) === hex(S.assemble(cur.asm)) && hex(mem) === p.mem && lim === 10000;
      let wb;
      if (sameInput && w) {
        const m = w.status === r.status && w.cu === r.cu && BigInt(w.r0) === r.r0;
        wb = resultBox('Precomputed: weights machine', 'weights', { status: w.status, r0: BigInt(w.r0), cu: w.cu }, `<span class="match ${m ? 'yes' : 'no'}">${m ? 'matches live reference' : 'MISMATCH'}</span>`);
      } else wb = `<div class="res-box muted small"><span class="tag weights">Precomputed: weights machine</span><br>You changed the program, input or CU limit, so there is no recorded weights run for this exact case. Run it offline with <code>python -m svllm.demo</code> or <code>machine.run()</code>.</div>`;
      const outW = words(r.mem, 6).map((v, i) => `+${i * 8}: ${fmt(v)}`).join(' &middot; ');
      out.innerHTML = resultBox('Live: reference interpreter (JS)', 'ref', r) + wb + `<div class="muted small">input window after the run: ${outW}</div>`;
      $('d-trace').innerHTML = traceTable(prog, tr);
      $('d-trace-meta').innerHTML = `${tr.length} steps from the live reference run${sameInput && w ? `, the recorded weights trace has ${w.steps.length} steps and ${w.steps.every((s, i) => tr[i] && s.pc === tr[i].pc && s.cu === tr[i].cu) ? 'identical pc and CU at every step' : 'differences'}` : ''}`;
    } catch (e) { out.innerHTML = `<div class="res-box err">${esc(e.message)}</div>`; $('d-trace').innerHTML = ''; }
  }
  $('d-run').onclick = () => { runDemo(); [...$('d-presets').children].forEach(clearRan); markRan($('d-run')); reveal($('d-out')); }; pick(0);

  // ---------------------------------------------------------------- paste box
  const MAX_SRC = 8192, MAX_INSN = 256;
  function pasteSize() { const n = $('p-asm').value.length; $('p-size').textContent = `${n} / ${MAX_SRC} characters, max ${MAX_INSN} instructions`; }
  $('p-asm').oninput = pasteSize; pasteSize();
  $('p-run').onclick = () => {
    try {
      const src = $('p-asm').value; if (src.length > MAX_SRC) throw new Error('source is over 8 KB');
      if (!sourceHasInsn(src)) throw new Error(EMPTY_PROG);
      const prog = S.assemble(src); if (prog.length / 8 > MAX_INSN) throw new Error(`program has ${prog.length / 8} slots; the limit is ${MAX_INSN}`);
      const lim = Math.min(100000, Math.max(1, +$('p-lim').value || 1000)), tr = [], r = S.run(prog, parseInput($('p-words').value), lim, tr);
      $('p-out').innerHTML = resultBox('Live: reference interpreter (JS)', 'ref', r);
      $('p-trace').innerHTML = traceTable(prog, tr.slice(0, 500)) + (tr.length > 500 ? `<p class="muted">showing 500 of ${tr.length} steps</p>` : '');
    } catch (e) { $('p-out').innerHTML = `<div class="res-box err">${esc(e.message)}</div>`; $('p-trace').innerHTML = ''; }
    markRan($('p-run'));
    reveal($('p-out'));
  };

  // ---------------------------------------------------------------- judgement filter
  const AMT_IDX = { vault: 3, transfer: 2, 'spend-cap': 0 };
  const demoById = Object.fromEntries(D.demos.map((d) => [d.id, d]));
  const vBadge = (v) => `<i class="v ${v.decision}">${v.decision}</i>`;
  D.judge.forEach((e, i) => {
    const card = document.createElement('div'); card.className = 'card jcard';
    const s = e.verdict.signals;
    card.innerHTML = `<h4>${i + 1}. ${esc(e.title)}</h4><p>${esc(e.story)}</p>
      <div class="kv"><span>program</span><b>${esc(demoById[e.program].title)}</b><span>amount</span><b>${fmt(e.tx.amount)}</b><span>destination</span><b>${esc(e.tx.destination)}</b>
      <span>history</span><b>${e.history.length} past tx, median ${fmt(s.median_past_amount)}</b></div>
      <div class="res-box"><div class="k"><span class="tag weights">VM: weights machine (precomputed)</span></div><div class="${e.vm.status === 'ok' ? 'ok' : 'err'}"><b>${e.vm.status === 'ok' ? 'exit ' + exitHTML(e.vm.r0) : esc(e.vm.status)}</b> &middot; ${e.vm.cu} CU ${e.vm.r0 === 0 && e.vm.status === 'ok' ? '&middot; valid by code' : '&middot; rejected by code'}</div></div>
      <div class="kv"><span>ratio to median</span><b>${s.ratio_to_median >= 1000 ? "over 1,000x" : s.ratio_to_median + "x"}</b><span>seen before</span><b>${s.destination_seen_before ? 'yes' : 'no'}</b><span>tx in 15 min</span><b>${s.tx_in_window}</b><span>15 min outflow</span><b>${Math.round(s.window_outflow_fraction * 100)}% of balance</b></div>
      <div class="verdict">${vBadge(e.verdict)}<div>${esc(e.verdict.reason)}<div class="muted small">${esc(e.verdict.source)}</div></div></div>
      <div class="try"><input type="number" min="0" value="${e.tx.amount}"><button type="button">Re-run live</button></div><div class="live" aria-live="polite"></div>`;
    const inp = card.querySelector('input'), live = card.querySelector('.live'), btn = card.querySelector('button');
    btn.onclick = async () => {
      if (btn.disabled) return;
      btn.disabled = true; btn.classList.add('busy'); btn.textContent = 'Running...';
      const started = performance.now();
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const stamp = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit' });
      try {
        const t0 = performance.now();
        const amt = BigInt(inp.value || '0'), w = e.words.map(BigInt); w[AMT_IDX[e.program]] = amt;
        const r = S.run(S.assemble(demoById[e.program].asm), S.u64s(w), 10000);
        const v = S.judge({ status: r.status, r0: r.r0 }, { ...e.tx, amount: Number(amt) }, e.history);
        const elapsed = performance.now() - t0;
        const ms = elapsed < 10 ? elapsed.toFixed(1) : String(Math.round(elapsed));
        const exitBit = r.status === 'ok' ? 'exit ' + exitHTML(r.r0) : esc(r.status);
        live.innerHTML = `<div class="ran-now"><i>Ran</i> Ran just now, ${esc(stamp)}, ${ms} ms</div><div class="verdict flash">${vBadge(v)}<div><span class="tag ref">live reference VM</span> ${exitBit}, ${r.cu} CU<br>${esc(v.reason)}<div class="muted small">${esc(v.source)}, in your browser</div></div></div>`;
      } catch (err) {
        const elapsed = performance.now() - started;
        const ms = elapsed < 10 ? elapsed.toFixed(1) : String(Math.round(elapsed));
        live.innerHTML = `<div class="ran-now"><i>Ran</i> Ran just now, ${esc(stamp)}, ${ms} ms</div><div class="res-box err">${esc(err.message)}</div>`;
      }
      reveal(live);
      const remain = 280 - (performance.now() - started);
      if (remain > 0) await new Promise((resolve) => setTimeout(resolve, remain));
      btn.textContent = 'Re-run live'; btn.disabled = false; btn.classList.remove('busy');
    };
    $('jgrid').appendChild(card);
  });
})();
