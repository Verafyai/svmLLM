// node tests/js_parity.mjs <site>/js/core.js : browser interpreter, assembler and judge vs Python
import { readFileSync } from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const SVM = require(process.argv[2] || new URL('../site/js/core.js', import.meta.url).pathname);
const fx = JSON.parse(readFileSync(new URL('./parity_fixture.json', import.meta.url)));
const hex = (b) => Buffer.from(b).toString('hex');
let ok = 0, bad = [];
for (const c of fx.cases) {
  const r = SVM.run(SVM.fromHex(c.prog), SVM.fromHex(c.mem), c.lim);
  const same = r.status === c.status && r.cu === c.cu && (c.status !== 'ok' || (r.r0.toString() === c.r0 && hex(r.mem) === c.out));
  same ? ok++ : bad.push(c.name);
}
let asmOk = fx.asm.filter((a) => hex(SVM.assemble(a.asm)) === a.hex).length;
let jOk = 0, pOk = 0, sOk = 0;
function firstDiff(a, b) {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return -1;
}
for (const e of fx.judge) {
  const v = SVM.judge(e.vm, e.tx, e.history);
  if (v.decision === e.verdict.decision && v.reason === e.verdict.reason) jOk++; else console.log('judge diff', e.id, v.reason, '|', e.verdict.reason);
  const prompt = SVM.buildPrompt(e.vm, e.tx, e.history, v, { programId: e.program, programTitle: e.program_title, kind: e.kind, wallet: e.wallet });
  if (prompt === e.prompt) pOk++;
  else {
    const i = firstDiff(prompt, e.prompt || '');
    console.log('prompt diff', e.id, 'at', i, JSON.stringify(prompt.slice(Math.max(0, i - 40), i + 40)), '|', JSON.stringify((e.prompt || '').slice(Math.max(0, i - 40), i + 40)));
  }
  const sim = SVM.simulatedComplete(prompt);
  if (sim === e.simulated.output) sOk++; else console.log('simulated diff', e.id, sim, '|', e.simulated && e.simulated.output);
}
const tighten = SVM.mergeDecision({ decision: 'block', reason: 'rule' }, { decision: 'allow', reason: 'model' });
if (tighten.decision !== 'block' || !tighten.tightened) { console.log('merge failed', tighten); bad.push('merge'); }
console.log(JSON.stringify({ interpreter: `${ok}/${fx.cases.length}`, assembler: `${asmOk}/${fx.asm.length}`, judge: `${jOk}/${fx.judge.length}`, prompt: `${pOk}/${fx.judge.length}`, simulated: `${sOk}/${fx.judge.length}`, bad: bad.slice(0, 5) }));
process.exit(bad.length || asmOk !== fx.asm.length || jOk !== fx.judge.length || pOk !== fx.judge.length || sOk !== fx.judge.length ? 1 : 0);
