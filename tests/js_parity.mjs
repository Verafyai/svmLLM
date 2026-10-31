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
let jOk = 0;
for (const e of fx.judge) {
  const v = SVM.judge(e.vm, e.tx, e.history);
  if (v.decision === e.verdict.decision && v.reason === e.verdict.reason) jOk++; else console.log('judge diff', e.id, v.reason, '|', e.verdict.reason);
}
console.log(JSON.stringify({ interpreter: `${ok}/${fx.cases.length}`, assembler: `${asmOk}/${fx.asm.length}`, judge: `${jOk}/${fx.judge.length}`, bad: bad.slice(0, 5) }));
process.exit(bad.length || asmOk !== fx.asm.length || jOk !== fx.judge.length ? 1 : 0);
