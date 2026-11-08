// The full machine: results, Real programs cards, CPI cards and real-transaction judge examples.
// Everything comes from window.SVM_FULL (export_full_site.py, from results/).
(function () {
  const F = window.SVM_FULL;
  if (!F) return;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const S = F.summary;
  const n = (x) => Number(x).toLocaleString('en-US');
  const secs = (s) => (s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);

  // ---------------------------------------------------------------- full machine overview
  $('full-chips').innerHTML = S.chips.map((c) => `<span class="chip">${c[0]} <b>${esc(c[1])}</b></span>`).join('');
  $('full-in').innerHTML = S.in_weights.map((x) => `<li>${x}</li>`).join('');
  $('full-out').innerHTML = S.outside.map((x) => `<li>${x}</li>`).join('');
  $('full-gallery').innerHTML = S.gallery.map(([img, cap]) =>
    `<figure class="gfig"><a href="assets/${img}" target="_blank" rel="noopener"><img src="assets/${img}" alt="${esc(cap)}" loading="lazy"></a><figcaption>${esc(cap)}</figcaption></figure>`).join('');
  const hc = $('hero-chips');
  if (hc) hc.insertAdjacentHTML('afterbegin', S.hero_chips.map((c) => `<span class="chip">${c[0]} <b>${esc(c[1])}</b></span>`).join(''));

  // ---------------------------------------------------------------- results tables
  const row = (cells, cls) => `<tr>${cells.map((c, i) => `<td class="${cls && cls[i] ? cls[i] : ''}">${c}</td>`).join('')}</tr>`;
  const passc = (x) => (x && /^(\d[\d,]*)\/\1$/.test(String(x).replace(/ .*/, '')) ? 'pass' : '');
  $('full-table').innerHTML = '<tr><th>Suite</th><th>Ground truth</th><th>Cases</th><th>Reference vs truth</th><th>Weights vs truth</th></tr>' +
    S.suites.map((s) => row([s.suite, s.truth, s.cases, s.ref, s.weights], [0, 0, 0, passc(s.ref), passc(s.weights)])).join('');
  $('crypto-table').innerHTML = '<tr><th>Family</th><th>Syscalls</th><th>Reference vs LiteSVM</th><th>In the weights</th><th>Time per call on the weights</th></tr>' +
    S.crypto.map((c) => row([c.family, `<code>${c.calls}</code>`, c.ref, c.weights, c.time], [0, 0, passc(c.ref), 0, 0])).join('');
  $('speed-table').innerHTML = '<tr><th>Operation</th><th>Weights</th><th>Comparison</th></tr>' +
    S.speed.map((s) => row([s[0], s[1], s[2]])).join('');

  // ---------------------------------------------------------------- Real programs cards
  const verdict = (same, t) => same === undefined ? '<span class="tag llm">weights run pending</span>' :
    same ? `<span class="tag ok">weights = LiteSVM${t !== undefined ? ', ' + secs(t) : ''}</span>` : '<span class="tag bad">weights differ</span>';
  const outcome = (t) => t.ok ? `<span class="v allow">success</span>` : `<span class="v block">${esc(t.err.replace('InstructionError(0, ', '').replace(/\)$/, ''))}</span>`;
  const logs = (l) => l.length ? `<pre class="logs">${l.map(esc).join('\n')}</pre>` : '';
  $('show-grid').innerHTML = F.showcase.map((c) => `
    <div class="card show">
      <div class="show-head"><h4>${esc(c.title)}</h4>${verdict(c.weights_same, c.weights_seconds)}</div>
      <p class="muted">${esc(c.story)}</p>
      <div class="calls">${c.calls.map((x) => `<code>${x}</code>`).join(' ')}</div>
      <div class="kv"><span>outcome</span>${outcome(c.truth)}<span>compute units</span><b>${n(c.truth.cu)} of ${n(c.cu_limit)}</b>${c.returns.length ? `<span>syscall returns</span><b>${c.returns.join(', ')}</b>` : ''}</div>
      ${logs(c.truth.logs)}
      ${c.output && /[1-9a-f]/.test(c.output) && !/^0+$/.test(c.output) ? `<label class="muted small">first output bytes in account memory</label><pre class="hex">${c.output.match(/.{1,32}/g).join('\n')}</pre>` : ''}
    </div>`).join('');
  $('cpi-grid').innerHTML = F.cpi.map((c) => `
    <div class="card show">
      <div class="show-head"><h4>${c.truth.ok ? 'Succeeds' : esc(c.truth.err.replace('InstructionError(0, ', '').replace(/\)$/, ''))}</h4>${verdict(c.weights_same, c.weights_seconds)}</div>
      <p class="muted">${esc(c.what)}</p>
      <div class="kv"><span>outcome</span>${outcome(c.truth)}<span>compute units</span><b>${n(c.truth.cu)}</b></div>
      ${logs(c.truth.logs)}
    </div>`).join('') || '<p class="muted">The CPI runs on the weights are still in progress.</p>';

  // ---------------------------------------------------------------- judging real transactions
  $('jgrid-real').innerHTML = F.judge_real.map((j) => `
    <div class="jcard card">
      <div class="show-head"><h4>${esc(j.title)}</h4><span class="v ${j.review.decision}">${j.review.decision}</span></div>
      <p class="muted">${esc(j.story)}</p>
      <div class="kv"><span>transaction</span>${outcome(j.truth)}<span>compute units</span><b>${n(j.truth.cu)}</b>
        <span>lamports that left the wallet</span><b>${n(j.review.moved_lamports)}</b><span>to</span><b>${esc(j.review.destination)}</b>
        <span>ratio to the usual amount</span><b>${j.review.signals.ratio_to_median}x</b></div>
      <p class="reason">${esc(j.review.reason)}</p>
      <div class="muted small">LiteSVM, reference ${j.ref_same ? 'matches' : 'differs'}${j.weights_same === undefined ? '' : ', weights ' + (j.weights_same ? 'match' : 'differ')}. The amount and destination are read from the post-state, not from the request.</div>
    </div>`).join('');
})();
