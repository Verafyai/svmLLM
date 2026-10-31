"""Judgement filter: a review step that runs AFTER the VM.

Pipeline:  sBPF program -> VM (exact result + CU count) -> judge -> allow / hold / block

The VM answers "is this valid by the code?". The judge answers "would a sensible operator
let this through?". It looks at the amount, the destination and the account's history.

IMPORTANT: the judge is NOT part of the transformer weights. RuleJudge is plain Python
heuristics. LLMJudge is a pluggable interface; the default backend is a stub that defers
to the rules, so nothing here calls a model unless you pass one in.
"""
from dataclasses import dataclass, field
from statistics import median

ALLOW, HOLD, BLOCK = 'allow', 'hold', 'block'


@dataclass
class Tx:
    amount: int
    destination: str
    time: int                      # seconds, any epoch
    balance_before: int = 0


@dataclass
class Context:
    history: list = field(default_factory=list)    # past Tx, oldest first
    known_destinations: set = field(default_factory=set)


@dataclass
class Verdict:
    decision: str
    reason: str
    signals: dict
    source: str


# thresholds (documented on the site and in the README)
HOLD_RATIO = 10          # amount >= 10x median past amount
BLOCK_RATIO = 50         # amount >= 50x median past amount AND new destination
NEW_DEST_RATIO = 3       # new destination and amount >= 3x median
DRAIN_WINDOW = 15 * 60   # seconds
DRAIN_COUNT = 3          # this tx plus >= 3 others inside the window
DRAIN_FRACTION = 0.5     # window outflow >= 50% of the balance at window start


def signals(tx, ctx):
    past = [h.amount for h in ctx.history] or [tx.amount]
    med = max(1, int(median(past)))
    seen = tx.destination in ctx.known_destinations or any(h.destination == tx.destination for h in ctx.history)
    window = [h for h in ctx.history if tx.time - h.time <= DRAIN_WINDOW]
    outflow = sum(h.amount for h in window) + tx.amount
    start_bal = (window[0].balance_before if window else tx.balance_before) or 1
    return {'median_past_amount': med, 'ratio_to_median': round(tx.amount / med, 2),
            'destination_seen_before': seen, 'tx_in_window': len(window) + 1,
            'window_outflow': outflow, 'window_outflow_fraction': round(outflow / start_bal, 3)}


class RuleJudge:
    name = 'rules (heuristics, not in the weights)'

    def review(self, vm, tx, ctx):
        s = signals(tx, ctx)
        if vm['status'] != 'ok' or vm['r0'] != 0:
            why = vm['status'] if vm['status'] != 'ok' else f'exit code {vm["r0"]}'
            return Verdict(BLOCK, f'The program itself rejected this ({why}). The judge never overrides a code-level rejection.', s, self.name)
        if s['tx_in_window'] > DRAIN_COUNT and s['window_outflow_fraction'] >= DRAIN_FRACTION:
            return Verdict(BLOCK, f"Drain pattern: {s['tx_in_window']} withdrawals in {DRAIN_WINDOW // 60} minutes moving {s['window_outflow_fraction']:.0%} of the balance. Each one passed the code checks.", s, self.name)
        if s['ratio_to_median'] >= BLOCK_RATIO and not s['destination_seen_before']:
            return Verdict(BLOCK, f"{s['ratio_to_median']:.0f}x the usual amount to a destination never seen before. Valid by code, but a sensible operator would stop it.", s, self.name)
        if s['ratio_to_median'] >= HOLD_RATIO:
            return Verdict(HOLD, f"{s['ratio_to_median']:.0f}x the usual amount. Hold for a human to confirm.", s, self.name)
        if not s['destination_seen_before'] and s['ratio_to_median'] >= NEW_DEST_RATIO:
            return Verdict(HOLD, f"New destination and {s['ratio_to_median']:.1f}x the usual amount. Hold for confirmation.", s, self.name)
        return Verdict(ALLOW, 'Valid by code and consistent with history.', s, self.name)


class LLMJudge:
    """Pluggable LLM judge. `complete(prompt) -> str` should return a JSON object like
    {"decision": "allow|hold|block", "reason": "..."}. With complete=None (the default stub)
    it returns the rule verdict, labeled as such. The rules always run first; the LLM may
    escalate (allow -> hold/block) but never downgrades a rule BLOCK."""
    name = 'llm-judge'

    def __init__(self, complete=None, rules=None):
        self.complete, self.rules = complete, rules or RuleJudge()

    def prompt(self, vm, tx, ctx, base):
        hist = '\n'.join(f'- t={h.time} amount={h.amount} to={h.destination}' for h in ctx.history[-20:])
        return (f'You review on-chain withdrawals after the program has executed.\n'
                f'Program result: status={vm["status"]} exit={vm["r0"]} compute_units={vm["cu"]}\n'
                f'Request: amount={tx.amount} to={tx.destination} t={tx.time}\nHistory:\n{hist or "- none"}\n'
                f'Rule verdict: {base.decision} ({base.reason})\n'
                'Answer with JSON {"decision": "allow|hold|block", "reason": "..."}.')

    def review(self, vm, tx, ctx):
        import json
        base = self.rules.review(vm, tx, ctx)
        if self.complete is None:
            return Verdict(base.decision, base.reason, base.signals, base.source + ' (LLM stub: no model configured)')
        out = json.loads(self.complete(self.prompt(vm, tx, ctx, base)))
        order = [ALLOW, HOLD, BLOCK]
        d = max(base.decision, out['decision'], key=order.index)
        return Verdict(d, out.get('reason', base.reason), base.signals, self.name)


# ------------------------------------------------------------------ runnable examples
H = 3600
def _hist(amounts_dests, t0=0, step=86400, bal=1_000_000):
    out = []
    for i, (a, d) in enumerate(amounts_dests):
        out.append(Tx(a, d, t0 + i * step, bal)); bal -= a
    return out


USUAL = _hist([(200, 'addr:alice-savings'), (150, 'addr:landlord'), (250, 'addr:alice-savings'),
               (180, 'addr:grocer'), (220, 'addr:alice-savings'), (210, 'addr:landlord')])
NOW = 10 * 86400


def examples():
    """Each example: program id + input words for the VM, plus the tx and context for the judge."""
    drain_hist = USUAL + [Tx(2400, 'addr:x7f3...unknown', NOW - 600, 10000), Tx(2400, 'addr:x7f3...unknown', NOW - 420, 7600),
                          Tx(2400, 'addr:x7f3...unknown', NOW - 240, 5200)]
    return [
        {'id': 'big-new-address', 'title': 'Valid by code, 100x usual, never-seen address',
         'story': 'The vault program checks balance and the daily limit, and this request passes both. It is also 100x what this account usually sends, to an address it has never paid.',
         'program': 'vault', 'words': [1_000_000, 0, 50_000, 20_500],
         'tx': Tx(20_500, 'addr:9qZ...never-seen', NOW, 1_000_000), 'ctx': Context(USUAL)},
        {'id': 'known-address', 'title': 'Normal transfer to a known address',
         'story': 'A routine transfer of a typical amount to an address this account pays every week.',
         'program': 'transfer', 'words': [1_000_000, 5_000, 210],
         'tx': Tx(210, 'addr:alice-savings', NOW, 1_000_000), 'ctx': Context(USUAL)},
        {'id': 'rapid-drain', 'title': 'Rapid repeated drain',
         'story': 'The fourth 2,400 withdrawal in ten minutes. Each request is under the daily limit on its own, so the code allows it. Together they move most of the balance.',
         'program': 'vault', 'words': [2_800, 7_200, 20_000, 2_400],
         'tx': Tx(2_400, 'addr:x7f3...unknown', NOW, 2_800), 'ctx': Context(drain_hist)},
        {'id': 'spend-cap', 'title': 'Spend cap: over $1B',
         'story': 'A $1.2B transfer against the $1B cap. The program exits with code 1 before the judge sees it; the judge records the code-level block.',
         'program': 'spend-cap', 'words': [1_200_000_000],
         'tx': Tx(1_200_000_000, 'addr:treasury-2', NOW, 5_000_000_000), 'ctx': Context(USUAL)},
        {'id': 'new-address-moderate', 'title': 'Moderate amount, new address',
         'story': 'Five times the usual amount to a first-time address. Not alarming enough to block, but worth a confirmation.',
         'program': 'transfer', 'words': [1_000_000, 0, 1_000],
         'tx': Tx(1_000, 'addr:new-merchant', NOW, 1_000_000), 'ctx': Context(USUAL)},
    ]


def run_examples(engine='ref'):
    from . import ref, machine
    from .programs import DEMOS, u64s
    from .isa import assemble
    progs = {d['id']: assemble(d['asm']) for d in DEMOS}
    out = []
    for ex in examples():
        p, m = progs[ex['program']], u64s(*ex['words'])
        vm = (machine if engine == 'weights' else ref).run(p, m, 10000)
        v = RuleJudge().review(vm, ex['tx'], ex['ctx'])
        out.append({**{k: ex[k] for k in ('id', 'title', 'story', 'program', 'words')},
                    'tx': vars(ex['tx']), 'history': [vars(h) for h in ex['ctx'].history],
                    'vm': {'engine': engine, 'status': vm['status'], 'r0': vm['r0'], 'cu': vm['cu']},
                    'verdict': vars(v)})
    return out


if __name__ == '__main__':
    for r in run_examples():
        print(f"{r['id']:22s} vm={r['vm']['status']}/{r['vm']['r0']} cu={r['vm']['cu']:3d} -> {r['verdict']['decision'].upper():5s} {r['verdict']['reason']}")
