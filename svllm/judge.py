"""Judgement filter: a review step that runs AFTER the VM.

Pipeline:  sBPF program -> VM (exact result + CU count) -> prompt -> judge -> allow / hold / block

The VM answers "is this valid by the code?". The judge answers "would a sensible operator
let this through?". It looks at the amount, the destination and the account's history.

IMPORTANT: the judge is NOT part of the transformer weights. RuleJudge is plain Python
heuristics. LLMJudge is a pluggable interface. The default backend is SimulatedLLM, which
returns a canned JSON reply and never calls a model. Pass complete(prompt) -> str to use a real one.
The label for every simulated reply is SIMULATED_LABEL.
"""
import json
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


# Shown on every canned reply, in the README, and on the demo site.
SIMULATED_LABEL = 'SIMULATED model output, no real LLM was called'
WALLET_PLACEHOLDER = 'Wallet X'

SYSTEM_ROLE = (
    'You are the transaction reviewer for a wallet. The sBPF virtual machine has already run the program exactly. '
    'Decide allow, hold or block based on everything you know about the sender and the receiver and the run rules. '
    'You may only make the decision stricter, never looser than the rule verdict, and you never override a code-level rejection.'
)

RULES_TEXT = (
    'The rule review is a heuristic outside the weights. You may only make the decision stricter, never looser than the rule verdict, and you never override a code-level rejection.\n'
    '1. Program error or nonzero exit code: block.\n'
    '2. More than 3 withdrawals in 15 minutes moving at least 50% of the balance: block.\n'
    '3. At least 50x the median past amount to a never-seen destination: block.\n'
    '4. At least 10x the median past amount: hold.\n'
    '5. New destination and at least 3x the median: hold.\n'
    '6. Otherwise: allow.'
)

# kind is part of the prompt wording. call = "has this program call been allowed?". withdraw = "should this withdrawal be allowed?".
EXAMPLE_KIND = {
    'big-new-address': 'withdraw',
    'known-address': 'call',
    'rapid-drain': 'withdraw',
    'spend-cap': 'call',
    'new-address-moderate': 'call',
}


def program_titles():
    from .programs import DEMOS
    return {d['id']: d['title'] for d in DEMOS}


def _field(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def num_text(x):
    """Number text that matches JSON.stringify in the browser port, including whole floats as integers."""
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return str(x)
    s = json.dumps(x)
    if s.endswith('.0'):
        s = s[:-2]
    return s


def return_data_hex(r0):
    """8-byte little-endian hex of r0. This toy VM has no separate return-data buffer."""
    return (int(r0) & ((1 << 64) - 1)).to_bytes(8, 'little').hex()


def _decision_of(base):
    return base['decision'] if isinstance(base, dict) else base.decision


def _reason_of(base):
    return base['reason'] if isinstance(base, dict) else base.reason


def _signals_of(base):
    return base['signals'] if isinstance(base, dict) else base.signals


def build_prompt(vm, tx, history, base, program_id, program_title, kind, wallet=WALLET_PLACEHOLDER, known=None):
    """Full reviewer prompt. kind is 'call' or 'withdraw'. history items are Tx or dicts."""
    if kind not in ('call', 'withdraw'):
        raise ValueError(f'kind must be call or withdraw, got {kind!r}')
    known = list(known or [])
    amount = int(_field(tx, 'amount'))
    dest = _field(tx, 'destination')
    hist = list(history)[-20:]
    sender_lines = [
        f"- t={int(_field(h, 'time'))} amount={int(_field(h, 'amount'))} to={_field(h, 'destination')} balance_before={int(_field(h, 'balance_before'))}"
        for h in hist
    ] or ['- none']
    recv = [h for h in hist if _field(h, 'destination') == dest]
    recv_lines = [
        f"- t={int(_field(h, 'time'))} amount={int(_field(h, 'amount'))} balance_before={int(_field(h, 'balance_before'))}"
        for h in recv
    ] or ['- none']
    sig = _signals_of(base)
    seen_txt = 'yes' if sig['destination_seen_before'] else 'no'
    call = f'{wallet} has requested calling this Solana program "{program_title}" (program id: {program_id}).'
    if kind == 'withdraw':
        ask = f'{wallet} would like to withdraw {amount}. Based on everything you know about the context and the run rules, should that be allowed?'
    else:
        ask = f'{wallet} has requested calling this smart contract. Based on everything you know about the context and the run rules, is this allowed?'
    known_txt = ', '.join(sorted(known)) if known else 'none'
    lines = [
        'SYSTEM',
        SYSTEM_ROLE,
        '',
        'TRANSACTION REQUEST',
        call,
        ask,
        '',
        f'Sender (wallet): {wallet}',
        f'Receiver (destination): {dest}',
        f'Amount: {amount}',
        f'Time (seconds): {int(_field(tx, "time"))}',
        f'Balance before this request: {int(_field(tx, "balance_before"))}',
        '',
        'VM RESULT',
        'The sBPF virtual machine has already run the program exactly. These numbers are that result.',
        f'status: {vm["status"]}',
        f'exit code: {int(vm["r0"])}',
        f'compute units: {int(vm["cu"])}',
        f'return data: {return_data_hex(vm["r0"])}',
        '',
        'SENDER AND RECEIVER HISTORY',
        'Past requests from this sender, oldest first:',
        *sender_lines,
        '',
        f'Past payments to this receiver ({dest}):',
        *recv_lines,
        f'Destination seen before: {seen_txt}',
        f'Known destinations on file: {known_txt}',
        '',
        'RUN RULES',
        RULES_TEXT,
        f'Rule verdict: {_decision_of(base)}',
        f'Rule reason: {_reason_of(base)}',
        f'median past amount: {num_text(sig["median_past_amount"])}',
        f'ratio to median: {num_text(sig["ratio_to_median"])}',
        f'destination seen before: {seen_txt}',
        f'transactions in 15 minutes: {num_text(sig["tx_in_window"])}',
        f'window outflow: {num_text(sig["window_outflow"])}',
        f'window outflow fraction: {num_text(sig["window_outflow_fraction"])}',
        '',
        'REQUIRED REPLY',
        'Reply with one JSON object and no other text. The keys are decision and reason.',
        '{"decision": "allow" or "hold" or "block", "reason": "short reason"}',
    ]
    return '\n'.join(lines) + '\n'


def _line_value(prompt, label):
    prefix = label + ': '
    for line in prompt.split('\n'):
        if line.startswith(prefix):
            return line[len(prefix):]
    return ''


# Needles are exact substrings of build_prompt() for the published scenarios.
# A live edit that changes the amount or the exit code misses the needles and uses the fallback.
CANNED = (
    {
        'id': 'known-address',
        'needles': ('\nAmount: 210\n', 'Receiver (destination): addr:alice-savings', 'program id: transfer', '\nexit code: 0\n', 'is this allowed?'),
        'decision': 'allow',
        'reason': 'The program exited 0 in 12 compute units. The amount is in line with what this wallet usually sends, and the receiver is an address it already pays. Allow it.',
    },
    {
        'id': 'big-new-address',
        'needles': ('\nAmount: 20500\n', 'Receiver (destination): addr:9qZ...never-seen', 'program id: vault', '\nexit code: 0\n', 'would like to withdraw 20500'),
        'decision': 'block',
        'reason': 'The program allowed this withdrawal (exit code 0), but 20500 is 100x the usual amount and the destination has never been seen before. Block it.',
    },
    {
        'id': 'rapid-drain',
        'needles': ('\nAmount: 2400\n', 'Receiver (destination): addr:x7f3...unknown', 'program id: vault', '\nexit code: 0\n', 'would like to withdraw 2400'),
        'decision': 'block',
        'reason': 'The program allowed this withdrawal, but the history shows 4 withdrawals in 15 minutes moving 96% of the balance. Block it.',
    },
    {
        'id': 'spend-cap',
        'needles': ('\nAmount: 1200000000\n', 'program id: spend-cap', '\nexit code: 1\n'),
        'decision': 'block',
        'reason': 'The program itself exited with code 1. A code-level rejection is never overridden. Block it.',
    },
    {
        'id': 'new-address-moderate',
        'needles': ('\nAmount: 1000\n', 'Receiver (destination): addr:new-merchant', 'program id: transfer', '\nexit code: 0\n', 'is this allowed?'),
        'decision': 'hold',
        'reason': 'The program allowed the call, but the amount is about 4.9x the usual size and the destination has never been seen before. Hold it for a person to confirm.',
    },
)


def dumps_reply(decision, reason):
    return json.dumps({'decision': decision, 'reason': reason}, separators=(',', ':'))


def fallback_reply(prompt):
    """Used when the prompt is not one of the five canned scenarios. Still a fake, still not looser than the rules."""
    decision = _line_value(prompt, 'Rule verdict')
    reason = _line_value(prompt, 'Rule reason')
    ratio = _line_value(prompt, 'ratio to median')
    exit_code = _line_value(prompt, 'exit code')
    seen = _line_value(prompt, 'destination seen before')
    if decision not in (ALLOW, HOLD, BLOCK):
        return {'decision': HOLD, 'reason': 'The simulated reviewer could not read a rule verdict, so it holds the request.'}
    if 'code-level rejection' in reason or (exit_code not in ('', '0') and decision == BLOCK):
        text = f'The program itself rejected this (exit code {exit_code}). A code-level rejection is never overridden.'
    elif decision == BLOCK and reason.startswith('Drain pattern'):
        text = f'The history shows a drain. {reason}'
    elif decision == BLOCK:
        where = 'a destination that has never been seen' if seen == 'no' else 'a destination this wallet has used'
        text = f'The program allowed the request, but the amount is {ratio}x the usual size and it goes to {where}. Block it.'
    elif decision == HOLD:
        text = f'The amount is {ratio}x the usual size (destination seen before: {seen}). Hold it for a person to confirm.'
    else:
        text = 'The program exited 0 and the amount fits this sender and receiver history. Allow it.'
    return {'decision': decision, 'reason': text}


def simulated_output(prompt):
    """Canned JSON for a known scenario, otherwise the fallback. Never calls a model."""
    for spec in CANNED:
        if all(n in prompt for n in spec['needles']):
            return dumps_reply(spec['decision'], spec['reason'])
    fb = fallback_reply(prompt)
    return dumps_reply(fb['decision'], fb['reason'])


class SimulatedLLM:
    """Fake model backend. complete(prompt) -> JSON text.

    The string is a canned reply for one of the published scenarios, or a short fallback
    that follows the rule verdict already written into the prompt. No network call.
    """

    label = SIMULATED_LABEL

    def complete(self, prompt):
        return simulated_output(prompt)


RANK = {ALLOW: 0, HOLD: 1, BLOCK: 2}


def merge_decision(base_decision, base_reason, model_decision, model_reason):
    """The model may only make the decision stricter. A looser reply keeps the rule verdict."""
    if model_decision not in RANK:
        return base_decision, base_reason
    if RANK[model_decision] >= RANK[base_decision]:
        return model_decision, model_reason
    return base_decision, model_reason + ' Kept the stricter rule verdict (' + base_decision + ').'


class LLMJudge:
    """Pluggable LLM judge. complete(prompt) -> str returns JSON {"decision", "reason"}.

    With complete=None (the default) the backend is SimulatedLLM. The rules always run
    first. The reply may move allow to hold or block, but it never makes a verdict looser,
    and a code-level rejection is already a rule block.
    """

    name = 'llm-judge'

    def __init__(self, complete=None, rules=None):
        self.complete = complete
        self.rules = rules or RuleJudge()
        self.last_prompt = ''
        self.last_output = ''

    def prompt(self, vm, tx, ctx, base, program_id, program_title, kind, wallet=WALLET_PLACEHOLDER):
        known = sorted(ctx.known_destinations) if ctx.known_destinations else []
        return build_prompt(vm, tx, ctx.history, base, program_id, program_title, kind, wallet, known)

    def review(self, vm, tx, ctx, program_id, program_title, kind, wallet=WALLET_PLACEHOLDER):
        base = self.rules.review(vm, tx, ctx)
        prompt = self.prompt(vm, tx, ctx, base, program_id, program_title, kind, wallet)
        simulated = self.complete is None
        complete = SimulatedLLM().complete if simulated else self.complete
        raw = complete(prompt)
        out = json.loads(raw)
        decision, reason = merge_decision(base.decision, base.reason, out.get('decision'), out.get('reason', base.reason))
        self.last_prompt, self.last_output = prompt, raw
        source = SIMULATED_LABEL if simulated else self.name
        return Verdict(decision, reason, base.signals, source)


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
         'program': 'vault', 'kind': 'withdraw', 'wallet': WALLET_PLACEHOLDER,
         'words': [1_000_000, 0, 50_000, 20_500],
         'tx': Tx(20_500, 'addr:9qZ...never-seen', NOW, 1_000_000), 'ctx': Context(USUAL)},
        {'id': 'known-address', 'title': 'Normal transfer to a known address',
         'story': 'A routine transfer of a typical amount to an address this account pays every week.',
         'program': 'transfer', 'kind': 'call', 'wallet': WALLET_PLACEHOLDER,
         'words': [1_000_000, 5_000, 210],
         'tx': Tx(210, 'addr:alice-savings', NOW, 1_000_000), 'ctx': Context(USUAL)},
        {'id': 'rapid-drain', 'title': 'Rapid repeated drain',
         'story': 'The fourth 2,400 withdrawal in ten minutes. Each request is under the daily limit on its own, so the code allows it. Together they move most of the balance.',
         'program': 'vault', 'kind': 'withdraw', 'wallet': WALLET_PLACEHOLDER,
         'words': [2_800, 7_200, 20_000, 2_400],
         'tx': Tx(2_400, 'addr:x7f3...unknown', NOW, 2_800), 'ctx': Context(drain_hist)},
        {'id': 'spend-cap', 'title': 'Spend cap: over $1B',
         'story': 'A $1.2B transfer against the $1B cap. The program exits with code 1 before the judge sees it; the judge records the code-level block.',
         'program': 'spend-cap', 'kind': 'call', 'wallet': WALLET_PLACEHOLDER,
         'words': [1_200_000_000],
         'tx': Tx(1_200_000_000, 'addr:treasury-2', NOW, 5_000_000_000), 'ctx': Context(USUAL)},
        {'id': 'new-address-moderate', 'title': 'Moderate amount, new address',
         'story': 'Five times the usual amount to a first-time address. Not alarming enough to block, but worth a confirmation.',
         'program': 'transfer', 'kind': 'call', 'wallet': WALLET_PLACEHOLDER,
         'words': [1_000_000, 0, 1_000],
         'tx': Tx(1_000, 'addr:new-merchant', NOW, 1_000_000), 'ctx': Context(USUAL)},
    ]


def _attach_llm(vm, ex):
    """Rule verdict plus the default simulated reply. vm is a dict with status, r0, cu."""
    titles = program_titles()
    kind = ex.get('kind', EXAMPLE_KIND[ex['id']])
    wallet = ex.get('wallet', WALLET_PLACEHOLDER)
    base = RuleJudge().review(vm, ex['tx'], ex['ctx'])
    llm = LLMJudge()
    reviewed = llm.review(vm, ex['tx'], ex['ctx'], ex['program'], titles[ex['program']], kind, wallet)
    return {
        'kind': kind,
        'wallet': wallet,
        'program_title': titles[ex['program']],
        'verdict': vars(base),
        'prompt': llm.last_prompt,
        'simulated': {
            'decision': reviewed.decision,
            'reason': reviewed.reason,
            'source': reviewed.source,
            'output': llm.last_output,
        },
    }


def annotate_saved(records):
    """Add prompt and simulated fields to saved example dicts, using the VM result already stored.

    Used so results/judge_examples.json can gain the prompt without re-running the weights machine.
    """
    by_id = {ex['id']: ex for ex in examples()}
    out = []
    for rec in records:
        ex = by_id[rec['id']]
        vm = {'status': rec['vm']['status'], 'r0': rec['vm']['r0'], 'cu': rec['vm']['cu']}
        # The saved tx and history are the scenario. Rebuild Tx objects so the rule judge matches.
        packed = dict(ex)
        extra = _attach_llm(vm, packed)
        merged = dict(rec)
        merged.update(extra)
        out.append(merged)
    return out


def run_examples(engine='ref'):
    from . import ref
    from .programs import DEMOS, u64s
    from .isa import assemble
    if engine == 'weights':
        from . import machine
        runner = machine
    else:
        runner = ref
    progs = {d['id']: assemble(d['asm']) for d in DEMOS}
    out = []
    for ex in examples():
        p, m = progs[ex['program']], u64s(*ex['words'])
        vm = runner.run(p, m, 10000)
        vm_pub = {'status': vm['status'], 'r0': vm['r0'], 'cu': vm['cu']}
        extra = _attach_llm(vm_pub, ex)
        out.append({**{k: ex[k] for k in ('id', 'title', 'story', 'program', 'words')},
                    'tx': vars(ex['tx']), 'history': [vars(h) for h in ex['ctx'].history],
                    'vm': {'engine': engine, **vm_pub},
                    **extra})
    return out


if __name__ == '__main__':
    for r in run_examples():
        print(f"{r['id']:22s} vm={r['vm']['status']}/{r['vm']['r0']} cu={r['vm']['cu']:3d} rules={r['verdict']['decision'].upper():5s} simulated={r['simulated']['decision'].upper():5s}")
        print(f"  rules:     {r['verdict']['reason']}")
        print(f"  simulated: {r['simulated']['reason']}")
        print(f"  label:     {r['simulated']['source']}")
