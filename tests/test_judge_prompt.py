"""Prompt builder, simulated backend, and the stricter-only merge. No torch, no network."""
import json, os, sys, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svllm.judge import (
    ALLOW, BLOCK, HOLD, SIMULATED_LABEL, SYSTEM_ROLE, WALLET_PLACEHOLDER,
    LLMJudge, RuleJudge, SimulatedLLM, annotate_saved, build_prompt, examples,
    merge_decision, return_data_hex, run_examples, simulated_output,
)


def _ex(eid):
    return next(e for e in examples() if e['id'] == eid)


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run_examples('ref')
        cls.by = {r['id']: r for r in cls.rows}

    def test_system_role_and_sections(self):
        for r in self.rows:
            p = r['prompt']
            self.assertIn(SYSTEM_ROLE, p)
            for head in ('SYSTEM\n', 'TRANSACTION REQUEST\n', 'VM RESULT\n', 'SENDER AND RECEIVER HISTORY\n', 'RUN RULES\n', 'REQUIRED REPLY\n'):
                self.assertIn(head, p)
            self.assertIn('{"decision": "allow" or "hold" or "block", "reason": "short reason"}', p)
            self.assertIn(f'exit code: {r["vm"]["r0"]}', p)
            self.assertIn(f'compute units: {r["vm"]["cu"]}', p)
            self.assertIn('return data: ' + return_data_hex(r['vm']['r0']), p)
            self.assertTrue(p.endswith('\n'))

    def test_call_example_allows(self):
        r = self.by['known-address']
        self.assertIn(
            'Wallet X has requested calling this smart contract. Based on everything you know about the context and the run rules, is this allowed?',
            r['prompt'])
        self.assertIn('Wallet X has requested calling this Solana program "Balance transfer" (program id: transfer).', r['prompt'])
        out = json.loads(r['simulated']['output'])
        self.assertEqual(out['decision'], ALLOW)
        self.assertIn('already pays', out['reason'])
        self.assertEqual(r['simulated']['source'], SIMULATED_LABEL)
        self.assertEqual(r['simulated']['decision'], ALLOW)

    def test_withdrawal_example_blocks_on_history(self):
        r = self.by['big-new-address']
        self.assertIn(
            'Wallet X would like to withdraw 20500. Based on everything you know about the context and the run rules, should that be allowed?',
            r['prompt'])
        self.assertIn('addr:9qZ...never-seen', r['prompt'])
        self.assertIn('amount=200 to=addr:alice-savings', r['prompt'])
        out = json.loads(r['simulated']['output'])
        self.assertEqual(out['decision'], BLOCK)
        self.assertIn('100x the usual amount', out['reason'])
        self.assertIn('never been seen', out['reason'])
        self.assertEqual(r['verdict']['decision'], BLOCK)

    def test_other_canned_scenarios(self):
        self.assertEqual(json.loads(self.by['rapid-drain']['simulated']['output'])['decision'], BLOCK)
        self.assertIn('4 withdrawals', self.by['rapid-drain']['simulated']['reason'])
        self.assertEqual(json.loads(self.by['spend-cap']['simulated']['output'])['decision'], BLOCK)
        self.assertIn('code 1', self.by['spend-cap']['simulated']['reason'])
        self.assertEqual(self.by['spend-cap']['verdict']['decision'], BLOCK)
        hold = json.loads(self.by['new-address-moderate']['simulated']['output'])
        self.assertEqual(hold['decision'], HOLD)
        self.assertIn('never been seen', hold['reason'])

    def test_return_data_is_little_endian_r0(self):
        self.assertEqual(return_data_hex(0), '0000000000000000')
        self.assertEqual(return_data_hex(1), '0100000000000000')

    def test_saved_weights_rows_match_reference_prompts(self):
        path = Path(__file__).resolve().parents[1] / 'results' / 'judge_examples.json'
        saved = json.loads(path.read_text())
        # VM numbers in the published file are from the weights machine. They must still match the reference.
        for rec, fresh in zip(saved, self.rows):
            self.assertEqual(rec['vm']['status'], fresh['vm']['status'])
            self.assertEqual(rec['vm']['r0'], fresh['vm']['r0'])
            self.assertEqual(rec['vm']['cu'], fresh['vm']['cu'])
        annotated = annotate_saved(saved)
        for rec, fresh in zip(annotated, self.rows):
            self.assertEqual(rec['prompt'], fresh['prompt'])
            self.assertEqual(rec['simulated']['output'], fresh['simulated']['output'])
            self.assertEqual(rec['vm']['engine'], 'weights')

    def test_fallback_when_amount_changes(self):
        r = self.by['big-new-address']
        edited = r['prompt'].replace('\nAmount: 20500\n', '\nAmount: 400\n').replace('would like to withdraw 20500', 'would like to withdraw 400')
        # The rule block about 100x is still in the edited text, so the fallback must not invent an allow.
        raw = simulated_output(edited)
        out = json.loads(raw)
        self.assertNotIn('20500 is 100x', out['reason'])
        self.assertEqual(out['decision'], BLOCK)
        self.assertNotEqual(raw, r['simulated']['output'])

    def test_plugin_cannot_loosen_and_is_not_labeled_simulated(self):
        calls = []

        def complete(prompt):
            calls.append(prompt)
            return json.dumps({'decision': ALLOW, 'reason': 'I would allow this'})

        ex = _ex('big-new-address')
        vm = {'status': 'ok', 'r0': 0, 'cu': 13}
        reviewed = LLMJudge(complete=complete).review(vm, ex['tx'], ex['ctx'], ex['program'], 'Vault with withdrawal limit', 'withdraw')
        self.assertEqual(len(calls), 1)
        self.assertIn(SYSTEM_ROLE, calls[0])
        self.assertEqual(reviewed.decision, BLOCK)
        self.assertIn('Kept the stricter rule verdict (block).', reviewed.reason)
        self.assertEqual(reviewed.source, 'llm-judge')
        self.assertNotIn('SIMULATED', reviewed.source)

    def test_plugin_may_escalate(self):
        def complete(prompt):
            return json.dumps({'decision': HOLD, 'reason': 'Hold it. The receiver is new to me.'})

        ex = _ex('known-address')
        vm = {'status': 'ok', 'r0': 0, 'cu': 12}
        reviewed = LLMJudge(complete=complete).review(vm, ex['tx'], ex['ctx'], ex['program'], 'Balance transfer', 'call')
        self.assertEqual(reviewed.decision, HOLD)
        self.assertEqual(reviewed.reason, 'Hold it. The receiver is new to me.')

    def test_default_backend_is_simulated(self):
        ex = _ex('known-address')
        vm = {'status': 'ok', 'r0': 0, 'cu': 12}
        j = LLMJudge()
        reviewed = j.review(vm, ex['tx'], ex['ctx'], ex['program'], 'Balance transfer', 'call')
        self.assertIsInstance(SimulatedLLM().complete(j.last_prompt), str)
        self.assertEqual(reviewed.source, SIMULATED_LABEL)
        self.assertEqual(json.loads(j.last_output)['decision'], ALLOW)

    def test_code_rejection_stays_blocked(self):
        r = self.by['spend-cap']
        self.assertIn('exit code 1', r['verdict']['reason'])
        self.assertEqual(r['simulated']['decision'], BLOCK)
        self.assertIn('never overridden', r['simulated']['reason'])

    def test_merge_rank(self):
        self.assertEqual(merge_decision(ALLOW, 'a', BLOCK, 'b'), (BLOCK, 'b'))
        self.assertEqual(merge_decision(BLOCK, 'a', ALLOW, 'b')[0], BLOCK)
        self.assertEqual(merge_decision(HOLD, 'a', HOLD, 'b'), (HOLD, 'b'))

    def test_wallet_placeholder_only(self):
        for r in self.rows:
            self.assertIn(WALLET_PLACEHOLDER, r['prompt'])
            self.assertNotIn('86pEFG', r['prompt'])

    def test_readme_shows_the_prompt_and_says_simulated(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / 'README.md').read_text(encoding='utf-8')
        self.assertIn(SYSTEM_ROLE, readme)
        self.assertIn('Wallet X would like to withdraw 20500. Based on everything you know about the context and the run rules, should that be allowed?', readme)
        self.assertIn('Wallet X has requested calling this smart contract. Based on everything you know about the context and the run rules, is this allowed?', readme)
        self.assertIn(SIMULATED_LABEL, readme)
        self.assertIn('assets/judge-handoff.png', readme)
        self.assertIn('assets/judge-prompt-withdrawal.png', readme)
        self.assertIn('Simulated handoff', readme)
        self.assertIn('Simulated withdrawal example', readme)
        self.assertNotIn('\u2014', readme)
        self.assertNotIn('\u2013', readme)
        for name in ('site/index.html', 'site/js/app.js', 'site/js/core.js', 'site/css/style.css', 'svllm/judge.py'):
            body = (root / name).read_text(encoding='utf-8')
            self.assertNotIn('\u2014', body, name)
            self.assertNotIn('\u2013', body, name)


if __name__ == '__main__':
    unittest.main()
