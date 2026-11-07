"""The settlement core: every named case on the reference interpreter against the oracle
(a transcription of Verafy Harness settle.js), a few on the weights machine, and the
weights inventory. The full suites are in run_settlement.py."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svllm import settle, settle_oracle, machine
from svllm.settle_cases import named_cases, random_cases


def strip(r):
    return {k: v for k, v in r.items() if k != 'cu'}


class SettlementTests(unittest.TestCase):
    def test_named_cases_match_the_oracle(self):
        for name, (p, c) in named_cases().items():
            with self.subTest(name):
                self.assertEqual(strip(settle.run_program(p, c, 'ref')), settle_oracle.run_program(p, c))

    def test_random_cases_match_the_oracle(self):
        for p, c in random_cases(100, seed=7):
            self.assertEqual(strip(settle.run_program(p, c, 'ref')), settle_oracle.run_program(p, c))

    def test_weights_equal_the_interpreter(self):
        cases = named_cases()
        for name in ['graded/43pct', 'graded/101pct', 'bond/slash-over', 'milestone/again', 'ledger/move']:
            p, c = cases[name]
            with self.subTest(name):
                self.assertEqual(settle.run_program(p, c, 'weights'), settle.run_program(p, c, 'ref'))

    def test_only_the_subset_decodes(self):
        prog = bytes([0xaf, 0x12, 0, 0, 0, 0, 0, 0]) + bytes([0x95] + [0] * 7)   # xor64 r2, r1; exit
        self.assertEqual(machine.run(prog, ops=settle.OPS, win=settle.WIN)['status'], 'InvalidInstruction')
        self.assertEqual(machine.run(prog)['status'], 'ok')

    def test_inventory_adds_up(self):
        inv = settle.weights_inventory()
        self.assertEqual(inv['svmllm_machine_params'], machine.PARAMS['count'])
        self.assertEqual(inv['settlement_machine_params'], sum(t['params'] for t in inv['tables'] if t['in_settlement_machine']))

    def test_layout_limits_refuse_before_running(self):
        p, c = named_cases()['ledger/move']
        big = {**c, 'capacity': settle.LEDGER_ROWS + 1}
        with self.assertRaises(settle.Unsupported):
            settle.run_program(p, big, 'ref')


if __name__ == '__main__':
    unittest.main()
