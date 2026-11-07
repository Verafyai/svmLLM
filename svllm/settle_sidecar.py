"""The settlement core as a sidecar: JSON lines over stdio, for Verafy Harness's
cross-runtime test (packages/tinysvm/src/svmllm.js).

    python -m svllm.settle_sidecar

First line out: {"hello": {...}}. Then one request per line in:
    {"id": 1, "method": "exit" | "bond" | "ledger", "input": <jsRuntime's input>}
and one reply per line out, in completion order, matched by id:
    {"id": 1, "result": {...runtime.js-shaped, plus "cu"}}
    {"id": 1, "unsupported": "why"}      an input this core does not encode (never a disagreement)
    {"id": 1, "error": "why"}            anything else

Every run is on the weights machine. Requests run in a process pool
(SVMLLM_WORKERS, default min(8, cpus)), one torch thread each. Integers within
+-(2^53 - 1) are JSON numbers and larger ones are decimal strings, which is exactly
what runtime.js's num() returns, so results compare as they are.
"""
import json, os, sys, threading
import multiprocessing as mp

SAFE = (1 << 53) - 1


def num(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x if -SAFE <= x <= SAFE else str(x)
    if isinstance(x, list):
        return [num(v) for v in x]
    if isinstance(x, dict):
        return {k: num(v) for k, v in x.items()}
    return x


def _init():
    import torch
    torch.set_num_threads(1)


def handle(req):
    from . import settle, settle_oracle
    rid = req.get('id')
    try:
        method, inp = req['method'], req['input']
        program, call = settle.call_of(method, inp)
        if program is None:
            if method == 'exit':
                return {'id': rid, 'result': {'ok': False, 'code': 1, 'cu': 0}}
            return {'id': rid, 'unsupported': f'the settlement core does not cover {method}'}
        if call.get('state') is None and program in ('graded_payout', 'milestone_escrow', 'ranked_split'):
            call = {**call, 'state': settle_oracle.initial_state(program)}
        return {'id': rid, 'result': num(settle.run_program(program, call, engine='weights'))}
    except settle.Unsupported as e:
        return {'id': rid, 'unsupported': str(e)}
    except Exception as e:                                  # noqa: BLE001 - reported to the caller
        return {'id': rid, 'error': f'{type(e).__name__}: {e}'}


def main():
    from . import settle
    workers = int(os.environ.get('SVMLLM_WORKERS', min(8, os.cpu_count() or 1)))
    pool = mp.get_context('spawn').Pool(workers, initializer=_init)
    lock = threading.Lock()

    def send(msg):
        with lock:
            sys.stdout.write(json.dumps(msg) + '\n')
            sys.stdout.flush()
    inv = settle.weights_inventory()
    send({'hello': {'engine': 'svmllm-weights', 'programs': sorted(settle.PROGRAMS),
                    'instructions': {k: len(v) // 8 for k, v in settle.PROGRAMS.items()},
                    'window': settle.WIN, 'workers': workers, 'weights_sha256': settle.weights_digest(),
                    'params': inv['settlement_machine_params']}})
    pending = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({'id': None, 'error': f'bad JSON: {e}'})
            continue
        pending.append(pool.apply_async(handle, (req,), callback=send,
                                        error_callback=lambda e, rid=req.get('id'): send({'id': rid, 'error': str(e)})))
    for p in pending:
        p.wait()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
