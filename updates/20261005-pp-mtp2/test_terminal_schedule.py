"""Run the real scheduler admission statements with synthetic drained states.

This reproduces the terminal async-MTP deadlock without loading model weights.
The full engine and token accounting are validated separately through the API.
"""
import ast
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

parser = argparse.ArgumentParser()
parser.add_argument('--before', type=Path, required=True)
parser.add_argument('--after', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('terminal-scheduler-tests.json'))
args = parser.parse_args()


def load_gate(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    loop = next(n for n in method.body if isinstance(n, ast.While))
    body = loop.body
    first = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                 and 'terminal_slots' in ast.unparse(n))
    last = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                and 'self.max_model_len' in ast.unparse(n))
    guard = next(n for n in body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == '0 < num_new_tokens < min_verification_tokens')
    return compile(ast.fix_missing_locations(ast.Module(body[first:last+1] + [guard], [])),
                   str(path), 'exec')


def run(code, length, remaining, asynchronous=True, placeholders=0, in_flight=0,
        drafts=2, input_budget=8192):
    computed = length - remaining - 1
    class Request(NS):
        @property
        def num_tokens_with_spec(self):
            return self.num_tokens + len(self.spec_token_ids)
    request = Request(num_computed_tokens=computed, num_tokens=computed+1,
                      spec_token_ids=[-1]*drafts, num_output_placeholders=placeholders,
                      num_in_flight_tokens=in_flight, is_prefill_chunk=False)
    scheduler = NS(use_v2_model_runner=True, max_model_len=length,
                   num_sampled_tokens_per_step=1,
                   scheduler_config=NS(async_scheduling=asynchronous, long_prefill_token_threshold=1280))
    scope = dict(self=scheduler, request=request, token_budget=8192,
                 input_budget=input_budget, draft_slots=2)
    exec(code, scope)
    return scope['num_new_tokens'], len(request.spec_token_ids)


before = load_gate(args.before)
after = load_gate(args.after)
assert run(before, 262144, 1) == (0, 2), 'Must reproduce pre-fix deadlock'
assert run(after, 262144, 1) == (1, 0)
checks = []
for length in (64, 262144):
    for drafts in (2, 5):
        for remaining in (1, 2, 3, 8):
            for asynchronous in (False, True):
                result = run(after, length, remaining, asynchronous, drafts=drafts)
                assert result == (min(remaining, drafts+1), min(remaining-1, drafts)), result
                if not asynchronous:
                    assert result == run(before, length, remaining, False, drafts=drafts)
                checks.append(dict(length=length, remaining=remaining, drafts=drafts,
                                   asynchronous=asynchronous, result=result))
    for placeholders, in_flight in ((1,0),(0,1),(3,3)):
        result = run(after, length, 1, placeholders=placeholders, in_flight=in_flight)
        assert result == run(before, length, 1, placeholders=placeholders, in_flight=in_flight)
        assert result[1] == 2, 'Never rewrite in-flight drafts'
        checks.append(dict(length=length, placeholders=placeholders, in_flight=in_flight, result=result))
    assert run(after,length,20,input_budget=4) == (0,2), 'Other partial-batch guards remain active'
report = dict(pass_count=len(checks)+4, regression_reproduced=True, checks=checks,
              scope='Actual scheduler draft/admission AST only; full live API validation is separate.')
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
