"""Incremental feedback -> exact gradient update -> validation gate -> frozen test."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .client import CachedChatClient
from .evaluation import compare_paired, evaluate_matrix
from .policy import FEATURE_NAMES, LinearPolicy, extract_features
from .strategies import STRATEGIES, build_messages, build_review_messages
from .tasks import Task, extract_answer, is_correct, load_gsm8k


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def collect_task(task, split, client, directory):
    """Three actual requests measure routes whose deployment costs are 1/1/2."""
    path = directory / f'{task.id}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['task'] != task.to_dict():
            raise ValueError('Task cache does not match this dataset')
        return saved
    responses = {}
    for action in ('direct', 'reason'):
        responses[action] = client.complete(build_messages(task.question, action), tag=f'{split}:{task.id}:{action}')
    responses['review'] = client.complete(
        build_review_messages(task.question, responses['reason']['text']), tag=f'{split}:{task.id}:review')
    outcomes = {}
    for action in STRATEGIES:
        response = responses[action]
        steps = [response] if action != 'review' else [responses['reason'], response]
        answer = extract_answer(response['text'])
        outcomes[action] = {
            'correct': int(is_correct(response['text'], task.answer)),
            'prediction': str(answer) if answer is not None else None,
            'text': response['text'],
            'input_tokens': sum(s['input_tokens'] for s in steps),
            'output_tokens': sum(s['output_tokens'] for s in steps),
            'total_tokens': sum(s['total_tokens'] for s in steps),
            'reasoning_tokens': sum(s['reasoning_tokens'] for s in steps),
            'calls': len(steps),
            'latency_s': sum(s['latency_s'] for s in steps),
            'request_ids': [s['request_id'] for s in steps],
            'finish_reason': response['finish_reason'],
        }
    row = {'task': task.to_dict(), 'split': split, 'outcomes': outcomes}
    save_json(path, row)
    return row


def collect(tasks, split, client, output, workers):
    directory = output / 'observations' / split
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f'Collecting {split}: {len(tasks)} tasks; requests so far={client.requests_made}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_task, task, split, client, directory): task for task in tasks}
        try:
            for future in as_completed(futures):
                task = futures[future]
                results[task.id] = future.result()
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(f'{split}: {len(results)}/{len(tasks)} tasks; requests={client.requests_made}', flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return [results[task.id] for task in tasks]


def matrices(rows):
    fields = ('correct', 'total_tokens', 'calls', 'latency_s')
    return tuple(np.array([[row['outcomes'][a][field] for a in STRATEGIES] for row in rows], dtype=float)
                 for field in fields)


def metrics(matrix, actions, config):
    return evaluate_matrix(*matrix, actions=actions,
                           cost_weight=config['cost_weight'], token_unit=config['token_unit'])


def ledger_summary(path):
    events = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    successes = [x for x in events if x['event'] == 'success']
    return {
        'attempts': sum(x['event'] == 'attempt' for x in events),
        'successes': len(successes), 'failures': sum(x['event'] == 'failure' for x in events),
        **{key: sum(x[key] for x in successes) for key in ('input_tokens', 'output_tokens', 'total_tokens')},
        'failed_request_cost': 'unknown; any provider-billed failed requests are not measurable here',
    }


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    credentials = json.loads(Path(args.credentials).read_text(encoding='utf-8')) if args.credentials else {}
    base_url = args.base_url or credentials.get('base_url') or os.getenv('HARNESS_BASE_URL')
    model = args.model or credentials.get('model') or os.getenv('HARNESS_MODEL')
    key = os.getenv(args.key_env) or credentials.get('api_key')
    if not all((base_url, model, key)):
        raise ValueError('Provide HARNESS_BASE_URL, HARNESS_MODEL, HARNESS_API_KEY or --credentials')
    base_url = base_url.rstrip('/')
    for suffix in ('/chat/completions', '/messages'):
        if base_url.endswith(suffix):
            base_url = base_url[:-len(suffix)]
    config = {
        'dataset': 'GSM8K (official train/test)', 'model': model, 'base_url': base_url,
        'train_size': args.train_size, 'val_size': args.val_size, 'test_size': args.test_size,
        'seed': args.seed, 'rounds': args.rounds, 'steps': args.steps, 'lr': .2,
        'entropy': .01, 'l2': .03, 'cost_weight': args.cost_weight, 'token_unit': 1000,
        'max_requests': args.max_requests, 'max_tokens': args.max_tokens,
        'gate_accuracy_tolerance': .03, 'gate_utility_margin': 1e-6,
        'features': list(FEATURE_NAMES), 'strategies': list(STRATEGIES),
        'deployment_rule': 'argmax', 'temperature': 0,
        'draft_sharing': 'reason output is reused as the first review step; deployment counts both steps',
        'source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in [('train', args.train_data), ('test', args.test_data)]},
        'implementation_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in Path(__file__).parent.glob('*.py') if path.name != 'report.py'},
    }
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing run has a different configuration; use a new output directory')
    save_json(config_path, config)
    splits = load_gsm8k(args.train_data, args.test_data, train_size=args.train_size,
                       val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    save_json(output / 'split_manifest.json', {k: [t.to_dict() for t in tasks] for k, tasks in splits.items()})
    client = CachedChatClient(base_url, model, key, output / 'cache', output / 'requests.jsonl',
                              max_requests=args.max_requests, max_tokens=args.max_tokens,
                              timeout=args.timeout, max_parallel=args.workers)
    validation = collect(splits['val'], 'val', client, output, args.workers)
    vm = matrices(validation)
    vx = np.array([extract_features(t.question) for t in splits['val']])
    policy = LinearPolicy(len(FEATURE_NAMES), len(STRATEGIES), seed=args.seed)
    policy.weights[:] = 0
    policy.bias[1] = .5
    incumbent_metrics = metrics(vm, np.ones(len(validation), dtype=int), config)
    initial_val_accuracy = incumbent_metrics['accuracy']
    history = []
    training = []
    selected_version = 0
    started = time.perf_counter()
    for round_index, batch in enumerate(np.array_split(np.arange(len(splits['train'])), args.rounds), 1):
        tasks = [splits['train'][int(i)] for i in batch]
        training.extend(collect(tasks, 'train', client, output, args.workers))
        tm = matrices(training)
        tx = np.array([extract_features(row['task']['question']) for row in training])
        rewards = tm[0] - config['cost_weight'] * tm[1] / config['token_unit']
        candidate = LinearPolicy.from_dict(policy.to_dict())
        objectives = candidate.fit(tx, rewards, steps=args.steps, lr=config['lr'],
                                    entropy=config['entropy'], l2=config['l2'])
        candidate_metrics = metrics(vm, candidate.predict(vx), config)
        # Guard against accumulating a 3pp accuracy loss at every iteration.
        accuracy_floor = max(initial_val_accuracy, incumbent_metrics['accuracy']) - config['gate_accuracy_tolerance']
        accepted = (candidate_metrics['utility'] > incumbent_metrics['utility'] + config['gate_utility_margin']
                    and candidate_metrics['accuracy'] >= accuracy_floor)
        entry = {'round': round_index, 'train_tasks': len(training), 'accepted': bool(accepted),
                 'val_incumbent': incumbent_metrics, 'val_candidate': candidate_metrics,
                 'objective_start': objectives[0], 'objective_end': objectives[-1],
                 'reason': 'validation utility improved and accuracy guard passed' if accepted else 'validation gate rejected candidate'}
        history.append(entry)
        save_json(output / 'versions' / f'candidate_{round_index}.json', candidate.to_dict())
        if accepted:
            policy = candidate
            incumbent_metrics = candidate_metrics
            selected_version = round_index
        save_json(output / 'rounds.json', history)
        print(f'Round {round_index}: accepted={accepted}, val utility={candidate_metrics["utility"]:.5f}, '
              f'val accuracy={candidate_metrics["accuracy"]:.3f}', flush=True)
    fixed_values = [metrics(vm, np.full(len(validation), a), config) for a in range(3)]
    best_fixed = int(np.argmax([m['utility'] for m in fixed_values]))
    # This manifest exists BEFORE any test response is requested.
    freeze = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'policy': policy.to_dict(),
              'selected_version': selected_version, 'best_fixed': STRATEGIES[best_fixed],
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    freeze_path = output / 'frozen_policy.json'
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text(encoding='utf-8'))
        if any(old[k] != freeze[k] for k in ('policy', 'selected_version', 'best_fixed', 'config_sha256')):
            raise ValueError('Frozen policy changed; refusing to reuse the final test')
    else:
        save_json(freeze_path, freeze)
    print(f'Frozen version={selected_version}; best fixed={STRATEGIES[best_fixed]}; starting final test', flush=True)
    test = collect(splits['test'], 'test', client, output, args.workers)
    test_matrix = matrices(test)
    test_x = np.array([extract_features(t.question) for t in splits['test']])
    chosen = policy.predict(test_x)
    n = len(test)
    fixed_actions = {f'fixed_{name}': np.full(n, index) for index, name in enumerate(STRATEGIES)}
    actions = {'initial_reason': np.ones(n, dtype=int), **fixed_actions,
               'best_fixed': np.full(n, best_fixed), 'learned': chosen,
               'uniform': np.full((n, 3), 1 / 3),
               'oracle': np.argmax(test_matrix[0] - config['cost_weight'] * test_matrix[1] / config['token_unit'], axis=1)}
    measured = {name: metrics(test_matrix, selection, config) for name, selection in actions.items()}
    indices = np.arange(n)
    comparisons = {}
    for label, baseline in [('vs_initial', 1), ('vs_best_fixed', best_fixed)]:
        comparisons[label] = compare_paired(test_matrix[0][indices, chosen], test_matrix[1][indices, chosen],
                                             test_matrix[0][:, baseline], test_matrix[1][:, baseline],
                                             cost_weight=config['cost_weight'], token_unit=config['token_unit'], seed=args.seed)
    summary = {
        'run_name': output.name, 'config': config, 'metrics': measured,
        'best_fixed_strategy': STRATEGIES[best_fixed], 'comparisons': comparisons,
        'ledger': ledger_summary(output / 'requests.jsonl'), 'rounds': history,
        'selected_actions': {name: int(np.sum(chosen == i)) for i, name in enumerate(STRATEGIES)},
        'review_effect': {'fixes': int(np.sum((test_matrix[0][:, 1] == 0) & (test_matrix[0][:, 2] == 1))),
                          'breaks': int(np.sum((test_matrix[0][:, 1] == 1) & (test_matrix[0][:, 2] == 0)))},
        'validation_best_fixed_metrics': fixed_values[best_fixed], 'selected_version': selected_version,
        'wall_seconds': time.perf_counter() - started,
        'format_failures': {name: sum(row['outcomes'][name]['prediction'] is None for row in test) for name in STRATEGIES},
        'truncated_outputs': {name: sum(row['outcomes'][name]['finish_reason'] == 'length' for row in test) for name in STRATEGIES},
    }
    save_json(output / 'summary.json', summary)
    save_json(output / 'test_predictions.json', [{'task_id': row['task']['id'], 'action': STRATEGIES[int(a)],
                                                **row['outcomes'][STRATEGIES[int(a)]]} for row, a in zip(test, chosen)])
    from .report import write_report
    write_report(summary, output)
    print(json.dumps({'report': str(output / 'REPORT.md'), 'metrics': measured, 'requests': client.requests_made}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', help='Local JSON (ignored by Git) with api_key,base_url,model')
    parser.add_argument('--base-url')
    parser.add_argument('--model')
    parser.add_argument('--key-env', default='HARNESS_API_KEY')
    parser.add_argument('--train-data', default='data/raw/train.jsonl')
    parser.add_argument('--test-data', default='data/raw/test.jsonl')
    parser.add_argument('--output', default='artifacts/gsm8k_flash_v1')
    parser.add_argument('--train-size', type=int, default=128)
    parser.add_argument('--val-size', type=int, default=48)
    parser.add_argument('--test-size', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cost-weight', type=float, default=.05)
    parser.add_argument('--max-requests', type=int, default=980)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.test_size, args.rounds) < 1 or args.rounds > args.train_size:
        parser.error('Split sizes and rounds must be positive; rounds cannot exceed training tasks')
    if 3 * (args.train_size + args.val_size + args.test_size) > args.max_requests:
        parser.error('Full-action collection requires three requests per task, exceeding the request cap')
    run(args)


if __name__ == '__main__':
    main()
