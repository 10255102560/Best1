"""Single auxiliary convergence check; uses only first 64 train and validation tasks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from harness_rsi.policy import FEATURE_NAMES, LinearPolicy, extract_features
from harness_rsi.evaluation import evaluate_matrix
from harness_rsi.strategies import STRATEGIES


def main():
    source = ROOT / 'artifacts/gsm8k_flash_v1'
    output = ROOT / 'artifacts/convergence_check'
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / 'frozen_policy.json'
    if frozen_path.exists():
        raise SystemExit('Auxiliary check already frozen; refusing to overwrite it.')
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'split_manifest.json').read_text(encoding='utf-8'))
    train_tasks = manifest['train'][:64]
    val_tasks = manifest['val']
    del manifest  # No test questions, outcomes, or summary are used.
    def rows(tasks, split):
        return [json.loads((source / 'observations' / split / f"{t['id']}.json").read_text(encoding='utf-8')) for t in tasks]
    def matrices(records):
        return tuple(np.asarray([[r['outcomes'][a][field] for a in STRATEGIES] for r in records], dtype=float) for field in ('correct','total_tokens','calls','latency_s'))
    training = rows(train_tasks, 'train')
    validation = rows(val_tasks, 'val')
    vm = matrices(validation)
    vx = np.asarray([extract_features(t['question']) for t in val_tasks])
    options = {'cost_weight':config['cost_weight'], 'token_unit':config['token_unit']}
    def metrics(actions):
        return evaluate_matrix(*vm, actions=actions, **options)
    policy = LinearPolicy(len(FEATURE_NAMES), len(STRATEGIES), seed=config['seed'])
    policy.weights[:] = 0
    policy.bias[1] = .5
    initial = metrics(np.ones(len(validation),dtype=int))
    incumbent = initial
    history = []
    selected_version = 0
    for round_index, size in enumerate((32,64),1):
        tm = matrices(training[:size])
        tx = np.asarray([extract_features(r['task']['question']) for r in training[:size]])
        rewards = tm[0] - config['cost_weight'] * tm[1] / config['token_unit']
        candidate = LinearPolicy.from_dict(policy.to_dict())
        objectives = candidate.fit(tx,rewards,steps=4000,lr=.2,entropy=.01,l2=.03)
        actions = candidate.predict(vx)
        candidate_metrics = metrics(actions)
        accepted = bool(candidate_metrics['utility'] > incumbent['utility'] + config['gate_utility_margin'] and candidate_metrics['accuracy'] >= max(initial['accuracy'],incumbent['accuracy']) - config['gate_accuracy_tolerance'])
        history.append({'round':round_index,'train_tasks':size,'steps':4000,'accepted':accepted,'val_incumbent':incumbent,'val_candidate':candidate_metrics,'candidate_actions':{a:int(np.sum(actions==i)) for i,a in enumerate(STRATEGIES)},'objective_start':objectives[0],'objective_end':objectives[-1]})
        if accepted:
            policy = candidate
            incumbent = candidate_metrics
            selected_version = round_index
    frozen_at = datetime.now(timezone.utc).isoformat()
    frozen = {'frozen_at':frozen_at,'policy':policy.to_dict(),'selected_version':selected_version,'steps_per_round':4000,'source_run':str(source.relative_to(ROOT)),'source_config':config,'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),'train_ids':[t['id'] for t in train_tasks],'validation_ids':[t['id'] for t in val_tasks],'rounds':history,'initial_validation':initial,'selected_validation':incumbent,'selected_actions':{a:int(np.sum(policy.predict(vx)==i)) for i,a in enumerate(STRATEGIES)},'disclosure':'One additional 4000-step convergence check chosen after seeing early validation results. Not originally preregistered. No test outcomes or summary were read; only the first 64 training tasks and validation set were used.'}
    with frozen_path.open('x',encoding='utf-8') as handle:
        json.dump(frozen,handle,ensure_ascii=False,indent=2)
    # Ledger contains request metadata, not test answer text. This establishes
    # whether freezing finished before the first final-test request was sent.
    freeze_completed = datetime.now(timezone.utc)
    test_times = []
    for line in (source/'requests.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event.get('event')=='attempt' and event.get('tag','').startswith('test:'):
            test_times.append(datetime.fromisoformat(event['timestamp']))
    eligible = not test_times or min(test_times) >= freeze_completed
    audit={'freeze_completed_at':freeze_completed.isoformat(),'first_test_request_at':min(test_times).isoformat() if test_times else None,'frozen_before_test_requests':eligible,'interpretation':'Auxiliary policy frozen before final test requests; still selected after early validation and not originally preregistered.' if eligible else 'Test requests had already started before this freeze; cannot be labeled a confirmatory pre-test-frozen auxiliary check.','frozen_policy_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    (output/'freeze_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'audit':audit,'rounds':history,'selected_validation':incumbent,'selected_actions':frozen['selected_actions']},indent=2))


if __name__=='__main__':
    main()
