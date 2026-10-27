"""Reload a real training checkpoint and distinguish core checks from clean exit."""
import argparse
import json
import math
import re

import flax.traverse_util
import numpy as np
from openpi.models import model
from paths import ROOT,DATASET_DIR


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-name',required=True)
    parser.add_argument('--training-exit-code',type=int)
    args=parser.parse_args()
    config=json.loads((ROOT/f'{args.exp_name}_config.json').read_text())
    from pathlib import Path
    checkpoint=Path(config['checkpoint_dir'])/str(config['steps']-1)
    params=model.restore_params(str(checkpoint/'params'),restore_type=np.ndarray)
    flat=flax.traverse_util.flatten_dict(params,sep='/')
    changes={}
    with np.load(ROOT/f'{args.exp_name}_initial_tensors.npz') as initial:
        for key in initial.files:
            current=flat[key]
            assert np.isfinite(current).all(),key
            changes[key]=float(np.abs(current.astype(np.float32)-initial[key].astype(np.float32)).max())
    assert max(changes.values())>0,changes
    expert_changes={key:value for key,value in changes.items() if '_1/' in key and 'lora' in key}
    assert expert_changes and max(expert_changes.values())>0,'Action expert LoRA did not update'
    assert params['action_in_proj']['kernel'].shape[0]==52
    assert params['action_out_proj']['kernel'].shape[1]==52
    assert params['action_out_proj']['bias'].shape[0]==52
    log=(ROOT/'logs'/f'{args.exp_name}.log').read_text()
    metrics=[]
    for match in re.finditer(r'Step (\d+): ([^\r\n]+)',log):
        values={key:float(value) for key,value in re.findall(r'(loss|grad_norm|param_norm)=([0-9.eE+\-]+)',match[2])}
        if len(values)==3:
            assert all(math.isfinite(value) for value in values.values())
            metrics.append({'step':int(match[1]),**values})
    interval=1 if config['steps']<=10 else 10
    assert {row['step'] for row in metrics}==set(range(0,config['steps'],interval)),metrics
    assert any(row['grad_norm']>0 for row in metrics)
    exit_code=args.training_exit_code
    if exit_code is None:
        status_path=ROOT/'logs'/f'{args.exp_name}.exit.json'
        if status_path.exists():exit_code=json.loads(status_path.read_text())['exit_code']
        elif 'Fatal Python error: Segmentation fault' in log:exit_code=139
    clean_exit=exit_code==0 and (ROOT/f'{args.exp_name}_complete.json').exists()
    report={'passed':clean_exit,'core_finetuning_checks_passed':True,
            'status':'passed' if clean_exit else 'core_checks_passed_but_runtime_not_clean',
            'training_process_exit_code':exit_code,'clean_training_process_exit':clean_exit,
            'checkpoint_reloaded':True,'checkpoint':str(checkpoint),
            'completed_training_steps':config['steps'],'batch_size':config['batch_size'],
            'action_expert_lora_updated':True,'max_abs_trainable_tensor_changes':changes,
            'training_metrics':metrics,'state_dim':52,'action_dim':52,'horizon':30,
            'dataset':json.loads((DATASET_DIR/'subset_receipt.json').read_text()),
            'scope':'Training execution check only; no convergence or task-success evaluation'}
    output=ROOT/'results'/f'{args.exp_name}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not clean_exit:raise SystemExit(2)


if __name__=='__main__':main()
