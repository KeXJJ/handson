"""Fine-tune pi05 with HandsOn's official trainer and a task-specific config."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from task_data import ROOT, install_dataset_factory
from paths import DATASET_DIR, UPSTREAM_ROOT


def get_config(steps=1000, batch_size=1, exp_name='pilot_1000', resume=False):
    from openpi.training.handson_configs import HandsOnConfig, make_dual_arm_config
    from openpi.training import optimizer
    import dataclasses
    base = make_dual_arm_config(HandsOnConfig(
        name='t09_insert_peg_pi05_lora', checkpoint_base_dir=str(ROOT/'checkpoints'),
        data_root=DATASET_DIR, single_arm=False,
        base_img_name='observation.images.front',
        wrist_left_img_name='observation.images.wrist_left',
        wrist_right_img_name='observation.images.wrist_right',
        policy_action_dim=52, policy_state_dim=52, normalized_clip=3.0))
    return dataclasses.replace(base, exp_name=exp_name, batch_size=batch_size, num_workers=0 if steps <= 10 else 2,
        data=dataclasses.replace(base.data, repo_id='handson_t09'),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=str(ROOT/'models/pi05_base_eef52/params')),
        num_train_steps=steps, save_interval=100, keep_period=1000, log_interval=1 if steps <= 10 else 10,
        wandb_enabled=False, assets_base_dir=str(ROOT/'assets'), resume=resume,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(100,max(1,steps//10)),
            peak_lr=5e-5, decay_steps=max(steps,1000), decay_lr=5e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--exp-name', default='pilot_1000')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.steps > 0 and args.batch_size > 0
    import jax
    devices = jax.devices()
    assert devices and all(device.platform == 'gpu' for device in devices), devices
    print('TRAINING_GPU', devices, flush=True)
    install_dataset_factory()
    config = get_config(args.steps, args.batch_size, args.exp_name, args.resume)
    ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT/f'{args.exp_name}_config.json'
    config_path.write_text(json.dumps(dict(task='t09_insert_peg', source_task='b03_peg_in_socket',
        base='gs://openpi-assets/checkpoints/pi05_base', hands_on_finetuned_weights_published=False,
        implementation='Unmodified TsinghuaC3I/HandsOn openpi trainer and make_dual_arm_config',
        action_dim=52, state_dim=52, horizon=30, model='pi05', method='LoRA plus full vision/projection/time-MLP updates',
        trainable_modules={'paligemma_lora_rank':16,'action_expert_lora_rank':32,
            'vision_encoder':'full','action_projections':'full','time_mlp':'full'},
        image_processing='official OpenPI 224x224 resize with padding from native training RGB',
        steps=args.steps, batch_size=args.batch_size, seed=config.seed,
        checkpoint_dir=str(config.checkpoint_dir), official_recipe_changes=[
            'select t09 data only with official session split', 'batch size reduced for shared GPU',
            'short pilot schedule with shortened warmup', 'expand base action head 32 to 52',
            'exclude terminal chunks requiring padding from training loss'],
        evaluation='No closed-loop success claim; validation episodes excluded from training'), indent=2))
    source = UPSTREAM_ROOT/'scripts/train.py'
    spec = importlib.util.spec_from_file_location('handson_official_train',source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Save a few real initialized trainable tensors so checkpoint verification
    # can prove that optimizer updates changed weights, independent of loss logs.
    original_init = module.init_train_state

    def audited_init(*positional, **keywords):
        import flax.traverse_util
        import jax
        import numpy as np
        state, sharding = original_init(*positional, **keywords)
        if keywords.get('resume', False):
            return state, sharding
        flat = flax.traverse_util.flatten_dict(
            state.params.filter(config.trainable_filter).to_pure_dict(), sep='/')
        keys = sorted(key for key in flat if 'lora' in key)[:8]
        assert keys, 'No trainable LoRA tensors'
        keys.append('action_out_proj/kernel')
        np.savez(ROOT/f'{args.exp_name}_initial_tensors.npz',
                 **{key: np.asarray(jax.device_get(flat[key])) for key in keys})
        return state, sharding

    module.init_train_state = audited_init
    module.main(config)
    (ROOT/f'{args.exp_name}_complete.json').write_text(json.dumps(
        {'steps': args.steps, 'checkpoint_dir':str(config.checkpoint_dir), 'completed':True},indent=2))


if __name__ == '__main__':
    main()
