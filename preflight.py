"""Check official model preprocessing, episode split and optional real CUDA execution."""
import argparse
import json
import subprocess

from task_data import ROOT,TaskDataset,install_dataset_factory
from paths import UPSTREAM_ROOT
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    import jax
    import numpy as np
    from train_task import get_config
    install_dataset_factory()
    from openpi.training import data_loader
    from openpi.training import config as official_config
    assert Path(official_config.__file__).resolve().is_relative_to(UPSTREAM_ROOT),official_config.__file__
    config=get_config(5,1,'preflight')
    data_config=config.data.create(config.assets_dirs,config.model)
    dataset=TaskDataset()
    item=data_loader.transform_dataset(dataset,data_config)[0]
    assert item['actions'].shape==(30,52)
    assert item['state'].shape==(52,)
    assert item['tokenized_prompt'].shape==(250,)
    assert all(image.shape==(224,224,3) for image in item['image'].values())
    assert np.isfinite(item['actions']).all()
    result={'input_shapes_passed':True,'training_windows':len(dataset),
            'model':'pi05','state_dim':52,'action_dim':52,'horizon':30,
            'action_expert_variant':config.model.action_expert_variant,
            'paligemma_variant':config.model.paligemma_variant,
            'gpu_checked':args.gpu,'jax':jax.__version__,'vendored_import_verified':True}
    if args.gpu:
        assert jax.default_backend()=='gpu',jax.devices()
        import jax.numpy as jnp
        matrix=jnp.ones((128,128),jnp.float32)
        assert float((matrix@matrix)[0,0])==128
        result['devices']=[str(device) for device in jax.devices()]
        result['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.free','--format=csv'],text=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
