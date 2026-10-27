"""Adapt the official HandsOn reader to the unchanged official OpenPI trainer."""
from bisect import bisect_right
import json
from pathlib import Path
import sys

import numpy as np

from paths import ROOT, DATASET_DIR, DATASET_READER_ROOT
sys.path.insert(0, str(DATASET_READER_ROOT))
from training_reader import HandsOnDataset


class TaskDataset:
    """Use only complete action chunks from the official session-level split."""

    def __init__(self, horizon=30, split='train'):
        self.reader = HandsOnDataset(DATASET_DIR, horizon=horizon, split=split)
        self.reader.set_epoch(0)
        self.offsets = [0]
        for row in self.reader.rows:
            self.offsets.append(self.offsets[-1] + max(0, row['frames']-horizon+1))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.offsets, index)-1
        source_index = self.reader.offsets[episode] + index-self.offsets[episode]
        sample = self.reader[source_index]
        assert not np.any(sample['action_is_pad'])
        return sample


def install_dataset_factory():
    """Select the official split-aware reader for this task's repo ID only."""
    import openpi.training.data_loader as data_loader
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if data_config.repo_id != 'handson_t09':
            return original(data_config, action_horizon, model_config)
        return TaskDataset(horizon=action_horizon)

    data_loader.create_torch_dataset = create


def compute_stats():
    """Use all training frames; validation never contributes normalization."""
    from openpi.shared import normalize
    reader = HandsOnDataset(DATASET_DIR, split='train', decode_images=False)
    values = {'state': [], 'actions': []}
    train_sessions = {r['collection_session_id'] for r in reader.rows}
    validation = HandsOnDataset(DATASET_DIR, split='validation', decode_images=False)
    assert train_sessions.isdisjoint({r['collection_session_id'] for r in validation.rows})
    for row in reader.rows:
        numeric = reader._numeric(row['data_path'])
        values['state'].append(numeric['observation.state'])
        values['actions'].append(numeric['action'])
    stats = {}
    for key, chunks in values.items():
        array = np.concatenate(chunks, axis=0).astype(np.float64)
        assert array.shape == (len(reader), 52) and np.isfinite(array).all()
        stats[key] = normalize.NormStats(mean=array.mean(axis=0), std=array.std(axis=0),
            q01=np.quantile(array, .01, axis=0), q99=np.quantile(array, .99, axis=0))
    output = ROOT/'assets/t09_insert_peg_pi05_lora/handson_t09'
    normalize.save(output, stats)
    receipt = dict(train_episodes=len(reader.rows), train_frames=len(reader),
                   validation_episodes=len(validation.rows), validation_frames=len(validation),
                   normalization_split='train only', disjoint_collection_sessions=True,
                   horizon=30, terminal_padding_policy='exclude incomplete action windows',
                   full_training_windows=len(TaskDataset()), full_validation_windows=len(TaskDataset(split='validation')))
    (ROOT/'data_validation.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
