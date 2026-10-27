"""Read HandsOn measured observations, command chunks, three cameras and prompts."""
from bisect import bisect_right
from functools import lru_cache
from fractions import Fraction
import json
from pathlib import Path

from prompt_sampler import PromptSampler


class HandsOnDataset:
    def __init__(self, root, *, horizon=16, split="train", language_seed=0,
                 prompt_split=None, decode_images=True):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.rows = [json.loads(s) for s in
                     (self.root / "source_manifest.jsonl").read_text().splitlines()]
        assert split in ("train", "validation", "all")
        self.rows = [r for r in self.rows if split == "all" or r["split"] == split]
        self.horizon = int(horizon)
        assert self.horizon > 0
        self.language = PromptSampler(self.root / "language/prompt_paraphrases.json",
                                      seed=language_seed)
        self.prompt_split = prompt_split or ("train" if split == "train" else "canonical")
        self.decode_images = bool(decode_images)
        self.epoch = 0
        self.offsets = [0]
        for row in self.rows:
            assert row["representation"] == "world_palm_command_eef52_v1"
            self.offsets.append(self.offsets[-1] + row["frames"])

    def __len__(self):
        return self.offsets[-1]

    def set_epoch(self, epoch):
        """Set before constructing each epoch's loader/worker replicas."""
        self.epoch = int(epoch)

    @lru_cache(maxsize=8)
    def _numeric(self, relative):
        import numpy as np
        import pyarrow.parquet as pq
        table = pq.read_table(self.root / relative,
                              columns=["observation.state", "action", "timestamp", "frame_index"])
        result = {}
        for name in ("observation.state", "action"):
            result[name] = np.asarray(table[name].to_pylist(), dtype=np.float32)
        for name in ("timestamp", "frame_index"):
            result[name] = table[name].to_numpy()
        assert result["observation.state"].shape == result["action"].shape
        assert result["action"].shape[1] == 52
        assert np.isfinite(result["observation.state"]).all()
        assert np.isfinite(result["action"]).all()
        return result

    def _image(self, relative, source_frame):
        import av
        import numpy as np
        fps = self.info["fps"]
        with av.open(str(self.root / relative)) as container:
            stream = container.streams.video[0]
            target = Fraction(int(source_frame), fps)
            container.seek(int(target / stream.time_base), stream=stream,
                           backward=True, any_frame=False)
            for frame in container.decode(stream):
                at = frame.pts * stream.time_base
                if at + Fraction(1, fps * 2) < target:
                    continue
                assert abs(at - target) < Fraction(1, fps * 2), (
                    relative, source_frame, float(at))
                return np.asarray(frame.to_image(), dtype=np.uint8)
        raise RuntimeError(f"Missing frame {source_frame} in {relative}")

    def __getitem__(self, index):
        return self.sample(index, epoch=self.epoch)

    def sample(self, index, *, epoch, draw_index=0):
        import numpy as np
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ordinal = bisect_right(self.offsets, index) - 1
        row = self.rows[ordinal]
        frame = index - self.offsets[ordinal]
        numeric = self._numeric(row["data_path"])
        assert len(numeric["action"]) == row["frames"]
        end = min(frame + self.horizon, row["frames"])
        length = end - frame
        action = np.repeat(numeric["action"][end - 1:end], self.horizon, axis=0)
        action[:length] = numeric["action"][frame:end]
        padding = np.arange(self.horizon) >= length
        active = np.concatenate([np.repeat(side in row["active_sides"], 26)
                                 for side in ("right", "left")])
        sample = {
            "observation.state": numeric["observation.state"][frame].copy(),
            "action": action,
            "action_is_pad": padding,
            "active_dimension_mask": active,
            "episode_index": row["episode_index"],
            "frame_index": int(numeric["frame_index"][frame]),
            "timestamp": float(numeric["timestamp"][frame]),
            "source_task": row["source_task"],
        }
        sample.update(self.language.sample(
            row["source_task"], row["prompt"], row["episode_uid"],
            sample["frame_index"], epoch=epoch, draw_index=draw_index, split=self.prompt_split))
        if self.decode_images:
            for camera, relative in row["video_paths"].items():
                sample["observation.images." + camera] = self._image(
                    relative, sample["frame_index"])
        return sample
