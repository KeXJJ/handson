"""Deterministic prompt sampling, independent of scene/camera/physics RNG."""
import hashlib
import json
from pathlib import Path


class PromptSampler:
    def __init__(self, bank_path, seed=0):
        self.bank = json.loads(Path(bank_path).read_text())
        self.seed = int(seed)
        self.by_task_prompt = {}
        for prompt_id, row in self.bank["prompts"].items():
            assert row["status"] == "generated"
            assert len(row["train"]) == 32 and len(row["unseen"]) == 8
            assert len(set(row["train"])) == 32 and len(set(row["unseen"])) == 8
            assert not set(row["train"]) & set(row["unseen"])
            assert all(isinstance(s, str) and s.strip()
                       for s in [row["canonical"], *row["train"], *row["unseen"]])
            key = (row["task"], row["canonical"])
            assert key not in self.by_task_prompt
            self.by_task_prompt[key] = (prompt_id, row)
        self.canonical_probability = self.bank["sampling"]["canonical_probability"]
        assert self.canonical_probability == 0.2

    def sample(self, task, canonical, episode_uid, frame_index, *, epoch=0,
               draw_index=0, split="train"):
        prompt_id, row = self.by_task_prompt[(task, canonical)]
        assert split in ("train", "canonical", "unseen")
        key = json.dumps(["v16-language-v1", self.seed, int(epoch), str(episode_uid),
                          int(frame_index), int(draw_index), split],
                         separators=(",", ":")).encode()
        value = hashlib.sha256(key).digest()
        if split == "canonical" or (split == "train" and
                int.from_bytes(value[:8], "big") / 2**64 < self.canonical_probability):
            text, variant = canonical, "canonical"
        else:
            choices = row["train" if split == "train" else "unseen"]
            index = int.from_bytes(value[8:16], "big") % len(choices)
            text, variant = choices[index], f"{split}:{index}"
        return dict(prompt=text, task=text, prompt_id=prompt_id, prompt_variant=variant,
                    language_seed=self.seed, language_epoch=int(epoch))
