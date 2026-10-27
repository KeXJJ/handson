"""Verify byte-identical vendored upstream files against the recorded manifest."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'UPSTREAM.json').read_text())
for item in manifest['files']:
    digest=hashlib.sha256((root/item['path']).read_bytes()).hexdigest()
    if digest != item['sha256']:
        raise RuntimeError(f"Upstream source changed: {item['path']}")
print(f"Verified {len(manifest['files'])} upstream files at {manifest['hands_on_commit']}")
