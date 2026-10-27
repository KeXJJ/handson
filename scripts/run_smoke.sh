#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RUNTIME="$HANDSON_REPO_ROOT/.venv/bin/python"
EXPERIMENT="${1:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo 'Use letters, digits, dot, underscore or hyphen for the experiment name.' >&2
    exit 1
fi
mkdir -p "$HANDSON_WORK_DIR/logs"
if [[ -e "$HANDSON_WORK_DIR/${EXPERIMENT}_config.json" || -e "$HANDSON_WORK_DIR/logs/${EXPERIMENT}.log" ]]; then
    echo 'Choose a new experiment name; existing run files will not be overwritten.' >&2
    exit 1
fi
JAX_PLATFORMS=cpu "$RUNTIME" prepare_stats.py
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

if [[ ! -s "$TOKENIZER" ]]; then
    "$RUNTIME" - <<'PY'
from pathlib import Path
from paths import ROOT
from download_assets import atomic_bytes,request_bytes
import os
path=Path(os.environ['OPENPI_DATA_HOME'])/'big_vision/paligemma_tokenizer.model'
atomic_bytes(path,request_bytes('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model'))
PY
fi
JAX_PLATFORMS=cpu "$RUNTIME" preflight.py
if [[ ! -f "$HANDSON_WORK_DIR/models/pi05_base_eef52/params/_METADATA" ||
      ! -f "$HANDSON_WORK_DIR/models/pi05_base_eef52/params/manifest.ocdbt" ]]; then
    if [[ ! -f "$HANDSON_WORK_DIR/base_download_receipt.json" ]]; then
        echo 'Download the base first: .venv/bin/python download_assets.py base' >&2
        exit 1
    fi
    JAX_PLATFORMS=cpu "$RUNTIME" convert_to_eef52.py \
        --input-path "$HANDSON_WORK_DIR/models/pi05_base" \
        --output-path "$HANDSON_WORK_DIR/models/pi05_base_eef52"
fi
set +e
"$RUNTIME" -u train_task.py --steps 5 --batch-size 1 --exp-name "$EXPERIMENT" \
    > "$HANDSON_WORK_DIR/logs/${EXPERIMENT}.log" 2>&1
TRAIN_EXIT=$?
set -e
"$RUNTIME" - "$EXPERIMENT" "$TRAIN_EXIT" <<'PY'
import json,sys
from paths import ROOT
(ROOT/'logs'/f'{sys.argv[1]}.exit.json').write_text(json.dumps({'exit_code':int(sys.argv[2])})+'\n')
PY
# A committed checkpoint can be checked independently even after a cleanup crash.
CHECKPOINT="$HANDSON_WORK_DIR/checkpoints/t09_insert_peg_pi05_lora/$EXPERIMENT/4"
if [[ -f "$CHECKPOINT/_CHECKPOINT_METADATA" && -f "$CHECKPOINT/params/_METADATA" ]]; then
    set +e
    JAX_PLATFORMS=cpu "$RUNTIME" verify_checkpoint.py --exp-name "$EXPERIMENT" --training-exit-code "$TRAIN_EXIT"
    VERIFY_EXIT=$?
    set -e
else
    echo "No committed checkpoint; inspect $HANDSON_WORK_DIR/logs/${EXPERIMENT}.log" >&2
    VERIFY_EXIT=1
fi
if (( TRAIN_EXIT != 0 )); then
    echo "Training process exited with $TRAIN_EXIT; do not treat the run as fully passed." >&2
    exit "$TRAIN_EXIT"
fi
exit "$VERIFY_EXIT"
