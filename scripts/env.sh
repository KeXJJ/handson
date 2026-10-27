#!/usr/bin/env bash
# Source this file so the unchanged upstream imports resolve without extra installs.
HANDSON_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HANDSON_WORK_DIR="${HANDSON_WORK_DIR:-$HANDSON_REPO_ROOT/workdir}"
export PYTHONPATH="$HANDSON_REPO_ROOT:$HANDSON_REPO_ROOT/third_party/handson/openpi/src:$HANDSON_REPO_ROOT/third_party/handson/openpi/packages/openpi-client/src:$HANDSON_REPO_ROOT/third_party/handson/legacy_lerobot${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HANDSON_WORK_DIR/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HANDSON_GPU:-0}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONFAULTHANDLER=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [[ "${HANDSON_DISABLE_AUTOTUNE:-1}" == 1 && "${XLA_FLAGS:-}" != *xla_gpu_autotune_level* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
fi
ulimit -c 0
cd "$HANDSON_REPO_ROOT"
