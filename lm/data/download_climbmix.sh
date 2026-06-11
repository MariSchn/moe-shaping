#!/bin/bash
# Download Nemotron-ClimbMix (climbmix_small) to shared capstor storage.
# Run directly on the login node: bash data/download_climbmix.sh
set -euo pipefail

source "$(dirname "$0")/../config.sh"

DATA_DIR="/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small"
LOGFILE="$WORKDIR/data/download_climbmix.log"

mkdir -p "$DATA_DIR"

{
echo "[$(date)] Starting download to $DATA_DIR"

uv run --with huggingface_hub python3 -c "
from huggingface_hub import snapshot_download
import shutil, glob, os

local_dir = '$DATA_DIR/tmp'
snapshot_download(
    'nvidia/Nemotron-ClimbMix',
    repo_type='dataset',
    allow_patterns='climbmix_small/*',
    local_dir=local_dir,
)

# Move parquet shards up
for f in glob.glob(os.path.join(local_dir, 'climbmix_small', '*.parquet')):
    shutil.move(f, '$DATA_DIR/')
shutil.rmtree(local_dir)

files = glob.glob('$DATA_DIR/*.parquet')
print(f'Downloaded {len(files)} shards to $DATA_DIR')
"

echo "[$(date)] Done."
} 2>&1 | tee "$LOGFILE"
