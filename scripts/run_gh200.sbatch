#!/bin/bash
#SBATCH --job-name=astrosim-fused-cuda
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=fused-%j.out
#SBATCH --error=fused-%j.err
#
# Runs astrosim_fused on a GH200 GPU node. Never invoke the binary directly
# on roihu-gpu.csc.fi's login shell -- it has no GPU driver and the program
# will refuse to start there; it only runs inside an allocation like this
# one (see the "No CUDA device visible" check in main()).
#
# The account is intentionally not hard-coded here -- export it once instead
# of editing every script:
#   export SBATCH_ACCOUNT=project_XXXXXXX
#
# Build first, from roihu-gpu.csc.fi:
#   module load nvhpc/26.3
#   make -C astrosimgpu_cuda -j
#
# Then, from the astrosimgpu_cuda directory:
#   sbatch scripts/run_gh200.sbatch
#
# Override the network size (and everything else astrosim_fused takes) on
# submission, e.g. a wider sweep across the same fixed-in-degree scaling law
# astrosimgpu/scripts/roihu/gen_neuron_scaling_configs.py uses:
#   sbatch --export=ALL,SIZES="500 5000 100000 10000000" scripts/run_gh200.sbatch

set -uo pipefail

module load nvhpc/26.3 2>/dev/null || true

BIN=./astrosim_fused
if [[ ! -x "$BIN" ]]; then
    echo "MISSING: $BIN -- build it first with 'make' from this directory." >&2
    exit 1
fi

echo "account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "host   : $(hostname)   arch: $(uname -m)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

SIZES="${SIZES:-500 5000 100000 1000000}"
SIM_MS="${SIM_MS:-5000}"
PRE_MS="${PRE_MS:-1000}"
SEED="${SEED:-1}"

printf "%-12s %14s %14s %14s %14s\n" "N_total" "wall_s" "us/step" "spikes" "mean_Hz"

for n in $SIZES; do
    out=$(srun --export=ALL "$BIN" --n "$n" --seed "$SEED" --pre-ms "$PRE_MS" --sim-ms "$SIM_MS" 2>&1)
    echo "$out" | sed 's/^/  /'
    wall=$(echo "$out" | awk -F'[: ]+' '/wall time/ {print $4}')
    usstep=$(echo "$out" | grep -oE '[0-9.]+ us/step' | awk '{print $1}')
    spikes=$(echo "$out" | awk -F': ' '/spikes \(recorded/ {print $2}')
    hz=$(echo "$out" | awk -F': ' '/mean firing rate/ {print $2}' | awk '{print $1}')
    printf "%-12s %14s %14s %14s %14s\n" "$n" "${wall:-FAILED}" "${usstep:-?}" "${spikes:-?}" "${hz:-?}"
done
