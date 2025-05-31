#!/bin/bash
#SBATCH --job-name=sbatch_run_pytorch
#SBATCH --output=sbatch_run_pytorch_%j.out
##SBATCH --error=sbatch_run_pytorch_%j.err
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p mi3008x
#SBATCH -q alloc_diwu_05142024_06302025

set -x

cd $SLURM_SUBMIT_DIR
CONTAINER_HOME=$SLURM_SUBMIT_DIR

COUNTERS_PROFILE=1

SIF_FILE=rocm-vllm-container.sif

export TORCHINDUCTOR_COMPILE_THREADS=1

SRUN_APPTAINER_ARGS=(
        srun
        apptainer exec
        --rocm
        --bind
        /opt/rocm-6.3.1/:/opt/rocm-6.3.1/
        "${CONTAINER_HOME}/${SIF_FILE}"
)

METRICS=(
        "--pmc SQ_VALU_MFMA_BUSY_CYCLES,GRBM_GUI_ACTIVE -d MFMA"
)

PROF_BIN=/opt/rocm-6.3.1/bin/rocprofv3

VLLM_ARGS=(
        python examples/offline_inference/llm_engine_example.py
        --model meta-llama/Llama-3.1-8B
)

echo "RUNANDTIME_START $(date +%s)"
if [[ $COUNTERS_PROFILE -eq 1 ]]; then
        for metric in "${METRICS[@]}"; do
                "${SRUN_APPTAINER_ARGS[@]}" $PROF_BIN $metric -- "${VLLM_ARGS[@]}"
        done
else
        "${SRUN_APPTAINER_ARGS[@]}" "${VLLM_ARGS[@]}"
fi
echo "RUNANDTIME_STOP $(date +%s)"
