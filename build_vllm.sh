#!/bin/bash
#SBATCH --job-name=sbatch_build_pytorch
#SBATCH --output=sbatch_build_pytorch_%j.out
##SBATCH --error=sbatch_build_pytorch_%j.err
#SBATCH -t 02:30:00
##SBATCH -p mi3008x
#SBATCH -p mi2104x
#SBATCH -q alloc_diwu_05142024_06302025

set -x

cd $SLURM_SUBMIT_DIR

apptainer pull rocm-vllm-container.sif docker://rocm/vllm
