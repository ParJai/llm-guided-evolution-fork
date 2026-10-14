#!/bin/bash
#SBATCH -A math
#SBATCH -q coe-ice
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out

RUN=${1:?usage: sbatch run_wine.sh <run_name>}

module load uv
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
export UV_NO_SYNC=1
export SBATCH_ACCOUNT=math
export SBATCH_QOS=coe-ice
export SERVER_HOSTNAME=$(hostname)

# Replication run: no RAG, no critic
export LLMGE_QISKIT_RAG=0
export LLMGE_CRITIC=0
export LLMGE_MIGRATION_GEN=${LLMGE_MIGRATION_GEN:-0}
export LLMGE_NUM_GENERATIONS=${LLMGE_NUM_GENERATIONS:-30}
export LLMGE_START_POPULATION=${LLMGE_START_POPULATION:-32}
export LLMGE_POPULATION=${LLMGE_POPULATION:-32}

uv run python run_improved.py \
    --checkpoints $RUN/island_llama3_QuantumVQC-Normal \
    --global_path $RUN/global_data \
    --llm_model llama3 \
    --prompt_group QuantumVQC/Normal