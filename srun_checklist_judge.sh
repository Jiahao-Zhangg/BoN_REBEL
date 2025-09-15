#!/bin/bash

#SBATCH --job-name=checklist_judge
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=zhiweiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=checklist_judge_%j.out
#SBATCH --error=checklist_judge_%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Run the command using srun within the allocated job
srun --jobid $SLURM_JOB_ID bash -c 'source /home/jiahaoz4/miniconda3/etc/profile.d/conda.sh && conda activate new_vllm \
&& export HUGGINGFACE_HUB_TOKEN="$(cat /home/jiahaoz4/.cache/huggingface/token 2>/dev/null || true)" \
&& export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
&& if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then echo "Warning: No Hugging Face token found at ~/.cache/huggingface/token"; fi \
&& PERSIST_BASE=/project/flame/$USER \
&& export HF_HOME=${PERSIST_BASE}/hf_cache \
&& export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub \
&& export TRANSFORMERS_CACHE=${HF_HOME}/transformers \
&& export HF_DATASETS_CACHE=${HF_HOME}/datasets \
&& export XDG_CACHE_HOME=${PERSIST_BASE}/.cache \
&& mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
&& cd /home/jiahaoz4/BoN_REBEL \
&& echo "Conda environment activated: $CONDA_DEFAULT_ENV" \
&& echo "Changed to directory: $(pwd)" \
&& echo "Using persistent caches under $PERSIST_BASE" \
&& echo "Starting Python script..." \
&& python src/ultrafeedback_judge/checklist_judge_local_explanation_clean.py \
    --judge_model Qwen/Qwen3-14B \
    --judge_type preference_5score \
    --input_repo zjhhhh/human-scored-1.5B \
    --switch_position \
    --world_size 1 \
    --n_samples 5 \
    --temperature 0.6 \
    --top_p 0.95'

echo "Script completed at $(date)"
