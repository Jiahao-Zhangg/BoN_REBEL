#!/bin/bash

# Master script to submit all 4 multi-generate node jobs
# This will process all 51.5k data points across 26 jobs on 4 nodes

echo "=== Submitting Multi-Generate Jobs Across 4 Nodes ==="
echo "Total data: 51,500 items"
echo "Chunk size: 2,000 per job"
echo "Total jobs: 26 (distributed across 4 nodes)"
echo

# Make scripts executable
chmod +x run_multi_generate_iter2_node1.sh
chmod +x run_multi_generate_iter2_node2.sh
chmod +x run_multi_generate_iter2_node3.sh
chmod +x run_multi_generate_iter2_node4.sh

# Submit Node 1: Jobs 0-7 (data indices 0-15999)
echo "Submitting Node 1: Jobs 0-7 (data indices 0-15999)"
JOB1=$(sbatch run_multi_generate_iter2_node1.sh | awk '{print $4}')
echo "Node 1 Job ID: $JOB1"

# Submit Node 2: Jobs 8-15 (data indices 16000-31999)
echo "Submitting Node 2: Jobs 8-15 (data indices 16000-31999)"
JOB2=$(sbatch run_multi_generate_iter2_node2.sh | awk '{print $4}')
echo "Node 2 Job ID: $JOB2"

# Submit Node 3: Jobs 16-23 (data indices 32000-47999)
echo "Submitting Node 3: Jobs 16-23 (data indices 32000-47999)"
JOB3=$(sbatch run_multi_generate_iter2_node3.sh | awk '{print $4}')
echo "Node 3 Job ID: $JOB3"

# Submit Node 4: Jobs 24-25 (data indices 48000-51999)
echo "Submitting Node 4: Jobs 24-25 (data indices 48000-51999)"
JOB4=$(sbatch run_multi_generate_iter2_node4.sh | awk '{print $4}')
echo "Node 4 Job ID: $JOB4"


echo "=== All Jobs Submitted ==="
echo "Node 1 Job ID: $JOB1"
echo "Node 2 Job ID: $JOB2"
echo "Node 3 Job ID: $JOB3"
echo "Node 4 Job ID: $JOB4"
echo

echo "=== Monitoring Commands ==="
echo "Check job status: squeue -u \$USER"
echo "Watch all jobs: watch -n 5 'squeue -u \$USER'"
echo "Check specific job: squeue -j JOB_ID"
echo "Cancel all jobs: scancel $JOB1 $JOB2 $JOB3 $JOB4"
echo

echo "=== Expected Output Repositories ==="
echo "Each job will create output repos with format: Qwen3b_iter1_min_ver2_<start_idx>"
echo "Node 1: Qwen3b_iter1_multi_0, Qwen3b_iter1_multi_2000, Qwen3b_iter1_multi_4000, ..., Qwen3b_iter1_multi_14000"
echo "Node 2: Qwen3b_iter1_multi_16000, Qwen3b_iter1_multi_18000, Qwen3b_iter1_multi_20000, ..., Qwen3b_iter1_multi_30000"
echo "Node 3: Qwen3b_iter1_multi_32000, Qwen3b_iter1_multi_34000, Qwen3b_iter1_multi_36000, ..., Qwen3b_iter1_multi_46000"
echo "Node 4: Qwen3b_iter1_multi_48000, Qwen3b_iter1_multi_50000"
echo

echo "=== Log Files ==="
echo "Node 1 logs: logs/multi_generate_iter2_node1_$JOB1.out/err"
echo "Node 2 logs: logs/multi_generate_iter2_node2_$JOB2.out/err"
echo "Node 3 logs: logs/multi_generate_iter2_node3_$JOB3.out/err"
echo "Node 4 logs: logs/multi_generate_iter2_node4_$JOB4.out/err"
