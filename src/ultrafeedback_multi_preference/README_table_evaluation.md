# Table Comparison Evaluation

This script implements a new evaluation metric that creates 3x3 comparison tables between three models (Base, REBEL, Ours) using pairwise reward comparisons.

## Overview

The evaluation generates one response from each model for each prompt in the validation dataset, then computes three different 3x3 tables:

1. **Table 1 (Helpfulness)**: Each entry (i,j) is averaged over x: `exp(r_1(x,y_i))/(exp(r_1(x,y_i))+exp(r_1(x,y_j)))` where `y_i` is from row model and `y_j` is from column model, and `r_1` is the helpfulness reward.

2. **Table 2 (Correctness)**: Each entry (i,j) is averaged over x: `exp(r_2(x,y_i))/(exp(r_2(x,y_i))+exp(r_2(x,y_j)))` where `r_2` is the correctness reward.

3. **Table 3 (Min of both)**: Each entry (i,j) is averaged over x: `min_i exp(r_i(x,y))/(exp(r_i(x,y))+exp(r_i(x,y')))` where we take the minimum of the helpfulness and correctness ratios.

## Usage

### Basic Usage

```bash
python table_comparison_evaluation.py \
    --models meta-llama/Llama-3.2-3B-Instruct zjhhhh/REBEL_1e4_ver2 zjhhhh/Multi_Preference_REBEL_1e4 \
    --input_dataset zjhhhh/Whole-Data-Llama-3.2-3B-Instruct-20_armo_tokenized_Llama-3.2-3B-Instruct_slice30 \
    --split test \
    --end_idx 100 \
    --output_dir ./table_evaluation_results \
    --output_dataset_name zjhhhh/table_comparison_evaluation_data
```

### Parameters

- `--models`: List of 3 models to compare (Base, REBEL, Ours)
- `--input_dataset`: The dataset to use for evaluation
- `--split`: Dataset split to use (default: "test")
- `--end_idx`: Limit number of prompts (default: -1 for all)
- `--reward_model`: Reward model for scoring (default: "RLHFlow/ArmoRM-Llama3-8B-v0.1")
- `--maxlen`: Maximum response length (default: 1024)
- `--world_size`: Number of GPUs (default: 1)
- `--rm_batch_size`: Batch size for reward model (default: 2)
- `--output_dir`: Directory to save results (default: "./table_evaluation_results")
- `--output_dataset_name`: Name for dataset upload to hub

## Outputs

The script generates:

1. **Heatmap visualization** (`pairwise_comparison_tables_*.png`): Visual representation of all three tables
2. **Text results** (`pairwise_tables_*.txt`): Numerical values of the three tables
3. **Dataset** (`comparison_dataset_*`): Contains responses and rewards for all models, ready for hub upload

### Dataset Structure

The output dataset contains:
- `prompt`: Original prompt
- `response_{model_name}`: Response from each model
- `reward_helpfulness_{model_name}`: Helpfulness reward for each model's response
- `reward_correctness_{model_name}`: Correctness reward for each model's response

## Uploading to Hub

After evaluation, upload the dataset using the helper script:

```bash
python upload_to_hub.py \
    --dataset_path ./table_evaluation_results/comparison_dataset_20241201_123456 \
    --hub_name your_username/your_dataset_name \
    --private
```

Or manually in Python:
```python
from datasets import Dataset
dataset = Dataset.load_from_disk("./table_evaluation_results/comparison_dataset_20241201_123456")
dataset.push_to_hub("your_username/your_dataset_name")
```

## Interpretation

- **Diagonal entries**: Always 0.5 (model compared to itself)
- **Off-diagonal entries**: Higher values indicate the row model is preferred over the column model
- **Table 1**: Focuses on helpfulness dimension
- **Table 2**: Focuses on correctness dimension  
- **Table 3**: Conservative metric taking minimum of both dimensions

Values closer to 1.0 indicate strong preference for the row model, while values closer to 0.0 indicate strong preference for the column model.
