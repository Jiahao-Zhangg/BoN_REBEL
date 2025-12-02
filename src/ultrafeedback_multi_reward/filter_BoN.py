import argparse
import re
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from tqdm import tqdm

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="BoN filter: compute multi-dimensional LSE g-values per selection and expand pairwise."
    )
    parser.add_argument("--input_repo", type=str, required=True,
                        help="HF dataset repo pushed by preprocess_common.py (expects response_i and response_i_reward)")
    parser.add_argument("--selection_pairs", type=int, required=True,
                        help="Number of selection responses per row (first indices after sorting)")
    parser.add_argument("--M", type=int, required=True,
                        help="Number of current responses per BoN group; total current responses = BoN * M")
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use as repo prefix for push_to_hub; otherwise reuse input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split for debugging")
    parser.add_argument("--BoN", type=int, default=1,
                        help="Best of N; number of current groups to average")
    parser.add_argument("--base_pairs", type=int, default=1,
                        help="Number of base responses")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta parameter for beta-BoN (temperature > 0)")
    return parser.parse_args()


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def logsumexp_axis0(matrix: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-sum-exp along axis 0 for a 2D array.
    """
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    max_vals = np.max(matrix, axis=0)
    # Handle all -inf case
    stable = np.where(np.isfinite(max_vals), max_vals, 0.0)
    summed = np.sum(np.exp(matrix - stable), axis=0)
    with np.errstate(divide='ignore'):
        result = stable + np.log(summed)
    return result


def main():
    args = parse_arguments()

    # Load both splits from preprocessed dataset
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    # Optionally limit rows per split
    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

    def process_split(dataset):
        print('split length:', len(dataset))

        # Identify response indices present (response_i and response_i_reward)
        resp_pat = re.compile(r'^response_(\d+)$')
        response_ids_all = sorted([
            int(m.group(1))
            for name in dataset.column_names
            if (m := resp_pat.match(name)) and (f"response_{m.group(1)}_reward" in dataset.column_names)
        ])
        if not response_ids_all:
            raise ValueError("Dataset must contain 'response_i' and matching 'response_i_reward' columns.")

        total_needed = args.selection_pairs + args.BoN * args.M + args.base_pairs
        if len(response_ids_all) < total_needed:
            raise ValueError(f"Found {len(response_ids_all)} responses but require at least {total_needed} "
                             f"(selection_pairs + BoN * M + base_pairs).")

        selection_ids = response_ids_all[:args.selection_pairs]
        current_ids = response_ids_all[args.selection_pairs:args.selection_pairs + args.BoN * args.M]
        base_ids = response_ids_all[args.selection_pairs + args.BoN * args.M:
                                    args.selection_pairs + args.BoN * args.M + args.base_pairs]

        def row_generator():
            for row in tqdm(dataset):
                # Build reward matrices
                try:
                    selection_rewards = [np.array(row[f"response_{sid}_reward"], dtype=float) for sid in selection_ids]
                    current_rewards = [np.array(row[f"response_{cid}_reward"], dtype=float) for cid in current_ids]
                    base_rewards = [np.array(row[f"response_{bid}_reward"], dtype=float) for bid in base_ids]
                except KeyError as e:
                    raise ValueError(f"Missing reward column: {e}") from e

                if not selection_rewards or not current_rewards or not base_rewards:
                    # Require both selections and gradients
                    continue
                if len(base_rewards) != args.base_pairs:
                    # Unexpected number of base responses; skip this row
                    continue
                if args.beta <= 0:
                    raise ValueError("beta must be > 0 for beta-BoN.")

                # Validate consistent reward dimension H
                reward_dims = [arr.shape[0] for arr in selection_rewards + current_rewards + base_rewards]
                if len(set(reward_dims)) != 1:
                    # Skip malformed row with inconsistent reward dimensions
                    continue
                H = reward_dims[0]

                sel_mat = np.vstack(selection_rewards)  # (S, H)
                # Build gradient tensor with shape (M, H, BoN)
                expected_gradients = args.BoN * args.M
                if len(current_rewards) != expected_gradients:
                    # Skip malformed row with unexpected number of gradient entries
                    continue
                # Stack all gradients then reshape into (BoN, M, H) and transpose to (M, H, BoN)
                grads_stacked = np.vstack([g[None, :] for g in current_rewards])  # (BoN*M, H)
                grads_bgn = grads_stacked.reshape(args.BoN, args.M, H)  # (BoN, M, H)
                grad_mat = np.transpose(grads_bgn, (1, 2, 0))  # (M, H, BoN)

                # r_hat = argmin over dimensions of averaged LSE across BoN using gradients
                # For each h, compute mean over i of LSE over BoN of grad_mat[i, h, :], minus base baseline
                lse_over_selection = np.empty(H, dtype=float)  # (H,)
                # Base baseline vector (per-dimension): beta * (log(mean_j exp(base_j[h]/beta)))
                base_mat = np.vstack(base_rewards)  # (B, H)
                base_lse_vec = args.beta * (logsumexp_axis0(base_mat / args.beta) - np.log(len(base_rewards)))  # (H,)
                for h_index in range(H):
                    block = grad_mat[:, h_index, :]  # (M, BoN)
                    # Transpose to (BoN, M) so logsumexp_axis0 reduces over BoN -> (M,)
                    lse_per_i = logsumexp_axis0(block.T)  # (M,)
                    lse_over_selection[h_index] = float(np.mean(lse_per_i)) - float(base_lse_vec[h_index])
                r_hat = int(np.argmin(lse_over_selection))

                # For each selection i, compute vector h_i where:
                # h_i[h] = mean_j LSE( [sel_mat[i, h]] U grad_mat[j, h, :] ) over j in [0, M)
                g_values = []
                for i in range(sel_mat.shape[0]):
                    sel_vec = sel_mat[i]  # (H,)
                    h_i_vals = np.empty(H, dtype=float)
                    for h_index in range(H):
                        sel_val = sel_vec[h_index]
                        # Vectorize across j: build matrix (BoN, M) for this h_index
                        sel_row = np.full((1, args.M), sel_val)  # (1, M)
                        grad_block = grad_mat[:, h_index, :-1].T  # (BoN-1, M)
                        stacked = np.vstack([sel_row, grad_block])  # (BoN, M)
                        lse_per_j = logsumexp_axis0(stacked)  # (M,)
                        h_i_vals[h_index] = float(np.mean(lse_per_j))
                    g_values.append(float(h_i_vals[r_hat])*args.BoN)

                # Expand to all pairwise comparisons among selection responses
                for idx_a in range(len(selection_ids)):
                    for idx_b in range(idx_a + 1, len(selection_ids)):
                        higher_idx, lower_idx = idx_a, idx_b
                        if g_values[higher_idx] < g_values[lower_idx]:
                            higher_idx, lower_idx = lower_idx, higher_idx

                        higher_sel_id = selection_ids[higher_idx]
                        lower_sel_id = selection_ids[lower_idx]

                        # Preserve original columns and append pairwise outputs
                        example = {col: row[col] for col in dataset.column_names}
                        example.update({
                            "chosen": row[f"response_{higher_sel_id}"],
                            "reject": row[f"response_{lower_sel_id}"],
                            "chosen_reward": float(g_values[higher_idx]),
                            "reject_reward": float(g_values[lower_idx]),
                            "g_chosen": float(g_values[higher_idx]),
                            "g_reject": float(g_values[lower_idx]),
                        })
                        yield example

        # Build explicit features to include new columns
        features = dataset.features.copy()
        features.update({
            "chosen": Value("string"),
            "reject": Value("string"),
            "chosen_reward": Value("float64"),
            "reject_reward": Value("float64"),
            "g_chosen": Value("float64"),
            "g_reject": Value("float64"),
        })

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        streamed = streamed.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(streamed))
        return streamed

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    out.push_to_hub(repo_prefix + "_BoN_tokenized")

if __name__ == "__main__":
    main()


