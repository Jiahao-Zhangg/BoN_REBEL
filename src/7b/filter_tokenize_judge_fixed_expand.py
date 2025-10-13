import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter + tokenize using ONLY a single fixed check per prompt (1D case).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors + requirements)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=24,
                        help="Fallback slicing index if model-specific detection not used")
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority score vectors when computing preferences")
    # Fixed check text we will search for in each prompt's requirements
    parser.add_argument(
        "--fixed_check",
        type=str,
        default=(
            "Does the response satisfy the following two criteria: "
            "1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction? "
            "2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality."
        ),
        help="Exact check text to locate per prompt; we use only this check's coordinate.",
    )
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use this as the repo prefix for push_to_hub instead of input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split to first N rows for debugging")
    parser.add_argument("--gap_ratio", type=float, default=0.0,
                        help="If >0, filter top ratio by (g_chosen - g_reject) per split and push *_gap")
    parser.add_argument("--gap_shuffle_seed", type=int, default=None,
                        help="Shuffle seed used after gap filtering (None = no fixed seed)")
    return parser.parse_args()


def get_message(instruction=None, response=None):
    assert instruction is not None or response is not None

    if response is None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction is None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    return message


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Map curly quotes to ascii and collapse whitespace, lowercase
    trans = {
        ord('’'): "'",
        ord('‘'): "'",
        ord('“'): '"',
        ord('”'): '"',
        ord('\u00A0'): ' ',
    }
    s = s.translate(trans)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_requirements_to_checks(requirements: str):
    """Parse the enumerated requirements string into a list of check texts (without importance).

    Mirrors the parsing logic used during scoring (run_inference_on_shard.py) so
    that indices match the vectors in the dataset.
    """
    if not isinstance(requirements, str) or len(requirements.strip()) == 0:
        return []

    req_str = requirements
    counter = 1
    chunks = []
    while len(req_str) > 0:
        # Expect like "1) ... (importance: XX/100)\n2) ..."
        prefix = f"{counter})"
        assert req_str.startswith(prefix), (
            f"Malformed requirements format: expected prefix '{prefix}' but got: {req_str[:40]}...")
        marker = f"/100)\n{counter+1})"
        pos = req_str.find(marker)
        if pos > 0:
            curr = req_str[len(prefix): pos + len("/100)\n")]
        else:
            curr = req_str[len(prefix):]
        chunks.append(curr)
        req_str = req_str[len(prefix) + len(curr):]
        counter += 1

    # Strip and remove trailing importance suffix
    checks = []
    for c in chunks:
        c = c.strip()
        if "(importance:" in c:
            c = c.split("(importance:")[0].strip()
        checks.append(c)
    return checks


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')

    # Ensure PAD handling sensible
    if "Qwen" in args.model:
        if tokenizer.pad_token != "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        if tokenizer_left.pad_token != "[PAD]":
            tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer_left.pad_token = "[PAD]"
    else:
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if tokenizer_left.pad_token is None:
            if tokenizer_left.eos_token is not None:
                tokenizer_left.pad_token = tokenizer_left.eos_token
            else:
                tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
    else:
        slicing_idx_used = args.slicing_idx

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
        required_cols = ["qwen_prompt", "qwen_prompt_tokens"]
        for c in required_cols:
            if c not in dataset.column_names:
                raise ValueError(f"Expected preprocessed dataset to contain column '{c}'. Please run preprocess_common.py first.")

        if 'requirements' not in dataset.column_names:
            raise ValueError("Dataset is missing 'requirements' column required to locate fixed check index per prompt.")

        response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for downstream tokenization.")

        expanded_data = {col: [] for col in dataset.column_names}
        expanded_data.update({
            "chosen": [],
            "chosen_reward": [],
            "qwen_chosen": [],
            "qwen_chosen_tokens": [],
            "reject": [],
            "reject_reward": [],
            "qwen_reject": [],
            "qwen_reject_tokens": [],
            "g_chosen": [],
            "g_reject": [],
            "j_fixed": [],
        })

        norm_target = _normalize_text(args.fixed_check)

        for row in tqdm(dataset):
            checks = parse_requirements_to_checks(row.get('requirements', ''))
            j_fixed = None
            for i, ch in enumerate(checks):
                if _normalize_text(ch) == norm_target:
                    j_fixed = i
                    break
            assert j_fixed is not None, "Row passed filter but fixed check index not found."
            beta = args.beta

            current_base_scores = []
            for base_j in range(1, 3):
                rows = []
                for cur_k in range(1, 3):
                    key = f"current_{cur_k}_base_{base_j}_{args.score_type}"
                    p_vec = np.array(row[key], dtype=float)
                    rows.append(p_vec)
                current_base_scores.append(np.stack(rows, axis=0))  # (2, L)

            exp_terms = []
            for cb in current_base_scores:
                inner = np.sum(cb[:, j_fixed], axis=0) / (beta * cb.shape[0])
                exp_terms.append(np.exp(-inner))

            A_fixed = exp_terms[0] + exp_terms[1]

            g_values = []
            for sel_i in range(1, 4):
                p_z_vs_bases = []
                for base_j in range(1, 3):
                    key = f"selection_{sel_i}_base_{base_j}_{args.score_type}"
                    p_vec = np.array(row[key], dtype=float)
                    p_z_vs_bases.append(p_vec[j_fixed])
                B_fixed = p_z_vs_bases[0] * exp_terms[0] + p_z_vs_bases[1] * exp_terms[1]
                g_z = B_fixed / A_fixed
                g_values.append(float(g_z))

            selection_tokens = []
            selection_texts = []
            for sel_i in range(1, 4):
                sel_token = tokenizer.apply_chat_template(
                    get_message(response=row[f"selection_response_{sel_i}"]),
                    add_generation_prompt=False,
                    tokenize=True,
                    padding='max_length',
                    max_length=args.maxlen + slicing_idx_used,
                )[slicing_idx_used:]
                selection_tokens.append(sel_token)
                sel_text = tokenizer.decode(sel_token, skip_special_tokens=False)
                selection_texts.append(sel_text)
                assert len(sel_token) == args.maxlen
                if "Qwen" in args.model:
                    assert not sel_text.lstrip().startswith("<|im_start|>assistant"), "Qwen selection should not include assistant header"
                    assert ("<|eot_id|>" in sel_text) or ("<|im_end|>" in sel_text), "Qwen selection text should include end-of-turn marker"
                    last_id = int(sel_token[-1])
                    pid = tokenizer.pad_token_id
                    eid = tokenizer.eos_token_id
                    assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen selection last token should be PAD or EOS"

            for pair in ((0, 1), (1, 2), (0, 2)):
                higher_idx, lower_idx = pair
                if g_values[higher_idx] < g_values[lower_idx]:
                    higher_idx, lower_idx = lower_idx, higher_idx

                for col in dataset.column_names:
                    expanded_data[col].append(row[col])

                expanded_data["chosen"].append(row[f"selection_response_{higher_idx+1}"])
                expanded_data["reject"].append(row[f"selection_response_{lower_idx+1}"])

                expanded_data["qwen_chosen_tokens"].append(selection_tokens[higher_idx])
                expanded_data["qwen_reject_tokens"].append(selection_tokens[lower_idx])

                expanded_data["qwen_chosen"].append(selection_texts[higher_idx])
                expanded_data["qwen_reject"].append(selection_texts[lower_idx])

                expanded_data["chosen_reward"].append(g_values[higher_idx])
                expanded_data["reject_reward"].append(g_values[lower_idx])

                expanded_data["g_chosen"].append(g_values[higher_idx])
                expanded_data["g_reject"].append(g_values[lower_idx])

                expanded_data["j_fixed"].append(j_fixed)

        dataset = Dataset.from_dict(expanded_data)
        dataset = dataset.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(dataset))
        return dataset

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_' + args.score_type + '_beta_' + str(args.beta) + '_fixed_expand_tokenized'
    out.push_to_hub(base_repo)

    if args.gap_ratio and args.gap_ratio > 0.0:
        def gap_filter(split_ds):
            if "g_chosen" not in split_ds.column_names or "g_reject" not in split_ds.column_names:
                raise ValueError("Missing g_chosen or g_reject for gap filtering")
            with_gap = split_ds.map(lambda row: {"_gap": float(row["g_chosen"]) - float(row["g_reject"])})
            sorted_by_gap = with_gap.sort("_gap", reverse=True)
            keep_count = max(1, int(len(sorted_by_gap) * float(args.gap_ratio)))
            filtered = sorted_by_gap.select(range(keep_count)).remove_columns(["_gap"]).shuffle(seed=args.gap_shuffle_seed)
            return filtered

        gap_out = DatasetDict({
            "train": gap_filter(out["train"]),
            "test": gap_filter(out["test"]),
        })
        gap_out.push_to_hub(base_repo + "_gap")


if __name__ == "__main__":
    main()
