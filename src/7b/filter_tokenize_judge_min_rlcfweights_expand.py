import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter + tokenize using importance-weighted averages of per-check scores.",
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
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use this as the repo prefix for push_to_hub instead of input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split to first N rows for debugging")
    parser.add_argument("--gap_ratio", type=float, default=0.0,
                        help="If >0, filter top ratio by (g_chosen - g_reject) per split and push *_gap")
    parser.add_argument("--gap_shuffle_seed", type=int, default=None,
                        help="Shuffle seed used after gap filtering (None = no fixed seed)")
    parser.add_argument("--debug_schema", action="store_true",
                        help="If set, print debug info about detected score columns and exit on mismatch")
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


def parse_importance_weights(requirements: str):
    """
    Parse the enumerated requirements string and return a list of importance weights
    aligned with per-check vectors. If a check has no explicit importance, returns None
    for that position.
    """
    if not isinstance(requirements, str) or len(requirements.strip()) == 0:
        return []
    req_str = requirements
    counter = 1
    chunks = []
    while len(req_str) > 0:
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
        req_str = req_str[len(curr) + len(prefix):]
        counter += 1
    weights = []
    for c in (x.strip() for x in chunks):
        if "(importance:" in c:
            try:
                w = int(c.split("(importance:")[1].split("/")[0].strip())
            except Exception:
                w = None
            weights.append(w)
        else:
            weights.append(None)
    return weights


def weighted_average(values: np.ndarray, weights):
    """
    Compute weighted average over 1D values with given weights.
    - Ignores non-finite values and non-positive/None weights.
    - Falls back to unweighted mean if no valid weighted entries.
    """
    v = np.atleast_1d(np.array(values, dtype=float))
    if weights is None:
        valid = np.isfinite(v)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(v[valid]))
    w_arr = np.array([0.0 if (w is None or w <= 0) else float(w) for w in weights], dtype=float)
    mask = np.isfinite(v) & (w_arr > 0)
    if not np.any(mask):
        valid = np.isfinite(v)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(v[valid]))
    return float(np.sum(v[mask] * w_arr[mask]) / np.sum(w_arr[mask]))


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

    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

    def process_split(dataset, is_train):
        print('split length:', len(dataset))
        required_cols = ["qwen_prompt", "qwen_prompt_tokens"]
        for c in required_cols:
            if c not in dataset.column_names:
                raise ValueError(f"Expected preprocessed dataset to contain column '{c}'. Please run preprocess_common.py first.")

        # Allow datasets that use either 'base' or 'adversary' terminology
        response_pattern = re.compile(r'^(selection|current|base|adversary)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for downstream tokenization.")

        escaped_score_type = re.escape(args.score_type)
        # Detect whether score columns consistently use 'base' or 'adversary'.
        # We require both selection_*_<key>_* and current_*_<key>_* to exist for the SAME key.
        counts = {}
        sel_present_keys = set()
        cur_present_keys = set()
        for candidate in ("base", "adversary"):
            # Use single backslash \d in raw strings to match digits
            sel_pat = re.compile(rf"^selection_(\d+)_{candidate}_(\d+)_({escaped_score_type})$")
            cur_pat = re.compile(rf"^current_(\d+)_{candidate}_(\d+)_({escaped_score_type})$")
            sel_count = sum(1 for n in dataset.column_names if sel_pat.match(n))
            cur_count = sum(1 for n in dataset.column_names if cur_pat.match(n))
            counts[candidate] = (sel_count, cur_count)
            if sel_count > 0:
                sel_present_keys.add(candidate)
            if cur_count > 0:
                cur_present_keys.add(candidate)

            if args.debug_schema:
                sel_examples = [n for n in dataset.column_names if sel_pat.match(n)][:5]
                cur_examples = [n for n in dataset.column_names if cur_pat.match(n)][:5]
                print(f"[debug] key={candidate} score_type={args.score_type}: sel_count={sel_count}, cur_count={cur_count}")
                if sel_examples:
                    print(f"[debug] sample selection cols: {sel_examples}")
                if cur_examples:
                    print(f"[debug] sample current cols: {cur_examples}")

        valid_keys = sel_present_keys & cur_present_keys
        if len(valid_keys) == 1:
            detected_key = next(iter(valid_keys))
        elif len(valid_keys) > 1:
            # Ambiguous: both base and adversary appear complete. Ask user to disambiguate by cleaning dataset.
            raise ValueError(
                "Both 'base' and 'adversary' score families are present; please keep only one naming scheme. "
                f"Counts: base sel={counts.get('base',(0,0))[0]}, cur={counts.get('base',(0,0))[1]}; "
                f"adversary sel={counts.get('adversary',(0,0))[0]}, cur={counts.get('adversary',(0,0))[1]}."
            )
        else:
            # No single key has both families; error with specifics.
            msg = (
                "Could not find matching selection/current score columns for the same key. "
                f"Observed counts — base: sel={counts.get('base',(0,0))[0]}, cur={counts.get('base',(0,0))[1]}; "
                f"adversary: sel={counts.get('adversary',(0,0))[0]}, cur={counts.get('adversary',(0,0))[1]}."
            )
            if args.debug_schema:
                print("[debug] columns:")
                print(dataset.column_names)
                # Also surface similar-looking names to help spot typos/case/spacing issues
                near_sel = [n for n in dataset.column_names if 'selection_' in n and '_adversary_' in n]
                near_cur = [n for n in dataset.column_names if 'current_' in n and '_adversary_' in n]
                if near_sel:
                    print(f"[debug] near selection adversary-like cols (first 10): {near_sel[:10]}")
                if near_cur:
                    print(f"[debug] near current adversary-like cols (first 10): {near_cur[:10]}")
            raise ValueError(msg)

        selection_score_pattern = re.compile(rf"^selection_(\d+)_{detected_key}_(\d+)_({escaped_score_type})$")
        current_score_pattern = re.compile(rf"^current_(\d+)_{detected_key}_(\d+)_({escaped_score_type})$")
        selection_ids = set()
        current_ids = set()
        base_ids = set()
        for name in dataset.column_names:
            sel_match = selection_score_pattern.match(name)
            if sel_match:
                selection_ids.add(int(sel_match.group(1)))
                base_ids.add(int(sel_match.group(2)))
            cur_match = current_score_pattern.match(name)
            if cur_match:
                current_ids.add(int(cur_match.group(1)))
                base_ids.add(int(cur_match.group(2)))

        selection_ids = sorted(selection_ids)
        current_ids = sorted(current_ids)
        base_ids = sorted(base_ids)
        if not selection_ids:
            raise ValueError("Dataset is missing selection score columns for the specified score type.")
        if not current_ids:
            raise ValueError("Dataset is missing current score columns for the specified score type.")
        if not base_ids:
            raise ValueError("Dataset is missing base indices for the specified score type.")

        # Stream rows via generator to avoid building a huge dict-of-lists
        def row_generator():
            for row in tqdm(dataset):
                # Build importance weights aligned to per-check vectors
                importance_weights = None
                if 'requirements' in row and isinstance(row['requirements'], str):
                    try:
                        importance_weights = parse_importance_weights(row['requirements'])
                    except AssertionError:
                        importance_weights = None

                g_values = []
                selection_tokens = []
                selection_texts = []
                for sel_id in selection_ids:
                    per_base_scores = []
                    for base_id in base_ids:
                        key = f"selection_{sel_id}_{detected_key}_{base_id}_{args.score_type}"
                        raw_scores = row.get(key, None)
                        if raw_scores is None:
                            continue
                        p_vec = np.array(raw_scores, dtype=float)
                        p_vec = np.atleast_1d(p_vec)
                        # If we have matching-length weights, use weighted average; otherwise unweighted mean.
                        if (importance_weights is not None) and (len(importance_weights) == p_vec.shape[0]):
                            per_base_scores.append(float(weighted_average(p_vec, importance_weights)))
                        else:
                            per_base_scores.append(float(np.mean(p_vec)))

                    if not per_base_scores:
                        raise ValueError(
                            f"No selection-{detected_key} scores available for selection {sel_id} to compute gradient."
                        )

                    g_values.append(float(np.mean(per_base_scores)))

                    sel_token = tokenizer.apply_chat_template(
                        get_message(response=row[f"selection_response_{sel_id}"]),
                        add_generation_prompt=False,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen + slicing_idx_used,
                    )[slicing_idx_used:]
                    selection_tokens.append(list(sel_token))
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

                selected_pairs = []
                if is_train:
                    for idx_a in range(len(selection_ids)):
                        for idx_b in range(idx_a + 1, len(selection_ids)):
                            higher_idx, lower_idx = idx_a, idx_b
                            if g_values[higher_idx] < g_values[lower_idx]:
                                higher_idx, lower_idx = lower_idx, higher_idx
                            selected_pairs.append((higher_idx, lower_idx))
                else:
                    if g_values:
                        g_array = np.array(g_values, dtype=float)
                        if g_array.size >= 2:
                            sorted_indices = np.argsort(g_array)
                            lower_idx = int(sorted_indices[0])
                            higher_idx = int(sorted_indices[-1])
                            if higher_idx != lower_idx:
                                selected_pairs.append((higher_idx, lower_idx))

                for higher_idx, lower_idx in selected_pairs:
                    higher_sel_id = selection_ids[higher_idx]
                    lower_sel_id = selection_ids[lower_idx]

                    example = {col: row[col] for col in dataset.column_names}
                    example.update({
                        "chosen": row[f"selection_response_{higher_sel_id}"],
                        "reject": row[f"selection_response_{lower_sel_id}"],
                        "qwen_chosen_tokens": selection_tokens[higher_idx],
                        "qwen_reject_tokens": selection_tokens[lower_idx],
                        "qwen_chosen": selection_texts[higher_idx],
                        "qwen_reject": selection_texts[lower_idx],
                        "chosen_reward": float(g_values[higher_idx]),
                        "reject_reward": float(g_values[lower_idx]),
                        "g_chosen": float(g_values[higher_idx]),
                        "g_reject": float(g_values[lower_idx]),
                    })

                    yield example

        # Explicit features for faster Arrow construction; use float64 for rewards/g-values
        features = dataset.features.copy()
        features.update({
            "chosen": Value("string"),
            "reject": Value("string"),
            "qwen_chosen": Value("string"),
            "qwen_reject": Value("string"),
            "qwen_chosen_tokens": Sequence(Value("int64")),
            "qwen_reject_tokens": Sequence(Value("int64")),
            "chosen_reward": Value("float64"),
            "reject_reward": Value("float64"),
            "g_chosen": Value("float64"),
            "g_reject": Value("float64"),
        })

        generated = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        generated = generated.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(generated))
        return generated

    train_processed = process_split(ds_dict['train'], is_train=True)
    test_processed = process_split(ds_dict['test'], is_train=False)

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_min_expand_rlcfweights_tokenized'
    out.push_to_hub(base_repo)

    # Optional gap filtering and secondary upload
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
            "test": out["test"],
        })
        gap_out.push_to_hub(f"{base_repo}_gap_ratio_{args.gap_ratio}")


if __name__ == "__main__":
    main()
