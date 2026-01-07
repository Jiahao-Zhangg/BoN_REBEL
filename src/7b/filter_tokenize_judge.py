import argparse
import numpy as np
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

torch.set_printoptions(threshold=10_000)

SYS_PROMPT_LEN = 24

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    # Default to MisDrifter/test_dataset which contains selection/current/base score vectors
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=24)
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority score vectors when computing preferences")
    return parser.parse_args()


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
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


# BTL no longer used; probabilities are provided by dataset


def main():

    # init
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    # Prefer explicit [PAD] token for Qwen so it shows in decoded strings
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
        print(f'slicing index used (fixed): {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx

    # Load both splits (typically preprocessed via preprocess_common or preprocess_common_stage2)
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    def process_split(dataset):
        print('split length:', len(dataset))

        has_qp = "qwen_prompt" in dataset.column_names
        has_qpt = "qwen_prompt_tokens" in dataset.column_names
        preprocessed = has_qp and has_qpt

        if preprocessed:
            # Dataset was already filtered and had prompts tokenized via preprocess_common/preprocess_common_stage2.
            print("Detected preprocessed dataset with qwen_prompt columns; skipping length/PAD filtering.")
        else:
            # Apply the same style of filtering as the preprocess_common scripts for raw datasets.
            # Filter overly long prompts
            dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
                get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt'
            ).shape[-1] <= args.maxlen_prompt)
            print('filtered long prompts:', len(dataset))

            # Identify all response columns (include adversary to mirror stage2 preprocessing)
            response_pattern = re.compile(r'^(selection|current|base|adversary)_response_\d+$')
            response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
            if not response_columns:
                raise ValueError("Dataset is missing response columns required for length filtering.")
            print(f'response columns length-filtered: {response_columns}')

            # Filter by max response length
            def responses_within_limit(row):
                for col in response_columns:
                    resp = row[col]
                    if not isinstance(resp, str):
                        return False
                    tokens = tokenizer.apply_chat_template(
                        get_message(response=resp),
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors='pt',
                    )[:, slicing_idx_used:]
                    if tokens.shape[-1] > args.maxlen:
                        return False
                return True

            dataset = dataset.filter(responses_within_limit)
            print('filtered responses by length:', len(dataset))

            # Ensure responses end with PAD/EOS
            def responses_end_properly(row):
                for col in response_columns:
                    resp = row[col]
                    if not isinstance(resp, str):
                        return False
                    response_token = tokenizer.apply_chat_template(
                        get_message(response=resp),
                        add_generation_prompt=False,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen + slicing_idx_used,
                    )[slicing_idx_used:]

                    if len(response_token) != args.maxlen:
                        return False

                    pad_id = tokenizer.pad_token_id
                    eos_id = tokenizer.eos_token_id
                    if not response_token:
                        return False
                    last_id = int(response_token[-1])
                    if pad_id is None and eos_id is None:
                        continue
                    pad_ok = pad_id is not None and last_id == pad_id
                    eos_ok = eos_id is not None and last_id == eos_id
                    if not (pad_ok or eos_ok):
                        return False
                return True

            dataset = dataset.filter(responses_end_properly)
            print('filtered responses not ending with PAD/EOS:', len(dataset))

            # Ensure prompt columns exist; compute only if missing
            if not (has_qp and has_qpt):
                qwen_prompts = []
                qwen_prompt_tokens = []
                for row in tqdm(dataset):
                    qwen_prompt_token = tokenizer_left.apply_chat_template(
                        get_message(row['prompt']),
                        add_generation_prompt=True,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen_prompt,
                    )
                    qwen_prompt = tokenizer_left.decode(qwen_prompt_token, skip_special_tokens=False)
                    assert len(qwen_prompt_token) == args.maxlen_prompt
                    if "Qwen" in args.model:
                        assert ("<|start_header_id|>" in qwen_prompt or "<|im_start|>" in qwen_prompt), "Qwen prompt missing chat header markers"
                    qwen_prompts.append(qwen_prompt)
                    qwen_prompt_tokens.append(qwen_prompt_token)
                if not has_qp:
                    dataset = dataset.add_column("qwen_prompt", qwen_prompts)
                if not has_qpt:
                    dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

        # Compute chosen/reject once per example (no expand), streaming via generator.
        # Allow datasets that use either 'base' or 'adversary' terminology for scores, and auto-detect all ids.
        escaped_score_type = re.escape(args.score_type)
        counts = {}
        sel_present_keys = set()
        cur_present_keys = set()
        for candidate in ("base", "adversary"):
            sel_pat = re.compile(rf"^selection_(\d+)_{candidate}_(\d+)_({escaped_score_type})$")
            cur_pat = re.compile(rf"^current_(\d+)_{candidate}_(\d+)_({escaped_score_type})$")
            sel_count = sum(1 for n in dataset.column_names if sel_pat.match(n))
            cur_count = sum(1 for n in dataset.column_names if cur_pat.match(n))
            counts[candidate] = (sel_count, cur_count)
            if sel_count > 0:
                sel_present_keys.add(candidate)
            if cur_count > 0:
                cur_present_keys.add(candidate)

        valid_keys = sel_present_keys & cur_present_keys
        if len(valid_keys) == 1:
            detected_key = next(iter(valid_keys))
        elif len(valid_keys) > 1:
            raise ValueError(
                "Both 'base' and 'adversary' score families are present; please keep only one naming scheme. "
                f"Counts: base sel={counts.get('base',(0,0))[0]}, cur={counts.get('base',(0,0))[1]}; "
                f"adversary sel={counts.get('adversary',(0,0))[0]}, cur={counts.get('adversary',(0,0))[1]}."
            )
        else:
            raise ValueError(
                "Could not find matching selection/current score columns for the same key. "
                f"Observed counts — base: sel={counts.get('base',(0,0))[0]}, cur={counts.get('base',(0,0))[1]}; "
                f"adversary: sel={counts.get('adversary',(0,0))[0]}, cur={counts.get('adversary',(0,0))[1]}."
            )

        # Collect all selection/current/base ids for the detected key
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

        def row_generator():
            for row in tqdm(dataset):
                beta = args.beta
                # Build current vs base/adversary matrices per base: shape (K=len(current_ids), L)
                current_base_scores = []
                vector_len = None
                for base_id in base_ids:
                    rows = []
                    for cur_id in current_ids:
                        key = f"current_{cur_id}_{detected_key}_{base_id}_{args.score_type}"
                        raw_scores = row.get(key, None)
                        if raw_scores is None:
                            continue
                        p_vec = np.array(raw_scores, dtype=float)
                        p_vec = np.atleast_1d(p_vec)
                        if vector_len is None:
                            vector_len = p_vec.shape[0]
                        elif p_vec.shape[0] != vector_len:
                            raise ValueError(
                                f"Mismatched score vector length for {key}: expected {vector_len}, got {p_vec.shape[0]}"
                            )
                        rows.append(p_vec)
                    if not rows:
                        continue
                    current_base_scores.append(np.stack(rows, axis=0))
                if not current_base_scores:
                    raise ValueError("No current-base/adversary scores available to estimate A_vec and j_star.")

                # Compute exp_terms per base: exp(- (1/K) * sum_k P_m / beta) for each m
                exp_terms_list = []
                for cb in current_base_scores:
                    inner_sums = np.sum(cb, axis=0) / (beta * cb.shape[0])
                    exp_terms_list.append(np.exp(-inner_sums))

                # A_m(x) = sum_i exp_terms_i[m] across bases
                A_vec = exp_terms_list[0]
                for i in range(1, len(exp_terms_list)):
                    A_vec = A_vec + exp_terms_list[i]
                j_star = int(np.argmax(A_vec))

                # Compute g(z) using weighted B at m*
                g_values = []
                for sel_id in selection_ids:
                    p_z_vs_bases = []
                    for base_index, base_id in enumerate(base_ids):
                        key = f"selection_{sel_id}_{detected_key}_{base_id}_{args.score_type}"
                        raw_scores = row.get(key, None)
                        if raw_scores is None:
                            continue
                        p_vec = np.array(raw_scores, dtype=float)
                        p_vec = np.atleast_1d(p_vec)
                        if j_star >= p_vec.shape[0]:
                            raise ValueError(
                                f"Score vector for {key} missing coordinate j*={j_star} (length={p_vec.shape[0]})."
                            )
                        weight = exp_terms_list[base_index][j_star]
                        p_z_vs_bases.append(float(p_vec[j_star]) * float(weight))

                    if not p_z_vs_bases:
                        raise ValueError(
                            f"No selection-{detected_key} scores available for selection {sel_id} to compute g."
                        )

                    B_star = float(np.sum(p_z_vs_bases))
                    g_z = B_star / float(A_vec[j_star])
                    g_values.append(float(g_z))

                if not g_values or len(g_values) < 2:
                    continue

                chosen_idx_in_z = int(np.argmax(g_values))
                reject_idx_in_z = int(np.argmin(g_values))
                if chosen_idx_in_z == reject_idx_in_z:
                    continue

                chosen_sel_id = selection_ids[chosen_idx_in_z]
                reject_sel_id = selection_ids[reject_idx_in_z]

                chosen_key = f"selection_response_{chosen_sel_id}"
                reject_key = f"selection_response_{reject_sel_id}"

                qwen_chosen_token = tokenizer.apply_chat_template(
                        get_message(response=row[chosen_key]),
                        add_generation_prompt=False,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen+slicing_idx_used,
                )[slicing_idx_used:]
                chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
                assert len(qwen_chosen_token) == args.maxlen
                if "Qwen" in args.model:
                    assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
                    assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
                    last_id = int(qwen_chosen_token[-1])
                    pid = tokenizer.pad_token_id
                    eid = tokenizer.eos_token_id
                    assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

                qwen_reject_token = tokenizer.apply_chat_template(
                        get_message(response=row[reject_key]),
                        add_generation_prompt=False,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen+slicing_idx_used,
                )[slicing_idx_used:]
                reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
                assert len(qwen_reject_token) == args.maxlen
                if "Qwen" in args.model:
                    assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
                    assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
                    last_id = int(qwen_reject_token[-1])
                    pid = tokenizer.pad_token_id
                    eid = tokenizer.eos_token_id
                    assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

                example = {col: row[col] for col in dataset.column_names}
                example.update({
                    "chosen": row[chosen_key],
                    "reject": row[reject_key],
                    "qwen_chosen_tokens": list(qwen_chosen_token),
                    "qwen_reject_tokens": list(qwen_reject_token),
                    "qwen_chosen": chosen_text,
                    "qwen_reject": reject_text,
                    "chosen_reward": float(g_values[chosen_idx_in_z]),
                    "reject_reward": float(g_values[reject_idx_in_z]),
                    "g_chosen": float(g_values[chosen_idx_in_z]),
                    "g_reject": float(g_values[reject_idx_in_z]),
                    "j_star": int(j_star),
                })

                yield example

        # Explicit features to speed up Arrow construction and use float64 for rewards/g-values
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
            "j_star": Value("int64"),
        })

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        streamed = streamed.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(streamed))
        return streamed

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    out.push_to_hub(args.input_repo + '_beta_' + str(args.beta) + '_multi_noexpand_tokenized')



if __name__ == "__main__":
    main()
