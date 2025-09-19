import argparse
import numpy as np
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter + tokenize using ONLY a single fixed check per prompt (1D case).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/sw_maxlen_8192",
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

    def compute_qwen_slicing_idx(tok):
        """Find the start-of-content offset by aligning raw content tokens inside the templated assistant message."""
        sample = "__QWEN_SLICE_PROBE__"
        full = tok.apply_chat_template(get_message(response=sample), tokenize=True, add_generation_prompt=False)
        full_ids = full["input_ids"] if isinstance(full, dict) else full
        content_ids = tok(sample, add_special_tokens=False)["input_ids"]

        # Simple subsequence search
        def find_subseq(haystack, needle):
            n, m = len(haystack), len(needle)
            for i in range(0, n - m + 1):
                if haystack[i:i+m] == needle:
                    return i
            return -1

        pos = find_subseq(full_ids, content_ids)
        if pos < 0:
            raise RuntimeError("Failed to locate content within Qwen assistant template; cannot compute slicing index.")
        return pos

    dataset = load_dataset(args.input_repo, split='train')
    print('initial length:', len(dataset))

    # Filter overly long prompts
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
        get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))

    # Filter responses by length
    for i in range(1, 4):
        key = f'selection_response_{i}'
        dataset = dataset.filter(lambda row, _key=key: tokenizer.apply_chat_template(
            get_message(response=row[_key]), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 5:].shape[-1] <= args.maxlen)
        print(f'filtered {key}:', len(dataset))

    # Ensure requirements column exists for locating the fixed check index
    if 'requirements' not in dataset.column_names:
        raise ValueError("Dataset is missing 'requirements' column required to locate fixed check index per prompt.")

    # Pre-compute Qwen slicing index if needed
    if "Qwen" in args.model:
        slicing_idx_used = compute_qwen_slicing_idx(tokenizer)
        print(f'slicing index used: {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx

    # Add prompt tokens
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
    dataset = dataset.add_column("qwen_prompt", qwen_prompts)
    dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

    # Locate and filter rows to only those that contain the fixed check text
    norm_target = _normalize_text(args.fixed_check)

    def has_fixed_check(row):
        checks = parse_requirements_to_checks(row.get('requirements', ''))
        for ch in checks:
            if _normalize_text(ch) == norm_target:
                return True
        return False

    n_before = len(dataset)
    dataset = dataset.filter(has_fixed_check)
    print(f"kept rows with fixed check present: {len(dataset)} / {n_before}")

    # select chosen and reject using only the fixed check coordinate
    chosen, reject = [], []
    qwen_chosen, qwen_reject = [], []
    qwen_chosen_tokens, qwen_reject_tokens = [], []
    chosen_reward, reject_reward = [], []
    g_chosen_list, g_reject_list = [], []
    j_fixed_list = []

    # Iterate rows; compute the fixed index per-row on the fly
    for row in tqdm(dataset):
        # Find fixed index for this row
        checks = parse_requirements_to_checks(row.get('requirements', ''))
        j_fixed = None
        for i, ch in enumerate(checks):
            if _normalize_text(ch) == norm_target:
                j_fixed = i
                break
        assert j_fixed is not None, "Row passed filter but fixed check index not found."
        beta = args.beta

        # Collect current vs base score vectors (shape per base: (K=2, L))
        current_base_scores = []
        for base_j in range(1, 3):
            rows = []
            for cur_k in range(1, 3):
                key = f"current_{cur_k}_base_{base_j}_{args.score_type}"
                p_vec = np.array(row[key], dtype=float)
                rows.append(p_vec)
            current_base_scores.append(np.stack(rows, axis=0))  # (2, L)

        # Compute exp_terms for the fixed coordinate only
        exp_terms = []  # length 2
        for cb in current_base_scores:
            # cb shape: (2, L)
            inner = np.sum(cb[:, j_fixed], axis=0) / (beta * cb.shape[0])  # scalar for this coord
            exp_terms.append(np.exp(-inner))

        # Denominator A at fixed coord
        A_fixed = exp_terms[0] + exp_terms[1]

        # For each selection z, compute B_fixed and g
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

        chosen_idx_in_z = int(np.argmax(g_values))  # 0..2
        reject_idx_in_z = int(np.argmin(g_values))  # 0..2

        # Raw strings
        chosen.append(row[f"selection_response_{chosen_idx_in_z+1}"])
        reject.append(row[f"selection_response_{reject_idx_in_z+1}"])

        # Tokenize chosen
        qwen_chosen_token = tokenizer.apply_chat_template(
            get_message(response=row[f"selection_response_{chosen_idx_in_z+1}"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
        )[slicing_idx_used:]
        qwen_chosen_tokens.append(qwen_chosen_token)
        chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
        qwen_chosen.append(chosen_text)
        chosen_reward.append(g_values[chosen_idx_in_z])
        assert len(qwen_chosen_token) == args.maxlen
        if "Qwen" in args.model:
            assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
            assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
            last_id = int(qwen_chosen_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

        # Tokenize reject
        qwen_reject_token = tokenizer.apply_chat_template(
            get_message(response=row[f"selection_response_{reject_idx_in_z+1}"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
        )[slicing_idx_used:]
        qwen_reject_tokens.append(qwen_reject_token)
        reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
        qwen_reject.append(reject_text)
        reject_reward.append(g_values[reject_idx_in_z])
        assert len(qwen_reject_token) == args.maxlen
        if "Qwen" in args.model:
            assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
            assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
            last_id = int(qwen_reject_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

        # Store g-values and fixed index
        g_chosen_list.append(g_values[chosen_idx_in_z])
        g_reject_list.append(g_values[reject_idx_in_z])
        j_fixed_list.append(j_fixed)

    # Attach columns
    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("qwen_chosen", qwen_chosen)
    dataset = dataset.add_column("qwen_chosen_tokens", qwen_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("qwen_reject", qwen_reject)
    dataset = dataset.add_column("qwen_reject_tokens", qwen_reject_tokens)
    dataset = dataset.add_column("g_chosen", g_chosen_list)
    dataset = dataset.add_column("g_reject", g_reject_list)
    dataset = dataset.add_column("j_fixed", j_fixed_list)

    # Remove rows where chosen == reject
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    # Split and push (keep naming consistent)
    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_' + args.score_type + '_maxlenprompt_' + str(args.maxlen_prompt) + '_fixed_tokenized')


if __name__ == "__main__":
    main()
