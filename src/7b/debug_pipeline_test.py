import argparse
import subprocess
import sys
from datasets import load_dataset, concatenate_datasets


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Debug pipeline runner for preprocess + filters")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--raw_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale")
    parser.add_argument("--nocheck_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_nocheck_rescale")
    parser.add_argument("--preprocessed_repo", type=str, default="MisDrifter/1013_preprocessed")
    parser.add_argument("--output_repo_prefix", type=str, default="MisDrifter/1013")
    parser.add_argument("--limit_rows", type=int, default=1)
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--score_type", type=str, default="mean")
    parser.add_argument("--slicing_idx", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=1)
    args = parser.parse_args()

    py = sys.executable

    # 1) Preprocess common with a single row
    run([
        py, "preprocess_common.py",
        "--model", args.model,
        "--input_repo", args.raw_repo,
        "--output_repo", args.preprocessed_repo,
        "--maxlen", str(args.maxlen),
        "--maxlen_prompt", str(args.maxlen_prompt),
        "--slicing_idx", str(args.slicing_idx),
        "--test_size", str(args.test_size),
        "--seed", str(args.seed),
        "--limit_rows", str(args.limit_rows),
    ])

    # Derive distinct prefixes to avoid repo collisions for comparison
    out_prefix_7b = args.output_repo_prefix + "_7b"
    out_prefix_chk = args.output_repo_prefix + "_chk"

    # 2) Run 7b filters on preprocessed repo (vector-based methods)
    seven_b_vec = [
        ("filter_tokenize_judge_expand.py", []),
        ("filter_tokenize_judge_fixed_expand.py", []),
        ("filter_tokenize_judge_min_expand.py", []),
    ]
    for script, extra in seven_b_vec:
        run([
            py, script,
            "--model", args.model,
            "--input_repo", args.preprocessed_repo,
            "--maxlen", str(args.maxlen),
            "--maxlen_prompt", str(args.maxlen_prompt),
            "--beta", str(args.beta),
            "--slicing_idx", str(args.slicing_idx),
            "--score_type", args.score_type,
            "--output_repo_prefix", out_prefix_7b,
            "--limit_rows", str(args.limit_rows),
            *extra,
        ])

    # 2b) Prepare nocheck-matched dataset and run 7b nocheck filter
    matched_nocheck_repo = args.preprocessed_repo + "_nocheck_matched"
    run([
        py, "match_nocheck.py",
        "--preprocessed_repo", args.preprocessed_repo,
        "--nocheck_repo", args.nocheck_repo,
        "--output_repo", matched_nocheck_repo,
    ])

    run([
        py, "filter_tokenize_judge_nocheck_expand.py",
        "--model", args.model,
        "--input_repo", matched_nocheck_repo,
        "--maxlen", str(args.maxlen),
        "--maxlen_prompt", str(args.maxlen_prompt),
        "--beta", str(args.beta),
        "--slicing_idx", str(args.slicing_idx),
        "--score_type", args.score_type,
        "--output_repo_prefix", out_prefix_7b,
        "--limit_rows", str(args.limit_rows),
    ])

    # 3) Run checklist filters directly on raw repo (vector-based methods)
    checklist_dir = "../checklist_judge_data_parallel"
    checklist_vec = [
        (f"{checklist_dir}/filter_tokenize_judge_expand.py", []),
        (f"{checklist_dir}/filter_tokenize_judge_fixed_expand.py", []),
        (f"{checklist_dir}/filter_tokenize_judge_min_expand.py", []),
    ]
    for script, extra in checklist_vec:
        run([
            py, script,
            "--model", args.model,
            "--input_repo", args.raw_repo,
            "--maxlen", str(args.maxlen),
            "--maxlen_prompt", str(args.maxlen_prompt),
            "--beta", str(args.beta),
            "--slicing_idx", str(args.slicing_idx),
            "--score_type", args.score_type,
            "--output_repo_prefix", out_prefix_chk,
            "--test_size", str(args.test_size),
            "--seed", str(args.seed),
            "--limit_rows", str(args.limit_rows),
            *extra,
        ])

    # 3b) Run checklist nocheck filter on nocheck repo (scalar-based)
    run([
        py, f"{checklist_dir}/filter_tokenize_judge_nocheck_expand.py",
        "--model", args.model,
        "--input_repo", args.nocheck_repo,
        "--maxlen", str(args.maxlen),
        "--maxlen_prompt", str(args.maxlen_prompt),
        "--beta", str(args.beta),
        "--slicing_idx", str(args.slicing_idx),
        "--score_type", args.score_type,
        "--output_repo_prefix", out_prefix_chk,
        "--test_size", str(args.test_size),
        "--seed", str(args.seed),
        "--limit_rows", str(args.limit_rows),
    ])

    # 4) Compare g values between 7b outputs and checklist outputs
    def method_suffixes():
        return [
            ("expand", f"_{args.score_type}_beta_{args.beta}_multi_expand_tokenized"),
            ("fixed_expand", f"_{args.score_type}_beta_{args.beta}_fixed_expand_tokenized"),
            ("min_expand", f"_{args.score_type}_min_expand_ver2_tokenized"),
            ("nocheck_expand", f"_{args.score_type}_maxlenp_{args.maxlen_prompt}_beta_{args.beta}_nocheck_tokenized"),
        ]

    def nearly_equal(a: float, b: float, tol: float = 1e-8) -> bool:
        return abs(a - b) <= tol

    def compare_repo(repo_7b: str, repo_chk: str):
        # Merge train and test
        ds7_train = load_dataset(repo_7b, split="train")
        ds7_test = load_dataset(repo_7b, split="test")
        ds7 = concatenate_datasets([ds7_train, ds7_test])

        dsc_train = load_dataset(repo_chk, split="train")
        dsc_test = load_dataset(repo_chk, split="test")
        dsc = concatenate_datasets([dsc_train, dsc_test])

        # Build a quick lookup on checklist side by (prompt, qwen_chosen, qwen_reject)
        index_c = {}
        for row in dsc:
            prompt = row.get("prompt")
            qchosen = row.get("qwen_chosen", row.get("chosen"))
            qreject = row.get("qwen_reject", row.get("reject"))
            key = (prompt, qchosen, qreject)
            index_c[key] = (float(row["g_chosen"]), float(row["g_reject"]))

        # Find first matching row in 7b and compare g values
        for row in ds7:
            prompt = row.get("prompt")
            qchosen = row.get("qwen_chosen", row.get("chosen"))
            qreject = row.get("qwen_reject", row.get("reject"))
            key = (prompt, qchosen, qreject)
            if key in index_c:
                g7 = (float(row["g_chosen"]), float(row["g_reject"]))
                gc = index_c[key]
                if not (nearly_equal(g7[0], gc[0]) and nearly_equal(g7[1], gc[1])):
                    raise AssertionError(f"g mismatch for pair={key}: 7b={g7} chk={gc}")
                print(f"OK: matched single pair for {repo_7b} vs {repo_chk}")
                return
        raise AssertionError("No common (prompt, qwen_chosen, qwen_reject) row found between 7b and checklist outputs")

    for tag, suffix in method_suffixes():
        repo7 = out_prefix_7b + suffix
        repoc = out_prefix_chk + suffix
        print(f"Comparing method {tag}: {repo7} vs {repoc}")
        compare_repo(repo7, repoc)


if __name__ == "__main__":
    main()


