import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and plot A_vec[j] and j* vs N.")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/test_n_1_sel10_cur20_base20",
                        help="HF dataset repo containing current/base score vectors.")
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use 'mean' or 'majority' score vectors.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta scaling in A_vec computation.")
    parser.add_argument("--n_min", type=int, default=2, help="Minimum N (responses per side).")
    parser.add_argument("--n_max", type=int, default=20, help="Maximum N (responses per side).")
    parser.add_argument("--split", type=str, default=None,
                        help="Optional split to use (e.g., 'train'/'test'). If omitted, concatenates all.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save plots.")
    parser.add_argument("--sel_idx", type=int, default=1,
                        help="Selection response index to use when plotting g(z) vs N (1-based).")
    return parser.parse_args()


def collect_current_base_keys(column_names: List[str], score_type: str) -> Dict[Tuple[int, int], str]:
    pattern = re.compile(rf"^current_(\d+)_base_(\d+)_{score_type}$")
    mapping: Dict[Tuple[int, int], str] = {}
    for name in column_names:
        m = pattern.match(name)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            mapping[(i, j)] = name
    if not mapping:
        raise ValueError(f"No current_i_base_j_{score_type} columns found in dataset.")
    return mapping


def get_max_i_j(mapping: Dict[Tuple[int, int], str]) -> Tuple[int, int]:
    max_i = max(i for (i, _j) in mapping.keys())
    max_j = max(j for (_i, j) in mapping.keys())
    return max_i, max_j


def compute_A_vec_for_row(row, key_map: Dict[Tuple[int, int], str], N: int, beta: float) -> np.ndarray:
    # Determine available Ns per side for safety
    max_i, max_j = get_max_i_j(key_map)
    Ni = min(N, max_i)
    Nj = min(N, max_j)

    # Determine L from the first available vector
    first_key = key_map[(1, 1)]
    L = len(row[first_key])

    # For each base j in 1..Nj, aggregate over currents i in 1..Ni
    # inner_sums_j = (1/(beta*Ni)) * sum_i P_{i,j}  (element-wise over L)
    # A_vec_avg = (1/Nj) * sum_j exp(-inner_sums_j)  (element-wise over L)
    A_vec = np.zeros(L, dtype=float)
    for j in range(1, Nj + 1):
        acc = np.zeros(L, dtype=float)
        for i in range(1, Ni + 1):
            key = key_map.get((i, j))
            if key is None:
                raise ValueError(f"Missing column for current_{i}_base_{j} in row.")
            vec = np.asarray(row[key], dtype=float)
            if vec.shape[0] != L:
                raise ValueError(f"Inconsistent vector length for {key}: {vec.shape[0]} vs {L}.")
            acc += vec
        inner = acc / (beta * float(Ni))
        A_vec += np.exp(-inner)
    return A_vec / float(Nj)


def collect_selection_base_keys(column_names: List[str], score_type: str) -> Dict[Tuple[int, int], str]:
    pattern = re.compile(rf"^selection_(\d+)_base_(\d+)_{score_type}$")
    mapping: Dict[Tuple[int, int], str] = {}
    for name in column_names:
        m = pattern.match(name)
        if m:
            sel = int(m.group(1))
            base = int(m.group(2))
            mapping[(sel, base)] = name
    if not mapping:
        raise ValueError(f"No selection_i_base_j_{score_type} columns found in dataset.")
    return mapping


def compute_g_for_row(row,
                      cur_key_map: Dict[Tuple[int, int], str],
                      sel_key_map: Dict[Tuple[int, int], str],
                      N: int,
                      beta: float,
                      sel_idx: int) -> Tuple[float, int]:
    # Determine available Ns per side for safety
    max_i, max_j = get_max_i_j(cur_key_map)
    Ni = min(N, max_i)
    Nj = min(N, max_j)

    # Determine L from first key
    first_key = cur_key_map[(1, 1)]
    L = len(row[first_key])

    # Compute exp_terms per base (vectors length L)
    exp_terms_list: List[np.ndarray] = []
    for j in range(1, Nj + 1):
        acc = np.zeros(L, dtype=float)
        for i in range(1, Ni + 1):
            key = cur_key_map[(i, j)]
            vec = np.asarray(row[key], dtype=float)
            acc += vec
        inner = acc / (beta * float(Ni))
        exp_terms_list.append(np.exp(-inner))  # (L,)

    # A_sum and j* from sums (not averaged) to match g(z) definition
    A_sum = np.zeros(L, dtype=float)
    for term in exp_terms_list:
        A_sum += term
    j_star = int(np.argmax(A_sum))

    # Numerator: sum_b P_{j*}(x, z_sel, b) * exp_terms_b[j*]
    numer = 0.0
    for b in range(1, Nj + 1):
        sel_key = sel_key_map.get((sel_idx, b))
        if sel_key is None:
            raise ValueError(f"Missing selection_{sel_idx}_base_{b} column in row.")
        p_vec = np.asarray(row[sel_key], dtype=float)
        numer += float(p_vec[j_star]) * float(exp_terms_list[b - 1][j_star])

    denom = float(A_sum[j_star])
    g_val = numer / denom if denom > 0 else 0.0
    return g_val, j_star


def main():
    args = parse_args()

    ds = load_dataset(args.input_repo)
    if isinstance(ds, DatasetDict):
        if args.split is not None:
            if args.split not in ds:
                raise ValueError(f"Requested split '{args.split}' not in dataset: {list(ds.keys())}")
            dataset: Dataset = ds[args.split]
        else:
            # Concatenate all available splits
            datasets = [ds[k] for k in ds.keys()]
            dataset = datasets[0]
            for extra in datasets[1:]:
                dataset = Dataset.from_list(list(dataset) + list(extra))
    else:
        dataset = ds  # type: ignore

    key_map = collect_current_base_keys(dataset.column_names, args.score_type)
    sel_key_map = collect_selection_base_keys(dataset.column_names, args.score_type)
    max_i, max_j = get_max_i_j(key_map)
    n_max_possible = min(args.n_max, max_i, max_j)
    n_values = list(range(max(args.n_min, 2), n_max_possible + 1))

    # Establish L from first row
    L = len(dataset[0][key_map[(1, 1)]])

    # For each N, compute mean A_vec across rows and derive j* from that mean
    mean_A_by_N: Dict[int, np.ndarray] = {}
    mean_h_by_N: Dict[int, np.ndarray] = {}
    jstar_by_N: Dict[int, int] = {}
    mean_g_by_N: Dict[int, float] = {}

    for N in n_values:
        A_acc = np.zeros(L, dtype=float)
        H_acc = np.zeros(L, dtype=float)
        g_acc = 0.0
        for row in dataset:
            A_vec = compute_A_vec_for_row(row, key_map, N, args.beta)
            A_acc += A_vec
            # h_j(x) = -beta * log A_avg_j(x)
            H_acc += (-args.beta) * np.log(np.clip(A_vec, 1e-12, None))
            # g(z) for the chosen selection index (using sums in denominator)
            g_val, _ = compute_g_for_row(row, key_map, sel_key_map, N, args.beta, args.sel_idx)
            g_acc += g_val
        mean_A = A_acc / float(len(dataset))
        mean_H = H_acc / float(len(dataset))
        mean_A_by_N[N] = mean_A
        mean_h_by_N[N] = mean_H
        jstar_by_N[N] = int(np.argmax(mean_A))
        mean_g_by_N[N] = g_acc / float(len(dataset))

    # Plot A_vec[j] vs N (for each j)
    plt.figure(figsize=(10, 6))
    for j in range(L):
        y = [mean_A_by_N[N][j] for N in n_values]
        plt.plot(n_values, y, marker='o', linewidth=1.5, label=f"j={j}")
    plt.xlabel("N (num currents/bases)")
    plt.ylabel("A_vec_avg[j] (avg bases, avg rows)")
    plt.title(f"A_vec_avg[j] vs N  |  score={args.score_type}, beta={args.beta}")
    if L <= 12:
        plt.legend(ncol=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    a_fig_path = f"{args.output_dir}/A_vecAvg_vs_N_{args.score_type}_beta{args.beta}.png"
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(a_fig_path, dpi=150)
    print(f"Saved A_vec plot to {a_fig_path}")

    # Plot h_j vs N (for each j): h_j = -beta log E_base[exp(-E_y P_j / beta)]
    plt.figure(figsize=(10, 6))
    for j in range(L):
        y = [mean_h_by_N[N][j] for N in n_values]
        plt.plot(n_values, y, marker='o', linewidth=1.5, label=f"j={j}")
    plt.xlabel("N (num currents/bases)")
    plt.ylabel("h_j (avg across rows)")
    plt.title(f"h_j vs N  |  score={args.score_type}, beta={args.beta}")
    if L <= 12:
        plt.legend(ncol=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    h_fig_path = f"{args.output_dir}/h_j_vs_N_{args.score_type}_beta{args.beta}.png"
    plt.savefig(h_fig_path, dpi=150)
    print(f"Saved h_j plot to {h_fig_path}")

    # Plot g(z_sel) vs N
    plt.figure(figsize=(8, 4))
    y_g = [mean_g_by_N[N] for N in n_values]
    plt.plot(n_values, y_g, marker='d', linewidth=1.5)
    plt.xlabel("N (num currents/bases)")
    plt.ylabel(f"g(z_sel={args.sel_idx}) (avg across rows)")
    plt.title(f"g(z_sel={args.sel_idx}) vs N  |  score={args.score_type}, beta={args.beta}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    g_fig_path = f"{args.output_dir}/g_sel{args.sel_idx}_vs_N_{args.score_type}_beta{args.beta}.png"
    plt.savefig(g_fig_path, dpi=150)
    print(f"Saved g(z) plot to {g_fig_path}")

    # Plot j* vs N
    plt.figure(figsize=(8, 4))
    ystar = [jstar_by_N[N] for N in n_values]
    plt.plot(n_values, ystar, marker='s', linewidth=1.5)
    plt.yticks(sorted(set(ystar)))
    plt.xlabel("N (num currents/bases)")
    plt.ylabel("j* (argmax_j mean A_vec_avg[j])")
    plt.title(f"j* vs N  |  score={args.score_type}, beta={args.beta}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    j_fig_path = f"{args.output_dir}/j_star_vs_N_{args.score_type}_beta{args.beta}.png"
    plt.savefig(j_fig_path, dpi=150)
    print(f"Saved j* plot to {j_fig_path}")


if __name__ == "__main__":
    main()
