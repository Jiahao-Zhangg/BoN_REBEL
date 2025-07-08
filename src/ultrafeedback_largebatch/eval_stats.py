import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List
import numpy as np

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    return parser.parse_args()


def main():

    # init
    args = parse_arguments()

    # load dataset
    dataset = load_dataset(args.input_repo, split='train')
    best_scores = np.maximum(dataset["response_0_reward"], dataset["response_1_reward"])
    mean_best = best_scores.mean()
    print("Mean of best-of-two scores:", mean_best)


if __name__ == "__main__":
    main()