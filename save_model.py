import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hf_repo", type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize accelerator and load the checkpoint
    accelerator = Accelerator()
    device = accelerator.device

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Load your trained checkpoint
    accelerator.load_state(args.checkpoint_path)

    # Unwrap model
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Save to disk
    if args.output_dir is not None:
        tokenizer.save_pretrained(args.output_dir)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    # Upload to HuggingFace
    if args.hf_repo is not None:
        tokenizer.push_to_hub(args.hf_repo)
        unwrapped_model.push_to_hub(args.hf_repo)
