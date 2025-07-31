import deepspeed
import math
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb

from accelerate import Accelerator
from dataclasses import asdict, dataclass, field
from typing import List
from datasets import load_dataset, DatasetDict
from functools import partial
from rich.console import Console
from rich.pretty import pprint
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
from typing import Literal, Optional


@dataclass
class REBELHParams:
    num_updates: tyro.conf.Suppress[int] = 1000
    eta: float = 1e4
    bon: bool = False
    winrate: bool = False
    """If True, use current method. If False, use original REBEL with reward gap."""
    n_reward_models: int = 1
    """The number of reward models to use"""
    weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    """The weights of the reward models"""


@dataclass
class TaskHParams:
    input_repo: str = None
    """the output repo of filter_tokenize.py"""
    maxlen_prompt: int = 1024
    maxlen: int = 2048
    temperature: float = 0.8


@dataclass
class Args:
    # common args
    test: bool = False
    """if test is true, the dataset will be reduced to 100 samples"""
    exp_name: str = "ultrafeedback_rebel"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ultrafeedback"
    """the wandb's project name"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    print_sample_output_freq: int = 50
    """How often to print sample output"""

    # optimizer args
    eps: float = 1e-8
    """the epsilon value for the optimizer"""
    lr: float = 3e-7
    """learning rate"""
    weight_decay: float = 1e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    warmup_ratio: float = 0.1
    """warmup ratio"""
    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping"""
    enable_grad_clip: bool = False
    """whether to enable gradient clipping (set to False to just monitor norms first)"""

    gradient_accumulation_steps: int = 64 # used to be 16
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 1
    """per rank eval batch size"""
    total_episodes: int = 60000
    """The total number of episodes to train"""

    # optional args filled while running
    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    batch_size: Optional[int] = 128
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_batch_size: Optional[int] = 128
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""

    # other args
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    """the name of the pretrained model to use"""
    output_dir: str = None
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    rebel: REBELHParams = field(default_factory=REBELHParams)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def gather_logprob(args, model, tokenizer, query, response, device):

    query_response = torch.cat((query, response), dim=-1).long().to(device).unsqueeze(0)
    response = response.long().to(device).unsqueeze(0)
    attention_mask = query_response != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_response, ~attention_mask, tokenizer.eos_token_id)
    with torch.no_grad():
        output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                 )
        logits = output.logits[:, args.task.maxlen_prompt - 1 : -1]
        logits /= args.task.temperature + 1e-7
        all_logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, input_ids[:, args.task.maxlen_prompt:].unsqueeze(-1)).squeeze(-1)
        sequence_length = first_true_indices(response == tokenizer.pad_token_id) - 1
        seq_mask = torch.arange(args.task.maxlen, device=device).unsqueeze(0).expand_as(response) <= sequence_length.unsqueeze(1)
        
        return (logprob * seq_mask).sum(-1)


def gather_all_logprob(args, process_idx, policy, tokenizer, dataset, device):

    batch_size = len(dataset) // args.world_size + 1
    start_idx = batch_size * process_idx

    # make batch size same for accelerator.gather
    if start_idx + batch_size > len(dataset):
        start_idx = len(dataset) - batch_size

    chosen_logprob, reject_logprob, index = [], [], []

    with torch.no_grad():
        for i in tqdm(range(start_idx, start_idx + batch_size)):

            chosen_logprob.append(gather_logprob(args, policy, tokenizer, dataset[i]["llama_prompt_tokens"], dataset[i]["llama_chosen_tokens"], device))
            reject_logprob.append(gather_logprob(args, policy, tokenizer, dataset[i]["llama_prompt_tokens"], dataset[i]["llama_reject_tokens"], device))
            index.append(i)

        chosen_logprob = torch.cat(chosen_logprob)
        reject_logprob = torch.cat(reject_logprob)
        index = torch.LongTensor(index).to(device)

    chosen_logprob = accelerator.gather(chosen_logprob).cpu().tolist()
    reject_logprob = accelerator.gather(reject_logprob).cpu().tolist()
    index = accelerator.gather(index).cpu().tolist()

    chosen_logprobs = [0] * len(dataset)
    reject_logprobs = [0] * len(dataset)

    for i, data_i in enumerate(index):
        chosen_logprobs[data_i] = chosen_logprob[i]
        reject_logprobs[data_i] = reject_logprob[i]
        
    return chosen_logprobs, reject_logprobs


def evaluate(args, policy, tokenizer, dataloader):

    device = policy.device
    loss, sign_align = [], []
    with torch.no_grad():
        for data in tqdm(dataloader):
            
            responses = torch.cat((data["llama_chosen_tokens"], data["llama_reject_tokens"]), dim=0)
            logprobs = torch.cat((data["chosen_logprob"], data["reject_logprob"]), dim=0)
            query_responses = torch.cat((torch.cat((data["llama_prompt_tokens"], data["llama_prompt_tokens"]), dim=0), responses), dim=1)
            sequence_length = first_true_indices(responses == tokenizer.pad_token_id) - 1
            seq_mask = torch.arange(args.task.maxlen, device=device).unsqueeze(0).expand_as(responses) <= sequence_length.unsqueeze(1)

            attention_mask = query_responses != tokenizer.pad_token_id
            input_ids = torch.masked_fill(query_responses, ~attention_mask, tokenizer.eos_token_id)

            output = policy(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = output.logits[:, args.task.maxlen_prompt - 1 : -1]
            logits /= args.task.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, args.task.maxlen_prompt:].unsqueeze(-1)).squeeze(-1)
            new_logprobs = (new_logprobs * seq_mask).sum(-1)
            ratio_logprob = new_logprobs - logprobs
            ratio_logprob = ratio_logprob[:args.per_device_eval_batch_size] - ratio_logprob[args.per_device_eval_batch_size:]

            if args.rebel.bon:
                # Use precomputed weighted difference if available, otherwise compute on the fly
                weighted_g_diff = data["weighted_g_diff"]
                reg_diff = ratio_logprob - args.rebel.eta * weighted_g_diff
            elif args.rebel.winrate:
                # Use precomputed weighted difference if available, otherwise compute on the fly
                weighted_g_winrate_diff = data["weighted_g_winrate_diff"]
                reg_diff = ratio_logprob - args.rebel.eta * weighted_g_winrate_diff
            else:
                chosen_reward = data["chosen_reward_1"]
                reject_reward = data["reject_reward_1"]
                reg_diff = ratio_logprob - args.rebel.eta * (chosen_reward - reject_reward)
            loss.append((reg_diff ** 2).mean().reshape(1))

            sign_align.append((ratio_logprob > 0).float().mean().reshape(1))

    loss = torch.cat(loss)
    sign_align = torch.cat(sign_align)
    return {"val_loss" : loss, "sign_align" : sign_align}


if __name__ == '__main__':

    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.rebel.num_updates = args.total_episodes // args.batch_size 
    print(f"total_episodes: {args.total_episodes}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_updates: {args.rebel.num_updates}")

    # logging
    console = Console(force_terminal=True)
    accelerator.wait_for_everyone()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    accelerator.print("Wandb run name: ", run_name)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    # policy
    tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model, 
                    padding_side='right',
                    trust_remote_code=True,
                )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Loading model from {args.base_model}")
    policy = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
    disable_dropout_in_model(policy)

    base_columns = ["llama_prompt_tokens", "llama_chosen_tokens","llama_reject_tokens"]
    for i in range(args.rebel.n_reward_models):
        base_columns.append(f"chosen_reward_{i}")
        base_columns.append(f"reject_reward_{i}")
        base_columns.append(f"g_chosen_{i}")
        base_columns.append(f"g_reject_{i}")
        base_columns.append(f"g_winrate_chosen_{i}")
        base_columns.append(f"g_winrate_reject_{i}")
    logprob_columns = base_columns + ["chosen_logprob", "reject_logprob"]

    # Prompt Collection Dataset
    compute_log = False
    try:
        dataset = load_dataset(args.task.input_repo + '_logprob', split='train')
        dataset = dataset.with_format("torch", columns=logprob_columns)
        if args.test:
            dataset = dataset.select(range(100))
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.input_repo + '_logprob', split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=logprob_columns)
        if args.test:
            validation_dataset = validation_dataset.select(range(100))
    except:
        dataset = load_dataset(args.task.input_repo, split='train')
        dataset = dataset.with_format("torch", columns=base_columns)
        if args.test:
            dataset = dataset.select(range(100))
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.input_repo, split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=base_columns)
        compute_log = True
        if args.test:
            validation_dataset = validation_dataset.select(range(100))

    if accelerator.is_main_process:
        pprint(policy.config)

    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            policy.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.rebel.num_updates * args.warmup_ratio * args.world_size), args.rebel.num_updates * args.world_size)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, _, scheduler = accelerator.prepare(policy, optimizer, temp_dataloader, scheduler)

    if compute_log:
        accelerator.print('gathering validation logprob')
        chosen_logprob, reject_logprob = gather_all_logprob(args, accelerator.process_index, accelerator.unwrap_model(policy), tokenizer, validation_dataset, device)
        validation_dataset = validation_dataset.add_column("chosen_logprob", chosen_logprob)
        validation_dataset = validation_dataset.add_column("reject_logprob", reject_logprob)
        validation_dataset = validation_dataset.with_format("torch", columns=logprob_columns)

        accelerator.print('gathering logprob')
        chosen_logprob, reject_logprob = gather_all_logprob(args, accelerator.process_index, accelerator.unwrap_model(policy), tokenizer, dataset, device)
        dataset = dataset.add_column("chosen_logprob", chosen_logprob)
        dataset = dataset.add_column("reject_logprob", reject_logprob)
        dataset = dataset.with_format("torch", columns=logprob_columns)
        if accelerator.is_main_process:
            temp = DatasetDict({
                "train" : dataset,
                "test"  : validation_dataset,
            })
            temp.push_to_hub(args.task.input_repo + '_logprob')
    
    # Precompute weighted combinations for efficiency if using multiple reward models
    if args.rebel.n_reward_models > 1 and (args.rebel.bon or args.rebel.winrate):
        accelerator.print("Precomputing weighted combinations for efficiency...")
        
        # Precompute for training dataset
        weighted_g_diff = []
        weighted_g_winrate_diff = []
        
        for i in range(len(dataset)):
            if args.rebel.bon:
                g_diff = []
                for j in range(args.rebel.n_reward_models):
                    g_chosen = dataset[i][f"g_chosen_{j}"]
                    g_reject = dataset[i][f"g_reject_{j}"]
                    g_diff.append(g_chosen - g_reject)
                weighted_g_diff.append(sum(args.rebel.weights[j] * g_diff[j] for j in range(args.rebel.n_reward_models)).item())
            
            if args.rebel.winrate:
                g_winrate_diff = []
                for j in range(args.rebel.n_reward_models):
                    g_winrate_chosen = dataset[i][f"g_winrate_chosen_{j}"]
                    g_winrate_reject = dataset[i][f"g_winrate_reject_{j}"]
                    g_winrate_diff.append(g_winrate_chosen - g_winrate_reject)
                weighted_g_winrate_diff.append(sum(args.rebel.weights[j] * g_winrate_diff[j] for j in range(args.rebel.n_reward_models)).item())
        
        # Add precomputed columns to dataset
        if args.rebel.bon:
            dataset = dataset.add_column("weighted_g_diff", weighted_g_diff)
        if args.rebel.winrate:
            dataset = dataset.add_column("weighted_g_winrate_diff", weighted_g_winrate_diff)
            
        # Precompute for validation dataset
        weighted_g_diff_val = []
        weighted_g_winrate_diff_val = []
        
        for i in range(len(validation_dataset)):
            if args.rebel.bon:
                g_diff = []
                for j in range(args.rebel.n_reward_models):
                    g_chosen = validation_dataset[i][f"g_chosen_{j}"]
                    g_reject = validation_dataset[i][f"g_reject_{j}"]
                    g_diff.append(g_chosen - g_reject)
                weighted_g_diff_val.append(sum(args.rebel.weights[j] * g_diff[j] for j in range(args.rebel.n_reward_models)).item())
            
            if args.rebel.winrate:
                g_winrate_diff = []
                for j in range(args.rebel.n_reward_models):
                    g_winrate_chosen = validation_dataset[i][f"g_winrate_chosen_{j}"]
                    g_winrate_reject = validation_dataset[i][f"g_winrate_reject_{j}"]
                    g_winrate_diff.append(g_winrate_chosen - g_winrate_reject)
                weighted_g_winrate_diff_val.append(sum(args.rebel.weights[j] * g_winrate_diff[j] for j in range(args.rebel.n_reward_models)).item())
        
        # Add precomputed columns to validation dataset
        if args.rebel.bon:
            validation_dataset = validation_dataset.add_column("weighted_g_diff", weighted_g_diff_val)
        if args.rebel.winrate:
            validation_dataset = validation_dataset.add_column("weighted_g_winrate_diff", weighted_g_winrate_diff_val)
            
        # Recreate dataloaders with updated datasets including precomputed columns
        updated_logprob_columns = logprob_columns.copy()
        if args.rebel.bon:
            updated_logprob_columns.append("weighted_g_diff")
        if args.rebel.winrate:
            updated_logprob_columns.append("weighted_g_winrate_diff")
        
        dataset = dataset.with_format("torch", columns=updated_logprob_columns)
        validation_dataset = validation_dataset.with_format("torch", columns=updated_logprob_columns)
        
        dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader)
        
        accelerator.print("Weighted combinations precomputed successfully!")
    
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())

    accelerator.print("===training policy===")
    torch.manual_seed(local_seed)  # reset the local seed again
    global_step = 0
    start_time = time.time()

    kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    chosen_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    reject_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    ratio_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    grad_norm_stats = torch.zeros(args.gradient_accumulation_steps, device=device)

    policy.train()
    for update in range(1, args.rebel.num_updates + 1):

        # update parameters
        global_step += 1 * args.batch_size
        lrnow = optimizer.param_groups[0]["lr"]

        # save model
        if (update - 1) % args.print_sample_output_freq == 0:
            eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
            writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
            writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
            if args.output_dir:
                accelerator.wait_for_everyone()
                output_dir = os.path.join(args.output_dir, run_name)
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                accelerator.save_state(output_dir=output_dir)
                accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        # training
        data = next(iter_dataloader)

        gradient_accumulation_idx = 0
        for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
            mini_batch_end = mini_batch_start + args.per_device_train_batch_size
            with accelerator.accumulate(policy):
                mb_query = data["llama_prompt_tokens"][mini_batch_start : mini_batch_end]

                mb_chosen_response = data["llama_chosen_tokens"][mini_batch_start : mini_batch_end]
                mb_chosen_logprob = data["chosen_logprob"][mini_batch_start : mini_batch_end]

                mb_reject_response = data["llama_reject_tokens"][mini_batch_start : mini_batch_end]
                mb_reject_logprob = data["reject_logprob"][mini_batch_start : mini_batch_end]

                mb_g_chosen = []
                mb_g_reject = []
                mb_g_winrate_chosen = []
                mb_g_winrate_reject = []
                for i in range(args.rebel.n_reward_models):
                    mb_g_chosen.append(data[f"g_chosen_{i}"][mini_batch_start : mini_batch_end])
                    mb_g_reject.append(data[f"g_reject_{i}"][mini_batch_start : mini_batch_end])
                    mb_g_winrate_chosen.append(data[f"g_winrate_chosen_{i}"][mini_batch_start : mini_batch_end])
                    mb_g_winrate_reject.append(data[f"g_winrate_reject_{i}"][mini_batch_start : mini_batch_end])
                mb_g_chosen = torch.stack(mb_g_chosen, dim=1)
                mb_g_reject = torch.stack(mb_g_reject, dim=1)
                mb_g_winrate_chosen = torch.stack(mb_g_winrate_chosen, dim=1)
                mb_g_winrate_reject = torch.stack(mb_g_winrate_reject, dim=1)

                mb_chosen_reward = []
                mb_reject_reward = []
                for i in range(args.rebel.n_reward_models):
                    mb_chosen_reward.append(data[f"chosen_reward_{i}"][mini_batch_start : mini_batch_end])
                    mb_reject_reward.append(data[f"reject_reward_{i}"][mini_batch_start : mini_batch_end])
                mb_chosen_reward = torch.stack(mb_chosen_reward, dim=1)
                mb_reject_reward = torch.stack(mb_reject_reward, dim=1)

                mb_responses = torch.cat((mb_chosen_response, mb_reject_response), dim=0)
                mb_logprobs = torch.cat((mb_chosen_logprob, mb_reject_logprob), dim=0)
                mb_query_responses = torch.cat((torch.cat((mb_query, mb_query), dim=0), mb_responses), dim=1)
                mb_sequence_length = first_true_indices(mb_responses == tokenizer.pad_token_id) - 1
                mb_seq_mask = torch.arange(args.task.maxlen, device=device).unsqueeze(0).expand_as(mb_responses) <= mb_sequence_length.unsqueeze(1)

                attention_mask = mb_query_responses != tokenizer.pad_token_id
                input_ids = torch.masked_fill(mb_query_responses, ~attention_mask, tokenizer.eos_token_id)

                output = policy(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output.logits[:, args.task.maxlen_prompt - 1 : -1]
                logits /= args.task.temperature + 1e-7
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, args.task.maxlen_prompt:].unsqueeze(-1)).squeeze(-1)
                new_logprobs = (new_logprobs * mb_seq_mask).sum(-1)

                if update == 1:
                    print(('logprobs:', new_logprobs, mb_logprobs))

                ratio_logprob = new_logprobs - mb_logprobs
                ratio_logprob = ratio_logprob[:args.per_device_train_batch_size] - ratio_logprob[args.per_device_train_batch_size:]
                
                if args.rebel.bon:
                    # Use precomputed weighted difference if available, otherwise compute on the fly
                    weighted_g_diff = data["weighted_g_diff"][mini_batch_start : mini_batch_end]
                    reg_diff = ratio_logprob - args.rebel.eta * weighted_g_diff
                elif args.rebel.winrate:
                    # Use precomputed weighted difference if available, otherwise compute on the fly
                    weighted_g_winrate_diff = data["weighted_g_winrate_diff"][mini_batch_start : mini_batch_end]
                    reg_diff = ratio_logprob - args.rebel.eta * weighted_g_winrate_diff
                else:
                    reward_diff = mb_chosen_reward - mb_reject_reward
                    reg_diff = ratio_logprob - args.rebel.eta * reward_diff
                
                    
                loss = (reg_diff ** 2).mean()

                accelerator.backward(loss)
                
                # Gradient norm computation and optional clipping
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    # Always compute gradient norm for monitoring
                    total_norm = 0.0
                    for p in policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** (1. / 2)
                    
                    # Only clip if enabled
                    if args.enable_grad_clip:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logprobs_diff = new_logprobs - mb_logprobs
                    ratio = torch.exp(logprobs_diff)
                    kl_stats[gradient_accumulation_idx] = logprobs_diff.mean()
                    chosen_kl_stats[gradient_accumulation_idx] = logprobs_diff[:args.per_device_train_batch_size].mean()
                    reject_kl_stats[gradient_accumulation_idx] = logprobs_diff[args.per_device_train_batch_size:].mean()
                    loss_stats[gradient_accumulation_idx] = loss
                    ratio_stats[gradient_accumulation_idx] = ratio.mean()
                    grad_norm_stats[gradient_accumulation_idx] = grad_norm
            gradient_accumulation_idx += 1
        if accelerator.is_main_process:
            grad_norm_mean = grad_norm_stats.mean().item()
            clip_status = f"(clip={args.enable_grad_clip})" if args.enable_grad_clip else "(monitoring)"
            console.print(
                f"update",
                update,
                "kl_stats",
                kl_stats.mean().item(),
                "loss",
                loss_stats.mean().item(),
                "grad_norm",
                f"{grad_norm_mean:.4f}{clip_status}",
            )

        with torch.no_grad():
            writer.add_scalar("objective/kl", accelerator.gather(kl_stats).mean().item(), update)
            writer.add_scalar("objective/chosen_kl", accelerator.gather(chosen_kl_stats).mean().item(), update)
            writer.add_scalar("objective/reject_kl", accelerator.gather(reject_kl_stats).mean().item(), update)
            writer.add_scalar("rebel/loss/policy", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("rebel/loss/policy_avg", accelerator.gather(loss_stats).mean().item(), update)
            
            writer.add_scalar("rebel/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("rebel/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("rebel/grad_norm", accelerator.gather(grad_norm_stats).mean().item(), update)
            writer.add_scalar("rebel/lr", lrnow, update)
            writer.add_scalar("rebel/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("rebel/eps", eps, update)
            accelerator.print("rebel/eps", eps, update)
            torch.cuda.empty_cache()

    # save model
    eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
    writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
    writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
    if args.output_dir:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        accelerator.save_state(output_dir=output_dir)
        accelerator.wait_for_everyone()
    torch.cuda.empty_cache()