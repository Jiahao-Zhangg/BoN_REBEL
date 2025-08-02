import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional


class ArmoRMPipeline:
    def __init__(
        self,
        model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        truncation: bool = True,
        trust_remote_code: bool = False,
        max_length: int = 4096,
        reward_names: Optional[List[str]] = None,
    ):
        """
        Args:
            reward_names: If None, pipeline will return a single aggregated score for each input.
                Otherwise, will return a tensor of scores for each reward specified in the list.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

        self.reward_indices = []
        for reward_name in (reward_names or []):    # Keep as empty list if reward_names is None
            try:
                self.reward_indices.append(self.get_all_reward_names().index(reward_name))
            except ValueError:
                raise ValueError(f"Reward name {reward_name} not found in the model. "
                                f"Available reward names: {self.get_all_reward_names()}")

    def get_all_reward_names(self) -> List[str]:
        return [
            'helpsteer-helpfulness',
            'helpsteer-correctness',
            'helpsteer-coherence',
            'helpsteer-complexity',
            'helpsteer-verbosity',
            'ultrafeedback-overall_score',
            'ultrafeedback-instruction_following',
            'ultrafeedback-truthfulness',
            'ultrafeedback-honesty',
            'ultrafeedback-helpfulness',
            'beavertails-is_safe',
            'prometheus-score',
            'argilla-overall_quality',
            'code-style',
            'code-explanation',
            'code-instruction-following',
            'code-readability',
        ]

    def tokenized_len(self, message: Dict[str, str]) -> int:
        """Returns the tokenized length of the given message. Helpful for sorting messages by tokenized length."""
        return len(self.tokenizer.apply_chat_template(
            message,
            truncation=self.truncation,
            max_length=self.max_length,
        ))

    def __call__(self, messages_batch: List[List[Dict[str, str]]]) -> List[float]:
        """Process a batch of messages and return a list of scores."""
        if not isinstance(messages_batch[0], list):
            # Handle single message case for backward compatibility
            messages_batch = [messages_batch]
            single_message = True
        else:
            single_message = False

        input_ids_batch = self.tokenizer.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids_batch)

        if len(self.reward_indices) == 0:
            scores = output.score.cpu()
        else:
            scores = output.rewards[:, self.reward_indices].cpu()

        if single_message:
            scores = scores.squeeze(0)
        
        return scores
