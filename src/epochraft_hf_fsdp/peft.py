from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from peft import LoraConfig, PeftConfig


@dataclass
class PeftConfigFromFile:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_model: bool = False


def generate_peft_config(config: PeftConfigFromFile) -> PeftConfig:
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
        target_modules=config.target_modules,
        inference_mode=config.inference_model,
    )
    return lora_config
