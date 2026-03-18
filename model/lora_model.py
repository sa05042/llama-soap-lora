"""
Loads base model and applies LoRA fine-tuning.
"""

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from config.config import MODEL_NAME


def load_lora_model():
    """
    Load base LLaMA model and apply LoRA.
    """

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.03,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model