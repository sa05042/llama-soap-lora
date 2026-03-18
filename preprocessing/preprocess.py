"""
Handles prompt construction and tokenization.
"""

from transformers import AutoTokenizer
from config.config import MODEL_NAME, MAX_SEQ_LENGTH, MAX_INPUT_LENGTH


def load_tokenizer():
    """
    Load tokenizer for the LLaMA model.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    return tokenizer


def make_input(prompt, dialogue):
    """
    Construct model input prompt.
    """
    return (
        "You are a clinical summarizer. Generate a structured SOAP note from the conversation.\n\n"
        f"Instruction: {prompt}\n\nDialogue:\n{dialogue}\n\n"
        "Output format (exact):\nS:\nO:\nA:\nP:\n"
    )


def preprocess_function(batch, tokenizer):
    """
    Convert dataset examples into tokenized inputs
    suitable for causal language model training.
    """

    inputs_batch, attention_batch, labels_batch = [], [], []

    for p, d, s in zip(batch["prompt"], batch["dialogue"], batch["soap"]):

        prompt_text = make_input(p, d)
        full_text = prompt_text + " " + s

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_attention_mask=True,
        )

        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=False,
        )

        prompt_len = len(tokenized_prompt["input_ids"])

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        labels = input_ids.copy()

        # Mask prompt tokens so loss is only computed on target
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # Mask padding tokens
        labels = [tok if tok != tokenizer.pad_token_id else -100 for tok in labels]

        inputs_batch.append(input_ids)
        attention_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": inputs_batch,
        "attention_mask": attention_batch,
        "labels": labels_batch,
    }