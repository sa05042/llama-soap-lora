"""
Load LoRA adapter and generate SOAP summaries.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config.config import *
from preprocessing.preprocess import make_input
from utils.helpers import format_soap


def load_inference_model():

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    return model, tokenizer


def generate_single(model, tokenizer, example):

    input_text = make_input(example["prompt"], example["dialogue"])

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_TARGET_LENGTH,
        num_beams=4,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):].strip()

    return format_soap(decoded)