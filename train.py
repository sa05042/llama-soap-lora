"""
Main training pipeline.
Run this file to train the model.
"""

import os
from transformers import Trainer, TrainingArguments, default_data_collator

from config.config import *
from data.dataset_loader import load_medical_dataset
from preprocessing.preprocess import load_tokenizer, preprocess_function
from model.lora_model import load_lora_model


def main():

    dataset = load_medical_dataset()

    tokenizer = load_tokenizer()

    remove_cols = dataset["train"].column_names

    tokenized = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=remove_cols,
    )

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    train_full = tokenized["train"]
    eval_dataset = tokenized["validation"]

    if TRAIN_SAMPLES is not None:
        train_dataset = train_full.select(range(min(TRAIN_SAMPLES, len(train_full))))
    else:
        train_dataset = train_full

    print("Train size:", len(train_dataset))
    print("Eval size:", len(eval_dataset))

    model = load_lora_model()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print("LoRA adapter saved to:", ADAPTER_DIR)


if __name__ == "__main__":
    main()