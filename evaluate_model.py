"""
Evaluation using ROUGE and BERTScore.
"""

import numpy as np
import evaluate
from tqdm import tqdm

from inference import load_inference_model, generate_single
from data.dataset_loader import load_medical_dataset

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def evaluate_model(num_samples=100):

    dataset = load_medical_dataset()

    model, tokenizer = load_inference_model()

    preds = []
    refs = []

    subset = dataset["validation"].select(range(num_samples))

    for ex in tqdm(subset):

        pred = generate_single(model, tokenizer, ex)

        preds.append(pred)
        refs.append(ex["soap"])

    rouge_res = rouge.compute(predictions=preds, references=refs)

    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    print("ROUGE:", rouge_res)
    print("BERTScore:", np.mean(bert_res["f1"]))


if __name__ == "__main__":
    evaluate_model()