"""
Loads the medical SOAP summarization dataset.
"""

from datasets import load_dataset


def load_medical_dataset():
    """
    Load dataset from HuggingFace hub.
    """
    dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary")
    print(dataset)
    return dataset