# LLaMA SOAP Note Generator

A Large Language Model (LLM) fine-tuning project that generates structured SOAP clinical notes from doctor–patient medical conversations.

This project fine-tunes LLaMA 3.2 (1B) using LoRA (Low-Rank Adaptation) to efficiently generate structured SOAP summaries from dialogue data.

The model is trained on the Medical Dialogue to SOAP Summary Dataset and evaluated using ROUGE and BERTScore metrics.

## Features

- Fine-tuning LLaMA 3.2 1B
- LoRA parameter-efficient training
- Structured SOAP note generation
- Automatic ROUGE evaluation
- Automatic BERTScore evaluation
- Clean modular codebase
- Reproducible training pipeline
- Modular folder architecture for easy extension

## Project Structure

    llama-soap-lora/
    				│
    				├── README.md
    				├── requirements.txt
    				├── train.py
    				├── inference.py
    				├── evaluate_model.py
    				│
    				├── scripts/
    				│   └── install_dependencies.sh
    				│
    				├── config/
    				│   └── config.py
    				│
    				├── data/
    				│   └── dataset_loader.py
    				│
    				├── preprocessing/
    				│   └── preprocess.py
    				│
    				├── model/
    				│   └── lora_model.py
    				│
    				└── utils/
    					└── helpers.py
### Structure Overview

- README.md – Project overview and instructions.
- requirements.txt – Python dependencies for training, inference, and evaluation.
- train.py – Main script to fine-tune the LLaMA model with LoRA adapters on SOAP summarization.
- inference.py – Script for generating SOAP notes from new dialogues using the trained adapter.
- evaluate_model.py – Script to evaluate model outputs using ROUGE and BERTScore metrics.

### Directories
- scripts/ – Utility scripts (e.g., install_dependencies.sh) for setup.
- config/ – Configuration files (config.py) for model, training, and dataset parameters.
- data/ – Dataset loader and preprocessing helpers (dataset_loader.py).
- preprocessing/ – Data preprocessing functions for tokenization and formatting (preprocess.py).
- model/ – LoRA model definition and setup (lora_model.py).
- utils/ – Helper functions for training, inference, and evaluation (helpers.py).

### Workflow
- Load and preprocess medical dialogue datasets.
- Fine-tune a pre-trained LLaMA model using LoRA adapters for parameter-efficient training.
- Generate structured SOAP notes using the trained adapter.
- Evaluate predictions with ROUGE and BERTScore.

## Dataset

Dataset used:

    omi-health/medical-dialogue-to-soap-summary

Dataset contains:

- Doctor–patient conversations
- Prompt instructions
- Ground-truth SOAP summaries

The dataset is automatically downloaded using HuggingFace Datasets.

## Requirements

Python version:

    Python >= 3.9
    
Main libraries used:

    transformers 
    datasets 
    peft 
    accelerate 
    evaluate 
    rouge_score 
    bert_score 
    sentencepiece 
    torch 
    numpy 
    tqdm 
    huggingface_hub

## Install
### Clone Repository
    git clone https://github.com/sa05042/llama-soap-summarizer.git
    cd llama-soap-summarizer
    
### Install Dependencies
    pip install -r requirements.txt

## Train

Run the training script:
    
    python train.py

## Inference

Generate SOAP summaries using the trained adapter.
    
    python inference.py

## Evaluate

Run evaluation:
    
    python evaluate_model.py
## Author
Sabbir

Research Interests:
- Machine Learning
- Deep Learning
- Large Language Models
- Generative AI
- Privacy & Security
