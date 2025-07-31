# Reddit Finance SFT (Supervised Fine-Tuning)

A machine learning project that fine-tunes language models on Reddit finance subreddit data using LoRA (Low-Rank Adaptation) for efficient training.

## Overview

This project implements supervised fine-tuning (SFT) on the Qwen3-0.6B model using data from finance-related Reddit subreddits. The goal is to create a model that can better understand and respond to financial discussions, questions, and analysis in a "reddity" way.


## Overview

- **Data Preprocessing**: Filters and cleans Reddit finance data from the `winddude/reddit_finance_43_250k` dataset
- **Pattern Filtering**: Removes low-quality posts using predefined patterns
- **Text Cleaning**: Removes URLs and user mentions for better training data quality
- **LoRA Training**: Efficient fine-tuning using Low-Rank Adaptation to reduce computational requirements

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

## Usage

### 1. Data Preprocessing

The preprocessing step filters the Reddit finance dataset and formats it for training:

- Load the Reddit finance dataset from Hugging Face
- Filter out posts matching patterns in `title_patterns.txt`
- Clean text by removing URLs and user mentions
- Format data into instruction-response pairs
- Save processed data to `src/lora/training_data.jsonl`

### 2. LoRA Training

Run the LoRA fine-tuning process:

This will:
- Load the Qwen3-0.6B model
- Apply LoRA configuration for efficient training
- Train the model for 5 epochs
- Save model checkpoints
- Generate comparison results in `results.md`

## Model Configuration

The project uses the following LoRA configuration:
- **Rank (r)**: 8
- **Alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Dropout**: 0.05
- **Task type**: Causal Language Modeling

## Training Parameters

- **Base model**: Qwen/Qwen3-0.6B
- **Batch size**: 4 per device
- **Epochs**: 5
- **Learning rate**: 2e-4
- **Weight decay**: 0.01
- **Warmup ratio**: 0.03
- **Mixed precision**: FP16

## Data Sources

The project uses the `winddude/reddit_finance_43_250k` dataset, which contains posts from various finance-related Reddit subreddits including:
- r/wallstreetbets
- r/investing
- r/stocks
- r/personalfinance
- And many others

## Results

Training results and model comparisons are automatically saved to `results.md`, showing:
- Input examples
- Ground truth responses
- Pre-SFT model outputs
- Post-SFT model outputs
