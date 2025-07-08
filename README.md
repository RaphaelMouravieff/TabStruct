# TabStruct: Structural Deep Encoding for Table QA

This repository contains the codebase accompanying the paper:
**"Structural Deep Encoding for Table Question Answering"** (ACL 2025).

## 🧠 Overview

TabStruct explores how different structural encoding strategies, including sparse attention masks and absolute positional embeddings, impact generalization in table-based question answering.

The repo supports:
- Synthetic data generation
- Robust evaluation on structure, compositionality, and consistency
- Real benchmark evaluation (e.g., WikiSQL, WikiTableQuestions)

## 📁 Repository Structure
data/        # Scripts and configs for synthetic table generation
jobs/        # Configurations for different experiments (128x10)
script/      # Utilities to generate job batches
tabstruct/   # Source code for models, encodings, attention, etc.
run.py       # Main controller script for experiments


## 🔧 Setup

(Instructions on creating the conda/env, installing packages)

## 📊 Running Experiments

Instructions on using `run.py` to run predefined jobs or custom experiments.

## 📜 Citation

BibTeX entry.

## 📂 License

MIT or Apache-2.0 (choose one)