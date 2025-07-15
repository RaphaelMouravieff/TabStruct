# TabStruct 🧮  
**Structural Deep Encoding for Table Question Answering (ACL 2025)**

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue.svg)](https://aclanthology.org/)  
[![Stars](https://img.shields.io/github/stars/RaphaelMouravieff/TabStruct?style=social)](https://github.com/RaphaelMouravieff/TabStruct/stargazers)

![TabStruct Overview](./figures/main.png)


🚀 **[Paper](https://arxiv.org/abs/2503.01457es)** | 📘 **[Project Page](https://raphaelmouravieff.github.io/Structural-Deep-Encoding-for-Table-Question-Answering/)** | 🎥 **[Video Demo](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

> TabStruct is a flexible and modular framework for exploring and evaluating structural encodings in Transformer-based table QA. It combines sparse attention, structural embeddings, and token formatting to build robust, scalable models for real and synthetic data.

---

## 🔥 Key Features

- 128 model configurations combining 5 structural components
- Synthetic and real dataset evaluation 
- Structural generalization, compositionality, and robustness tests
- Fast training with sparse attention (up to **50× speedup** with M3)
- Fully reproducible with one-line setup and job scripts

---


## 📁 Repository Structure
```bash
data/        # Scripts and configs for synthetic table generation
jobs/        # Configurations for different experiments (128)
script/      # Utilities to generate job batches and download data
tabstruct/   # Source code for models, encodings, attention, etc.
run.py       # Main controller script for experiments
```

---

## 🔧 Installation

> 📌 **Reproducing results?** Just copy-paste this to get started with TabStruct.

```bash
# Clone the repo
git clone https://github.com/RaphaelMouravieff/TabStruct.git TabStruct
cd TabStruct

# Set up the environment
conda create -n tabstruct python=3.11.11 -y
conda activate tabstruct

# Install dependencies
pip install -r requirements.txt
```

## 📥 Data & Preprocessing

📦 Step 1: Download all datasets used in the paper (WikiSQL, Synthetic)
```bash
# Download all necessary datasets (WikiSQL, Synthetic)
bash script/download_data.sh

🧪 Step 2: Auto-generate all training & evaluation jobs (128 .sh scripts matching our experiments)
After this, you can directly run any script to reproduce results from the paper.
# Generate all train/test jobs for the 128 model variants
bash script/generate_all_jobs.sh
```

## 🧪 Running Experiments

🚂 Train on WikiSQL (single benchmark)
Test is automatically included in the training script.
```bash
# Train on WikiSQL
bash jobs/train/{model_name}/wikisql.sh
```

🧬 Train on Synthetic Data (multi-benchmark setup)
```bash
# Train on synthetic data
bash jobs/train/{model_name}/synthetic.sh
```
🧪 After training, run one or more generalization tests:
```bash
# Evaluate on compositional generalization
bash jobs/test/{model_name}/compositional.sh

# (Recommended) Run full synthetic tests: compositional, robustness, and structural
bash jobs/test/{model_name}/synthetic.sh
```
📄 See all valid {model_name} variants in: [all_models.txt](./all_models.txt)

## 🧬 Model Variants

Each model is defined by a unique combination of 5 structural components:

- **T: Token Structure (T0, T1, T2)** – T0 = no special tokens, T2 = row+column+cell markers  
- **E: Structural Embeddings (E0, E1)** – E0 = no structure embeddings, E1 = row+column embeddings  
- **PE: Positional Embedding (TPE, CPE)** – PE = cell-level, TPE = standard Transformer position encoding  
- **B: Attention Bias (B0, B1)** – B0 = no attention bias, B1 = TableFormer-style relational bias  
- **M: Sparse Attention Mask (M0–M6)** – M0 = no sparsity


Example:
T2-M3-TPE-B1-E1

This means:

- **Tokens**: Row+Column+Cell tokens  
- **Mask**: Sparse mask M3 (ultra-efficient)  
- **Positional Embedding**: Table-wise (TPE)  
- **Bias**: Enabled  
- **Structure Embedding**: Row+Column Embedding

📄 See [all_models.txt](./all_models.txt) for the full list of 128 variants.
📓 For a visual overview of how each component is applied during model generation, check out:
[Project_Overview.ipynb](./Notebooks/Project_Overview.ipynb)


## 🎥 Demo & GitHub Pages

📘 [Project Page](https://raphaelmouravieff.github.io/Structural-Deep-Encoding-for-Table-Question-Answering/)

🎥 [Video Demo](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)


⸻

## 🧪 Reproducing Paper Results

```bash
# Train the best variant on WikiSQL
bash jobs/train/T2-M3-TPE-B1-E1/wikisql.sh

# Evaluate on all generalization tests
bash jobs/test/T2-M3-TPE-B1-E1/synthetic.sh
```

⸻

## 📜 **Citation**

If you use TabStruct in your research, please cite us:

```bibtex
@article{mouravieff2025structural,
  title={Structural Deep Encoding for Table Question Answering},
  author={Mouravieff, Rapha{\"e}l and Piwowarski, Benjamin and Lamprier, Sylvain},
  journal={arXiv preprint arXiv:2503.01457},
  year={2025}
}
```

⸻

## 📂 License

This repository is licensed under the MIT License.

Portions of the data are adapted from the WikiSQL dataset by Salesforce.com, Inc.,
which is licensed under the BSD 3-Clause License.



This project uses the following datasets:

- **WikiSQL** (Zhong et al., 2017)  
  → We include a preprocessed version  
  → License: BSD 3-Clause  
  → See `LICENSE.wikisql` for full terms

- **Synthetic Data**  
  → Fully auto-generated