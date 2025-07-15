# TabStruct 🧮  
**Structural Deep Encoding for Table Question Answering (ACL 2025)**

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue.svg)](https://aclanthology.org/)  
[![Stars](https://img.shields.io/github/stars/RaphaelMouravieff/TabStruct?style=social)](https://github.com/RaphaelMouravieff/TabStruct/stargazers)

🚀 **[Paper](https://arxiv.org/abs/XXXX.YYYY)** | 📘 **[Project Page](https://raphaelmouravieff.github.io/TabStruct/)** | 🎥 **[Video Demo](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

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

```bash
git clone https://github.com/RaphaelMouravieff/TabStruct.git TabStruct
cd TabStruct
conda create -n tabstruct python=3.11.11 -y
conda activate tabstruct
pip install -r requirements.txt
```

## 📥 Data & Preprocessing

```bash
# Download all necessary datasets (WikiSQL, Synthetic)
bash script/download_data.sh
```

## 🧪 Running Experiments

```bash
# Generate all train/test jobs for the 128 model variants
bash script/generate_all_jobs.sh

# Train on WikiSQL
bash jobs/train/{model_name}/wikisql.sh

# Train on synthetic data
bash jobs/train/{model_name}/synthetic.sh

# Evaluate on compositional generalization
bash jobs/test/{model_name}/compositional.sh

# (Recommended) Run full synthetic tests: compositional, robustness, and structural
bash jobs/test/{model_name}/synthetic.sh
```

## 🧬 Model Variants

Each model is defined by a unique combination of 5 structural components:
	•	T: Token Structure (T0, T1, T2) - T0 = no special tokens, T2 = row+column+cell markers
	•	E: Structural Embeddings (E0, E1) - E0 = no structure embeddings, E1 = row+column embeddings
	•	PE: Positional Embedding (TPE, CPE) - PE = cell-level, TPE = standard Transformer position encoding
	•	B: Attention Bias (B0, B1) - B0 = no attention bias, B1 = TableFormer-style relational bias
	•	M: Sparse Attention Mask (M0–M6) — M0 = no sparsity


Example:
T2-M3-TPE-B1-E1

This means:
	•	Tokens: Row+Column+Cell tokens
	•	Mask: Sparse mask M3 (ultra-efficient)
	•	Positional Embedding: Table-wise (TPE)
	•	Bias: Enabled
	•	Structure Embedding: Row+Column Embedding

📄 See [all_models.txt](./all_models.txt) for the full list of 128 variants.
📓 For a visual overview of how each component is applied during model generation, check out:
[Project_Overview.ipynb](./Notebooks/Project_Overview.ipynb)


## 🎥 Demo & GitHub Pages

📘 Project Page
🎥 Watch the Demo


⸻

📈 Results & Benchmarks

TabStruct achieves strong generalization and robustness across synthetic and real datasets.

Model Variant	WikiSQL	Synthetic Avg	Speedup (M3)
T2-M0-TPE-B1-E1	78.5%	79.3%	1×
T2-M3-TPE-B1-E1	80.3%	79.4%	50×
TAPEX (baseline)	74.7%	79.5%	N/A
TableFormer (baseline)	60.5%	23.1%	N/A


⸻

## 🧪 Reproducing Paper Results

```bash
# Train the best variant on WikiSQL
bash jobs/train/T2-M3-TPE-B1-E1/wikisql.sh

# Evaluate on all generalization tests
bash jobs/test/T2-M3-TPE-B1-E1/synthetic.sh
```

⸻

📊 Datasets & Licenses

This project uses the following datasets:
	•	WikiSQL (Zhong et al., 2017)
→ We include a preprocessed version
→ License: BSD 3-Clause
→ See LICENSE.wikisql for full terms
	•	Synthetic Data
→ Fully auto-generated, no human annotation involved
	•	WikiTableQuestions (WTQ)
→ Downloaded externally using the provided script

⸻

📜 Citation

If you use TabStruct in your research, please cite us:

@inproceedings{mouravieff2025tabstruct,
  title     = {Structural Deep Encoding for Table Question Answering},
  author    = {Raphael Mouravieff and others},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}


⸻

📂 License

This repository is licensed under the MIT License.

Portions of the data are adapted from the WikiSQL dataset by Salesforce.com, Inc.,
which is licensed under the BSD 3-Clause License.

