# Supplementary Materials for "DCTR: Dual-Constraint Subgraph Optimization for Knowledge Graph-based Retrieval-Augmented Generation"

This repository contains the supplementary materials accompanying our paper. It includes the full source code for our proposed DCTR framework and the mathematical appendix detailing theoretical derivations, algorithms, and implementation-related details.

## 📂 Contents

### 📁 `DCTR/`
This folder contains the complete implementation of our DCTR framework, including data preprocessing, model training, inference, and prompt templates.

Key files and folders:
- `configs/` – Configuration files for experiments.
- `preprocess/` – Scripts for data formatting and preprocessing.
- `requirements/` – Dependencies and environment specifications.
- `src/` – Core source code and model components.
- `emb.py` – Embedding generator.
- `train.py` – Model training script.
- `inference.py` – Inferencepipeline.
- `prompts.py` – Prompt templates for LLM-based submodules.
- `reason.py` – Implementation of reasoning.
- `llm_utils.py` – Utility functions for language model interaction.

### 📄 `Appendix.pdf`
This file provides the following detailed content not included in the main paper:
- **Maximum Flow Retrieval Algorithm**: Mathematical formulation of subgraph extraction via maximum flow.
- **Error Propagation Analysis**: Theoretical analysis of perturbation accumulation across multi-hop paths.
- **Posterior Inference Path Ranking**: A probabilistic derivation of the optimal path ranking objective.
- **Positional Bias in Attention**: A complete derivation of how positional attention weight affects model prediction.
- **Differentiable Subgraph Ranking Loss (DSRS)**: Derivation from Kendall’s Tau to our differentiable ranking loss.
- **Prompt Design**: Examples and formatting for query reasoning and triple evaluation tasks.
- **Hyperparameter Analysis**: Implementation-related details.
