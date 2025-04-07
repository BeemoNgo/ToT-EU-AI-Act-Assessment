# Tree of Thought (ToT) Based EU AI Act Assessment

This project evaluates AI systems for compliance with the **EU AI Act** using a **Tree of Thought (ToT)** reasoning approach and Large Language Models (LLMs). It offers both:

1. **Staged Risk Classification** — classifies an AI system into one of four categories: Unacceptable, High, Limited, or Minimal Risk.
2. **Comprehensive Compliance Tree** — performs detailed regulatory assessments based on AI Act requirements (e.g., data governance, risk management, transparency).

The system is powered by models like **LLaMA 2 (7B)** or any Hugging Face-compatible LLM.

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- [Hugging Face account](https://huggingface.co/) with access to gated models (e.g., LLaMA 2)
- Access token from Hugging Face

### Set Up Environment

```bash
# (Optional) Create virtual environment
conda create -n ai-act python=3.10
conda activate ai-act

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (only needed once)
huggingface-cli login
```

---

## 🚀 Usage

### 1. Prepare Your Dataset

- Place your input CSV (e.g., `AI_apps_full_dataset (2).csv`) in the root directory.
- Each row should describe an AI application, including fields like:
  - Full Description
  - Collected Data
  - Shared Data
  - Security Practices

### 2. Run the Assessment

```bash
python main.py
```

### 3. Outputs

The script generates:

- `risk_classification_results.csv` — AI system risk level with reasoning and confidence score
- `compliance_results.csv` — Detailed compliance map (each criterion with verdict and justification)
- `compliance_results.json` — JSON format for structured review or downstream use

---

## Supported Models

This system supports any Hugging Face-compatible model. Recommended:

- `meta-llama/Llama-2-7b-hf` (requires gated access)
- `TheBloke/Nous-Hermes-Llama2-GPTQ` (requires GPU)
- `tiiuae/falcon-7b-instruct` (open access)
- `mistralai/Mistral-7B-Instruct-v0.1` (strong performance, open)

> GPTQ models require GPU. For CPU testing, use unquantized or lighter models.

---

## About Tree of Thought (ToT)

ToT enhances reasoning by:

- Exploring multiple branches (thoughts) per compliance or risk criterion
- Self-evaluating and selecting the most promising reasoning paths
- Providing full transparency into why a system was classified a certain way

This mirrors how human auditors deliberate over legal compliance decisions.

---

## Optional: Visualizing Compliance Trees

If `networkx` and `pydot` are installed, you can generate compliance tree diagrams:

```python
from output.visualizer import visualize_compliance_tree
visualize_compliance_tree(compliance_result, "output.png")
```

---

## License & Credits

- Based on EU AI Act documents
- LLM reasoning powered by Hugging Face models
- ToT logic inspired by [Yao et al., 2023 - Tree of Thoughts](https://arxiv.org/abs/2305.10601)

---

