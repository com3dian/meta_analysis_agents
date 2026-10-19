<div align="center">

# Generative AI for Meta-Analysis Data Extraction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

</div>

## Description

This repository contains the code used to evaluate generative-AI approaches for extracting structured data from intercropping research papers. It compares direct zero-shot prompting, a staged workflow, and a multi-agent system against manually curated ground truth and evaluates their effects on a downstream meta-analysis.

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/com3dian/meta_analysis_agents
cd meta_analysis_agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and add the credentials for the LLM provider you want to use:

```bash
cp .env.example .env
```

Never commit the completed `.env` file or API keys.

## Data

The source papers and ground-truth data are not included in the repository. The experiment notebooks expect the paper inputs under `data/wopke_100/` and the ground-truth table under `data/wopke_paper_code/`. Access and reuse of these materials remain subject to their original licenses.

## Running the Experiments

- Run the extraction experiments with the notebooks in `notebooks/experiment/`.
- Evaluate field- and record-level extraction quality with `notebooks/eval/eval_wopke_100.ipynb`.
- Reproduce the downstream TND-LER analysis with `notebooks/eval/reproduce_ler_tnd.ipynb`.

Provider and model settings are configured in `.env`; the supported options are documented in `.env.example`.

## Project Structure

```text
.
├── docs/                   # Architecture and usage notes
├── notebooks/
│   ├── experiment/         # Extraction experiments
│   └── eval/               # Evaluation and downstream analysis
├── src/
│   ├── direct_llm_call/    # Direct zero-shot extraction
│   ├── static_workflow/    # Staged extraction workflow
│   ├── orchestrator/       # Multi-agent orchestration
│   └── experimentutils/    # Evaluation and reproduction utilities
├── .env.example            # LLM provider configuration template
└── requirements.txt        # Python dependencies
```

## Citation

Citation information will be added after publication.
