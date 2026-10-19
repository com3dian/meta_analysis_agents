<div align="center">

# Generative AI for Meta-Analysis Data Extraction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2609.35089-b31b1b)](https://arxiv.org/abs/2609.35089)
[![WASP 2026](https://img.shields.io/badge/WASP-2026-0A7B5A)](https://ui.adsabs.harvard.edu/WIESP/2026/)

</div>

## Description

This repository contains the code for [Can Generative AI Automate Data Extraction for Meta-Analysis? A Case Study on Intercropping Research](https://arxiv.org/abs/2609.35089), accepted at the 4th Workshop on Artificial Intelligence for Scientific Publications ([WASP 2026](https://ui.adsabs.harvard.edu/WIESP/2026/)), co-located with IJCNLP-AACL 2026.

It compares direct zero-shot prompting, a staged workflow, and a multi-agent system against manually curated ground truth and evaluates their effects on a downstream meta-analysis.

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
├── figures/                # Paper figures
├── notebooks/
│   ├── experiment/         # Extraction experiments
│   └── eval/               # Evaluation and TND–LER reproduction
├── src/
│   ├── direct_llm_call/    # Direct zero-shot extraction
│   ├── static_workflow/    # Staged extraction workflow
│   ├── orchestrator/       # Multi-agent orchestration
│   ├── players/            # Agent roles and configs
│   ├── tools/              # Agent tools
│   ├── context/            # Document context
│   ├── core/               # Shared schemas and state
│   └── experimentutils/    # Evaluation and reproduction utilities
├── data/                   # Papers and ground truth (not in git)
├── outputs/                # Extraction and evaluation outputs
├── .env.example            # LLM provider configuration template
└── requirements.txt        # Python dependencies
```

## Citation

```bibtex
@misc{lu2026generativeaiautomatedata,
      title={Can Generative AI Automate Data Extraction for Meta-Analysis? A Case Study on Intercropping Research}, 
      author={Zehao Lu and Xingguo Xiong and Wopke van der Werf and Thijs L. van der Plas and Ioannis N. Athanasiadis},
      year={2026},
      eprint={2609.35089},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.35089}, 
}
```
