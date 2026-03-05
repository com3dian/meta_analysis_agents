# MAS Usage Guide

This guide shows practical ways to run the multi-agent system (MAS), including multi-paper and batch workflows.

## What this system can do

- Analyze one paper (`.pdf`, `.md`, `.txt`)
- Analyze multiple papers in one run
- Analyze all supported papers in a directory
- Run repeated jobs over many papers and aggregate outputs
- Optionally enforce structured final output with a Pydantic schema

## Supported input formats

`Orchestrator.run(source=...)` accepts:

- `str`: a single file path or a directory path
- `List[str]`: multiple file paths
- `Dict[str, str]`: named resources (`resource_name -> file_path`)

Supported file types:

- `.pdf`
- `.md`
- `.txt`

## Quick start (single paper)

```python
from src.orchestrator import Orchestrator
from src.standards import METADATA_STANDARDS

objective = f"""
Extract meta-analytic records from this paper.

META-ANALYTIC SCHEMA:
{METADATA_STANDARDS["climate_vs_cropyield"]}
"""

orchestrator = Orchestrator(topology_name="default")
result = orchestrator.run(
    source="./data/paper_01.pdf",
    objective=objective,
    name="single_paper_run",
)

print(result.success)
print(result.final_output)
```

## One run with multiple papers

### Option A: pass a directory

```python
result = orchestrator.run(
    source="./data/papers",  # all supported files in folder
    objective=objective,
    name="multi_paper_dir_run",
)
```

### Option B: pass an explicit list

```python
paper_paths = [
    "./data/papers/paper_01.pdf",
    "./data/papers/paper_02.pdf",
    "./data/papers/paper_03.pdf",
]

result = orchestrator.run(
    source=paper_paths,
    objective=objective,
    name="multi_paper_list_run",
)
```

### Option C: pass named resources

```python
resources = {
    "paper_alpha": "./data/papers/paper_01.pdf",
    "paper_beta": "./data/papers/paper_02.pdf",
}

result = orchestrator.run(
    source=resources,
    objective=objective,
    name="multi_paper_named_run",
)
```

## Batch many jobs (recommended for large collections)

For large corpora, run one orchestrator job per paper and persist each result.

```python
import glob
import json
import os

from src.orchestrator import Orchestrator
from src.standards import METADATA_STANDARDS

objective = f"""
Extract meta-analytic records from this paper.

META-ANALYTIC SCHEMA:
{METADATA_STANDARDS["climate_vs_cropyield"]}
"""

orchestrator = Orchestrator(topology_name="fast")
paper_paths = sorted(glob.glob("./data/papers/*.pdf"))
os.makedirs("./outputs", exist_ok=True)

for paper_path in paper_paths:
    run_name = os.path.splitext(os.path.basename(paper_path))[0]
    result = orchestrator.run(
        source=paper_path,
        objective=objective,
        name=run_name,
    )

    out_path = f"./outputs/{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "success": bool(result and result.success),
                "final_output": result.final_output if result else None,
                "error": result.error if result else "No result returned",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
```

## Use structured final output (optional but recommended)

If you want strict schema compliance in the final synthesis step:

```python
from src.core.schema_factory import create_output_schema
from src.standards import METADATA_STANDARDS

OutputSchema = create_output_schema(
    METADATA_STANDARDS["climate_vs_cropyield"],
    record_class_name="YieldRecord",
    output_class_name="MetaAnalysisOutput",
    records_key="records",
)

result = orchestrator.run(
    source="./data/papers",
    objective=objective,
    output_schema=OutputSchema,
    name="structured_multi_paper_run",
)
```

Notes:

- Structured output is applied in the final synthesis step.
- Without `output_schema`, synthesis returns plain text/JSON-like free-form output.

## Topology guidance

- `fast`: lower latency, less debate
- `default`: balanced quality/speed
- `deep_analysis`: highest quality, highest cost/latency

Choose topology based on quality/throughput needs.

## Troubleshooting

- `FileNotFoundError`: verify paths exist
- `Unsupported context type`: use `.pdf`, `.md`, or `.txt`
- Empty/weak outputs: strengthen `objective` with explicit extraction criteria and schema
- Inconsistent keys in output: pass `output_schema` for strict structure

