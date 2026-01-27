# Metadata Agent Architecture

This document describes the architecture of the multi-agent metadata extraction system.

## Overview

The Metadata Agent is a multi-agent system that extracts metadata from datasets using:
1. **Unified DataSource**: Abstract data layer that handles any data format (CSV, SQLite, etc.)
2. **Planning**: An LLM generates a step-by-step plan based on the data source and metadata standard
3. **Parallel Execution**: Multiple players execute each step simultaneously
4. **Debate**: Players critique and revise each other's work to improve quality
5. **Synthesis**: A synthesizer consolidates results into a final output

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                         │
│     (file path, list of paths, dict, directory, SQLite, DataSource)          │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DataSourceFactory                                     │
│              (Auto-detects type, creates appropriate DataSource)             │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DataSource (Unified Interface)                            │
│                                                                              │
│  CSVDataSource │ SQLiteDataSource │ ParquetDataSource (future) │ ...        │
│                                                                              │
│  Properties: name, tables, is_multi_table, source_type                      │
│  Methods: get_table_info(), read_table(), get_relationships(), ...          │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│         orchestrator.run(source, metadata_standard)                         │
│         (Unified entry point for ALL data sources)                          │
│                                                                              │
│  Note: For multi-table datasets, 'relationship_analyst' is automatically    │
│        added to the player pool.                                            │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ Plan: [Step1, Step2, Step3, ...]
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PLAN EXECUTOR                                      │
│                    (Iterates through plan steps)                            │
│                    (Maintains workspace of artifacts)                       │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ For each step:
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STEP EXECUTOR (LangGraph)                                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  1. PARALLEL EXECUTION                                              │     │
│  │     Player1 ──┐                                                     │     │
│  │     Player2 ──┼──► Execute same task with different perspectives   │     │
│  │     Player3 ──┘                                                     │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  2. DEBATE LOOP (critique → revise → repeat)                        │     │
│  │     - Each player critiques others' work                           │     │
│  │     - Each player revises based on critiques                       │     │
│  │     - Repeat for N debate rounds                                   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  3. SYNTHESIS                                                       │     │
│  │     Synthesizer consolidates all results into final answer         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. DataSource (`src/datasource/`)

The unified data access layer that abstracts away differences between data formats.

```python
from src.datasource import create_datasource

# All of these create appropriate DataSource objects:
ds = create_datasource("./data/users.csv")           # Single CSV
ds = create_datasource("./data/mydb.sqlite")         # SQLite database
ds = create_datasource("./data/my_dataset/")         # Directory of CSVs
ds = create_datasource(["./a.csv", "./b.csv"])       # List of files
ds = create_datasource({                              # Named tables
    "users": "./users.csv",
    "orders": "./orders.csv"
})
```

**DataSource Interface:**
- `tables` - List of table names
- `is_multi_table` - Boolean indicating multiple tables
- `source_type` - Type of data source (csv, sqlite, etc.)
- `get_table_info(table)` - Get metadata for a table
- `read_table(table)` - Read table data as DataFrame
- `get_relationships()` - Get discovered relationships

### 2. Orchestrator (`src/orchestrator/orchestrator.py`)

The main entry point that coordinates planning and execution with a **single unified interface**.

```python
from src.orchestrator import Orchestrator, run_metadata_extraction
from src.standards import METADATA_STANDARDS

# Create orchestrator
orchestrator = Orchestrator(topology_name="default")

# Run on ANY data source - same interface for all
result = orchestrator.run(
    source="./data/users.csv",  # or dict, or list, or directory, or sqlite
    metadata_standard=METADATA_STANDARDS["basic"]
)

# Or use convenience function
result = run_metadata_extraction(
    source={"users": "./users.csv", "orders": "./orders.csv"},
    metadata_standard=METADATA_STANDARDS["relational"]
)
```

**Multi-table Auto-adaptation:**
The orchestrator automatically adds `relationship_analyst` to the player pool when analyzing multi-table datasets. No separate topology needed.

### 3. Player (`src/players/player.py`)

A unified agent class that can execute tasks and participate in debates.

```python
from src.players import Player, create_player_from_config, PLAYER_CONFIGS

player = create_player_from_config(PLAYER_CONFIGS["data_analyst"], name="analyst_1")

# Execute a task with DataSource
result = player.execute_task(
    task="Analyze dataset structure",
    datasource_key="ds_abc123",
    datasource_info={...},
    workspace={},
    inputs={}
)
```

### 4. Topology & Player Configs

Configuration is split into two modules:
- **Player Configs** (`src/players/configs.py`): Defines player roles, prompts, and tools
- **Execution Topologies** (`src/topology.py`): Defines how plans are executed

#### Execution Topologies (`src/topology.py`)

```python
EXECUTION_TOPOLOGIES = {
    "default": {
        "description": "Standard execution with 3 parallel players, 2 debate rounds",
        "players_per_step": 3,
        "debate_rounds": 2,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist"],
    },
    "fast": {
        "description": "Quick execution with 2 players and minimal debate",
        "players_per_step": 2,
        "debate_rounds": 1,
        "player_pool": ["data_analyst", "schema_expert"],
    },
    "thorough": {
        "description": "Thorough execution with more players and extended debate",
        "players_per_step": 4,
        "debate_rounds": 3,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist", "critic"],
    },
    "single": {
        "description": "Single player execution with no debate. Fastest but least robust.",
        "players_per_step": 1,
        "debate_rounds": 0,
        "player_pool": ["data_analyst"],
    },
}
```

**Note:** For multi-table DataSources, `relationship_analyst` is automatically added to the player pool by the orchestrator. No separate multi-table topologies are needed.

## Tools (`src/tools/`)

### Unified DataSource Tools (`src/tools/datasource_tools.py`)

All tools work with the DataSource abstraction:

| Tool | Description |
|------|-------------|
| `get_dataset_overview` | Overview of the entire dataset |
| `list_tables` | List all tables |
| `get_dataset_schema` | Complete schema with relationships |
| `get_table_info` | Detailed table information |
| `get_row_count` | Row count for a table |
| `get_column_names` | Column names |
| `get_column_types` | Column data types |
| `get_sample_rows` | Preview rows |
| `get_column_statistics` | Statistics for all columns |
| `get_missing_values` | Missing value counts |
| `get_unique_values` | Unique values in a column |
| `get_relationships` | Get discovered relationships |
| `analyze_potential_relationship` | Analyze a specific relationship |
| `preview_join` | Preview joining two tables |
| `find_common_columns` | Find shared columns across tables |
| `compare_table_schemas` | Compare table structures |

## Metadata Standards (`src/standards.py`)

Predefined output formats:

- `basic`: Simple title, description, schema
- `dublin_core`: Dublin Core metadata standard
- `relational`: Full relational dataset metadata with tables and relationships
- `relational_simple`: Simplified relational format for quick analysis
- `ecological_data`: Specialized format for ecological/scientific datasets

## Usage Examples

### Single File

```python
from src.orchestrator import run_metadata_extraction
from src.standards import METADATA_STANDARDS

result = run_metadata_extraction(
    source="./data/users.csv",
    metadata_standard=METADATA_STANDARDS["basic"]
)
print(result.final_metadata)
```

### Multiple Related Files

```python
result = run_metadata_extraction(
    source={
        "users": "./data/users.csv",
        "orders": "./data/orders.csv",
        "products": "./data/products.csv"
    },
    metadata_standard=METADATA_STANDARDS["relational"]
)

# Access per-table metadata
for table, metadata in result.table_metadata.items():
    print(f"{table}: {metadata}")

# Access discovered relationships
for rel in result.relationships:
    print(f"{rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")
```

### SQLite Database

```python
result = run_metadata_extraction(
    source="./data/mydb.sqlite",
    metadata_standard=METADATA_STANDARDS["relational"]
)
```

### Directory of Files

```python
result = run_metadata_extraction(
    source="./data/my_dataset/",
    metadata_standard=METADATA_STANDARDS["relational"]
)
```

### Using DataSource Directly

```python
from src.datasource import create_datasource
from src.orchestrator import Orchestrator

# Create and inspect DataSource first
ds = create_datasource("./data/my_dataset/")
print(f"Tables: {ds.tables}")
print(f"Relationships: {ds.get_relationships()}")

# Then run orchestration
orchestrator = Orchestrator(topology_name="default")
result = orchestrator.run(ds, METADATA_STANDARDS["relational"])
```

## File Structure

```
src/
├── main.py                    # CLI entry point
├── standards.py               # Metadata standards
├── topology.py                # Execution topology configs
├── config.py                  # LLM and system configuration
├── datasource/                # Unified data access layer
│   ├── __init__.py            # Exports DataSource, create_datasource
│   ├── base.py                # Abstract DataSource base class
│   ├── csv_source.py          # CSV file(s) implementation
│   ├── sqlite_source.py       # SQLite database implementation
│   └── factory.py             # DataSourceFactory with auto-detection
├── orchestrator/
│   ├── orchestrator.py        # Main Orchestrator class (unified interface)
│   ├── plan_executor.py       # Executes full plans
│   ├── step_executor.py       # LangGraph for step debates
│   ├── prompts.py             # All prompt templates
│   ├── schemas.py             # Pydantic models
│   └── state.py               # State TypedDicts
├── players/
│   ├── __init__.py            # Exports Player, PLAYER_CONFIGS
│   ├── player.py              # Unified Player class
│   └── configs.py             # Player role configurations
└── tools/
    ├── __init__.py            # Exports all tools
    └── datasource_tools.py    # Unified DataSource tools
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

### Adding New Data Source Types

1. Create a new class extending `DataSource` in `src/datasource/`
2. Implement required abstract methods
3. Add type detection to `DataSourceFactory`

Example for Parquet:
```python
class ParquetDataSource(DataSource):
    @property
    def source_type(self) -> SourceType:
        return SourceType.PARQUET
    
    @property
    def tables(self) -> List[str]:
        # Implementation
        
    def _load_table_info(self, table: str) -> TableInfo:
        # Implementation
        
    def read_table(self, table: str, ...) -> pd.DataFrame:
        # Implementation
```

### Adding New Tools

Add to `src/tools/datasource_tools.py`:

```python
@tool
def my_new_tool(datasource_key: str, ...) -> Dict[str, Any]:
    """Description of what this tool does."""
    ds = get_datasource(datasource_key)
    # Implementation using DataSource API
    return result
```

Then add to player configs in `src/players/configs.py`.
