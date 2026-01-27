"""
Main entry point for the Metadata Agent application.

This script uses the Orchestrator with planning + parallel execution + debate
to extract metadata from datasets.

Example Usage:
    # Single CSV file
    python -m src.main --source ./data/my_data.csv --topology default
    
    # Directory of CSVs
    python -m src.main --source ./data/my_dataset/ --topology default
    
    # SQLite database
    python -m src.main --source ./data/mydb.sqlite --metadata-standard relational
"""
import argparse
import logging
import os
import sys
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.standards import METADATA_STANDARDS
from src.topology import EXECUTION_TOPOLOGIES
from src.orchestrator import Orchestrator
from src.datasource import create_datasource


def load_metadata_standard(standard_arg: str) -> str:
    """
    Loads the metadata standard content.

    It first checks if the argument is a key in the predefined METADATA_STANDARDS.
    If not, it assumes the argument is a file path and tries to read it.
    """
    if standard_arg in METADATA_STANDARDS:
        return METADATA_STANDARDS[standard_arg]
    elif os.path.exists(standard_arg):
        with open(standard_arg, 'r') as f:
            return f.read()
    else:
        raise ValueError(
            f"Metadata standard '{standard_arg}' not found as a predefined standard or as a valid file path."
        )


def main():
    """
    Main function to run the metadata agent.
    """
    parser = argparse.ArgumentParser(
        description="Run metadata extraction using multi-agent orchestration."
    )
    
    # Required arguments
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Path to the data source. Can be: "
            "a single CSV file, a directory of CSVs, or a SQLite database."
        )
    )
    
    # Configuration arguments
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Name for the dataset (used in metadata output)."
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="default",
        choices=list(EXECUTION_TOPOLOGIES.keys()),
        help="The execution topology to use (defines parallelism and debate rounds)."
    )
    parser.add_argument(
        "--metadata-standard",
        type=str,
        default="basic",
        help=(
            "The name of a predefined metadata standard "
            "(e.g., 'basic', 'dublin_core', 'relational') or a file path."
        )
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate source exists
    if not os.path.exists(args.source):
        logging.error(f"Source not found: {args.source}")
        return
    
    # Create DataSource to get info for logging
    try:
        datasource = create_datasource(args.source, name=args.name)
    except Exception as e:
        logging.error(f"Failed to create DataSource: {e}")
        return
    
    logging.info("=" * 60)
    logging.info("METADATA AGENT")
    logging.info("=" * 60)
    logging.info(f"Source: {args.source}")
    logging.info(f"Dataset Name: {datasource.name}")
    logging.info(f"Source Type: {datasource.source_type.value}")
    logging.info(f"Tables: {datasource.tables}")
    logging.info(f"Multi-table: {datasource.is_multi_table}")
    logging.info(f"Topology: {args.topology}")
    logging.info(f"Metadata Standard: {args.metadata_standard}")
    logging.info("=" * 60)
    
    try:
        metadata_standard_content = load_metadata_standard(args.metadata_standard)
    except ValueError as e:
        logging.error(str(e))
        return
    
    # Initialize and run the orchestrator
    orchestrator = Orchestrator(topology_name=args.topology)
    
    result = orchestrator.run(
        source=datasource,
        metadata_standard=metadata_standard_content
    )
    
    if result is None:
        logging.error("Orchestration failed. No result produced.")
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Steps Completed: {result.steps_completed}/{result.plan_steps_count}")
    
    print("\n--- Step Results ---")
    for step_result in result.step_results:
        print(f"\nStep {step_result.step_index + 1}: {step_result.task}")
        print(f"  Player: {step_result.player_role}")
        print(f"  Success: {step_result.success}")
        print(f"  Debate Rounds: {step_result.debate_rounds_completed}")
        if step_result.consolidated_result:
            print(f"  Result Preview: {step_result.consolidated_result[:200]}...")
    
    print("\n--- Final Workspace Artifacts ---")
    for name, value in result.final_workspace.items():
        preview = str(value)[:100] if value else "None"
        print(f"  {name}: {preview}...")
    
    if result.final_metadata:
        print("\n--- Final Metadata ---")
        pprint(result.final_metadata)


if __name__ == "__main__":
    main()
