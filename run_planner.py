"""
This script demonstrates the orchestrator architecture.

It shows a full execution of a plan to achieve a given objective.

Usage:
    python run_planner.py
"""
import sys
import os
import logging
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.orchestrator.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """
    Main demo function.
    """
    print("=" * 60)
    print("MULTI-AGENT SYSTEM DEMO")
    print("=" * 60)

    # Configuration
    SOURCE_DATA = "./data/test_data.csv"
    OBJECTIVE = "Analyze the provided data and generate a summary of the key insights."
    TOPOLOGY = "default"

    print(f"\nConfiguration:")
    print(f"  Source Data: {SOURCE_DATA}")
    print(f"  Objective: {OBJECTIVE}")
    print(f"  Topology: {TOPOLOGY}")

    # Check if data exists
    if not os.path.exists(SOURCE_DATA):
        print(f"\nWarning: Source data not found at {SOURCE_DATA}")
        print("Creating a sample dataset for demo...")
        os.makedirs("./data", exist_ok=True)
        import pandas as pd
        sample_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "city": ["NYC", "LA", "Chicago", "NYC", "Boston"]
        })
        sample_df.to_csv(SOURCE_DATA, index=False)
        print(f"Sample dataset created at {SOURCE_DATA}")

    # Initialize and run
    orchestrator = Orchestrator(topology_name=TOPOLOGY)
    
    result = orchestrator.run(
        source=SOURCE_DATA,
        objective=OBJECTIVE,
    )
    
    if result:
        print("\n" + "=" * 60)
        print("EXECUTION RESULTS")
        print("=" * 60)
        print(f"\nSuccess: {result.success}")
        print(f"Steps Completed: {result.steps_completed}/{result.plan_steps_count}")
        
        print("\n--- Step Results Summary ---")
        for step_result in result.step_results:
            print(f"\nStep {step_result.step_index + 1}: {step_result.task}")
            print(f"  Player Role: {step_result.player_role}")
            print(f"  Success: {step_result.success}")
            print(f"  Debate Rounds: {step_result.debate_rounds_completed}")
            print(f"  Artifacts: {list(step_result.artifacts.keys())}")
        
        print("\n--- Final Output ---")
        pprint(result.final_output)
    else:
        print("Execution failed.")
    
    return result


if __name__ == "__main__":
    main()