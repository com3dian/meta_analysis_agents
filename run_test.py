"""
A simple test script to run the metadata agent with a specified standard.
"""
import os

# --- Using a predefined standard ---
print("--- Running with 'basic' predefined standard ---")
os.system(
    "python -m src.main "
    "--dataset-path ./data/test_data.csv "
    "--topology fast "
    "--metadata-standard basic"
)
print("\n" * 3)


# --- Using a custom standard from a file ---
custom_standard_content = """
{
    "custom_field_1": "...",
    "custom_field_2": "..."
}
"""
custom_standard_path = "./data/custom_standard.json"
with open(custom_standard_path, "w") as f:
    f.write(custom_standard_content)

print(f"--- Running with custom standard from file: {custom_standard_path} ---")
os.system(
    f"python -m src.main "
    f"--dataset-path ./data/test_data.csv "
    f"--topology fast "
    f"--metadata-standard {custom_standard_path}"
)

# Clean up the custom standard file
os.remove(custom_standard_path)
