import json
import re

# save_json_records_to_csv has moved to src.experimentutils.output_utils
from src.experimentutils.output_utils import save_json_records_to_csv  # noqa: F401


def print_json_records(json_string, record_key="yield_records"):
    """
    Parse and print JSON records in a readable format.
    
    Args:
        json_string: String containing JSON (may include markdown code blocks)
        record_key: Key in the JSON object that contains the array of records
    """
    # Strip markdown code blocks if present
    cleaned = json_string.strip()
    if cleaned.startswith("```json"):
        cleaned = re.sub(r'^```json\s*', '', cleaned)
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\s*', '', cleaned)
    if cleaned.endswith("```"):
        cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    try:
        # Parse JSON
        data = json.loads(cleaned)
        
        # Handle different JSON structures
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Try common keys
            for key in [record_key, "records", "yield_records", "data"]:
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            else:
                # If no array found, treat the dict itself as a single record
                records = [data]
        else:
            print(f"Unexpected JSON structure: {type(data)}")
            return
        
        # Print each record
        print(f"Found {len(records)} record(s):\n")
        print("=" * 80)
        
        for i, record in enumerate(records, 1):
            print(f"\n📋 Record {i}:")
            print("-" * 80)
            for key, value in record.items():
                if value is None:
                    print(f"  {key}: null")
                elif isinstance(value, (dict, list)):
                    print(f"  {key}: {json.dumps(value, indent=4, ensure_ascii=False)}")
                else:
                    # Truncate very long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        print(f"  {key}: {value_str[:100]}...")
                    else:
                        print(f"  {key}: {value_str}")
            print()
        
        print("=" * 80)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"\nFirst 500 chars of input:")
        print(cleaned[:500])
    except Exception as e:
        print(f"Error: {e}")
