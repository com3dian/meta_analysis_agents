import csv
import json
import os
import re

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

def _normalize_cell_value(value):
    """
    Normalize a Python value into a CSV cell string.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        # Store complex structures as JSON strings
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_json_records_to_csv(json_string, path, record_key="yield_records"):
    """
    Parse JSON records and save them to a CSV file.

    Behavior:
    - If the file does not exist or is empty, write header + rows.
    - If the file exists and is non-empty, verify that the header (column names)
      matches the new records' fields. If they match, append rows. If not, raise
      a ValueError with details about the mismatch.

    Args:
        json_string: String containing JSON (may include markdown code blocks), a dict, or a list.
        path: Path to the CSV file (str or Path-like). This is required.
        record_key: Key in the JSON object that contains the array of records.

    Returns:
        The number of records written.

    Raises:
        ValueError: If the existing CSV header does not match the new records' schema.
    """
    # Handle list input directly
    if isinstance(json_string, list):
        records = json_string
    # Handle dict input - extract records using record_key or convert to JSON string
    elif isinstance(json_string, dict):
        # Try to extract records from dict using record_key
        if record_key in json_string:
            value = json_string[record_key]
            # If it's already a list, use it directly
            if isinstance(value, list):
                records = value
            # If it's a string, try to parse it as JSON
            elif isinstance(value, str):
                # Strip markdown code blocks if present
                cleaned = value.strip()
                if cleaned.startswith("```json"):
                    cleaned = re.sub(r'^```json\s*', "", cleaned)
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```\s*', "", cleaned)
                if cleaned.endswith("```"):
                    cleaned = re.sub(r'\s*```$', "", cleaned)
                cleaned = cleaned.strip()
                
                # Parse JSON
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    records = parsed
                elif isinstance(parsed, dict) and "records" in parsed and isinstance(parsed["records"], list):
                    records = parsed["records"]
                else:
                    records = [parsed] if isinstance(parsed, dict) else []
            else:
                records = []
        else:
            # Try common keys
            for key in ["records", "yield_records", "data", "final_meta_analysis_records"]:
                if key in json_string:
                    value = json_string[key]
                    if isinstance(value, list):
                        records = value
                        break
                    elif isinstance(value, str):
                        # Parse JSON string
                        cleaned = value.strip()
                        if cleaned.startswith("```json"):
                            cleaned = re.sub(r'^```json\s*', "", cleaned)
                        if cleaned.startswith("```"):
                            cleaned = re.sub(r'^```\s*', "", cleaned)
                        if cleaned.endswith("```"):
                            cleaned = re.sub(r'\s*```$', "", cleaned)
                        cleaned = cleaned.strip()
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, list):
                            records = parsed
                            break
                        elif isinstance(parsed, dict) and "records" in parsed:
                            records = parsed["records"]
                            break
            else:
                # If no array found, treat the dict itself as a single record
                records = [json_string]
    else:
        # Handle string input (same logic as print_json_records)
        # Strip markdown code blocks if present
        cleaned = json_string.strip()
        if cleaned.startswith("```json"):
            cleaned = re.sub(r'^```json\s*', "", cleaned)
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\s*', "", cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r'\s*```$', "", cleaned)
        cleaned = cleaned.strip()

        # Parse JSON and extract records
        data = json.loads(cleaned)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Try common keys
            for key in [record_key, "records", "yield_records", "data", "final_meta_analysis_records"]:
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            else:
                # If no array found, treat the dict itself as a single record
                records = [data]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

    if not records:
        # Nothing to write
        return 0

    # Determine fieldnames from union of all keys
    fieldnames = sorted({key for rec in records for key in rec.keys()})

    csv_path = str(path)
    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or os.path.getsize(csv_path) == 0

    if file_empty:
        # New file or empty file: write header and rows
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = {fn: _normalize_cell_value(rec.get(fn)) for fn in fieldnames}
                writer.writerow(row)
        return len(records)

    # Existing non-empty file: check header compatibility
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            # File has no rows but is non-empty (corrupt or custom) - treat as mismatch
            existing_header = []

    if existing_header != fieldnames:
        existing_set = set(existing_header)
        new_set = set(fieldnames)
        only_in_existing = sorted(existing_set - new_set)
        only_in_new = sorted(new_set - existing_set)
        raise ValueError(
            "CSV schema mismatch between existing file and new records.\n"
            f"Existing header: {existing_header}\n"
            f"New header: {fieldnames}\n"
            f"Columns only in existing file: {only_in_existing}\n"
            f"Columns only in new records: {only_in_new}"
        )

    # Schemas match: append rows
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_header)
        for rec in records:
            row = {fn: _normalize_cell_value(rec.get(fn)) for fn in existing_header}
            writer.writerow(row)

    return len(records)