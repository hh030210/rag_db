import os
import json
import glob

data_dir = r"C:\Users\胡铭强\Desktop\chunk_code\data"
output_file = r"C:\Users\胡铭强\Desktop\chunk_code\_scan_results.txt"

def extract_numeric_metrics(data, indent=2):
    """Extract and display numeric metrics from data."""
    metrics = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics.append(f"{'  ' * indent}{key}: {value}")
            elif isinstance(value, dict):
                metrics.append(f"{'  ' * indent}{key}:")
                metrics.extend(extract_numeric_metrics(value, indent + 1))
            elif isinstance(value, list) and len(value) > 0:
                if all(isinstance(item, (int, float)) for item in value):
                    mean_val = sum(value) / len(value) if value else 0
                    metrics.append(f"{'  ' * indent}{key}: [{len(value)} items, mean={mean_val:.4f}]")
                elif all(isinstance(item, dict) for item in value):
                    metrics.append(f"{'  ' * indent}{key}: [list of {len(value)} dicts]")
                    # Sample first item's structure
                    if value:
                        first = value[0]
                        for k, v in list(first.items())[:5]:
                            if isinstance(v, (int, float)):
                                metrics.append(f"{'  ' * (indent+1)}{k}: {v}")
                            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                                mean_v = sum(v) / len(v)
                                metrics.append(f"{'  ' * (indent+1)}{k}: [mean={mean_v:.4f}]")
    return metrics

results = []

# Find all JSON files recursively
json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)

for filepath in sorted(json_files):
    try:
        rel_path = os.path.relpath(filepath, r"C:\Users\胡铭强\Desktop\chunk_code")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results.append(f"=== {rel_path} ===")
        results.append(f"keys: {list(data.keys()) if isinstance(data, dict) else 'N/A (not a dict)'}")
        results.append("data preview:")
        
        if isinstance(data, dict):
            metrics = extract_numeric_metrics(data)
            if metrics:
                results.extend(metrics)
            else:
                results.append("  (no numeric values found)")
        else:
            results.append(f"  type: {type(data).__name__}")
            if isinstance(data, list):
                results.append(f"  length: {len(data)}")
        
        results.append("")
        
    except Exception as e:
        results.append(f"=== {os.path.relpath(filepath, r'C:\Users\胡铭强\Desktop\chunk_code')} ===")
        results.append(f"ERROR reading file: {e}")
        results.append("")

# Write to output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

# Also print to console
print('\n'.join(results))
print(f"\n\nTotal JSON files found: {len(json_files)}")
print(f"Results saved to: {output_file}")
