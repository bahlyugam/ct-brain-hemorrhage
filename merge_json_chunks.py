import json
import glob

# Load parent JSON
with open('data/cleaned_test.reports_brain_ct.json', 'r') as f:
    parent_data = json.load(f)

# Convert parent to a dict keyed by patientId
parent_dict = {item['patient_id']: item for item in parent_data}

# Load and merge chunk files
chunk_files = glob.glob('chunks/*.json')  # Adjust path as needed

for chunk_num in range(1, 15):
    chunk_file = f'data/reports_brain_ct_id_text_chunks/chunk_{chunk_num}_gemini.json'
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
        for entry in chunk_data:
            pid = entry['patient_id']
            if pid in parent_dict:
                parent_dict[pid].update(entry)  # Merge the chunk data into parent

# Final merged data as a list
merged_data = list(parent_dict.values())

# Optional: save the result
with open('data/pathology_processed_reports.json', 'w') as f:
    json.dump(merged_data, f, indent=2)