import json

# Load JSON files
with open("data/cleaned_test.reports_brain_ct.json", "r") as f1, open("data/suggestive_text_brain_ct.json", "r") as f2:
    data1 = json.load(f1)  # Load first JSON file
    data2 = json.load(f2)  # Load second JSON file

print(len(data1), len(data2))

# Create a dictionary with patient_id as key for easy merging
merged_data = {}

# Add data from first JSON
for record in data1:
    patient_id = record["patient_id"]
    merged_data[patient_id] = record  # Store record using patient_id as key

# Merge data from second JSON
for record in data2:
    patient_id = record["patient_id"]
    if patient_id in merged_data:
        merged_data[patient_id].update(record)  # Merge records with same patient_id
    else:
        merged_data[patient_id] = record  # Add new records if not in first file

# Convert merged dictionary back to a list
merged_list = list(merged_data.values())

# Save the merged JSON
with open("data/hemorrhage_brain_ct.json", "w") as outfile:
    json.dump(merged_list, outfile, indent=4)
