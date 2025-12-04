import json

with open("data/hemorrhage_brain_ct.json", "r", encoding="utf-8") as file:
    data = json.load(file)

filtered_data = [{"patient_id": entry["patient_id"], "report_text": entry["report_text"], "is_hemorrhage": entry["is_hemorrhage"], "type": entry["type"]} for entry in data]

with open("data/test.reports_brain_ct_id_text.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4)

print(f"Extracted JSON saved as 'extracted_data.json' with {len(filtered_data)} entries.")