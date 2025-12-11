import json

with open("data/test.reports_brain_ct.json", "r", encoding="utf-8") as file:
    data = json.load(file)

keyword = 'BRAIN'

filtered_data = [entry for entry in data if keyword in entry["report_text"].upper()]

with open("data/cleaned_test.reports_brain_ct.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered JSON saved as 'cleaned_test.reports_brain_ct.json' with {len(filtered_data)} entries.")