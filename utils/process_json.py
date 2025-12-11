import json

with open("data/test.reports_brain_ct_id_text.json", "r", encoding="utf-8") as file:
    input_json = json.load(file)

def process_json_reports(json_data):
    processed_reports = []
    hemorrhage_keywords = ["hemorrhage", "hematoma", "hemorrhagic", "haemorrhage", "haematoma", "haemorrhagic"]

    for report_obj in json_data:
        processed_obj = report_obj.copy()
        report_text = processed_obj.pop('report_text')

        if processed_obj['is_hemorrhage']:
            supporting_snippets = []
            sentences = report_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                for keyword in hemorrhage_keywords:
                    if keyword in sentence.lower():
                        supporting_snippets.append(sentence + ".")
                        break # Avoid adding the same sentence multiple times if multiple keywords present
            processed_obj['supporting_text'] = supporting_snippets
        else:
            processed_obj['supporting_text'] = []
            pass # report_text is already removed via pop()

        processed_reports.append(processed_obj)
    return processed_reports

processed_output = process_json_reports(input_json)

with open("data/suggestive_text_brain_ct.json", "w", encoding="utf-8") as file:
    json.dump(processed_output, file, indent=4)