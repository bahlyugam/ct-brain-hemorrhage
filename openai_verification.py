import json
import random
import argparse
import os
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define all pathology groups
groups = [
    "calcification_or_granuloma",
    "cerebral_edema",
    "cystic_lesion",
    "diffuse_cerebral_atrophy",
    "hydrocephalus",
    "intracranial_hemorrhage",
    "ischemic_injury_or_infarct",
    "neoplasm_or_mass",
    "artifact",
    "rare_brain",
    "pns",
    "trauma_or_skull_injury",
    "vascular_lesion",
    "inaccurate_study"
]

def verify_fields_combined(patient_obj):
    report_text = patient_obj["report_text"]
    extracted_info = []

    for group in groups:
        if group not in patient_obj:
            continue
        group_data = patient_obj[group]
        extracted_info.append({
            "group": group,
            "is_present": group_data.get("is_present", False),
            "type": group_data.get("type", None),
            "supporting_text": group_data.get("supporting_text", None)
        })

    prompt = f"""
You are a radiology expert AI.

Here is a CT brain radiology report:
---
{report_text}
---

The following are extracted findings from this report for different pathology groups. For each group, please evaluate if the extracted data is accurate, based on the report.

Respond as a JSON list, where each object contains:
- group: pathology group name (as given),
- extraction_is_correct: true or false,
- reason: brief explanation.

Here are the extracted findings:
{json.dumps(extracted_info, indent=2)}

Respond in this format:
[
  {{
    "group": "cystic_lesion",
    "extraction_is_correct": true,
    "reason": "The report mentions 'arachnoid cyst', which confirms the extraction."
  }}
]
"""

    try:
        response = openai.ChatCompletion.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a radiology expert AI."},
                {"role": "user", "content": prompt}
            ]
        )

        usage = response.usage
        print(f"[{patient_obj.get('patient_id')}] Input tokens: {usage.prompt_tokens} | Output tokens: {usage.completion_tokens}")

        reply = response.choices[0].message.content
        result = json.loads(reply)

        return [res for res in result if not res.get("extraction_is_correct", True)]

    except Exception as e:
        print(f"Error verifying patient {patient_obj.get('patient_id')}: {e}")
        return [{
            "group": "all",
            "extraction_is_correct": False,
            "reason": f"Error during validation: {str(e)}"
        }]

def main(input_file, output_file, test_run):
    with open(input_file, "r") as f:
        data = json.load(f)

    if test_run:
        data = random.sample(data, 1)

    results = []
    for patient in tqdm(data):
        patient_id = patient.get("patient_id", "unknown")
        issues = verify_fields_combined(patient)
        if issues:
            results.append({"patient_id": patient_id, "issues": issues})

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Verification complete. {len(results)} reports had issues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pathology_processed_reports.json", help="Input JSON file")
    parser.add_argument("--output", default="data/verification_results.json", help="Output file for results")
    parser.add_argument("--test-run", action="store_true", help="Run on only 10 random patients")
    args = parser.parse_args()

    main(args.input, args.output, args.test_run)
