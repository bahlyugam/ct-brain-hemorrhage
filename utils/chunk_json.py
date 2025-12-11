import json
import os

def chunk_json_with_patient_info(input_file, output_dir, chunk_size=50):
    """
    Reads a large JSON file, splits it into smaller chunks, keeping only 'patient_id' and 'report_text',
    and saves each chunk as a separate JSON file.

    Args:
        input_file (str): The path to the input JSON file.
        output_dir (str): The directory where the output chunks will be saved.
        chunk_size (int, optional): The maximum number of JSON objects per chunk. Defaults to 50.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{input_file}': {e}")
        return
    except FileNotFoundError:
        print(f"Error: File not found at '{input_file}'")
        return

    if not isinstance(data, list):
        print(f"Error: Input JSON file should contain a list of objects, but found {type(data)}")
        return

    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(data))
        chunk = data[start_index:end_index]

        # Process each item in the chunk to keep only patient_id and report_text
        processed_chunk = []
        for item in chunk:
            if isinstance(item, dict):
                processed_item = {}
                if 'patient_id' in item:
                    processed_item['patient_id'] = item['patient_id']
                if 'report_text' in item:
                    processed_item['report_text'] = item['report_text']
                processed_chunk.append(processed_item)
            #handle the case where the item is not a dict.
            else:
                print(f"Warning: Item at index {start_index + (end_index - start_index)} in chunk {i+1} is not a dictionary. Skipping.")

        output_file = os.path.join(output_dir, f"chunk_{i + 1}.json")
        try:
            with open(output_file, 'w') as outfile:
                json.dump(processed_chunk, outfile, indent=4)
            print(f"Saved chunk {i + 1} to '{output_file}'")
        except Exception as e:
            print(f"Error writing chunk {i + 1} to '{output_file}': {e}")



if __name__ == "__main__":
    input_file = "data/test.reports_brain_ct_id_text.json"
    output_dir = "data/reports_brain_ct_id_text_chunks"
    chunk_size = 50

    chunk_json_with_patient_info(input_file, output_dir, chunk_size)
