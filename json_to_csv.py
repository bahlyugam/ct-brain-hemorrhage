import json
import csv
import sys

def convert_json_to_csv(input_file, output_file):
    """
    Reads JSON data from input_file, extracts patient_id and hemorrhage type
    for cases where hemorrhage.is_hemorrhage is true, and writes the results
    to a CSV file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output CSV file
    """
    try:
        # Read the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Open the CSV file for writing
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write the header row
            writer.writerow(['patient_id', 'type'])
            
            # Process each JSON object in the array
            for item in data:
                # Check if this is a hemorrhage case
                if 'hemorrhage' in item and item['hemorrhage'].get('is_hemorrhage', False):
                    patient_id = item.get('patient_id', '')
                    hemorrhage_type = item['hemorrhage'].get('type', '')
                    
                    # Write the row to the CSV file
                    writer.writerow([patient_id, hemorrhage_type])
        
        print(f"CSV file has been created successfully at {output_file}")
        
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {input_file} does not contain valid JSON.")
    except KeyError as e:
        print(f"Error: Missing expected key in JSON structure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input.json output.csv")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_json_to_csv(input_file, output_file)