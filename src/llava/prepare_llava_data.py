"""
prepare_llava_data.py

This script is responsible for processing an input JSON file (default: `filtered_dataset.json`)
to produce an output JSON file (default: `final_data.json`). The primary function,
`process_captions()`, generates `final_data.json` with image-caption pairs, which is
the format expected by the LLaVA model training script (`train_llava.py`).

The script can also generate a question-answer formatted dataset using `process_answers()`.

Key Libraries:
-   json: For reading and writing JSON data.
-   argparse: For command-line interface arguments.
-   os: For file system checks (e.g., os.path.exists).

Note:
-   This script depends on an input JSON file (e.g., `filtered_dataset.json`). This input
    file is NOT included in the repository and must be provided by the user. It is
    expected to be a JSON file where each entry contains at least an 'image' field
    (image filename) and a 'conversations' field (a list of dialogue turns).
"""
import json
import argparse
import os

def process_answers(input_filename="filtered_dataset.json", output_filename="final_data.json"):
    """
    Reads data from the input JSON file, reformats conversations into a
    Question & Answer (Q&A) structure, and saves it to the output JSON file.

    Args:
        input_filename (str): Path to the input JSON file.
        output_filename (str): Path to the output JSON file for Q&A data.
    """
    try:
        with open(input_filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{input_filename}'. Please ensure it's a valid JSON file.")
        return

    answers_data = []
    for i, item in enumerate(data):
        gpt_responses = []
        for turn in item.get('conversations', []):
            if turn.get('from') == "gpt":
                gpt_responses.append(str(turn.get('value', ''))) # Ensure value is string
        
        if gpt_responses:
            obj = {'id': i + 1, 'image': item.get('image', 'Unknown_image_path')}
            obj['conversations'] = [
                {"from": "human", "value": "[INST] <image>\nWhat is shown in this image? [/INST]"},
                {"from": "gpt", "value": ' '.join(gpt_responses)}
            ]
            answers_data.append(obj)
            
    try:
        with open(output_filename, "w+") as output_file: 
            json.dump(answers_data, output_file, indent=4)
        print(f"Successfully processed {len(answers_data)} items into '{output_filename}' (Q&A format).")
    except IOError as e:
        print(f"Error writing to '{output_filename}': {e}")


def process_captions(input_filename="filtered_dataset.json", output_filename="final_data.json"):
    """
    Reads data from the input JSON file, extracts GPT responses to form
    a single caption for each image, and saves this to the output JSON file. 
    This is the primary format expected by `train_llava.py`.

    Args:
        input_filename (str): Path to the input JSON file.
        output_filename (str): Path to the output JSON file for caption data.
    """
    try:
        with open(input_filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{input_filename}'. Please ensure it's a valid JSON file.")
        return

    captions_data = []
    for i, item in enumerate(data):
        gpt_responses = []
        for turn in item.get('conversations', []):
            if turn.get('from') == "gpt":
                gpt_responses.append(str(turn.get('value', ''))) # Ensure value is string
        
        if gpt_responses:
            obj = {
                'id': i + 1, 
                'image': item.get('image', 'Unknown_image_path'), 
                'caption': ' '.join(gpt_responses)
            }
            captions_data.append(obj)
            
    try:
        with open(output_filename, "w+") as output_file: 
            json.dump(captions_data, output_file, indent=4)
        print(f"Successfully processed {len(captions_data)} items into '{output_filename}' (caption format).")
        print(f"This '{output_filename}' is formatted for use with train_llava.py.")
    except IOError as e:
        print(f"Error writing to '{output_filename}': {e}")


def get_length(input_filename="filtered_dataset.json"):
    """
    Prints the number of entries (items) in the specified input JSON file.

    Args:
        input_filename (str): Path to the input JSON file.
    """
    try:
        with open(input_filename, "r") as f:
            data = json.load(f)
        print(f"The file '{input_filename}' contains {len(data)} entries.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Cannot get length.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{input_filename}'. Please ensure it's a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to get length of '{input_filename}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for LLaVA model training from an input JSON file. "
                    "Processes data into 'captions' format (default) or 'Q&A' format."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="filtered_dataset.json",
        help="Path to the input JSON file (e.g., filtered_dataset.json). Default: filtered_dataset.json"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="final_data.json",
        help="Path to the output JSON file (e.g., final_data.json). Default: final_data.json"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        default="captions", 
        choices=["captions", "qa"],
        help="Output format for the output file: 'captions' (default, for train_llava.py) or 'qa'."
    )
    parser.add_argument(
        "--get_input_length",
        action='store_true',
        help="If specified, prints the length of the input file after processing."
    )

    args = parser.parse_args()

    print(f"Starting data preparation script...")
    print(f"Input file: '{args.input_file}'")
    print(f"Output file: '{args.output_file}'")
    print(f"Processing format: '{args.format}'")

    if not os.path.exists(args.input_file):
        print(f"CRITICAL ERROR: Input file '{args.input_file}' not found. Please create it or specify the correct path.")
    else:
        if args.format == "captions":
            process_captions(input_filename=args.input_file, output_filename=args.output_file)
        elif args.format == "qa":
            process_answers(input_filename=args.input_file, output_filename=args.output_file)
        
        if args.get_input_length:
            print("\n---")
            get_length(input_filename=args.input_file)
    
    print("\nScript finished.")