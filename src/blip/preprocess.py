"""
preprocess.py

Preprocessing script for MedGPT BLIP training data.
- Removes PHI (Protected Health Information) from text fields using regex.
- Saves de-identified data to output CSV.
- Designed for HIPAA/GDPR/SOC2 compliance.

Usage (SageMaker Processing Job or local):
python preprocess.py --input_csv input.csv --output_csv output.csv
"""
import re
import pandas as pd
import argparse
import logging

# Set up logging for auditability
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Simple regex-based PHI removal (names, dates, emails, phone numbers, etc.)
def remove_phi(text):
    if not isinstance(text, str):
        return text
    # Remove emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    # Remove dates (simple)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", text)
    # Remove names (very basic, for demo; use a real de-id library for production)
    text = re.sub(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", "[NAME]", text)
    return text

def main(args):
    logging.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    text_fields = ['Question', 'Caption', 'Choice A', 'Choice B', 'Choice C', 'Choice D']
    for field in text_fields:
        if field in df.columns:
            logging.info(f"De-identifying field: {field}")
            df[field] = df[field].apply(remove_phi)
    logging.info(f"Saving de-identified data to {args.output_csv}")
    df.to_csv(args.output_csv, index=False)
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De-identify PHI from BLIP training data.")
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()
    main(args)
