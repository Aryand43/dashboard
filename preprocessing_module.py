import csv
import hashlib
import os
import json

def parse_csv_to_record(csv_path: str) -> dict:
    """
    Reads a CSV file, extracts metrics into key-value pairs, and generates a run_id.

    Args:
        csv_path: The path to the CSV file.

    Returns:
        A dictionary representing a single record with run_id, wer, and bleu.
    """
    record = {}
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metric = row["metric"]
            value = float(row["value"])
            record[metric] = value

    filename = os.path.basename(csv_path)
    model_name = os.path.splitext(filename)[0]
    hashid = hashlib.sha1(filename.encode()).hexdigest()
    record["run_id"] = f"{model_name}_{hashid}"

    return record

def process_all_csvs(data_raw_path: str = "data/raw/") -> list[dict]:
    """
    Loops over all CSVs in data/raw/ and returns a list of records.

    Args:
        data_raw_path: The path to the directory containing raw CSV files.

    Returns:
        A list of dictionaries, where each dictionary is a processed record.
    """
    all_records = []
    for filename in os.listdir(data_raw_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(data_raw_path, filename)
            record = parse_csv_to_record(csv_path)
            all_records.append(record)

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_file = os.path.join(processed_dir, "processed_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(all_records, f, indent=4)
    print(f"Processed data saved to {output_file}")

    return all_records

if __name__ == "__main__":
    records = process_all_csvs()
    # The print statements are now redundant as the data is saved to a file.
    # for record in records:
    #     print(record)

