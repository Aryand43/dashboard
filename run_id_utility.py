import json
import os

def shorten_run_id_hash(records: list[dict], hash_length: int = 6) -> list[dict]:
    """
    Shortens the hash suffix of run_id values in a list of records.

    Args:
        records: A list of dictionaries, each with 'run_id', 'wer', and 'bleu'.
        hash_length: The desired length for the truncated hash (default: 6).

    Returns:
        A new list of dictionaries with updated run_id values.
    """
    if not (6 <= hash_length <= 8):
        raise ValueError("hash_length must be between 6 and 8 characters.")

    shortened_records = []
    for record in records:
        run_id = record["run_id"]
        parts = run_id.split("_")
        model_name = "_".join(parts[:-1])
        original_hash = parts[-1]

        if len(original_hash) > hash_length:
            shortened_hash = original_hash[:hash_length]
            new_run_id = f"{model_name}_{shortened_hash}"
        else:
            new_run_id = run_id  # Hash is already shorter or equal to desired length

        shortened_records.append({
            "run_id": new_run_id,
            "wer": record["wer"],
            "bleu": record["bleu"]
        })
    return shortened_records

if __name__ == "__main__":
    # Load preprocessed data
    processed_data_path = "data/processed/processed_metrics.json"
    if not os.path.exists(processed_data_path):
        print(f"Error: {processed_data_path} not found. Please run preprocessing_module.py first.")
    else:
        with open(processed_data_path, 'r') as f:
            processed_data = json.load(f)

        # Shorten run_id hashes to 6 characters
        shortened_data = shorten_run_id_hash(processed_data, hash_length=6)
        
        # Save the shortened data back to the file
        with open(processed_data_path, 'w') as f:
            json.dump(shortened_data, f, indent=4)
        print(f"Shortened run_id hashes saved to {processed_data_path}")

        # Optional: Print first few records to confirm
        print("\nFirst 3 records with shortened run_ids:")
        for record in shortened_data[:3]:
            print(record)

