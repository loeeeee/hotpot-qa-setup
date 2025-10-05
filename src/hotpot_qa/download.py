#!/usr/bin/env python3
import os
import hashlib
import requests
import argparse
from pathlib import Path

EXPECTED_SHA256 = {
    "hotpot_train_v1.1.json": "26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316",
    "hotpot_dev_distractor_v1.json": "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
    "hotpot_dev_fullwiki_v1.json": "2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618",
}

def compute_sha256(filename: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_file_integrity(filename: str) -> bool:
    """Check if file exists and matches expected SHA256 if available."""
    if not os.path.isfile(filename):
        print(f"[INFO] File {filename} does not exist")
        return False

    base_filename = os.path.basename(filename)
    expected = EXPECTED_SHA256.get(base_filename)
    if expected:
        actual = compute_sha256(filename)
        if actual != expected:
            print(f"[ERROR] SHA256 checksum mismatch for {filename}. Expected: {expected}, Got: {actual}")
            return False

    print(f"[INFO] File {filename} integrity verified")
    return True

def download_if_missing(url: str, filename: str) -> None:
    """Download file if not present or invalid."""
    if check_file_integrity(filename):
        print(f"[INFO] {filename} already present and valid, skipping download")
        return

    print(f"[INFO] Downloading {filename} from {url}")
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

    if not check_file_integrity(filename):
        print(f"[ERROR] Downloaded {filename} failed integrity check")
        os.remove(filename)
        raise RuntimeError(f"Integrity check failed for {filename}")

    print(f"[INFO] Successfully downloaded and verified {filename}")

def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(description="Download HotpotQA datasets and Wikipedia dump.")
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=str,
        help="Directory to save raw downloaded data."
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=str,
        help="Directory for processed data (reserved for future use)."
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)  # Not used in this script

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting HotpotQA data download with integrity checks")

    # HotpotQA files
    hotpot_files = [
        ("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", "hotpot_dev_distractor_v1.json"),
        ("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json", "hotpot_dev_fullwiki_v1.json"),
        ("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json", "hotpot_train_v1.1.json"),
    ]

    for url, filename in hotpot_files:
        filepath = raw_dir / filename
        download_if_missing(url, str(filepath))

    # Wikipedia dump
    wiki_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
    wiki_filename = "enwiki-latest-pages-articles-multistream.xml.bz2"
    wiki_filepath = raw_dir / wiki_filename
    download_if_missing(wiki_url, str(wiki_filepath))

    print("[DONE] All downloads completed successfully")

if __name__ == "__main__":
    main()
