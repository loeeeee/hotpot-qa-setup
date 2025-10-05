#!/usr/bin/env bash
set -euo pipefail

declare -A EXPECTED_SHA256=(
    ["hotpot_train_v1.1.json"]="26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316"
    ["hotpot_dev_distractor_v1.json"]="4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
    ["hotpot_dev_fullwiki_v1.json"]="2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618"
)

# Function to compute SHA256 checksum
compute_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

# Function to check file integrity
check_file_integrity() {
    local file="$1"
    local expected_sha256="${EXPECTED_SHA256[$1]:-}"

    if [[ ! -f "$file" ]]; then
        echo "[INFO] File $file does not exist"
        return 1
    fi

    if [[ -n "$expected_sha256" ]]; then
        local actual_sha256
        actual_sha256=$(compute_sha256 "$file")
        if [[ "$actual_sha256" != "$expected_sha256" ]]; then
            echo "[ERROR] SHA256 checksum mismatch for $file. Expected: $expected_sha256, Got: $actual_sha256"
            return 1
        fi
    fi

    echo "[INFO] File $file integrity verified"
    return 0
}

# Function to download file with integrity check
download_if_missing() {
    local url="$1"
    local filename="$2"

    if check_file_integrity "$filename"; then
        echo "[INFO] $filename already present and valid, skipping download"
        return 0
    fi

    echo "[INFO] Downloading $filename from $url"
    wget "$url" -O "$filename"

    if ! check_file_integrity "$filename"; then
        echo "[ERROR] Downloaded $filename failed integrity check"
        rm -f "$filename"
        exit 1
    fi

    echo "[INFO] Successfully downloaded and verified $filename"
}

echo "[INFO] Starting HotpotQA data download with integrity checks"

# Download Hotpot Data
download_if_missing "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" "hotpot_dev_distractor_v1.json"
download_if_missing "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json" "hotpot_dev_fullwiki_v1.json"
download_if_missing "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" "hotpot_train_v1.1.json"

# Download Wikipedia Dump for Distractor Articles
download_if_missing "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2" "enwiki-latest-pages-articles-multistream.xml.bz2"

echo "[DONE] All downloads completed successfully"
