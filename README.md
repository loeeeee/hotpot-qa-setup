# Hotpot QA Setup

Dataset creation tools for long-context HotpotQA evaluation using distractor articles to test language models on reasoning across large documents.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

To create a virtual environment and install dependencies:

```bash
uv sync
```

## Usage

### Downloading Data

Download HotpotQA datasets and the Wikipedia dump:

```bash
uv run hotpot-download --raw-dir ./data/raw --processed-dir ./data/processed
```

- `--raw-dir`: Directory to save raw downloaded data (HotpotQA JSON files and Wikipedia dump).
- `--processed-dir`: Directory for processed data (reserved for future use in pipeline).

This replaces the legacy `scripts/00-download.sh` for cross-platform compatibility and package integration.

### Legacy Scripts

- `scripts/00-download.sh`: Bash script for downloading data. Use the CLI above instead for better compatibility.
