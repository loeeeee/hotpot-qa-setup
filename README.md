# Hotpot QA Setup

Dataset creation tools for long-context HotpotQA evaluation using distractor articles to test language models on reasoning across large documents.

## Quick Start

Get started with creating long-context HotpotQA datasets in minutes:

### 1. Install Dependencies

```bash
# Install uv package manager (if not already installed)
pip install uv

# Create virtual environment and install project dependencies
uv sync
```

### 2. Download Required Data

Download HotpotQA questions and Wikipedia dump:

```bash
# Download HotpotQA dev set and Wikipedia articles
uv run hotpot-download --raw-dir ./data --processed-dir ./data
```

This downloads:
- `hotpot_dev_fullwiki_v1.json` - HotpotQA questions with Wikipedia links
- `enwiki-*-processed.tar.bz2` - Wikipedia articles in JSON format

### 3. Process into Long-Context Datasets

Create datasets with varying context sizes (8k, 32k, 128k tokens):

```bash
# Process full dataset (may take hours for first run due to Wikipedia indexing)
uv run hotpot-process \
  --hotpot-path ./data/hotpot_dev_fullwiki_v1.json \
  --wikipedia-path ./data/enwiki-*-processed.tar.bz2 \
  --output-dir ./data

# For testing with limited questions
uv run hotpot-process \
  --hotpot-path ./data/hotpot_dev_fullwiki_v1.json \
  --wikipedia-path ./data/enwiki-*-processed.tar.bz2 \
  --output-dir ./data \
  --max-questions 10
```

### 4. Output Files

The processing script generates three JSON files in the output directory:
- `long_hotpot_8k.json` - Questions with ~8k token contexts
- `long_hotpot_32k.json` - Questions with ~32k token contexts
- `long_hotpot_128k.json` - Questions with ~128k token contexts

Each question includes supporting facts plus distractor articles to test long-context reasoning.

### Example Usage

```bash
# Complete workflow
uv sync
uv run hotpot-download --raw-dir ./data --processed-dir ./data
uv run hotpot-process \
  --hotpot-path ./data/raw/hotpot_dev_fullwiki_v1.json \
  --wikipedia-path ./data/raw/enwiki-*-processed.tar.bz2 \
  --output-dir ./data/processed \
  --tokenizer simple
```

## Detailed Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

To create a virtual environment and install dependencies:

```bash
uv sync
```

## Detailed Usage

### Downloading Data

Download HotpotQA datasets and the Wikipedia dump:

```bash
uv run hotpot-download --raw-dir ./data/raw --processed-dir ./data/processed
```

- `--raw-dir`: Directory to save raw downloaded data (HotpotQA JSON files and Wikipedia dump).
- `--processed-dir`: Directory for processed data (reserved for future use in pipeline).

This replaces the legacy `scripts/00-download.sh` for cross-platform compatibility and package integration.

### Processing Command Options

```bash
uv run hotpot-process --help
```

Key options:
- `--hotpot-path`: Path to HotpotQA JSON file (required)
- `--wikipedia-path`: Path to Wikipedia dump (.bz2 or directory, required)
- `--output-dir`: Output directory for JSON files (default: data)
- `--tokenizer`: Tokenizer type - "simple" (default) or "nltk"
- `--max-questions`: Limit number of questions to process (useful for testing)

### Legacy Scripts

- `scripts/00-download.sh`: Bash script for downloading data. Use the CLI above instead for better compatibility.
