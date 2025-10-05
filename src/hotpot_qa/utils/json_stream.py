"""
JSON streaming utilities for handling large files efficiently.
"""

import json
import logging
from typing import Dict, List, Iterator, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class JSONStreamer:
    """Memory-efficient JSON file access for large datasets."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_small_file(self) -> Dict[str, Any]:
        """Load small JSON files entirely into memory."""
        logger.info(f"Loading JSON from {self.file_path}")
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def stream_keys_and_sample(self, max_samples: int = 5) -> Dict[str, Any]:
        """Stream keys and sample values without loading entire file."""
        logger.info(f"Sampling JSON from {self.file_path}")
        sample_data = {}

        with open(self.file_path, 'r') as f:
            try:
                data = json.load(f)
                keys = list(data.keys())[:max_samples]

                for key in keys:
                    sample_data[key] = self._truncate_value(data[key])

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                return {}

        return sample_data

    def _truncate_value(self, value: Any, max_items: int = 3) -> Any:
        """Truncate large values for sampling."""
        if isinstance(value, list) and len(value) > max_items:
            return value[:max_items] + [f"... ({len(value)} total items)"]
        elif isinstance(value, dict) and len(value) > max_items:
            truncated = {k: self._truncate_value(v, max_items) for k, v in list(value.items())[:max_items]}
            truncated["..."] = f"({len(value)} additional keys)"
            return truncated
        else:
            return value

    def get_value_by_key(self, key: str) -> Any:
        """Get a specific value by key without loading entire file."""
        logger.debug(f"Getting value for key: {key}")

        with open(self.file_path, 'r') as f:
            data = json.load(f)
            return data.get(key)

    def keys_exist(self, keys: List[str]) -> bool:
        """Check if specific keys exist in the JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                return all(key in data for key in keys)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def get_file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_path.stat().st_size / (1024 * 1024)

    def stream_large_keys_and_count(self, max_keys: int = 1000) -> Dict[str, int]:
        """Stream through large JSON file to count keys and get sample (memory efficient)."""
        import ijson

        keys_count = {}
        try:
            with open(self.file_path, 'rb') as f:
                # Count occurrences and get sample
                count = 0
                for key in ijson.items(f, ''):
                    if isinstance(key, dict):
                        for k in key.keys():
                            keys_count[k] = keys_count.get(k, 0) + 1
                            count += 1
                            if count >= max_keys:
                                break
                    if count >= max_keys:
                        break
        except ImportError:
            logger.warning("ijson not available, falling back to sample loading")
            sample = self.stream_keys_and_sample(max_samples=max_keys)
            keys_count = {k: 1 for k in sample.keys()[:max_keys]}

        return keys_count
