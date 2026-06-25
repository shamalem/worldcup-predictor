"""Download results.csv from the public dataset repo into ../data/.

Usage (from backend/):  python -m scripts.download_data
"""
from __future__ import annotations

import sys
import urllib.request

from ml.config import DATA_DIR

# Primary source: the dataset author's GitHub repo (kept up to date).
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "results.csv"
    print(f"Downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, dest)
    except Exception as e:  # pragma: no cover - network dependent
        print(f"Download failed: {e}")
        print("Manual fallback: download results.csv from")
        print("  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
        print(f"and place it at {dest}")
        sys.exit(1)
    size = dest.stat().st_size
    print(f"Saved {dest} ({size/1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
