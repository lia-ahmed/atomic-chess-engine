#!/usr/bin/env python3
"""
Download Lichess Atomic Chess monthly PGN.ZST files with progress updates.

Usage:
    python download_atomic.py START_YEAR END_YEAR

Example:
    python download_atomic.py 2015 2016

DERIVED FROM THE WORK OF STEVEN E. PAV
"""

import os
import sys
import requests
from pathlib import Path

BASE_URL = "https://database.lichess.org/atomic"
SAVE_DIR = Path("raw_lichess_data") 


def download_file(url, dest):
    """Download a file from url to dest, skipping if already exists."""
    if dest.exists():
        print(f"✅ Already exists: {dest}")
        return True  # counted as processed

    print(f"⬇️  Downloading {url} -> {dest}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Saved: {dest}")
            return True
        else:
            print(f"❌ Failed: {url} (status {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python download_atomic.py START_YEAR END_YEAR")
        sys.exit(1)

    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])

    total_months = (end_year - start_year + 1) * 12
    completed = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filename = f"lichess_db_atomic_rated_{year}-{month:02d}.pgn.zst"
            url = f"{BASE_URL}/{filename}"
            dest = SAVE_DIR / filename

            success = download_file(url, dest)
            if success:
                completed += 1

            # Progress update
            print(f"📊 Progress: {completed}/{total_months} months downloaded "
                  f"({(completed / total_months) * 100:.2f}%)\n")


if __name__ == "__main__":
    main()
