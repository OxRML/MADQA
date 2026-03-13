"""
Download the MADQA PDF document collection from HuggingFace.

PDFs are hosted externally; the dataset's "documents" configuration provides
a mapping from document filenames to their download URLs.

Usage:
    pip install datasets tqdm
    python download_pdfs.py                    # download all PDFs
    python download_pdfs.py --limit 10         # download first 10
    python download_pdfs.py -o ./my_pdfs       # custom output directory
"""

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset, DownloadManager
from tqdm import tqdm

DATASET_REPO = "OxRML/MADQA"


def download_pdfs(
    repo_id: str = DATASET_REPO,
    output_dir: str = "./pdfs",
    limit: int | None = None,
) -> list[Path]:
    """Download PDF documents from the MADQA HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repository ID.
        output_dir: Local directory to store downloaded PDFs.
        limit: Maximum number of PDFs to download (None for all).

    Returns:
        List of local paths to downloaded PDFs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading document URLs from {repo_id}...")
    docs = load_dataset(repo_id, "documents", split="links")
    doc_urls = {r["document"]: r["url"] for r in docs}

    filenames = sorted(doc_urls.keys())
    if limit:
        filenames = filenames[:limit]

    print(f"Downloading {len(filenames)} PDFs to {output_dir}/")

    dm = DownloadManager()
    local_paths = []
    for filename in tqdm(filenames, desc="Downloading"):
        try:
            cached_path = dm.download(doc_urls[filename])
            dest = output_dir / filename
            if not dest.exists():
                shutil.copy2(cached_path, dest)
            local_paths.append(dest)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    print(f"\nDone. {len(local_paths)} PDFs saved to {output_dir}/")
    return local_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MADQA PDF collection")
    parser.add_argument(
        "-o", "--output-dir", default="./pdfs", help="Output directory (default: ./pdfs)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max PDFs to download (default: all)"
    )
    parser.add_argument(
        "--repo", default=DATASET_REPO, help=f"HuggingFace repo (default: {DATASET_REPO})"
    )
    args = parser.parse_args()

    paths = download_pdfs(repo_id=args.repo, output_dir=args.output_dir, limit=args.limit)

    # Show a few examples
    if paths:
        print(f"\nFirst files:")
        for p in paths[:5]:
            print(f"  {p}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more")
