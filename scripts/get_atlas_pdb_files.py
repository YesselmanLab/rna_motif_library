import pandas as pd
import wget

from rna_motif_library.util import get_pdb_ids
from rna_motif_library.parallel_utils import run_w_threads_in_batches

def download_hairpin_csv(pdb_id):
    url = f"https://rna.bgsu.edu/rna3dhub/loops/download/{pdb_id}"
    wget.download(url, out=f"data/atlas_csvs/hairpins/{pdb_id}.csv")


def main():
    """
    main function for script
    """
    pdb_ids = get_pdb_ids()
    results = run_w_threads_in_batches(
        items=pdb_ids,
        func=download_hairpin_csv,
        threads=30,
        batch_size=100,
        desc="Downloading hairpin csvs",
    )

if __name__ == '__main__':
    main()
