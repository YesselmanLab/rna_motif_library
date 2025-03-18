import click
import os
import pandas as pd

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.residue import get_cached_residues
from rna_motif_library.basepair import get_cached_basepairs
from rna_motif_library.util import canon_res_list, ion_list


def get_non_redundant_set_pdb_ids():
    df = pd.read_csv(os.path.join(DATA_PATH, "csvs", "non_redundant_set.csv"))
    return df["pdb_id"].to_list()


@click.group()
def cli():
    pass


@cli.command()
def basepairs():
    pdb_ids = get_non_redundant_set_pdb_ids()
    pdb_ids = ["1A9N"]
    for pdb_id in pdb_ids:
        df = pd.read_json(
            os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_id}.json")
        )
        print(df.iloc[0])
        exit()


@cli.command()
def small_molecule_count():
    pass


if __name__ == "__main__":
    cli()
