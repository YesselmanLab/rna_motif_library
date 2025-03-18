import pandas as pd

import click
import os
from rna_motif_library.settings import DATA_PATH


@click.group()
def cli():
    """Check for missing resource files in the library."""
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def pdbs(csv_path):
    """Check for missing PDB files."""
    df = pd.read_csv(csv_path)
    pdb_dir = os.path.join(DATA_PATH, "pdbs")
    missing = []

    for pdb_id in df.pdb_id:
        if not os.path.exists(os.path.join(pdb_dir, f"{pdb_id}.cif")):
            missing.append(pdb_id)

    if missing:
        click.echo(f"Missing {len(missing)} PDB files:")
        for pdb_id in missing:
            click.echo(pdb_id)
    else:
        click.echo("All PDB files present")


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def dssr(csv_path):
    """Check for missing DSSR output files."""
    df = pd.read_csv(csv_path)
    dssr_dir = os.path.join(DATA_PATH, "dssr_output")
    missing = []

    for pdb_id in df.pdb_id:
        if not os.path.exists(os.path.join(dssr_dir, f"{pdb_id}.json")):
            missing.append(pdb_id)

    if missing:
        click.echo(f"Missing {len(missing)} DSSR files:")
        for pdb_id in missing:
            click.echo(pdb_id)
    else:
        click.echo("All DSSR files present")


if __name__ == "__main__":
    cli()
