import pandas as pd
import click

from rna_motif_library.motif import MotifFactoryFromOther
from rna_motif_library.motif_factory import get_pdb_structure_data

@click.command()
@click.argument("motif_name")
def main(motif_name):
    pdb_id = motif_name.split("-")[-2]
    pdb_data = get_pdb_structure_data(pdb_id)
    mf = MotifFactoryFromOther(pdb_data)
    motifs = mf.get_motifs_from_dssr()
    motif_by_name = {m.name: m for m in motifs}
    motif = motif_by_name[motif_name]
    motif.to_cif()

if __name__ == "__main__":
    main()