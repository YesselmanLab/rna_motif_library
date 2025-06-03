import json
import click
import os

from pydssr.dssr import DSSROutput
from rna_motif_library.motif import DATA_PATH


@click.command()
@click.argument("pdb_code")
@click.argument("residue")
def main(pdb_code, residue):
    d = DSSROutput(json_path=os.path.join(DATA_PATH, "dssr_output", f"{pdb_code}.json"))
    motifs = d.get_motifs()
    for motif in motifs:
        for res in motif.nts_long:
            if res == residue:
                print(motif.mtype, motif.nts_long)
                break


if __name__ == "__main__":
    main()
