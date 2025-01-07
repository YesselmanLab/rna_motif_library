import os
import json
import click

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.classes import Residue


def write_cif_header(f):
    f.write("data_\n")
    f.write("_entry.id test\n")
    f.write("loop_\n")
    f.write("_atom_site.group_PDB\n")
    f.write("_atom_site.id\n")
    f.write("_atom_site.auth_atom_id\n")
    f.write("_atom_site.auth_comp_id\n")
    f.write("_atom_site.auth_asym_id\n")
    f.write("_atom_site.auth_seq_id\n")
    f.write("_atom_site.pdbx_PDB_ins_code\n")
    f.write("_atom_site.Cartn_x\n")
    f.write("_atom_site.Cartn_y\n")
    f.write("_atom_site.Cartn_z\n")


@click.command()
@click.argument("pdb_code")
@click.argument("residue_ids", nargs=-1, type=str)
def cli(pdb_code, residue_ids):

    residue_data = json.loads(
        open(os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")).read()
    )
    all_residues = {k: Residue.from_dict(v) for k, v in residue_data.items()}
    residues = [all_residues[residue_id] for residue_id in residue_ids]
    f = open("test.cif", "w")
    write_cif_header(f)
    acount = 1
    for residue in residues:
        s, acount = residue.to_cif_str(acount)
        f.write(s)
    f.close()


def main():
    cli()


if __name__ == "__main__":
    main()
