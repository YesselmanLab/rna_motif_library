import pandas as pd
import Bio.PDB
import numpy as np
import biopandas
from biopandas.mmcif import PandasMmcif

from rna_motif_library.residue import (
    Residue,
    get_residues_from_cif,
    residues_to_cif_file,
)
from rna_motif_library.logger import get_logger
from rna_motif_library.util import get_cif_header_str


log = get_logger("dataframe_tools")


def parse_residue_identifier(residue_identifier: str) -> tuple:
    """
    parse a residue identifier such as "A-1-1-A"
    into a tuple of (chain_id, res_id, res_num, ins_code)
    """
    spl = residue_identifier.split("-")
    return {
        "chain_id": spl[0],
        "res_id": spl[1],
        "res_num": spl[2],
        "ins_code": spl[3],
    }


def parse_motif_indentifier(motif_name: str) -> tuple:
    """
    parse a name such as HAIRPIN-1-CGG-7PWO-1
    into a tuple of (mtype, msize, msequence, pdb_id)
    """
    spl = motif_name.split("-")
    mtype = spl[0]
    # Count how many elements are numbers (size components)
    size_parts = []
    i = 1
    while i < len(spl) and spl[i].isdigit():
        size_parts.append(int(spl[i]))
        i += 1
    msize = "-".join([str(p) for p in size_parts])
    msequence = "-".join(spl[i:-2])
    pdb_id = spl[-2]
    return (mtype, msize, msequence, pdb_id)


def add_motif_indentifier_columns(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """Add columns for each component of a motif name.

    Args:
        df: DataFrame containing motif names
        name_col: Name of column containing motif names

    Returns:
        DataFrame with new columns for motif type, size, sequence and PDB ID
    """
    components = df[name_col].apply(parse_motif_indentifier)
    df["mtype"] = components.apply(lambda x: x[0])
    df["msize"] = components.apply(lambda x: x[1])
    df["msequence"] = components.apply(lambda x: x[2])
    df["pdb_id"] = components.apply(lambda x: x[3])
    return df


def get_cif_str_from_row(
    row: pd.Series, res_col: str, coord_col: str, atom_col: str
) -> str:
    res = row[res_col]
    coord = row[coord_col]
    atom_names = row[atom_col]
    cif_str = ""
    res_objs = []
    for res, coord, atom_name in zip(res, coord, atom_names):
        res_data = parse_residue_identifier(res)
        r = Residue(
            res_data["chain_id"],
            res_data["res_id"],
            res_data["res_num"],
            res_data["ins_code"],
            "",
            atom_name,
            coord,
        )
        res_objs.append(r)
    s = get_cif_header_str()
    acount = 1
    for residue in res_objs:
        s += residue.to_cif_str(acount)
        acount += len(residue.atom_names)
    return s


def generate_motif_cifs_from_cif_file(df, cif_file: str):
    """
    Generate CIF strings for each row in the DataFrame using the CIF file.

    Args:
        df: DataFrame with a 'residues' column containing lists of residues
        cif_file: Path to the CIF file containing coordinates and atom names

    Returns:
        list: List of CIF strings, one for each row in the DataFrame
    """
    pdb_id = cif_file.basename().split(".")[0]
    # Group atoms by residue
    residues_dict = get_residues_from_cif(cif_file)

    # Generate CIF strings for each row
    cif_strings = []
    for _, row in df.iterrows():
        if row["pdb_id"] != pdb_id:
            continue

        # Generate CIF string for this row
        s = get_cif_header_str()
        acount = 1
        res_objs = []
        for r in row["residues"]:
            if r in residues_dict:
                res_objs.append(residues_dict[r])
            else:
                log.warning(f"Residue {r} not found in CIF file")
        residues_to_cif_file(res_objs, f"{row['motif_id']}.cif")
