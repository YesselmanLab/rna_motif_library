"""
Module for parsing SNAP output files.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from rna_motif_library.x3dna import (
    X3DNAInteraction,
    X3DNAResidueFactory,
)
from rna_motif_library.util import sanitize_x3dna_atom_name
from rna_motif_library.settings import DSSR_EXE


def parse_snap_output(out_file: str) -> List[X3DNAInteraction]:
    """
    Retrieves RNA-Protein (RNP) interactions from an output file or a PDB file.

    Args:
        out_file (str): the file path to the output file where interactions are recorded.

    Returns:
        interactions (list): A list of RNPInteraction instances capturing the details of each interaction.

    """
    # Open and read the .out file containing RNPs
    with open(out_file, "r") as file:
        lines = file.readlines()

    s = "".join(lines)
    spl = s.split("List")
    interactions = []
    for section in spl:
        if "H-bonds" not in section:
            continue
        lines = section.split("\n")
        lines.pop(0)  # Remove the first two lines as they are headers
        lines.pop(0)
        for line in lines:
            i_spl = line.split()
            if len(i_spl) < 4:
                continue
            # Process interaction type
            inter_type = i_spl[5].split(":")[0]
            nt_part = {"po4": "phos", "sugar": "sugar", "base": "base"}.get(
                inter_type, inter_type
            )
            atom_1, res_1 = i_spl[2].split("@")
            atom_2, res_2 = i_spl[3].split("@")
            type_1, type_2 = nt_part, "aa"
            atom_1 = sanitize_x3dna_atom_name(atom_1)
            atom_2 = sanitize_x3dna_atom_name(atom_2)

            if res_1 > res_2:
                atom_1, atom_2 = atom_2, atom_1
                res_1, res_2 = res_2, res_1
                type_1, type_2 = type_2, type_1

            interactions.append(
                X3DNAInteraction(
                    atom_1,
                    X3DNAResidueFactory.create_from_string(res_1),
                    atom_2,
                    X3DNAResidueFactory.create_from_string(res_2),
                    float(i_spl[4]),
                    nt_part,
                    "aa",
                )
            )

    return interactions


def generate_out_file(pdb_path: str, out_path: str = "test.out") -> None:
    """
    Generates an .out file from DSSR data.

    Args:
        pdb_path (str): path to source PDB
        out_path (str): path to .out file

    Returns:
        None

    """
    dssr_exe = DSSR_EXE
    subprocess.run(f"{dssr_exe} snap -i={pdb_path} -o={out_path}", shell=True)
    files = [
        "dssr-2ndstrs.bpseq",
        "dssr-2ndstrs.ct",
        "dssr-2ndstrs.dbn",
        "dssr-atom2bases.pdb",
        "dssr-stacks.pdb",
        "dssr-torsions.txt",
    ]
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
