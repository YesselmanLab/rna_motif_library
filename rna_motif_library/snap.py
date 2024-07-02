import os
import subprocess
import settings
from typing import List, Optional


class RNPInteraction:
    """Class to represent an RNA-Protein interaction."""

    def __init__(self, nt_atom: str, aa_atom: str, dist: float, interaction_type: str):
        self.nt_atom = nt_atom  # Atom of nucleotide, e.g., 'N6@C.A1534'
        self.aa_atom = aa_atom  # Atom of amino acid, e.g., 'O@A.VAL281'
        self.dist = dist  # Distance between the two
        self.type = interaction_type  # Type of interaction, e.g., 'base:sidechain'
        self.nt_res = nt_atom.split("@")[1]  # Residue of nucleotide


def get_rnp_interactions(
    pdb_path: Optional[str] = None, out_file: Optional[str] = None
) -> List[RNPInteraction]:
    """Retrieves RNA-Protein (RNP) interactions from an output file or a PDB file.

    Args:
        pdb_path: Optional; the file path to the PDB file if available.
        out_file: Optional; the file path to the output file where interactions are recorded.

    Returns:
        A list of RNPInteraction instances capturing the details of each interaction.

    Raises:
        ValueError: If neither a pdb_path nor an out_file is provided.
    """
    if pdb_path is None and out_file is None:
        raise ValueError("Must supply either a pdb or out file")

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
            inter_type_new = f"{nt_part}:aa"

            interactions.append(
                RNPInteraction(i_spl[2], i_spl[3], float(i_spl[4]), inter_type_new)
            )

    return interactions


def __generate_out_file(pdb_path: str, out_path: str = "test.out") -> None:
    """Generates an .out file from DSSR data."""
    dssr_exe = settings.DSSR_EXE
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
