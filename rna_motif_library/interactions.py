import os
from typing import List
from dataclasses import dataclass


from pydssr.dssr_classes import DSSR_HBOND
from rna_motif_library.settings import LIB_PATH


from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class RNPInteraction:
    """
    Class to represent an RNA-Protein interaction.

    Args:
        nt_atom (str): atom of nucleotide in interaction
        aa_atom (str): atom of amino acid in interaction
        dist (float): distance between atoms in interaction (angstroms)
        interaction_type (str): type of interaction (base:sidechain/base:aa/etc)
    """

    nt_atom: str
    aa_atom: str
    dist: float
    interaction_type: str

    def __post_init__(self):
        object.__setattr__(self, "type", self.interaction_type)
        object.__setattr__(self, "nt_res", self.nt_atom.split("@")[1])


def get_interactions(pdb_name: str, hbonds: List[DSSR_HBOND]):
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{pdb_name}.out")
