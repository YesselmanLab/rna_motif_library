import pandas as pd

from biopandas.mmcif.pandas_mmcif import PandasMmcif
from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS
from typing import Dict, List, Any

canon_res_list = [
    "A",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "C",
    "G",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "U",
    "VAL",
]
canon_amino_acid_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


class PandasMmcifOverride(PandasMmcif):
    """
    Class to override standard behavior for handling mmCIF files in Pandas,
    particularly to address inconsistencies between ATOM and HETATM records.

    """

    def _construct_df(self, text: str) -> pd.DataFrame:
        """
        Constructs a DataFrame from mmCIF text.

        Args:
            text (str): The mmCIF file content as a string.

        Returns:
            combined_df (pd.DataFrame): A combined DataFrame of ATOM and HETATM records.

        """
        data = load_cif_data(text)
        data = data[list(data.keys())[0]]
        self.data = data

        df: Dict[str, pd.DataFrame] = {}
        full_df = pd.DataFrame.from_dict(data["atom_site"], orient="index").transpose()
        full_df = full_df.astype(mmcif_col_types, errors="ignore")

        # Combine ATOM and HETATM records into a single DataFrame
        combined_df = pd.DataFrame(
            full_df[(full_df.group_PDB == "ATOM") | (full_df.group_PDB == "HETATM")]
        )

        try:
            df["ANISOU"] = pd.DataFrame(data["atom_site_anisotrop"])
        except KeyError:
            df["ANISOU"] = pd.DataFrame(columns=ANISOU_DF_COLUMNS)

        return combined_df  # Return the combined DataFrame


class PotentialTertiaryContact:
    def __init__(
        self,
        motif_1,
        motif_2,
        res_1,
        res_2,
        atom_1,
        atom_2,
        type_1,
        type_2,
        distance,
        angle,
    ):
        """
        Holds data about potential tertiary contacts.
        Purpose is to get data ready for export to CSV and tertiary contacts.
        Some data will be left out.

        Args:


        """
        self.motif_1 = motif_1
        self.motif_2 = motif_2
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle


class SingleMotifInteraction:

    def __init__(
        self, motif_name, res_1, res_2, atom_1, atom_2, type_1, type_2, distance, angle
    ):
        """
        Holds data for H-bond interactions within a single motif.
        Purpose is to get data ready for export to CSV.
        Therefore, some data from the HBondInteraction class will be left out.

        Args:
            motif_name (str): name of motif the interaction comes from
            res_1 (str): residue 1 in the interaction
            res_2 (str): residue 2 in the interaction
            atom_1 (str): atom 1 in the interaction
            atom_2 (str): atom 2 in the interaction
            type_1 (str): component of residue 1 in interaction (base/sugar/phos/aa)
            type_2 (str): component of residue 2 in interaction (base/sugar/phos/aa)
            distance (float): distance of interaction, in angstroms
            angle (float): dihedral angle of interaction, in degrees

        """
        self.motif_name = motif_name
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle


class HBondInteraction:
    def __init__(
        self,
        res_1,
        res_2,
        atom_1,
        atom_2,
        type_1,
        type_2,
        distance,
        angle,
        pdb,
        first_atom_df,
        second_atom_df,
        third_atom_df,
        fourth_atom_df,
        pdb_name,
    ):
        """
        Holds data for H-bond interaction.
        Used to store all the data about interactions.

        Args:
            res_1 (str): residue 1 in the interaction
            res_2 (str): residue 2 in the interaction
            atom_1 (str): atom 1 in the interaction
            atom_2 (str): atom 2 in the interaction
            type_1 (str): residue type 1 in the interaction
            type_2 (str): residue type 2 in the interaction
            distance (float): distance between atoms in interaction
            angle (float): dihedral angle between two residues
            pdb (pd.DataFrame): interaction PDB
            first_atom_df (pd.DataFrame): PDB of the first atom in the interaction
            second_atom_df (pd.DataFrame): PDB of the second atom in the interaction
            third_atom_df (pd.DataFrame): PDB of the third atom connected to the first atom
            fourth_atom_df (pd.DataFrame): PDB of the fourth atom connected to the second atom
            pdb_name (str): name of PDB interaction comes from

        """
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type_1 = type_1
        self.type_2 = type_2
        self.distance = distance
        self.angle = angle
        self.pdb = pdb
        self.first_atom_df = first_atom_df
        self.second_atom_df = second_atom_df
        self.third_atom_df = third_atom_df
        self.fourth_atom_df = fourth_atom_df
        self.pdb_name = pdb_name


class HBondInteractionFactory:
    """
    Intermediate class to assist in building complete HBondInteraction data.

    Args:
        res_1 (str): residue 1 ID
        res_2 (str): residue 2 ID
        atom_1 (str): atom 1 ID
        atom_2 (str): atom 2 ID
        distance (float): distance between interacting atoms (in angstroms)
        residue_pair (str): pair comparing the types as returned by DSSR
        quality (str): quality of h-bond

    """

    def __init__(self, res_1, res_2, atom_1, atom_2, distance, residue_pair, quality):
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.distance = distance
        self.residue_pair = residue_pair
        self.quality = quality


class Motif:
    """
    Class to hold motif data. This data is final and should not be changed once built.
    """

    def __init__(
        self,
        motif_name: str,
        motif_type: str,
        pdb: str,
        size: str,
        sequence: str = None,
        res_list: List[str] = None,
        strands: Any = None,
        motif_pdb: pd.DataFrame = None,
    ) -> None:
        """
        Initialize a Motif object

        Args:
            motif_name (str): Name of the motif
            motif_type (str): Motif type
            pdb (str): PDB where motif is found
            size (str): Size of motif; reflects the structure and means different things depending on the type of motif
            sequence (str): Sequence in motif
            res_list (list): List of residues in the motif
            strands (list): List of strands in motif
            motif_pdb (pd.DataFrame): PDB data of the motif

        Returns:
            None
        """
        self.motif_name = motif_name
        self.motif_type = motif_type
        self.pdb = pdb
        self.size = size
        self.sequence = sequence
        self.res_list = res_list if res_list is not None else []
        self.strands = strands if strands is not None else []
        self.motif_pdb = motif_pdb if motif_pdb is not None else pd.DataFrame()


class DSSRRes:
    """
    Class that takes DSSR residue notation.
    Stores and dissects information from DSSR residue notation.
    """

    def __init__(self, s: str) -> None:
        """
        Initialize a DSSRRes object.

        Args:
            s (str): Given residue (something like "C.G1515")

        Returns:
            None

        """
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        i_num = 0
        for i, c in enumerate(spl[1]):
            if c.isdigit():
                cur_num = spl[1][i:]
                cur_num = extract_longest_numeric_sequence(cur_num)
                i_num = i
                break
        self.num = None
        try:
            if cur_num is not None:
                self.num = int(cur_num)
        except ValueError:
            pass
        self.chain_id = spl[0]
        self.res_id = spl[1][0:i_num]


class Residue:
    """
    Class to hold data on individual residues, used for building strands and sequences in find_strands

    Args:
        chain_id (str): chain ID
        res_id (str): residue ID
        ins_code (str): ID code, sometimes used instead of residue ID, often is None
        mol_name (str): molecule name
        pdb (pd.DataFrame): DataFrame to hold the actual contents of the residue obtained from the .cif file
    """

    def __init__(self, chain_id, res_id, ins_code, mol_name, pdb):
        self.chain_id = chain_id
        self.res_id = res_id
        self.ins_code = ins_code
        self.mol_name = mol_name
        self.pdb = pdb


def extract_longest_numeric_sequence(input_string: str) -> str:
    """
    Extracts the longest numeric sequence from a given string.

    Args:
        input_string (str): The string to extract the numeric sequence from.

    Returns:
        longest_sequence (str): The longest numeric sequence found in the input string.

    """
    longest_sequence = ""
    current_sequence = ""
    for c in input_string:
        if c.isdigit() or (
            c == "-" and (not current_sequence or current_sequence[0] == "-")
        ):
            current_sequence += c
            if len(current_sequence) >= len(longest_sequence):
                longest_sequence = current_sequence
        else:
            current_sequence = ""
    return longest_sequence
