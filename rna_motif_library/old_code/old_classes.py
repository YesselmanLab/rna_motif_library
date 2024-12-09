import pandas as pd
import json
from biopandas.mmcif.pandas_mmcif import PandasMmcif

from biopandas.mmcif.mmcif_parser import load_cif_data
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.engines import ANISOU_DF_COLUMNS


# TODO do I need this?
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


class HBondInteraction:
    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
        pdb: pd.DataFrame,
        first_atom_df: pd.DataFrame,
        second_atom_df: pd.DataFrame,
        third_atom_df: pd.DataFrame,
        fourth_atom_df: pd.DataFrame,
        pdb_name: str,
    ) -> None:
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

    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        distance: float,
        residue_pair: str,
        quality: str,
    ) -> None:
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

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a Motif object from a dictionary.

        Args:
            data (dict): Dictionary containing motif data

        Returns:
            Motif: A new Motif object with the data from the dictionary
        """

        # Convert motif_pdb back to DataFrame if it exists
        if data["motif_pdb"] is not None:
            data["motif_pdb"] = pd.DataFrame.from_records(data["motif_pdb"])

        # Convert strands back to Residue objects if they exist
        if data["strands"]:
            data["strands"] = [
                [Residue.from_dict(res) for res in strand] for strand in data["strands"]
            ]

        return cls(
            motif_name=data["motif_name"],
            motif_type=data["motif_type"],
            pdb=data["pdb"],
            size=data["size"],
            sequence=data["sequence"],
            res_list=data["res_list"],
            strands=data["strands"],
            motif_pdb=data["motif_pdb"],
        )

    def __eq__(self, other):
        """
        Check if two Motif objects are equal by comparing their dictionary representations.

        Args:
            other: Another Motif object to compare with

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if not isinstance(other, type(self)):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """
        Returns the object as a dictionary.

        Returns:
            The entire instance of a Motif class inside a dictionary.
            Intended for writing the motif to JSON.
        """
        return {
            "motif_name": self.motif_name,
            "motif_type": self.motif_type,
            "pdb": self.pdb,
            "size": self.size,
            "sequence": self.sequence,
            "res_list": self.res_list,
            "strands": [[res.to_dict() for res in strand] for strand in self.strands],
            "motif_pdb": (
                self.motif_pdb.to_dict(orient="records")
                if isinstance(self.motif_pdb, pd.DataFrame) and not self.motif_pdb.empty
                else None
            ),
        }

    def to_json(self, filepath: str) -> None:
        """
        Saves the motif object as a JSON file.

        Args:
            filepath (str): Path where the JSON file should be saved

        Returns:
            None
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)


class PotentialTertiaryContact:
    def __init__(
        self,
        motif_1: str,
        motif_2: str,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
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


class HBondInteraction:
    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        type_1: str,
        type_2: str,
        distance: float,
        angle: float,
        pdb: pd.DataFrame,
        first_atom_df: pd.DataFrame,
        second_atom_df: pd.DataFrame,
        third_atom_df: pd.DataFrame,
        fourth_atom_df: pd.DataFrame,
        pdb_name: str,
    ) -> None:
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

    def __init__(
        self,
        res_1: str,
        res_2: str,
        atom_1: str,
        atom_2: str,
        distance: float,
        residue_pair: str,
        quality: str,
    ) -> None:
        self.res_1 = res_1
        self.res_2 = res_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.distance = distance
        self.residue_pair = residue_pair
        self.quality = quality
