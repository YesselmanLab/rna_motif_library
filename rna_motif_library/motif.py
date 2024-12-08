import json
import os
from typing import List, Any, Tuple, Union, Dict
import pandas as pd
import numpy as np

from pydssr.dssr import DSSROutput
from pydssr.dssr_classes import DSSR_PAIR

from rna_motif_library.classes import (
    SingleMotifInteraction,
    PotentialTertiaryContact,
    Motif,
    extract_longest_numeric_sequence,
    sanitize_x3dna_atom_name,
    PandasMmcifOverride,
    X3DNAResidue,
    X3DNAResidueFactory,
    Residue,
    ResidueNew,  # TODO rename back to Residue when done
    Hbond,
    Basepair,
)
from rna_motif_library.dssr_hbonds import (
    dataframe_to_cif,
    assign_res_type,
    merge_hbond_interaction_data,
    assemble_interaction_data,
    build_complete_hbond_interaction,
    save_interactions_to_disk,
)
from rna_motif_library.settings import LIB_PATH, DATA_PATH
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.interactions import get_hbonds_and_basepairs
from rna_motif_library.logger import get_logger

log = get_logger("motif")


# TODO check other types of DSSR classes like kissing loops
def process_motif_interaction_out_data(pdb_name: str) -> List[Motif]:
    """Process motifs and interactions from a PDB file"""
    hbonds, basepairs = get_hbonds_and_basepairs(pdb_name)

    mp = MotifProcessor(pdb_name, hbonds, basepairs)
    mp.process()


class MotifProcessor:
    """Class for processing motifs and interactions from PDB files"""

    def __init__(self, pdb_name: str, hbonds: List[Hbond], basepairs: List[Basepair]):
        """
        Initialize the MotifProcessor

        Args:
            count (int): # of PDBs processed (loaded from outside)
            pdb_path (str): path to the source PDB
        """
        self.pdb_name = pdb_name
        self.hbonds = hbonds
        self.basepairs = basepairs

        self.pdb_model_df = None
        self.assembled_interaction_data = None
        self.discovered = []
        self.motif_count = 0
        self.motif_list = []
        self.single_motif_interactions = []
        self.potential_tert_contacts = []

    def process(self) -> List[Motif]:
        """
        Process the PDB file and extract motifs and interactions

        Returns:
            motif_list (list): list of motif names
        """
        log.debug(f"{self.pdb_name}")

        # Get the master PDB data
        df_atoms = pd.read_parquet(
            os.path.join(DATA_PATH, "pdbs_dfs", f"{self.pdb_name}.parquet")
        )
        json_path = os.path.join(DATA_PATH, "dssr_output", f"{self.pdb_name}.json")
        dssr_output = DSSROutput(json_path=json_path)
        dssr_motifs = dssr_output.get_motifs()
        dssr_tertiary_contacts = dssr_output.get_tertiary_contacts()
        # Process each motif
        for m in dssr_motifs:
            mtype = self._determine_motif_type(m)
            residues = self._generate_residues_for_motif(m, df_atoms)
            # strands = self._generate_strands(residues)
            # sequence = self._find_sequence(strands)
            # print(sequence)

        return []

    def _generate_residues_for_motif(
        self, m: Any, df_atoms: pd.DataFrame
    ) -> List[ResidueNew]:
        residues = []
        for nt in m.nts_long:
            x3dna_res = X3DNAResidueFactory.create_from_string(nt)
            df_res = df_atoms[
                (df_atoms["auth_comp_id"] == x3dna_res.res_id)
                & (df_atoms["auth_asym_id"] == x3dna_res.chain_id)
                & (df_atoms["auth_seq_id"] == x3dna_res.num)
            ]
            coords = df_res[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            atom_names = df_res["auth_atom_id"].tolist()
            atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
            residues.append(
                ResidueNew.from_x3dna_residue(x3dna_res, atom_names, coords)
            )
        return residues

        # return self.motif_list

    def _find_and_build_motif(self, m: Any) -> Union[Motif, str]:
        """
        Identifies motif in source PDB by ID and extracts to disk, setting its name and other data.

        Args:
            m (Any): DSSR_Motif object, contains motif data returned directly from DSSR

        Returns:
            our_motif (Motif): Motif object with all associated data inside
            or "UNKNOWN" if motif cannot be built
        """
        # We need to determine the data for the motif and build a class
        # First get the type
        motif_type = self._determine_motif_type(m)
        if motif_type == "UNKNOWN":
            log.debug(f"Unknown motif type for {self.name}")
            print(f"Unknown motif type for {self.name}")
            print(m)
            return "UNKNOWN"

        # Extract motif from source PDB
        motif_pdb = self._extract_motif_from_pdb(m.nts_long)

        # Now find the list of strands and sequence
        list_of_strands, sequence = self._generate_strands(motif_pdb)
        if list_of_strands == "UNKNOWN" or sequence == "UNKNOWN":
            return "UNKNOWN"

        # Get the size of the motif (as string)
        size = self._set_motif_size(list_of_strands, motif_type)
        if size == "UNKNOWN":
            return "UNKNOWN"

        # Real quick, set NWAY/TWOWAY junction based on return of size
        spl = size.split("-")
        if motif_type == "JCT":
            if len(spl) == 2:
                motif_type = "TWOWAY"
            else:
                motif_type = "NWAY"

        # fix that weird classification issue
        if motif_type == "NWAY" and len(size.split("-")) < 2:
            motif_type = "SSTRAND"

        # Pre-set motif name
        pre_motif_name = motif_type + "." + self.name + "." + str(size) + "." + sequence
        # Check if discovered; if so, then increment count
        if pre_motif_name in self.discovered:
            self.motif_count += 1
        else:
            self.discovered.append(pre_motif_name)
        # Set motif name
        motif_name = pre_motif_name + "." + str(self.motif_count)
        # Finally, set our motif
        our_motif = Motif(
            motif_name,
            motif_type,
            self.name,
            size,
            sequence,
            m.nts_long,
            list_of_strands,
            motif_pdb,
        )
        # And print the motif to the system
        if motif_type == "NWAY":
            spl = size.split("-")
            size_num = len(spl)
            size_str = str(size_num)
            size_str_name = size_str + "ways"
            motif_dir_path = os.path.join(
                LIB_PATH, "data/motifs", motif_type, size_str_name, size, sequence
            )
        else:
            motif_dir_path = os.path.join(
                LIB_PATH, "data/motifs", motif_type, size, sequence
            )
        os.makedirs(motif_dir_path, exist_ok=True)
        motif_cif_path = os.path.join(motif_dir_path, f"{motif_name}.cif")
        dataframe_to_cif(motif_pdb, motif_cif_path, motif_name)
        # Clear the PDB so it doesn't hog extra RAM
        our_motif.motif_pdb = None
        return our_motif

    def _set_motif_size(self, strands: List, motif_type: str) -> str:
        """
        Sets the size of the motif according to motif type and strands.

        Args:
            strands (list): List of all strands in motif (list of lists).
            motif_type (str): String describing the motif type.

        Returns:
            size (str): String specifying size.
        """
        if motif_type == "JCT":
            lens = []
            for strand in strands:
                len_strand = str(len(strand))
                lens.append(len_strand)
            size = "-".join(lens)
            return size
        elif motif_type == "HELIX":
            if len(strands) != 2:
                return "UNKNOWN"
            if len(strands[0]) != len(strands[1]):
                return "UNKNOWN"
            size = str(len(strands[0]))
            return size
        elif motif_type == "HAIRPIN":
            if len(strands) != 1:
                return "UNKNOWN"
            size = len(strands[0]) - 2
            if size < 3:
                return "UNKNOWN"
            else:
                return str(size)
        elif motif_type == "SSTRAND":
            if len(strands) != 1:
                return "UNKNOWN"
            size = len(strands[0])
            return str(size)

    def _determine_motif_type(self, motif):
        """Determine the type of motif"""
        motif_type_beta = motif.mtype
        if motif_type_beta in ["JUNCTION", "BULGE", "ILOOP"]:
            return "JCT"
        elif motif_type_beta in ["STEM", "HEXIX"]:
            return "HELIX"
        elif motif_type_beta in ["SINGLE_STRAND"]:
            return "SSTRAND"
        elif motif_type_beta in ["HAIRPIN"]:
            return "HAIRPIN"
        else:
            log.info(f"Unknown motif type: {motif_type_beta} for {self.pdb_name}")
            return "UNKNOWN"

    def _generate_strands(self, residues: List[ResidueNew]) -> List[List[ResidueNew]]:
        """
        Generates ordered strands of RNA residues by finding root residues and building 5' to 3'.

        Args:
            residues (List[ResidueNew]): List of RNA residues to analyze

        Returns:
            List[List[ResidueNew]]: List of RNA strands, where each strand is a list of residues ordered 5' to 3'
        """
        residue_roots, res_list_modified = self._find_residue_roots(residues)
        strands_of_rna = self._build_strands_5to3(residue_roots, res_list_modified)

        return strands_of_rna

    def _find_sequence(self, strands_of_rna: List[List[Residue]]) -> str:
        """Find sequences from found strands of RNA"""
        res_strands = []
        for strand in strands_of_rna:
            res_strand = []
            for residue in strand:
                mol_name = residue.res_id
                res_strand.append(mol_name)
            strand_sequence = "".join(res_strand)
            res_strands.append(strand_sequence)
        sequence = "-".join(res_strands)
        return sequence

    def _find_residue_roots(
        self, res_list: List[ResidueNew]
    ) -> Tuple[List[ResidueNew], List[ResidueNew]]:
        """
        Find the root residues that start each RNA chain.

        A root residue is one that has a 5' to 3' connection to another residue
        but no 3' to 5' connection from another residue (i.e. it's at the 5' end).

        Args:
            res_list: List of ResidueNew objects to analyze

        Returns:
            Tuple containing:
            - List of root residues found
            - Modified list with roots removed
        """
        roots = []

        # Check each residue to see if it's a root
        for source_res in res_list:
            has_outgoing = False  # 5' to 3' connection
            has_incoming = False  # 3' to 5' connection

            # Compare against all other residues
            for target_res in res_list:
                if source_res == target_res:
                    continue

                connection = self._are_residues_connected(source_res, target_res)

                if connection == 1:  # 5' to 3'
                    has_outgoing = True
                elif connection == -1:  # 3' to 5'
                    has_incoming = True
                    break  # Can stop checking once we find an incoming connection
            # Root residues have outgoing but no incoming connections
            if has_outgoing and not has_incoming:
                roots.append(source_res)

        # Return roots and remaining residues
        remaining = [res for res in res_list if res not in roots]
        return roots, remaining

    def _are_residues_connected(
        self,
        source_residue: ResidueNew,
        residue_in_question: ResidueNew,
        cutoff: float = 2.75,
    ) -> int:
        """Determine if another residue is connected to this residue"""
        # Get O3' coordinates from source residue
        o3_coords_1 = source_residue.get_atom_coords("O3'")
        p_coords_2 = residue_in_question.get_atom_coords("P")

        # Check 5' to 3' connection
        if o3_coords_1 is not None and p_coords_2 is not None:
            distance = np.linalg.norm(np.array(p_coords_2) - np.array(o3_coords_1))
            if distance < cutoff:
                return 1

        # Check 3' to 5' connection
        o3_coords_2 = residue_in_question.get_atom_coords("O3'")
        p_coords_1 = source_residue.get_atom_coords("P")

        if o3_coords_2 is not None and p_coords_1 is not None:
            distance = np.linalg.norm(np.array(o3_coords_2) - np.array(p_coords_1))
            if distance < cutoff:
                return -1

        return 0

    def _build_strands_5to3(
        self, residue_roots: List[Residue], res_list: List[Residue]
    ):
        """Build strands of RNA from the list of given residues"""
        built_strands = []

        for root in residue_roots:
            current_residue = root
            chain = [current_residue]

            while True:
                next_residue = None
                for res in res_list:
                    if self._are_residues_connected(current_residue, res) == 1:
                        next_residue = res
                        break

                if next_residue:
                    chain.append(next_residue)
                    current_residue = next_residue
                    res_list.remove(next_residue)
                else:
                    break

            built_strands.append(chain)

        return built_strands

    # TODO come back to this after other processing
    def _remove_duplicate_motifs(self, motifs: List[Any]) -> List[Any]:
        """Remove duplicate motifs from a list of motifs"""
        duplicates = []
        for m1 in motifs:
            if m1 in duplicates:
                continue

            m1_nts = [nt.split(".")[1] for nt in m1.nts_long]

            for m2 in motifs:
                if m1 == m2:
                    continue

                m2_nts = [nt.split(".")[1] for nt in m2.nts_long]

                if m1_nts == m2_nts:
                    duplicates.append(m2)

        unique_motifs = [m for m in motifs if m not in duplicates]
        return unique_motifs

    def _remove_large_motifs(self, motifs: List[Any]) -> List[Any]:
        """Remove motifs larger than 35 nucleotides"""
        return [m for m in motifs if len(m.nts_long) <= 35]

    def _merge_singlet_separated(self, motifs: List[Any]) -> List[Any]:
        """Merge singlet separated motifs into a unified list"""
        junctions = []
        others = []

        for m in motifs:
            if m.mtype in ["STEM", "HAIRPIN", "SINGLE_STRAND"]:
                others.append(m)
            else:
                junctions.append(m)

        merged = []
        used = []

        for m1 in junctions:
            m1_nts = m1.nts_long
            if m1 in used:
                continue

            for m2 in junctions:
                if m1 == m2:
                    continue

                included = sum(1 for r in m2.nts_long if r in m1_nts)

                if included < 2:
                    continue

                for nt in m2.nts_long:
                    if nt not in m1.nts_long:
                        m1.nts_long.append(nt)

                used.extend([m1, m2])
                merged.append(m2)

        new_motifs = others + [m for m in junctions if m not in merged]

        return new_motifs
