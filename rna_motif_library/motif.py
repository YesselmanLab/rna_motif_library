import json
import os
from typing import List, Any, Tuple, Union
import pandas as pd
import numpy as np

from pydssr.dssr import DSSROutput

from rna_motif_library.classes import (
    SingleMotifInteraction,
    PotentialTertiaryContact,
    Motif,
    extract_longest_numeric_sequence,
    PandasMmcifOverride,
    Residue,
    HBondInteraction,
    X3DNAResidue,
)
from rna_motif_library.dssr_hbonds import (
    dataframe_to_cif,
    assign_res_type,
    merge_hbond_interaction_data,
    assemble_interaction_data,
    build_complete_hbond_interaction,
    save_interactions_to_disk,
)
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.snap import parse_snap_output
from rna_motif_library.interactions import get_interactions
from rna_motif_library.logger import get_logger

log = get_logger("motif")


# TODO check other types of DSSR classes like kissing loops
def process_motif_interaction_out_data(count: int, pdb_path: str) -> List[Motif]:
    """Process motifs and interactions from a PDB file"""
    name = os.path.basename(pdb_path)[:-4]
    json_path = os.path.join(LIB_PATH, "data/dssr_output", f"{name}.json")
    d_out = DSSROutput(json_path=json_path)
    motifs = d_out.get_motifs()
    hbonds = d_out.get_hbonds()
    get_interactions(name, hbonds)
    exit()

    processor = MotifProcessor(count, pdb_path)
    return processor.process()


class MotifProcessor:
    """Class for processing motifs and interactions from PDB files"""

    def __init__(self, count: int, pdb_path: str):
        """
        Initialize the MotifProcessor

        Args:
            count (int): # of PDBs processed (loaded from outside)
            pdb_path (str): path to the source PDB
        """
        self.count = count
        self.pdb_path = pdb_path
        self.name = os.path.basename(pdb_path)[:-4]
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
        log.debug(f"{self.count}, {self.pdb_path}, {self.name}")

        # Get the master PDB data
        self.pdb_model_df = self._get_pdb_model_df()
        json_path = os.path.join(LIB_PATH, "data/dssr_output", f"{self.name}.json")

        # Get motifs, interactions, etc from DSSR
        motifs, hbonds = self._get_data_from_dssr(json_path)
        motif_out_path = os.path.join(LIB_PATH, "data/motifs")
        os.makedirs(motif_out_path, exist_ok=True)

        # Get RNP interactions from SNAP and merge with DSSR data
        rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{self.name}.out")
        unique_interaction_data = merge_hbond_interaction_data(
            parse_snap_output(out_file=rnp_out_path), hbonds
        )

        # This is the final interaction data in the temp class to assemble into the big H-Bond class
        pre_assembled_interaction_data = assemble_interaction_data(
            unique_interaction_data
        )

        # Assembly into big HBondInteraction class; this returns a big list of them
        self.assembled_interaction_data = build_complete_hbond_interaction(
            pre_assembled_interaction_data, self.pdb_model_df, self.name
        )

        # Process each motif
        for m in motifs:
            self._process_single_motif(m)

        return self.motif_list

    def _process_single_motif(self, m):
        """Process a single motif and its interactions"""
        built_motif = self._find_and_build_motif(m)
        if built_motif == "UNKNOWN":
            print("UNKNOWN")
            return

        self.motif_list.append(built_motif)
        print(built_motif.motif_name)

        # Process interactions for this motif
        residues_in_motif = built_motif.res_list
        interactions_in_motif = []
        potential_tert_contact_motif_1 = []
        potential_tert_contact_motif_2 = []

        self._categorize_interactions(
            residues_in_motif,
            interactions_in_motif,
            potential_tert_contact_motif_1,
            potential_tert_contact_motif_2,
        )

        self._process_motif_interactions(built_motif, interactions_in_motif)
        self._process_tertiary_contacts(
            built_motif, potential_tert_contact_motif_1, potential_tert_contact_motif_2
        )

    def _categorize_interactions(
        self,
        residues_in_motif,
        interactions_in_motif,
        potential_tert_contact_motif_1,
        potential_tert_contact_motif_2,
    ):
        """Categorize interactions based on their relationship to the motif"""
        for interaction in self.assembled_interaction_data:
            if (
                (
                    interaction.res_1 in residues_in_motif
                    and interaction.res_2 in residues_in_motif
                )
                or (
                    interaction.type_1 == "aa"
                    and interaction.res_2 in residues_in_motif
                )
                or (
                    interaction.type_2 == "aa"
                    and interaction.res_1 in residues_in_motif
                )
            ):
                interactions_in_motif.append(interaction)
            elif interaction.res_1 in residues_in_motif:
                potential_tert_contact_motif_1.append(interaction)
            elif interaction.res_2 in residues_in_motif:
                potential_tert_contact_motif_2.append(interaction)

    def _process_motif_interactions(self, built_motif, interactions_in_motif):
        """Process interactions within a motif"""
        for interaction in interactions_in_motif:
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            if type_1 == "nt":
                type_1 = assign_res_type(interaction.atom_1, type_1)
            if type_2 == "nt":
                type_2 = assign_res_type(interaction.atom_2, type_2)

            single_motif_interaction = SingleMotifInteraction(
                built_motif.motif_name,
                interaction.res_1,
                interaction.res_2,
                interaction.atom_1,
                interaction.atom_2,
                type_1,
                type_2,
                float(interaction.distance),
                float(interaction.angle),
            )
            self.single_motif_interactions.append(single_motif_interaction)

    def _process_tertiary_contacts(
        self,
        built_motif,
        potential_tert_contact_motif_1,
        potential_tert_contact_motif_2,
    ):
        """Process potential tertiary contacts"""
        for interaction in potential_tert_contact_motif_1:
            contact = PotentialTertiaryContact(
                built_motif.motif_name,
                "unknown",
                interaction.res_1,
                interaction.res_2,
                interaction.atom_1,
                interaction.atom_2,
                interaction.type_1,
                interaction.type_2,
                float(interaction.distance),
                float(interaction.angle),
            )
            self.potential_tert_contacts.append(contact)

        for interaction in potential_tert_contact_motif_2:
            contact = PotentialTertiaryContact(
                "unknown",
                built_motif.motif_name,
                interaction.res_1,
                interaction.res_2,
                interaction.atom_1,
                interaction.atom_2,
                interaction.type_1,
                interaction.type_2,
                float(interaction.distance),
                float(interaction.angle),
            )
            self.potential_tert_contacts.append(contact)

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
        list_of_strands, sequence = self._find_strands(motif_pdb)
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

    def _extract_motif_from_pdb(self, nts):
        """Extract motif data from PDB"""
        nt_list = []
        res = []

        # Extract identification data from nucleotide list
        for nt in nts:
            nt_spl = nt.split(".")
            chain_id = nt_spl[0]
            if "--" in nt_spl[1] and len(nt_spl) > 2:
                residue_id = extract_longest_numeric_sequence(nt_spl[2])
            else:
                residue_id = extract_longest_numeric_sequence(nt_spl[1])
            if "/" in nt_spl[1]:
                residue_id = nt_spl[1].split("/")[1]
            nt_list.append(chain_id + "." + residue_id)

        nucleotide_list_sorted, chain_list_sorted = self._group_residues_by_chain(
            nt_list
        )
        list_of_chains = []

        for chain_number, residue_list in zip(
            chain_list_sorted, nucleotide_list_sorted
        ):
            for residue in residue_list:
                chain_res = self.pdb_model_df[
                    self.pdb_model_df["auth_asym_id"].astype(str) == str(chain_number)
                ]
                res_subset = chain_res[
                    chain_res["auth_seq_id"].astype(str) == str(residue)
                ]
                res.append(res_subset)
            list_of_chains.append(res)

        df_list = [
            pd.DataFrame([line.split()], columns=self.pdb_model_df.columns)
            for r in self._remove_empty_dataframes(res)
            for line in r.to_string(index=False, header=False).split("\n")
        ]

        result_df = pd.concat(df_list, axis=0, ignore_index=True)

        return result_df

    def _get_pdb_model_df(self) -> pd.DataFrame:
        """
        Loads PDB model into a dataframe

        Returns:
            model_df (pd.DataFrame): PDB file as DataFrame
        """
        pdb_model = PandasMmcifOverride().read_mmcif(path=self.pdb_path)
        model_df = pdb_model.df[
            [
                "group_PDB",
                "id",
                "type_symbol",
                "label_atom_id",
                "label_alt_id",
                "label_comp_id",
                "label_asym_id",
                "label_entity_id",
                "label_seq_id",
                "pdbx_PDB_ins_code",
                "Cartn_x",
                "Cartn_y",
                "Cartn_z",
                "occupancy",
                "B_iso_or_equiv",
                "pdbx_formal_charge",
                "auth_seq_id",
                "auth_comp_id",
                "auth_asym_id",
                "auth_atom_id",
                "pdbx_PDB_model_num",
            ]
        ]

        return model_df

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
            return "UNKNOWN"

    def _find_strands(self, master_res_df: pd.DataFrame) -> Tuple[List[Any], str]:
        """
        Counts the number of strands in a motif and updates its name accordingly to better reflect structure.

        Args:
            master_res_df (pd.DataFrame): DataFrame containing motif data from PDB.

        Returns:
            strands_of_rna (list): list of strands of residues
            sequence (str): string containing sequence of the motif (AUCG-AUCG-AUCG)
        """
        # step 1: make a list of all known residues
        list_of_residues = self._extract_residue_list(master_res_df)

        # step 2: find the roots of the residues
        residue_roots, res_list_modified = self._find_residue_roots(list_of_residues)

        if residue_roots == "UNKNOWN":
            return "UNKNOWN", "UNKNOWN"

        # step 3: given the residue roots, build strands of RNA
        strands_of_rna = self._build_strands_5to3(residue_roots, res_list_modified)

        # step 4: find the sequence of the strands
        sequence = self._find_sequence(strands_of_rna)

        return strands_of_rna, sequence

    def _find_sequence(self, strands_of_rna: List[List[Residue]]) -> str:
        """Find sequences from found strands of RNA"""
        res_strands = []
        for strand in strands_of_rna:
            res_strand = []
            for residue in strand:
                mol_name = residue.mol_name
                res_strand.append(mol_name)
            strand_sequence = "".join(res_strand)
            res_strands.append(strand_sequence)
        sequence = "-".join(res_strands)
        return sequence

    def _extract_residue_list(self, master_res_df: pd.DataFrame) -> List[Residue]:
        """Extract PDB data per residue and put it in a list of Residue objects"""
        unique_ins_code_values = master_res_df["pdbx_PDB_ins_code"]
        unique_model_num_values = master_res_df["pdbx_PDB_model_num"]

        unique_ins_code_values_list = unique_ins_code_values.astype(str).tolist()
        unique_model_num_values_list = unique_model_num_values.astype(str).tolist()

        ins_code_set_list = sorted(set(unique_ins_code_values_list))
        model_num_set_list = sorted(set(unique_model_num_values_list))

        if len(ins_code_set_list) > 1:
            grouped_res_dfs = master_res_df.groupby(
                ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "auth_comp_id"]
            )
        elif len(model_num_set_list) > 1:
            filtered_master_df = master_res_df[
                master_res_df["pdbx_PDB_model_num"] == "1"
            ]
            grouped_res_dfs = filtered_master_df.groupby(
                ["auth_asym_id", "auth_seq_id", "pdbx_PDB_model_num", "auth_comp_id"]
            )
        else:
            grouped_res_dfs = master_res_df.groupby(
                ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "auth_comp_id"]
            )

        res_list = []
        for group in grouped_res_dfs:
            key = group[0]
            pdb = group[1]
            chain_id = key[0]
            res_id = key[1]
            ins_code = key[2]
            mol_name = key[3]
            residue = Residue(chain_id, res_id, ins_code, mol_name, pdb)
            res_list.append(residue)

        return res_list

    def _find_residue_roots(
        self, res_list: List[Residue]
    ) -> Tuple[List[Residue], List[Residue]]:
        """Find the roots of chains of RNA"""
        roots = []

        for source_res in res_list:
            has_5to3_connection = False
            has_3to5_connection = False

            for res_in_question in res_list:
                if source_res != res_in_question:
                    is_connected = self._connected_to(source_res, res_in_question)
                    if is_connected == 1:
                        has_5to3_connection = True
                    elif is_connected == -1:
                        has_3to5_connection = True
                    elif is_connected == 2:
                        return "UNKNOWN", "UNKNOWN"

                if has_3to5_connection:
                    break

            if has_5to3_connection and not has_3to5_connection:
                roots.append(source_res)

        res_list_modified = [res for res in res_list if res not in roots]

        return roots, res_list_modified

    def _connected_to(
        self,
        source_residue: Residue,
        residue_in_question: Residue,
        cutoff: float = 2.75,
    ) -> int:
        """Determine if another residue is connected to this residue"""
        residue_1 = source_residue.pdb
        residue_2 = residue_in_question.pdb

        residue_1[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_1[
            ["Cartn_x", "Cartn_y", "Cartn_z"]
        ].apply(pd.to_numeric)
        residue_2[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_2[
            ["Cartn_x", "Cartn_y", "Cartn_z"]
        ].apply(pd.to_numeric)

        o3_atom_1 = residue_1[residue_1["auth_atom_id"].str.contains("O3'", regex=True)]
        o3_atom_1 = o3_atom_1[~o3_atom_1["auth_atom_id"].str.contains("H", regex=False)]

        p_atom_2 = residue_2[residue_2["auth_atom_id"].isin(["P"])]

        if not o3_atom_1.empty and not p_atom_2.empty:
            try:
                distance = np.linalg.norm(
                    p_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values
                    - o3_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
                )
            except ValueError:
                return 2
            if distance < cutoff:
                return 1

        o3_atom_2 = residue_2[residue_2["auth_atom_id"].str.contains("O3'", regex=True)]
        o3_atom_2 = o3_atom_2[~o3_atom_2["auth_atom_id"].str.contains("H", regex=False)]

        p_atom_1 = residue_1[residue_1["auth_atom_id"].isin(["P"])]

        if not o3_atom_2.empty and not p_atom_1.empty:
            distance = np.linalg.norm(
                o3_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values
                - p_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            )
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
                    if self._connected_to(current_residue, res) == 1:
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

    def _remove_empty_dataframes(
        self, dataframes_list: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Remove empty DataFrames from a list"""
        return [df for df in dataframes_list if not df.empty]

    def _group_residues_by_chain(
        self, input_list: List[str]
    ) -> Tuple[List[List[int]], List[str]]:
        """Group residues into their own chains for counting"""
        chain_residues = {}
        chain_ids_for_residues = {}

        for item in input_list:
            chain_id, residue_id = item.split(".")
            if residue_id != "None":
                residue_id = int(residue_id)

            if chain_id not in chain_residues:
                chain_residues[chain_id] = []

            chain_residues[chain_id].append(residue_id)

            if residue_id not in chain_ids_for_residues:
                chain_ids_for_residues[residue_id] = []
            chain_ids_for_residues[residue_id].append(chain_id)

        sorted_chain_residues = []
        sorted_chain_ids = []

        unique_chain_ids = list(chain_residues.keys())
        sorted_unique_chain_ids = unique_chain_ids

        for chain_id in sorted_unique_chain_ids:
            sorted_residues = sorted(set(chain_residues[chain_id]))
            sorted_chain_residues.append(sorted_residues)
            sorted_chain_ids.append(chain_id)

        return sorted_chain_residues, sorted_chain_ids

    def _get_data_from_dssr(self, json_path: str) -> Tuple[List, List]:
        """Get motifs and hbonds from DSSR"""
        d_out = DSSROutput(json_path=json_path)
        motifs = d_out.get_motifs()
        hbonds = d_out.get_hbonds()
        motifs = self._merge_singlet_separated(motifs)
        motifs = self._remove_duplicate_motifs(motifs)
        motifs = self._remove_large_motifs(motifs)

        return motifs, hbonds

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
