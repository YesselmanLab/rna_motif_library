import os
import re
from json import JSONDecodeError
from typing import List, Any, Tuple, Dict
import pandas as pd
import numpy as np
from pandas import DataFrame

from pydssr.dssr import DSSROutput
from pydssr.dssr_classes import DSSR_HBOND

from rna_motif_library.update_library import PandasMmcifOverride, get_dssr_files
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.snap import get_rnp_interactions
from rna_motif_library.dssr_hbonds import extract_longest_numeric_sequence, dataframe_to_cif, canon_amino_acid_list, \
    HBondInteraction, HBondInteractionFactory, find_atoms, find_closest_atom, calculate_bond_angle, \
    SingleMotifInteraction, PotentialTertiaryContact


class Motif:
    """
    Class to hold motif data. This data is final and should not be changed once built.
    """

    def __init__(self, motif_name: str, motif_type: str, pdb: str, size: str,
                 sequence: str = None,
                 res_list: List[str] = None, strands: Any = None, motif_pdb: pd.DataFrame = None) -> None:
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


def process_motif_interaction_out_data(
        count: int,
        pdb_path: str):
    """
    Function for extracting motifs from a PDB in the loop

    Args:
        count (int): # of PDBs processed (loaded from outside)
        pdb_path (str): path to the source PDB
        limit (int): # of PDB files to process (loaded from outside)
        pdb_name (str): which specific PDB o process

    Returns:
        None

    """
    name = os.path.basename(pdb_path)[:-4]
    print(f"{count}, {pdb_path}, {name}")

    # Get the master PDB data
    pdb_model_df = get_pdb_model_df(pdb_path)
    json_path = os.path.join(LIB_PATH, "data/dssr_output", f"{name}.json")
    # Get motifs, interactions, etc from DSSR
    motifs, hbonds = get_data_from_dssr(json_path)
    motif_out_path = os.path.join(LIB_PATH, "data/motifs")
    os.makedirs(motif_out_path, exist_ok=True)
    # Get RNP interactions from SNAP and merge with DSSR data
    rnp_out_path = os.path.join(LIB_PATH, "data/snap_output", f"{name}.out")
    unique_interaction_data = merge_hbond_interaction_data(get_rnp_interactions(out_file=rnp_out_path), hbonds)
    # This is the final interaction data in the temp class to assemble into the big H-Bond class
    pre_assembled_interaction_data = assemble_interaction_data(unique_interaction_data)
    # Assembly into big HBondInteraction class; this returns a big list of them
    assembled_interaction_data = build_complete_hbond_interaction(pre_assembled_interaction_data, pdb_model_df, name)
    # Now for every interaction, print to PDB
    save_interactions_to_disk(assembled_interaction_data, name)

    discovered = []
    motif_count = 0
    motif_list = []
    single_motif_interactions = []
    potential_tert_contacts = []
    for m in motifs:
        built_motif = find_and_build_motif(m, name, pdb_model_df, discovered, motif_count)
        if built_motif == "UNKNOWN":
            print("UNKNOWN")
            continue
        else:
            motif_list.append(built_motif)
            print(built_motif.motif_name)
        # Also determine which interactions are involved with which motifs
        residues_in_motif = built_motif.res_list
        interactions_in_motif = []
        potential_tert_contact_motif_1 = []
        potential_tert_contact_motif_2 = []
        for interaction in assembled_interaction_data:
            if interaction.res_1 in residues_in_motif and interaction.res_2 in residues_in_motif:
                # H-bonds fully inside motif
                interactions_in_motif.append(interaction)
            elif interaction.res_1 in residues_in_motif:
                # H-bonds with 1 residue in motif and 1 outside
                potential_tert_contact_motif_1.append(interaction)
            elif interaction.res_2 in residues_in_motif:
                # H-bonds with 1 residue in motif and 1 outside
                potential_tert_contact_motif_2.append(interaction)
            else:
                # H-bonds uninvolved with the motif
                pass

        # For every single motif interaction in this motif, load up the class again and load the source motif into a new interaction class
        for interaction in interactions_in_motif:
            motif_name = built_motif.motif_name
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = float(interaction.distance)
            angle = float(interaction.angle)
            single_motif_interaction = SingleMotifInteraction(motif_name, res_1, res_2, atom_1, atom_2, type_1, type_2,
                                                              distance, angle)
            single_motif_interactions.append(single_motif_interaction)

        # For potential tert contacts where source is motif_1
        for interaction in potential_tert_contact_motif_1:
            motif_1 = built_motif.motif_name
            motif_2 = "unknown"
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = float(interaction.distance)
            angle = float(interaction.angle)
            potential_tert_contact_m1 = PotentialTertiaryContact(motif_1, motif_2, res_1, res_2, atom_1, atom_2, type_1,
                                                                 type_2, distance, angle)
            potential_tert_contacts.append(potential_tert_contact_m1)

        # For potential tert contacts where source is motif_2
        for interaction in potential_tert_contact_motif_2:
            motif_1 = "unknown"
            motif_2 = built_motif.motif_name
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = float(interaction.distance)
            angle = float(interaction.angle)
            potential_tert_contact_m2 = PotentialTertiaryContact(motif_1, motif_2, res_1, res_2, atom_1, atom_2, type_1,
                                                                 type_2, distance, angle)
            potential_tert_contacts.append(potential_tert_contact_m2)

    return motif_list, single_motif_interactions, potential_tert_contacts, assembled_interaction_data


### build functions down here


def save_interactions_to_disk(assembled_interaction_data, pdb):
    for interaction in assembled_interaction_data:
        interaction_name = str(
            pdb) + "." + interaction.res_1 + "." + interaction.atom_1 + "." + interaction.res_2 + "." + interaction.atom_2
        folder_path = os.path.join(LIB_PATH, "data/interactions",
                                   f"{DSSRRes(interaction.res_1).res_id}-{DSSRRes(interaction.res_2).res_id}")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{interaction_name}.cif")
        dataframe_to_cif(interaction.pdb, file_path=file_path, motif_name=interaction_name)


def build_complete_hbond_interaction(pre_assembled_interaction_data, pdb_model_df, pdb_name):
    built_interactions = []
    for interaction in pre_assembled_interaction_data:
        res_1 = DSSRRes(interaction.res_1)
        res_2 = DSSRRes(interaction.res_2)
        atom_1 = interaction.atom_1
        atom_2 = interaction.atom_2
        type_1 = interaction.residue_pair.split(":")[0]
        type_2 = interaction.residue_pair.split(":")[1]
        distance = interaction.distance
        pdb = get_interaction_pdb(res_1, res_2, pdb_model_df)
        first_atom, second_atom = extract_interacting_atoms(interaction, pdb)
        third_atom, fourth_atom = find_closest_atom(first_atom, pdb), find_closest_atom(second_atom, pdb)
        if first_atom.empty or second_atom.empty:
            # print(pdb)
            # print("EMPTY ATOM")
            # TODO come back to this, this is a placeholder; there aren't that many of these relative to the rest of interactions but need to look at
            continue
        # filter out protein-protein interactions
        if type_1 == "aa" and type_2 == "aa":
            continue
        dihedral_angle = calculate_bond_angle(first_atom, second_atom, third_atom, fourth_atom)

        built_interaction = HBondInteraction(interaction.res_1, interaction.res_2, atom_1, atom_2, type_1, type_2,
                                             distance, dihedral_angle, pdb, first_atom, second_atom, third_atom,
                                             fourth_atom, pdb_name)
        built_interactions.append(built_interaction)

    return built_interactions


def extract_interacting_atoms(interaction, pdb):
    atom_1 = interaction.atom_1
    atom_2 = interaction.atom_2

    res_1 = DSSRRes(interaction.res_1).res_id
    res_2 = DSSRRes(interaction.res_2).res_id

    chain_id_1 = DSSRRes(interaction.res_1).chain_id
    chain_id_2 = DSSRRes(interaction.res_2).chain_id

    res_id_1 = DSSRRes(interaction.res_1).num
    res_id_2 = DSSRRes(interaction.res_2).num

    first_atom = pdb[
        (pdb["auth_atom_id"] == atom_1) & (pdb["auth_comp_id"] == res_1) & (pdb["auth_asym_id"] == chain_id_1) & (
                pdb["auth_seq_id"] == res_id_1)]
    second_atom = pdb[
        (pdb["auth_atom_id"] == atom_2) & (pdb["auth_comp_id"] == res_2) & (pdb["auth_asym_id"] == chain_id_2) & (
                pdb["auth_seq_id"] == res_id_2)]

    if first_atom.empty:
        # Check for common prefixes or alternate namings
        prefixes = ["O1P", "O2P", "O3P", "OP1", "OP2", "OP3", "O2"]
        for prefix in prefixes:
            if prefix in atom_1:
                first_atom = pdb[
                    (pdb["auth_atom_id"].str.contains(prefix.replace("P", "")) & (pdb["auth_comp_id"] == res_1) & (
                            pdb["auth_asym_id"] == chain_id_1) & (
                             pdb["auth_seq_id"] == res_id_1))
                ]
                if not first_atom.empty:
                    break

    if second_atom.empty:
        # Check for common prefixes or alternate namings
        prefixes = ["O1P", "O2P", "O3P", "OP1", "OP2", "OP3", "O2"]
        for prefix in prefixes:
            if prefix in atom_2:
                second_atom = pdb[
                    (pdb["auth_atom_id"].str.contains(prefix.replace("P", "")) & (pdb["auth_comp_id"] == res_2) & (
                            pdb["auth_asym_id"] == chain_id_2) & (
                             pdb["auth_seq_id"] == res_id_2))
                ]
                if not first_atom.empty:
                    break

    return first_atom, second_atom


def get_interaction_pdb(res_1, res_2, pdb_model_df):
    res_1_chain_id, res_1_atom_type, res_1_res_id = res_1.chain_id, res_1.res_id, res_1.num
    res_2_chain_id, res_2_atom_type, res_2_res_id = res_2.chain_id, res_2.res_id, res_2.num
    res_1_inter_chain = pdb_model_df[pdb_model_df["auth_asym_id"].astype(str) == str(res_1_chain_id)]
    res_2_inter_chain = pdb_model_df[pdb_model_df["auth_asym_id"].astype(str) == str(res_2_chain_id)]
    res_1_inter_res = res_1_inter_chain[
        res_1_inter_chain["auth_seq_id"].astype(str) == str(res_1_res_id)
        ]
    res_2_inter_res = res_2_inter_chain[
        res_2_inter_chain["auth_seq_id"].astype(str) == str(res_2_res_id)
        ]
    res_1_res_2_result_df = pd.concat(
        [res_1_inter_res, res_2_inter_res], axis=0, ignore_index=True
    )

    return res_1_res_2_result_df


def assemble_interaction_data(unique_interaction_data):
    assembled_data = []
    # Load all data into HBondInteractionFactory class to prepare for processing
    for interaction in unique_interaction_data:
        # Filter out bad H-bonds and aa:aa
        if interaction[6] not in ["questionable", "unknown"] or interaction[5] not in ["aa:aa"]:

            new_interaction_atom_1 = interaction[2]
            new_interaction_atom_2 = interaction[3]

            # Also process the weird ones with . in their name
            if "." in new_interaction_atom_1:
                new_interaction_atom_1 = interaction[2].split(".")[0]
            elif "." in new_interaction_atom_2:
                new_interaction_atom_2 = interaction[3].split(".")[0]
            hbond_interaction_assembly = HBondInteractionFactory(interaction[0], interaction[1], new_interaction_atom_1,
                                                                 new_interaction_atom_2, float(interaction[4]),
                                                                 interaction[5],
                                                                 interaction[6])
            assembled_data.append(hbond_interaction_assembly)
    return assembled_data


def merge_hbond_interaction_data(rnp_interactions, hbonds):
    rnp_data = [
        (
            interaction.nt_atom.split("@")[1],
            interaction.aa_atom.split("@")[1],
            interaction.nt_atom.split("@")[0],
            interaction.aa_atom.split("@")[0],
            str(interaction.dist),
            interaction.type,
            "standard"
        )
        for interaction in rnp_interactions
    ]
    interaction_data = [
        (
            hbond.atom1_id.split("@")[1],
            hbond.atom2_id.split("@")[1],
            hbond.atom1_id.split("@")[0],
            hbond.atom2_id.split("@")[0],
            str(hbond.distance),
            hbond.residue_pair,
            hbond.donAcc_type
        )
        for hbond in hbonds
    ]
    interaction_data.extend(rnp_data)
    unique_interaction_data = list(set(interaction_data))

    return unique_interaction_data


def assign_residue_type(hbond: DSSR_HBOND):
    """
    Assign base, phosphate, sugar, or amino acid in interactions_detailed.csv.
    """
    residue_pair = hbond.residue_pair
    rt_1, rt_2 = residue_pair.split(":")
    atom_1 = str(hbond.atom1_id.split("@")[0])
    atom_2 = str(hbond.atom2_id.split("@")[0])

    # process each residue type to correct DSSR's mistakes
    hbond.residue_pair = determine_interaction_type_from_atoms(rt_1, rt_2, atom_1, atom_2)
    return hbond.residue_pair.split(":")


def determine_interaction_type_from_atoms(rt_1, rt_2, atom_1, atom_2):
    # type 1
    if rt_1 == "aa":
        res_type_1 = "aa"
    else:
        if atom_1 in canon_amino_acid_list:
            res_type_1 = "aa"
        elif "P" in atom_1:
            res_type_1 = "phos"
        elif atom_1.endswith("'"):
            res_type_1 = "sugar"
        else:
            res_type_1 = "base"

    # type 2
    if rt_2 == "aa":
        res_type_2 = "aa"
    else:
        if atom_2 in canon_amino_acid_list:
            res_type_2 = "aa"
        elif "P" in atom_1:
            res_type_2 = "phos"
        elif atom_1.endswith("'"):
            res_type_2 = "sugar"
        else:
            res_type_2 = "base"

    return f"{res_type_1}:{res_type_2}"


def find_and_build_motif(m, pdb_name, pdb_model_df, discovered, motif_count):
    # We need to determine the data for the motif and build a class
    # First get the type
    motif_type = determine_motif_type(m)
    if motif_type == "UNKNOWN":
        return "UNKNOWN"
    # list of long nucleotides (m.nts_long)
    # Extract motif from source PDB
    motif_pdb = extract_motif_from_pdb(m.nts_long, pdb_model_df)
    # Now find the list of strands and sequence
    list_of_strands, sequence = find_strands(motif_pdb)
    # Get the size of the motif (as string)
    size = size_up_motif(list_of_strands, motif_type)
    if size == "UNKNOWN":
        # print only for debugging purposes
        return "UNKNOWN"
    # Real quick, set NWAY/TWOWAY junction based on return of size
    spl = size.split("-")
    if motif_type == "JCT":
        if len(spl) == 2:
            motif_type = "TWOWAY"
        else:
            motif_type = "NWAY"
    # Pre-set motif name
    pre_motif_name = motif_type + "." + pdb_name + "." + str(size) + "." + sequence
    # Check if discovered; if so, then increment count
    if pre_motif_name in discovered:
        motif_count += 1
    else:
        discovered.append(pre_motif_name)
    # Set motif name
    motif_name = pre_motif_name + "." + str(motif_count)
    # Finally, set our motif
    our_motif = Motif(motif_name, motif_type, pdb_name, size, sequence, m.nts_long, list_of_strands, motif_pdb)
    # And print the motif to the system
    motif_dir_path = os.path.join(LIB_PATH, "data/motifs", motif_type, size, sequence)
    os.makedirs(motif_dir_path, exist_ok=True)
    motif_cif_path = os.path.join(motif_dir_path, f"{motif_name}.cif")
    dataframe_to_cif(motif_pdb, motif_cif_path, motif_name)
    return our_motif


def size_up_motif(strands, motif_type):
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


def extract_motif_from_pdb(nts, model_df):
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

    nucleotide_list_sorted, chain_list_sorted = group_residues_by_chain(nt_list)
    list_of_chains = []

    for chain_number, residue_list in zip(chain_list_sorted, nucleotide_list_sorted):
        for residue in residue_list:
            chain_res = model_df[
                model_df["auth_asym_id"].astype(str) == str(chain_number)
                ]
            res_subset = chain_res[chain_res["auth_seq_id"].astype(str) == str(residue)]
            res.append(res_subset)
        list_of_chains.append(res)

    df_list = [
        pd.DataFrame([line.split()], columns=model_df.columns)
        for r in remove_empty_dataframes(res)
        for line in r.to_string(index=False, header=False).split("\n")
    ]

    result_df = pd.concat(df_list, axis=0, ignore_index=True)

    return result_df


def get_pdb_model_df(pdb_path: str) -> pd.DataFrame:
    """
    Loads PDB model into a dataframe

    Args:
        pdb_path (str): path to PDB
    Returns:
        model_df (pd.DataFrame): PDB file as DataFrame
    """
    pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
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


def determine_motif_type(motif):
    # motif name
    motif_type_beta = motif.mtype
    if motif_type_beta in ["JUNCTION", "BULGE", "ILOOP"]:
        return "JCT"
    elif motif_type_beta in ["STEM"]:
        return "HELIX"
    elif motif_type_beta in ["SINGLE_STRAND"]:
        return "SSTRAND"
    elif motif_type_beta in ["HAIRPIN"]:
        return "HAIRPIN"
    else:
        return "UNKNOWN"


def find_strands(
        master_res_df):
    """
    Counts the number of strands in a motif and updates its name accordingly to better reflect structure.

    Args:
        master_res_df (pd.DataFrame): DataFrame containing motif data from PDB.

    Returns:
        len_chains (int): The number of strands in the motif.

    TODO in order to test this function
    we take a dataframe of residues
    and save it, knowing the right answer
    then write a bunch of edge cases for it

    """
    # step 1: make a list of all known residues
    list_of_residues = extract_residue_list(master_res_df)

    # step 2: find the roots of the residues
    residue_roots, res_list_modified = find_residue_roots(list_of_residues)

    # step 3: given the residue roots, build strands of RNA
    strands_of_rna = build_strands_5to3(residue_roots, res_list_modified)

    # step 4: find the sequence of the strands
    sequence = find_sequence(strands_of_rna)

    return strands_of_rna, sequence


def find_sequence(strands_of_rna):
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


def extract_residue_list(master_res_df: pd.DataFrame) -> List:
    """
    Extracts PDB data per residue and puts it in a list

    Args:
        master_res_df (pd.DataFrame): dataframe of the PDB data in the motif

    Returns:
        res_list (list): list of residues with their appropriate data

    """
    # there are several cases where the IDs don't represent the actual residues, so we have to account for each case
    # Extract unique values from pdbx_PDB_ins_code column
    unique_ins_code_values = master_res_df["pdbx_PDB_ins_code"]
    unique_model_num_values = master_res_df["pdbx_PDB_model_num"]

    # Convert each unique value to lists
    unique_ins_code_values_list = unique_ins_code_values.astype(str).tolist()
    unique_model_num_values_list = unique_model_num_values.astype(str).tolist()

    ins_code_set_list = sorted(set(unique_ins_code_values_list))
    model_num_set_list = sorted(set(unique_model_num_values_list))

    # lay out each case; also group by res comp to get sequence later?
    if len(ins_code_set_list) > 1:
        grouped_res_dfs = master_res_df.groupby(
            ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "auth_comp_id"]
        )
    elif len(model_num_set_list) > 1:
        filtered_master_df = master_res_df[master_res_df["pdbx_PDB_model_num"] == "1"]
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


def find_residue_roots(res_list):
    """
    Finds the roots of chains of RNA by finding the bottom of the chain first.
    Roots are residues that are only connected 5' to 3' to one other residue.

    Args:
        res_list (list): List of tuples, each containing a residue name and its corresponding DataFrame.

    Returns:
        roots (list): List containing the root residues, which are only connected in the 5' to 3' direction.

    """
    roots = []

    for source_res in res_list:
        has_5to3_connection = False
        has_3to5_connection = False

        for res_in_question in res_list:
            if source_res != res_in_question:
                is_connected = connected_to(source_res, res_in_question)
                if is_connected == 1:
                    has_5to3_connection = True
                elif is_connected == -1:
                    has_3to5_connection = True

            # If it's connected in the 3' to 5' direction, it cannot be a root
            if has_3to5_connection:
                break

        # A root is defined as having a 5' to 3' connection and no 3' to 5' connection
        if has_5to3_connection and not has_3to5_connection:
            roots.append(source_res)

    # Create a modified list with the root residues removed
    res_list_modified = [res for res in res_list if res not in roots]

    return roots, res_list_modified


def connected_to(source_residue, residue_in_question,
                 cutoff: float = 2.75):
    """
    Determine if another residue is connected to this residue.
    From 5' to 3'; if reverse, returns -1.

    Args:
        source_residue (Residue): Tuple containing the source residue name and its DataFrame.
        residue_in_question (Residue): Tuple containing the residue in question name and its DataFrame.
        cutoff (float): Distance cutoff to determine connectivity.

    Returns:
        connected (int): Whether the two residues are connected.
        Returns 1 if connected 5' to 3'.
        Returns -1 if connected 3' to 5'.
        Returns 0 if no connection.
    """

    residue_1 = source_residue.pdb
    residue_2 = residue_in_question.pdb

    # Convert 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns to numeric
    residue_1[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_1[["Cartn_x", "Cartn_y", "Cartn_z"]].apply(pd.to_numeric)
    residue_2[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_2[["Cartn_x", "Cartn_y", "Cartn_z"]].apply(pd.to_numeric)

    # Extract relevant atom data for both residues
    o3_atom_1 = residue_1[residue_1["auth_atom_id"].str.contains("O3'", regex=True)]
    o3_atom_1 = o3_atom_1[~o3_atom_1["auth_atom_id"].str.contains("H", regex=False)]

    p_atom_2 = residue_2[residue_2["auth_atom_id"].isin(["P"])]

    if not o3_atom_1.empty and not p_atom_2.empty:
        # Calculate the Euclidean distance between the two atoms
        distance = np.linalg.norm(
            p_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values - o3_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        )
        if distance < cutoff:
            return 1  # 5' to 3' direction

    # 3' to 5' direction
    o3_atom_2 = residue_2[residue_2["auth_atom_id"].str.contains("O3'", regex=True)]
    o3_atom_2 = o3_atom_2[~o3_atom_2["auth_atom_id"].str.contains("H", regex=False)]

    p_atom_1 = residue_1[residue_1["auth_atom_id"].isin(["P"])]

    if not o3_atom_2.empty and not p_atom_1.empty:
        # Calculate the Euclidean distance between the two atoms
        distance = np.linalg.norm(
            o3_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values - p_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        )
        if distance < cutoff:
            return -1  # 3' to 5' direction

    return 0  # No connection


def build_strands_5to3(residue_roots, res_list):
    """
    Given residue roots of strands, builds strands of RNA from the list of given residues.

    Args:
        residue_roots (list): List of Residue objs, each containing a root residue name and its corresponding DataFrame.
        res_list (list): List of Residue objs, each containing a residue name and its corresponding DataFrame.

    Returns:
        built_strands (list): List of tuples, each containing a root residue and its built chain of residues in the 5' to 3' direction.
    """
    built_strands = []

    for root in residue_roots:
        current_residue = root
        chain = [current_residue]

        while True:
            next_residue = None
            for res in res_list:
                if connected_to(current_residue, res) == 1:
                    next_residue = res
                    break

            if next_residue:
                chain.append(next_residue)
                current_residue = next_residue
                res_list.remove(next_residue)  # Remove the residue from the list to prevent reusing it
            else:
                break

        built_strands.append(chain)

    return built_strands


def remove_empty_dataframes(dataframes_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Removes empty DataFrames from a list.

    Args:
        dataframes_list (pd.DataFrame): A list of pandas DataFrames.

    Returns:
        dataframes_list (list): A list of pandas DataFrames with empty DataFrames removed.

    """
    dataframes_list = [df for df in dataframes_list if not df.empty]
    return dataframes_list


def extract_longest_letter_sequence(input_string: str) -> str:
    """
    Extracts the longest sequence of letters from a given string.

    Args:
        input_string (str): The string to extract the letter sequence from.

    Returns:
        str: The longest sequence of letters found in the input string.

    """
    # Find all sequences of letters using regular expression
    letter_sequences = re.findall("[a-zA-Z]+", input_string)

    # If there are no letter sequences, return an empty string
    if not letter_sequences:
        return ""

    # Find the longest letter sequence
    longest_sequence = max(letter_sequences, key=len)

    return str(longest_sequence)


def remove_duplicate_residues_in_chain(original_list: list) -> list:
    """
    Removes duplicate items in a list, meant for removing duplicate residues in a chain.

    Args:
        original_list: The list from which to remove duplicate items.

    Returns:
        unique_list: A list with duplicates removed.

    """
    unique_list = []
    for item in original_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def group_residues_by_chain(input_list: List[str]) -> Tuple[List[List[int]], List[str]]:
    """
    Groups residues into their own chains for sequence counting.

    Args:
        input_list (list): List of strings containing chain ID and residue ID separated by a dot.

    Returns:
        sorted_chain_residues (tuple): A tuple containing a list of lists with grouped and sorted residue IDs by chain ID.
        sorted_chain_ids (list): A list of chain IDs corresponding to each group of residues.
    """
    # Create a dictionary to hold grouped and sorted residue IDs by chain ID
    chain_residues = {}

    # Create a dictionary to hold chain IDs for the grouped residues
    chain_ids_for_residues = {}

    # Iterate through the input_list
    for item in input_list:
        chain_id, residue_id = item.split(".")
        if residue_id != "None":
            residue_id = int(residue_id)

            # Create a list for the current chain_id if not already present
        if chain_id not in chain_residues:
            chain_residues[chain_id] = []

            # Append the residue_id to the corresponding chain_id's list
        chain_residues[chain_id].append(residue_id)

        # Store the chain_id for this residue in the dictionary
        if residue_id not in chain_ids_for_residues:
            chain_ids_for_residues[residue_id] = []
        chain_ids_for_residues[residue_id].append(chain_id)

    # Sort each chain's residue IDs and store them in the list of lists
    sorted_chain_residues = []
    sorted_chain_ids = []

    # Sort the chain IDs based on the order they appeared in the input
    unique_chain_ids = list(chain_residues.keys())

    # Sort the chain IDs in the order of appearance
    sorted_unique_chain_ids = unique_chain_ids

    for chain_id in sorted_unique_chain_ids:
        sorted_residues = sorted(set(chain_residues[chain_id]))
        sorted_chain_residues.append(sorted_residues)
        sorted_chain_ids.append(chain_id)

    return sorted_chain_residues, sorted_chain_ids


def get_data_from_dssr(json_path: str) -> Tuple[List, List]:
    """
    Obtains motifs and hbonds from DSSR.

    Args:
        json_path (str): The path to the JSON file containing DSSR output.

    Returns:
        motifs (list): List of motifs.
        motif_hbonds (dict): Dictionary of motif hydrogen bonds.
        motif_interactions (dict): Dictionary of motif interactions.
        hbonds_in_motifs (list): List of hydrogen bonds in motifs.

    """
    d_out = DSSROutput(json_path=json_path)
    motifs = d_out.get_motifs()
    hbonds = d_out.get_hbonds()
    motifs = __merge_singlet_seperated(motifs)
    motifs = __remove_duplicate_motifs(motifs)
    motifs = __remove_large_motifs(motifs)

    return motifs, hbonds


def __remove_duplicate_motifs(motifs: list) -> list:
    """
    Removes duplicate motifs from a list of motifs.

    Args:
        motifs (list): A list of motifs.

    Returns:
        unique_motifs (list): A list of unique motifs.

    """
    # List of duplicates
    duplicates = []
    for m1 in motifs:
        # Skips motifs marked as duplicate
        if m1 in duplicates:
            continue

        m1_nts = [nt.split(".")[1] for nt in m1.nts_long]

        # Compares motif m1 with every other motif m2 in 'motifs' list
        for m2 in motifs:
            if m1 == m2:
                continue

            m2_nts = [nt.split(".")[1] for nt in m2.nts_long]

            # Check if nt sequences of m1 and m2 are identical
            if m1_nts == m2_nts:
                duplicates.append(m2)

    # List that stores unique motifs
    unique_motifs = [m for m in motifs if m not in duplicates]
    return unique_motifs


def __remove_large_motifs(motifs: list) -> list:
    """
    Removes motifs larger than 35 nucleotides.

    Args:
        motifs (list): A list of motifs.

    Returns:
        new_motifs (list): A list of motifs with 35 or fewer nucleotides.

    """
    new_motifs = []
    for m in motifs:
        if len(m.nts_long) > 35:
            continue
        new_motifs.append(m)
    return new_motifs


def __merge_singlet_seperated(motifs: list) -> list:
    """
    Merges singlet separated motifs into a unified list.

    Args:
        motifs (list): A list of motifs to be merged.

    Returns:
        new_motifs (list): A list of motifs that includes merged and non-merged motifs.

    """
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


def __sorted_res_int(item: str) -> Tuple[str, str]:
    """
    Sorts residues by their chain ID and residue number.

    Args:
        item (str): A string representing a residue in the format "chainID.residueID".

    Returns:
        chain_id, residue_id (tuple): A tuple containing the chain ID and residue number.
    """
    spl = item.split(".")
    return spl[0], spl[1][1:]


def __sort_res(item: Any) -> Tuple[str, str]:
    """
    Sorts motifs by the first residue's chain ID and residue number.

    Args:
        item: An object with an attribute 'nts_long' containing residues in the format "chainID.residueID".

    Returns:
        chain_id, residue_id (tuple): A tuple containing the chain ID and residue number of the first residue.
    """
    spl = item.nts_long[0].split(".")
    return spl[0], spl[1][1:]
