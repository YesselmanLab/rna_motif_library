import os
from typing import List, Any, Tuple
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
    HBondInteraction, DSSRRes
)
from rna_motif_library.dssr_hbonds import (
    dataframe_to_cif,
    assign_res_type, merge_hbond_interaction_data, assemble_interaction_data,
    build_complete_hbond_interaction, save_interactions_to_disk
)
from rna_motif_library.settings import LIB_PATH
from rna_motif_library.snap import get_rnp_interactions


def process_motif_interaction_out_data(count: int, pdb_path: str) -> List[Motif]:
    """
    Function for extracting motifs from a PDB in the loop

    Args:
        count (int): # of PDBs processed (loaded from outside)
        pdb_path (str): path to the source PDB

    Returns:
        motif_list (list): list of motif names

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
    unique_interaction_data = merge_hbond_interaction_data(
        get_rnp_interactions(out_file=rnp_out_path), hbonds
    )
    # This is the final interaction data in the temp class to assemble into the big H-Bond class
    pre_assembled_interaction_data = assemble_interaction_data(unique_interaction_data)
    # Assembly into big HBondInteraction class; this returns a big list of them
    assembled_interaction_data = build_complete_hbond_interaction(
        pre_assembled_interaction_data, pdb_model_df, name
    )
    # Now for every interaction, print to PDB
    save_interactions_to_disk(assembled_interaction_data, name)

    discovered = []
    motif_count = 0
    motif_list = []
    single_motif_interactions = []
    potential_tert_contacts = []
    for m in motifs:
        built_motif = find_and_build_motif(
            m, name, pdb_model_df, discovered, motif_count
        )
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
                # H-bonds fully inside motif (or RNP interactions with the motif)
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
            if type_1 == "nt":
                type_1 = assign_res_type(atom_1, type_1)
            if type_2 == "nt":
                type_2 = assign_res_type(atom_2, type_2)

            distance = float(interaction.distance)
            angle = float(interaction.angle)
            single_motif_interaction = SingleMotifInteraction(
                motif_name,
                res_1,
                res_2,
                atom_1,
                atom_2,
                type_1,
                type_2,
                distance,
                angle,
            )
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
            potential_tert_contact_m1 = PotentialTertiaryContact(
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
            )
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
            potential_tert_contact_m2 = PotentialTertiaryContact(
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
            )
            potential_tert_contacts.append(potential_tert_contact_m2)

    append_data_to_existing_csvs(potential_tert_contacts, assembled_interaction_data, single_motif_interactions)

    return motif_list


def append_data_to_existing_csvs(potential_tert_contacts: List[PotentialTertiaryContact],
                                 assembled_interaction_data: List[HBondInteraction],
                                 single_motif_interactions: List[SingleMotifInteraction]) -> None:
    """
    Writes/appends potential tertiary contact, single motif interaction, and detailed interaction data to the appropriate CSV files.
    Purpose of this function is to reduce RAM usage from the previous method.
    Previously large lists/dataframes were assembled and then dumped to CSV all at once.
    This dumps data periodically to CSV without holding large objects in memory.

    Args:
        potential_tert_contacts (list):
        assembled_interaction_data (list):
        single_motif_interactions (list):

    Returns:
        None

    """

    # establish if the destination CSVs exist or not
    potential_tert_contact_file_exists = os.path.isfile(
        os.path.join(LIB_PATH, "data/out_csvs/potential_tertiary_contacts.csv"))
    interactions_detailed_file_exists = os.path.isfile(
        os.path.join(LIB_PATH, "data/out_csvs/interactions_detailed.csv"))
    single_motif_interaction_file_exists = os.path.isfile(
        os.path.join(LIB_PATH, "data/out_csvs/single_motif_interaction.csv"))

    # First handle potential tert contacts
    potential_tert_contact_data = []
    for potential_contact in potential_tert_contacts:
        # First get the data out
        motif_1 = potential_contact.motif_1
        motif_2 = potential_contact.motif_2
        res_1 = potential_contact.res_1
        res_2 = potential_contact.res_2
        atom_1 = potential_contact.atom_1
        atom_2 = potential_contact.atom_2
        type_1 = potential_contact.type_1
        type_2 = potential_contact.type_2
        # Interactions with amino acids are absolutely not tertiary contacts
        if (
                type_1 == "aa"
                or type_2 == "aa"
                or type_1 == "ligand"
                or type_2 == "ligand"
        ):
            continue

        # Append the filtered data to the list as a dictionary
        potential_tert_contact_data.append(
            {
                "motif_1": motif_1,
                "motif_2": motif_2,
                "res_1": res_1,
                "res_2": res_2,
                "atom_1": atom_1,
                "atom_2": atom_2,
                "type_1": type_1,
                "type_2": type_2,
            }
        )

    # Create a DataFrame from the list of dictionaries and spit to CSV
    potential_tert_contact_df = pd.DataFrame(potential_tert_contact_data)
    if not potential_tert_contact_file_exists:
        # Write with header to the destination
        potential_tert_contact_df.to_csv(os.path.join(LIB_PATH, "data/out_csvs/potential_tertiary_contacts.csv"),
                                         mode='w', header=True, index=False)
    else:
        # Append data without header
        potential_tert_contact_df.to_csv(os.path.join(LIB_PATH, "data/out_csvs/potential_tertiary_contacts.csv"),
                                         mode='a', header=False, index=False)

    # Next handle the assembled interaction data
    interaction_data = []
    for interaction in assembled_interaction_data:
        res_1 = interaction.res_1
        res_2 = interaction.res_2
        atom_1 = interaction.atom_1
        atom_2 = interaction.atom_2
        type_1 = interaction.type_1
        type_2 = interaction.type_2
        distance = interaction.distance
        angle = interaction.angle
        pdb_name = interaction.pdb_name
        mol_1 = DSSRRes(res_1).res_id
        mol_2 = DSSRRes(res_2).res_id
        # filter out ligands
        if type_1 == "ligand" or type_2 == "ligand":
            continue
        # Append the data to the list as a dictionary
        interaction_data.append(
            {
                "pdb_name": pdb_name,
                "res_1": res_1,
                "res_2": res_2,
                "mol_1": mol_1,
                "mol_2": mol_2,
                "atom_1": atom_1,
                "atom_2": atom_2,
                "type_1": type_1,
                "type_2": type_2,
                "distance": distance,
                "angle": angle,
            }
        )
    # Create a DataFrame from the list of dictionaries
    interactions_detailed_df = pd.DataFrame(interaction_data)
    if not interactions_detailed_file_exists:
        # Write with header to the destination
        interactions_detailed_df.to_csv(os.path.join(LIB_PATH, "data/out_csvs/interactions_detailed.csv"),
                                         mode='w', header=True, index=False)
    else:
        # Append data without header
        interactions_detailed_df.to_csv(os.path.join(LIB_PATH, "data/out_csvs/interactions_detailed.csv"),
                                         mode='a', header=False, index=False)

    # Finally handle the single motif interactions
    single_motif_interaction_data = []
    for interaction in single_motif_interactions:
            res_1 = interaction.res_1
            res_2 = interaction.res_2
            atom_1 = interaction.atom_1
            atom_2 = interaction.atom_2
            type_1 = interaction.type_1
            type_2 = interaction.type_2
            distance = interaction.distance
            angle = interaction.angle
            motif_name = interaction.motif_name
            mol_1 = DSSRRes(res_1).res_id
            mol_2 = DSSRRes(res_2).res_id
            # filter out ligands
            if type_1 == "ligand" or type_2 == "ligand":
                continue
            # Append the data to the list as a dictionary
            single_motif_interaction_data.append(
                {
                    "motif_name": motif_name,
                    "res_1": res_1,
                    "res_2": res_2,
                    "mol_1": mol_1,
                    "mol_2": mol_2,
                    "atom_1": atom_1,
                    "atom_2": atom_2,
                    "type_1": type_1,
                    "type_2": type_2,
                    "distance": distance,
                    "angle": angle,
                }
            )
    single_motif_interaction_data_df = pd.DataFrame(single_motif_interaction_data)
    # Spit single motif inters to CSV
    if not single_motif_interaction_file_exists:
        single_motif_interaction_data_df.to_csv(
            os.path.join(LIB_PATH, "data/out_csvs/single_motif_interaction.csv"), mode='w', header=True, index=False
        )
    else:
        single_motif_interaction_data_df.to_csv(
            os.path.join(LIB_PATH, "data/out_csvs/single_motif_interaction.csv"), mode='a', header=False, index=False
        )




def find_and_build_motif(m: Any, pdb_name: str, pdb_model_df: pd.DataFrame, discovered: List[str], motif_count: int):
    """
    Identifies motif in source PDB by ID and extracts to disk, setting its name and other data.

    Args:
        m (Any): DSSR_Motif object, contains motif data returned directly from DSSR
        pdb_name (str): name of PDB
        pdb_model_df (pd.DataFrame): PDB structure as a dataframe
        discovered (list): list of discovered motifs (to count potential duplicates)
        motif_count (int): count of discovered motifs, used in conjunction with "discovered" to set motif name

    Returns:
        our_motif (Motif): Motif object with all associated data inside

    """

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
    size = set_motif_size(list_of_strands, motif_type)
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

    # fix that weird classification issue
    if motif_type == "NWAY" and len(size.split("-")) < 2:
        motif_type = "SSTRAND"

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
    our_motif = Motif(
        motif_name,
        motif_type,
        pdb_name,
        size,
        sequence,
        m.nts_long,
        list_of_strands,
        motif_pdb,
    )
    # And print the motif to the system
    motif_dir_path = os.path.join(LIB_PATH, "data/motifs", motif_type, size, sequence)
    os.makedirs(motif_dir_path, exist_ok=True)
    motif_cif_path = os.path.join(motif_dir_path, f"{motif_name}.cif")
    dataframe_to_cif(motif_pdb, motif_cif_path, motif_name)
    # Clear the PDB so it doesn't hog extra RAM
    our_motif.motif_pdb = None
    return our_motif


def set_motif_size(strands: List, motif_type: str) -> str:
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
    elif motif_type_beta in ["STEM", "HEXIX"]:
        return "HELIX"
    elif motif_type_beta in ["SINGLE_STRAND"]:
        return "SSTRAND"
    elif motif_type_beta in ["HAIRPIN"]:
        return "HAIRPIN"
    else:
        return "UNKNOWN"


def find_strands(master_res_df: pd.DataFrame) -> Tuple[List[Any], str]:
    """
    Counts the number of strands in a motif and updates its name accordingly to better reflect structure.

    Args:
        master_res_df (pd.DataFrame): DataFrame containing motif data from PDB.

    Returns:
        strands_of_rna (list): list of strands of residues
        sequence (str): string containing sequence of the motif (AUCG-AUCG-AUCG)

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


def find_sequence(strands_of_rna: List[List[Residue]]) -> str:
    """
    Finds sequences from found strands of RNA.

    Args:
        strands_of_rna (list): Strands of RNA found.

    Returns:
        sequence (str): RNA sequence of motif.

    """
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


def extract_residue_list(master_res_df: pd.DataFrame) -> List[Residue]:
    """
    Extracts PDB data per residue and puts it in a list of Residue objects.

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


def find_residue_roots(res_list: List[Residue]) -> Tuple[List[Residue], List[Residue]]:
    """
    Finds the roots of chains of RNA by finding the bottom of the chain first.
    Roots are residues that are only connected 5' to 3' to one other residue.

    Args:
        res_list (list): List of Residue objects.

    Returns:
        roots (list): List containing the root residues, which are only connected in the 5' to 3' direction.
        res_list_modified (list): List containing all other residues to build strands with.

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


def connected_to(source_residue: Residue, residue_in_question: Residue, cutoff: float = 2.75) -> int:
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
    residue_1[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_1[
        ["Cartn_x", "Cartn_y", "Cartn_z"]
    ].apply(pd.to_numeric)
    residue_2[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue_2[
        ["Cartn_x", "Cartn_y", "Cartn_z"]
    ].apply(pd.to_numeric)

    # Extract relevant atom data for both residues
    o3_atom_1 = residue_1[residue_1["auth_atom_id"].str.contains("O3'", regex=True)]
    o3_atom_1 = o3_atom_1[~o3_atom_1["auth_atom_id"].str.contains("H", regex=False)]

    p_atom_2 = residue_2[residue_2["auth_atom_id"].isin(["P"])]

    if not o3_atom_1.empty and not p_atom_2.empty:
        # Calculate the Euclidean distance between the two atoms
        distance = np.linalg.norm(
            p_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            - o3_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
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
            o3_atom_2[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            - p_atom_1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        )
        if distance < cutoff:
            return -1  # 3' to 5' direction

    return 0  # No connection


def build_strands_5to3(residue_roots: List[Residue], res_list: List[Residue]):
    """
    Given residue roots of strands, builds strands of RNA from the list of given residues.

    Args:
        residue_roots (list): List of Residue objs, each containing a root residue name and its corresponding DataFrame.
        res_list (list): List of Residue objs, each containing a residue name and its corresponding DataFrame.

    Returns:
        built_strands (list): List of residues, containing a root residue and its built chain of residues in the 5' to 3' direction.

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
                res_list.remove(
                    next_residue
                )  # Remove the residue from the list to prevent reusing it
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


def group_residues_by_chain(input_list: List[str]) -> Tuple[List[List[int]], List[str]]:
    """
    Groups residues into their own chains for counting.

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
    motifs = merge_singlet_separated(motifs)
    motifs = remove_duplicate_motifs(motifs)
    motifs = remove_large_motifs(motifs)

    return motifs, hbonds


def remove_duplicate_motifs(motifs: List[Any]) -> List[Any]:
    """
    Removes duplicate motifs from a list of motifs.

    Args:
        motifs (list): A list of motifs; motifs from DSSR_Motif class

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


def remove_large_motifs(motifs: List[Any]) -> List[Any]:
    """
    Removes motifs larger than 35 nucleotides.

    Args:
        motifs (list): A list of motifs; DSSR_Motif objects.

    Returns:
        new_motifs (list): A list of motifs with 35 or fewer nucleotides; DSSR_Motif objects.

    """
    new_motifs = []
    for m in motifs:
        if len(m.nts_long) > 35:
            continue
        new_motifs.append(m)
    return new_motifs


def merge_singlet_separated(motifs: List[Any]) -> List[Any]:
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
