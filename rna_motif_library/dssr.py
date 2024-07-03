import os
import re
import math
from typing import List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from pydssr.dssr import DSSROutput
import dssr_hbonds


def make_dir(directory: str) -> None:
    """Creates a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def count_strands(
        master_res_df: pd.DataFrame, motif_name: str, twoway_jct_csv: Any
) -> Tuple[int, str]:
    """Counts the number of strands in a motif and updates its name accordingly.

    Args:
        master_res_df: DataFrame containing motif data from PDB.
        motif_name: Name of the motif being processed.
        twoway_jct_csv: CSV file object to record data regarding two-way junctions.

    Returns:
        A tuple containing the number of strands in the motif and its updated name.
    """
    # step 1: make a list of all known residues
    # there are several cases where the IDs don't represent the actual residues, so we have to account for each case
    # Extract unique values from pdbx_PDB_ins_code column
    unique_ins_code_values = master_res_df["pdbx_PDB_ins_code"]
    unique_model_num_values = master_res_df["pdbx_PDB_model_num"]

    # Convert each unique value to lists
    unique_ins_code_values_list = unique_ins_code_values.astype(str).tolist()
    unique_model_num_values_list = unique_model_num_values.astype(str).tolist()

    ins_code_set_list = sorted(set(unique_ins_code_values_list))
    model_num_set_list = sorted(set(unique_model_num_values_list))

    # lay out each case
    if len(ins_code_set_list) > 1:
        grouped_res_dfs = master_res_df.groupby(
            ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]
        )
    elif len(model_num_set_list) > 1:
        # here we might need to filter the DFs to keep only 1 pdbx_PDB_model_num
        filtered_master_df = master_res_df[master_res_df["pdbx_PDB_model_num"] == "1"]
        grouped_res_dfs = filtered_master_df.groupby(
            ["auth_asym_id", "auth_seq_id", "pdbx_PDB_model_num"]
        )
    else:
        grouped_res_dfs = master_res_df.groupby(
            ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]
        )

    # puts the grouped residues in a list
    res_list = []
    for grouped_residue in grouped_res_dfs:
        res_list.append(grouped_residue)

    # step 2: find all possible dataframe combinations
    # this should include non-unique combinations
    combinations_of_residues = []

    # Nested loop to generate combinations
    for i in range(len(res_list)):
        for j in range(len(res_list)):
            combo = (res_list[i], res_list[j])
            combinations_of_residues.append(combo)
    # combinations_of_residues = list(itertools.combinations(res_list, 2))

    # step 3: calculate distances for each pair of dataframes and put them in a separate list
    distances_btwn_residues = []
    for pair_of_residues in combinations_of_residues:
        distance_btwn_residues = calc_residue_distances(
            pair_of_residues[0], pair_of_residues[1]
        )
        distances_btwn_residues.append(distance_btwn_residues)

    # step 4: put the two lists together into a big dataframe
    combined_combo_distance_df = pd.DataFrame(
        {"Residues": combinations_of_residues, "Distances": distances_btwn_residues}
    )

    # step 5: keep residues which are connected (2.7 seems a good cutoff)
    connected_residues_df = combined_combo_distance_df[
        combined_combo_distance_df["Distances"] < 2.7
        ]
    connected_residues_df_final = connected_residues_df[
        connected_residues_df["Distances"] != 0
        ]

    # step 6: extract the column with the combinations and put it back inside a list, take out the DFs
    list_of_residue_combos = connected_residues_df_final["Residues"].tolist()
    list_of_ids = []  # this is the list of connected residues to process

    for combo in list_of_residue_combos:
        res_1_data = combo[0]
        res_2_data = combo[1]

        res_1_id = res_1_data[0]
        res_2_id = res_2_data[0]

        small_list = [res_1_id, res_2_id]
        list_of_ids.append(small_list)

    # Step 7: take the list of pairs and extract chains from them
    ultra_refined_chains = find_continuous_chains(list_of_ids)

    # Step 8: count # of junctions and calculate structure
    # counting # of junctions
    len_chains = len(ultra_refined_chains)
    # calculating structure
    structure_list = []
    for chain in ultra_refined_chains:
        len_of_strand = len(chain)
        structure_list.append(len_of_strand - 2)
    # convert all data in structure_list to strings
    structure_list = [str(length) for length in structure_list]
    # join by delimiter "-"
    structure_result = "-".join(structure_list)

    # Step 9: print the contents of 2-way motifs into a CSV (nucleotide data, motif names)
    # also change the names to make them match TWOWAY motifs, this will cause problems if you don't!
    if len_chains == 2:
        # Concatenate all inner lists and keep only unique values
        result_0 = []
        for inner_list in ultra_refined_chains[0]:
            result_0.extend(inner_list)
        result_1 = []
        for inner_list in ultra_refined_chains[1]:
            result_1.extend(inner_list)
        # lists unique NTs inside each strand
        unique_nucleotides_0 = list(set(result_0))
        unique_nucleotides_1 = list(set(result_1))

        len_0 = len(unique_nucleotides_0)
        len_1 = len(unique_nucleotides_1)
        class_0 = len_0 - 2
        class_1 = len_1 - 2

        # change all NWAY names to TWOWAY
        motif_type_list = motif_name.split(".")
        motif_type = motif_type_list[0]

        if motif_type == "NWAY":
            new_motif_type = "TWOWAY"
            new_motif_class = str(class_0) + "-" + str(class_1)
            new_motif_name = (
                    new_motif_type
                    + "."
                    + motif_type_list[1]
                    + "."
                    + new_motif_class
                    + "."
                    + motif_type_list[2]
                    + "."
                    + motif_type_list[3]
            )
            motif_name = new_motif_name

        # there's only 1 of these in the entire library but I'll filter it to keep the graph normal
        if (class_0 > 1) or (class_1 > 1):
            # motif_name, motif_type (NWAY/TWOWAY), nucleotides_in_strand_1, nucleotides_in_strand_2
            twoway_jct_csv.write(
                motif_name
                + ","
                + motif_type
                + ","
                + str(class_0)
                + ","
                + str(class_1)
                + "\n"
            )  # + number of nucleotides, which can be found by length of each element in ultra refined chains

        '''
                    + str(len_0)
                    + ","
                    + str(len_1)'''

    # Rewrite motif names so the structure is correct
    elif len_chains > 2:
        # Write new motif name
        old_motif_name_spl = motif_name.split(".")
        motif_name = (
                "NWAY."
                + old_motif_name_spl[1]
                + "."
                + structure_result
                + "."
                + old_motif_name_spl[3]
                + "."
                + old_motif_name_spl[4]
        )

    return len_chains, motif_name


def find_continuous_chains(pair_list: List[List[str]]) -> List[List[str]]:
    """
    Finds and returns a list of lists of all the connected residues.

    Args:
        pair_list: List of pairs of residues (strings).

    Returns:
        A list of lists of all the connected residues.
    """
    chains = []
    chain_map = {}  # Dictionary to map the end points to their respective chains
    for pair in pair_list:
        start, end = pair
        matched_chain_start = chain_map.get(start)
        matched_chain_end = chain_map.get(end)
        if matched_chain_start and matched_chain_end:
            if matched_chain_start is not matched_chain_end:
                matched_chain_start.extend(matched_chain_end)
                for item in matched_chain_end:
                    chain_map[item[0]] = matched_chain_start
                    chain_map[item[1]] = matched_chain_start
                chains.remove(matched_chain_end)
            matched_chain_start.append(pair)
        elif matched_chain_start:
            matched_chain_start.append(pair)
            chain_map[end] = matched_chain_start
        elif matched_chain_end:
            matched_chain_end.append(pair)
            chain_map[start] = matched_chain_end
        else:
            new_chain = [pair]
            chains.append(new_chain)
            chain_map[start] = new_chain
            chain_map[end] = new_chain
    connected_chains = []
    start_map = (
        {}
    )  # Maps starting points of chains to the chain index in connected_chains
    end_map = {}  # Maps ending points of chains to the chain index in connected_chains
    for chain in chains:
        start, end = chain[0][0], chain[-1][1]
        connected = False
        if start in end_map:
            index = end_map[start]
            connected_chains[index].extend(chain)
            end_map[connected_chains[index][-1][1]] = index
            connected = True
        elif end in start_map:
            index = start_map[end]
            connected_chains[index] = chain + connected_chains[index]
            start_map[connected_chains[index][0][0]] = index
            connected = True
        else:
            connected_chains.append(chain)
            index = len(connected_chains) - 1
            start_map[start] = index
            end_map[end] = index
            connected = True
    parent = {}
    rank = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    # Initialize the union-find structure
    for sublist in connected_chains:
        for item in sublist:
            for element in item:
                if element not in parent:
                    parent[element] = element
                    rank[element] = 0
    # Union all connected items
    for sublist in connected_chains:
        first_item = sublist[0][0]
        for item in sublist:
            for element in item:
                union(first_item, element)
    # Group all items by their root
    clusters = {}
    for sublist in connected_chains:
        for item in sublist:
            for element in item:
                root = find(element)
                if root not in clusters:
                    clusters[root] = set()
                clusters[root].add(element)
    # Convert sets back to lists
    merged = [list(cluster) for cluster in clusters.values()]
    return merged


def write_res_coords_to_pdb(
        nts: List[str],
        interactions: Optional[List[str]],
        pdb_model: Any,
        pdb_path: str,
        motif_bond_list: List[str],
        csv_file: Any,
        residue_csv_list: Any,
        twoway_csv: Any,
        interactions_overview_csv: Any,
) -> None:
    """
    Writes motifs and interactions to PDB files, based on provided nucleotide and interaction data.

    Args:
        nts: List of nucleotides.
        interactions: Optional list of interactions.
        pdb_model: PDB model data, typically loaded from an external library.
        pdb_path: Path to the PDB file.
        motif_bond_list: List of motif bonds.
        csv_file: CSV file handler.
        residue_csv_list: CSV list for residues.
        twoway_csv: CSV file specific to two-way junctions.
        interactions_overview_csv: CSV file for interactions overview.
    """
    # directory setup for later
    dir_parts = pdb_path.split("/")
    sub_dir_parts = dir_parts[3].split(".")
    motif_name = dir_parts[3]

    nt_list = []
    res = []
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

    # Extract identification data from nucleotide list
    for nt in nts:
        nt_spl = nt.split(".")
        chain_id = nt_spl[0]
        residue_id = dssr_hbonds.extract_longest_numeric_sequence(nt_spl[1])
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

    if df_list:
        result_df = pd.concat(df_list, axis=0, ignore_index=True)
        if sub_dir_parts[0] in ["NWAY", "TWOWAY"]:
            basepair_ends, motif_name = count_strands(
                result_df, motif_name=motif_name, twoway_jct_csv=twoway_csv
            )
            if basepair_ends != 1:
                new_path = os.path.join(
                    dir_parts[0],
                    f"{basepair_ends}ways",
                    dir_parts[2],
                    motif_name.split(".")[2],
                    sub_dir_parts[3],
                )
                name_path = os.path.join(new_path, motif_name)
                make_dir(new_path)
                cif_path = f"{name_path}.cif"
            else:
                sub_dir_parts[0] = "HAIRPIN"

        if sub_dir_parts[0] == "HAIRPIN":
            hairpin_bridge_length = len(nts) - 2
            sub_dir_parts[2] = str(hairpin_bridge_length)
            motif_name = ".".join(sub_dir_parts)
            hairpin_path = (
                os.path.join(
                    dir_parts[0],
                    "hairpins",
                    str(hairpin_bridge_length),
                    sub_dir_parts[3],
                )
                if hairpin_bridge_length >= 3
                else None
            )
            if hairpin_path:
                make_dir(hairpin_path)
                name_path = os.path.join(hairpin_path, motif_name)
                cif_path = f"{name_path}.cif"
            else:
                sub_dir_parts[0] = "SSTRAND"

        if sub_dir_parts[0] == "HELIX":
            helix_path = os.path.join(
                dir_parts[0], "helices", sub_dir_parts[2], sub_dir_parts[3]
            )
            make_dir(helix_path)
            name_path = os.path.join(helix_path, motif_name)
            cif_path = f"{name_path}.cif"

        if sub_dir_parts[0] == "SSTRAND":
            sstrand_length = len(nts)
            sstrand_path = os.path.join(
                dir_parts[0], "sstrand", str(sstrand_length), sub_dir_parts[3]
            )
            make_dir(sstrand_path)
            name_path = os.path.join(sstrand_path, motif_name)
            cif_path = f"{name_path}.cif"

        if not os.path.exists(
                cif_path
        ):  # if motif already exists, don't bother overwriting
            dssr_hbonds.dataframe_to_cif(
                df=result_df, file_path=f"{name_path}.cif", motif_name=motif_name
            )

    residue_csv_list.write(motif_name + "," + ",".join(nts) + "\n")
    print(motif_name)

    if interactions is not None:
        interactions_filtered = remove_duplicate_residues_in_chain(interactions)
        dssr_hbonds.extract_individual_interactions(
            interactions_filtered,
            motif_bond_list,
            model_df,
            motif_name,
            csv_file,
            interactions_overview_csv,
            nts,
        )


def remove_empty_dataframes(dataframes_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Removes empty DataFrames from a list.

    Args:
        dataframes_list: A list of pandas DataFrames.

    Returns:
        A list of pandas DataFrames with empty DataFrames removed.
    """
    dataframes_list = [df for df in dataframes_list if not df.empty]
    return dataframes_list


def extract_longest_letter_sequence(input_string: str) -> str:
    """Extracts the longest sequence of letters from a given string.

    Args:
        input_string: The string to extract the letter sequence from.

    Returns:
        The longest sequence of letters found in the input string.
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
    """Removes duplicate items in a list, meant for removing duplicate residues in a chain.

    Args:
        original_list: The list from which to remove duplicate items.

    Returns:
        A list with duplicates removed.
    """
    unique_list = []
    for item in original_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def group_residues_by_chain(input_list: List[str]) -> Tuple[List[List[int]], List[str]]:
    """Groups residues into their own chains for sequence counting.

    Args:
        input_list: List of strings containing chain ID and residue ID separated by a dot.

    Returns:
        A tuple containing:
            - A list of lists with grouped and sorted residue IDs by chain ID.
            - A list of chain IDs corresponding to each group of residues.
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


def euclidean_distance_dataframe(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Calculates the Euclidean distance between two points represented by DataFrames.

    Args:
        df1: A pandas DataFrame with 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns.
        df2: A pandas DataFrame with 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns.

    Returns:
        The Euclidean distance between the two points.

    Raises:
        ValueError: If the DataFrames do not have the required columns.
    """
    required_columns = {"Cartn_x", "Cartn_y", "Cartn_z"}
    if not required_columns.issubset(df1.columns) or not required_columns.issubset(
            df2.columns
    ):
        raise ValueError(
            "DataFrames must have 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns"
        )

    try:
        point1 = df1[["Cartn_x", "Cartn_y", "Cartn_z"]].values[0]
        point2 = df2[["Cartn_x", "Cartn_y", "Cartn_z"]].values[0]
        squared_distances = [(float(x) - float(y)) ** 2 for x, y in zip(point1, point2)]
        distance = math.sqrt(sum(squared_distances))
    except IndexError:
        distance = 10.0

    return distance


def calc_residue_distances(
        res_1: Tuple[str, pd.DataFrame], res_2: Tuple[str, pd.DataFrame]
) -> float:
    """Calculate the Euclidean distance between two residues.

    Args:
        res_1: A tuple containing the residue ID and a DataFrame for the first residue.
        res_2: A tuple containing the residue ID and a DataFrame for the second residue.

    Returns:
        The Euclidean distance between the two residues.
    """
    residue1 = res_1[1]
    residue2 = res_2[1]

    # Convert 'Cartn_x', 'Cartn_y', and 'Cartn_z' columns to numeric
    residue1[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue1[
        ["Cartn_x", "Cartn_y", "Cartn_z"]
    ].apply(pd.to_numeric)
    residue2[["Cartn_x", "Cartn_y", "Cartn_z"]] = residue2[
        ["Cartn_x", "Cartn_y", "Cartn_z"]
    ].apply(pd.to_numeric)

    # Extract relevant atom data for both residues
    atom1 = residue1[residue1["auth_atom_id"].str.contains("O3'", regex=True)]
    # Remove hydrogen atoms from the selection if there are any
    atom1 = atom1[~atom1["auth_atom_id"].str.contains("H", regex=False)]

    atom2 = residue2[residue2["auth_atom_id"].isin(["P"])]

    # Calculate the Euclidean distance between the two atoms
    distance = np.linalg.norm(
        atom2[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        - atom1[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    )

    return float(distance)


class DSSRRes:
    """Class to hold residue data from DSSR notation."""

    def __init__(self, s: str) -> None:
        """Initialize a DSSRRes object.

        Args:
            s: A string representing the residue data.
        """
        s = s.split("^")[0]
        spl = s.split(".")
        cur_num = None
        i_num = 0
        for i, c in enumerate(spl[1]):
            if c.isdigit():
                cur_num = spl[1][i:]
                cur_num = dssr_hbonds.extract_longest_numeric_sequence(cur_num)
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


def get_motifs_from_structure(json_path):
    """Obtains motifs from DSSR.

    Args:
        json_path: The path to the JSON file containing DSSR output.

    Returns:
        A tuple containing:
            - List of motifs.
            - Dictionary of motif hydrogen bonds.
            - Dictionary of motif interactions.
            - List of hydrogen bonds in motifs.
    """
    name = os.path.splitext(os.path.basename(json_path))[0]
    d_out = DSSROutput(json_path=json_path)
    motifs = d_out.get_motifs()
    motifs = __merge_singlet_seperated(motifs)
    __name_motifs(motifs, name)
    shared = __find_motifs_that_share_basepair(motifs)
    hbonds = d_out.get_hbonds()
    motif_hbonds, motif_interactions, hbonds_in_motif = __assign_hbonds_to_motifs(
        motifs, hbonds, shared
    )
    motifs = __remove_duplicate_motifs(motifs)
    motifs = __remove_large_motifs(motifs)
    return motifs, motif_hbonds, motif_interactions, hbonds_in_motif


def assign_res_type(name: str, res_type: str) -> str:
    """Assign base, phosphate, sugar, or amino acid in interactions_detailed.csv.

    Args:
        name: The name of the residue.
        res_type: The type of the residue (e.g., "aa" for amino acid).

    Returns:
        The assigned residue type as a string.
    """
    if res_type == "aa":
        return "aa"
    else:
        if name in dssr_hbonds.canon_amino_acid_list:
            return "aa"
        elif "P" in name:
            return "phos"
        elif name.endswith("'"):
            return "sugar"
        else:
            return "base"


def __assign_atom_group(name: str) -> str:
    """Assigns atom groups (base, sugar, phosphate) for making interactions.csv.

    Args:
        name: The name of the atom.

    Returns:
        The assigned atom group as a string.
    """
    if "P" in name:
        return "phos"
    elif name.endswith("'"):
        return "sugar"
    else:
        return "base"


def __assign_hbond_class(atom1: str, atom2: str, rt1: str, rt2: str) -> list:
    """Assigns the hydrogen bond class for given atoms and residue types.

    Args:
        atom1: The name of the first atom.
        atom2: The name of the second atom.
        rt1: The residue type of the first atom (e.g., "nt" for nucleotide).
        rt2: The residue type of the second atom (e.g., "nt" for nucleotide).

    Returns:
        A list containing the assigned classes for the two atoms.
    """
    classes = []
    for atom, residue_type in zip([atom1, atom2], [rt1, rt2]):
        if residue_type == "nt":
            classes.append(__assign_atom_group(atom))
        else:
            classes.append("aa")
    return classes


def __assign_hbonds_to_motifs(motifs: list, hbonds: list, shared: dict) -> tuple:
    """Assigns hydrogen bonds to motifs and counts interactions.

    Args:
        motifs: A list of motifs from DSSR.
        hbonds: A list of hydrogen bonds.
        shared: A dictionary of shared motifs.

    Returns:
        A tuple containing motif_hbonds, motif_interactions, and hbonds_in_motif.
    """
    # All the data about the hbonds in this
    hbonds_in_motif = []

    motif_hbonds = {}
    motif_interactions = {}
    start_dict = {
        "base:base": 0,
        "base:sugar": 0,
        "base:phos": 0,
        "sugar:base": 0,
        "sugar:sugar": 0,
        "sugar:phos": 0,
        "phos:base": 0,
        "phos:sugar": 0,
        "phos:phos": 0,
        "base:aa": 0,
        "sugar:aa": 0,
        "phos:aa": 0,
    }

    hbond_quality_list = []

    for hbond in hbonds:
        # Delete unknown/questionable from consideration
        hbond_quality_list.append(hbond.donAcc_type)
        # If questionable then skip this entire loop
        if hbond.donAcc_type in {"questionable", "unknown"}:
            continue

        # This retrieves the data from the JSON extracted by DSSR
        # res1/res2 are the data we need, they define the actual residues/proteins in the interaction
        atom1, res1 = hbond.atom1_id.split("@")
        atom2, res2 = hbond.atom2_id.split("@")

        distance_bonds = str(hbond.distance)

        # Specifies what kind of biomolecules they are
        rt1, rt2 = hbond.residue_pair.split(":")
        # Not really needed ^ (for the process I was building at the time)

        m1, m2 = None, None

        # m1 and m2 are the actual motifs where:
        for m in motifs:
            # If one of the hbond residues interact with a residue in the motif then copy the hbond residue in
            if res1 in m.nts_long:
                m1 = m

            if res2 in m.nts_long:
                m2 = m

        if m1 == m2:
            continue

        # This creates a name for the interaction (I think); check if the motifs are not empty
        if m1 is not None and m2 is not None:
            names = sorted([m1.name, m2.name])
            key = names[0] + "-" + names[1]
            if key in shared:
                continue

        # This will specify the class of interaction (base/sugar/phos:aa, etc)
        hbond_classes = __assign_hbond_class(atom1, atom2, rt1, rt2)

        # Write residue pairs for export
        hbond_res_pair = (
            res1,
            res2,
            atom1,
            atom2,
            distance_bonds,
            str(hbond_classes[0]),
            str(hbond_classes[1]),
        )
        hbonds_in_motif.append(hbond_res_pair)

        # This counts the # of hydrogen bonds in each category
        if m1 is not None:
            if m1.name not in motif_hbonds:
                motif_hbonds[m1.name] = dict(start_dict)
                motif_interactions[m1.name] = []

            # Sets hydrogen bond class
            hbond_class = hbond_classes[0] + ":" + hbond_classes[1]

            # Increments the start_dict
            motif_hbonds[m1.name][hbond_class] += 1
            motif_interactions[m1.name].append(res2)

        if m2 is not None:
            if m2.name not in motif_hbonds:
                motif_hbonds[m2.name] = dict(start_dict)
                motif_interactions[m2.name] = []

            # Sets hydrogen bond class
            hbond_class = hbond_classes[1] + ":" + hbond_classes[0]
            # Increments the start dict
            if hbond_classes[1] == "aa":
                hbond_class = hbond_classes[0] + ":" + hbond_classes[1]
            motif_hbonds[m2.name][hbond_class] += 1
            motif_interactions[m2.name].append(res1)

    # Count the occurrences of each element
    """hbond_counts = Counter(hbond_quality_list)
    # Print the counts
    for hbond_type, count in hbond_counts.items():
        print(f"{hbond_type}: {count}")"""

    return motif_hbonds, motif_interactions, hbonds_in_motif


def __remove_duplicate_motifs(motifs: list) -> list:
    """Removes duplicate motifs from a list of motifs.

    Args:
        motifs: A list of motifs.

    Returns:
        A list of unique motifs.
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
    """Removes motifs larger than 35 nucleotides.

    Args:
        motifs: A list of motifs.

    Returns:
        A list of motifs with 35 or fewer nucleotides.
    """
    new_motifs = []
    for m in motifs:
        if len(m.nts_long) > 35:
            continue
        new_motifs.append(m)
    return new_motifs


def __merge_singlet_seperated(motifs: list) -> list:
    """Merges singlet separated motifs into a unified list.

    Args:
        motifs: A list of motifs to be merged.

    Returns:
        A list of motifs that includes merged and non-merged motifs.
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


def __find_motifs_that_share_basepair(motifs: list) -> dict:
    """Finds motifs that share base pairs.

    Args:
        motifs: A list of motifs to check for shared base pairs.

    Returns:
        A dictionary where the keys are motif pairs (sorted by name)
        that share base pairs, and the values are 1 indicating shared base pairs.
    """
    pairs = {}

    for m1 in motifs:
        m1_nts = m1.nts_long

        for m2 in motifs:
            if m1 == m2:
                continue

            included = sum(1 for r in m2.nts_long if r in m1_nts)

            if included < 2:
                continue

            names = sorted([m1.name, m2.name])
            key = names[0] + "-" + names[1]
            pairs[key] = 1

    return pairs


def __get_strands(motif) -> list:
    """Gets strands from a motif.

    Args:
        motif: A motif object containing nucleotide sequences.

    Returns:
        A list of strands, where each strand is a list of DSSRRes objects.
    """
    nts = motif.nts_long
    strands = []
    strand = []

    for nt in nts:
        r = DSSRRes(nt)
        if not strand:
            strand.append(r)
            continue

        if r.num is None:
            r.num = 0
        if strand[-1].num is None:
            strand[-1].num = 0

        diff = strand[-1].num - r.num
        if diff == -1:
            strand.append(r)
        else:
            strands.append(strand)
            strand = [r]

    strands.append(strand)
    return strands


def __name_junction(motif, pdb_name: str) -> str:
    """Assigns an initial name to junction motifs.

    This name is later overwritten if need be.

    Args:
        motif: The motif object containing nucleotide sequences.
        pdb_name: The name of the PDB file.

    Returns:
        The initial name assigned to the junction motif.
    """
    nts = motif.nts_long
    strands = __get_strands(motif)
    strs = []
    lens = []

    for strand in strands:
        s = "".join([x.res_id for x in strand])
        strs.append(s)
        lens.append(len(s) - 2)

    if len(strs) == 2:
        name = "TWOWAY."
    else:
        name = "NWAY."

    name += pdb_name + "."
    name += "-".join([str(l) for l in lens]) + "."
    name += "-".join(strs)

    return name


def __name_motifs(motifs, name: str) -> None:
    """Assigns names to motifs (helix, strand, junction, etc).

    Args:
        motifs: A list of motif objects to be named.
        name: The base name used in naming the motifs.

    Returns:
        None
    """
    for m in motifs:
        m.nts_long = sorted(m.nts_long, key=__sorted_res_int)
    motifs = sorted(motifs, key=__sort_res)
    count = {}

    for m in motifs:
        if m.mtype in {"JUNCTION", "BULGE", "ILOOP"}:
            m_name = __name_junction(m, name)
        else:
            mtype = m.mtype
            if mtype == "STEM":
                mtype = "HELIX"
            elif mtype == "SINGLE_STRAND":
                mtype = "SSTRAND"
            m_name = f"{mtype}.{name}."
            strands = __get_strands(m)
            strs = ["".join([x.res_id for x in strand]) for strand in strands]

            if mtype == "HELIX":
                if len(strs) != 2:
                    m.name = "UNKNOWN"
                    continue
                m_name += f"{len(strands[0])}."
                m_name += f"{strs[0]}-{strs[1]}"
            elif mtype == "HAIRPIN":
                m_name += f"{len(strs[0]) - 2}."
                m_name += strs[0]
            else:
                m_name += f"{len(strs[0])}."
                m_name += strs[0]

        if m_name not in count:
            count[m_name] = 0
        else:
            count[m_name] += 1
        m.name = f"{m_name}.{count[m_name]}"


def __sorted_res_int(item: str) -> Tuple[str, str]:
    """Sorts residues by their chain ID and residue number.

    Args:
        item: A string representing a residue in the format "chainID.residueID".

    Returns:
        A tuple containing the chain ID and residue number.
    """
    spl = item.split(".")
    return spl[0], spl[1][1:]


def __sort_res(item: Any) -> Tuple[str, str]:
    """Sorts motifs by the first residue's chain ID and residue number.

    Args:
        item: An object with an attribute 'nts_long' containing residues in the format "chainID.residueID".

    Returns:
        A tuple containing the chain ID and residue number of the first residue.
    """
    spl = item.nts_long[0].split(".")
    return spl[0], spl[1][1:]
