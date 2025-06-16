# Standard library imports
from dataclasses import dataclass
import glob
import os
from typing import Dict, List

# Third party imports
import click
import numpy as np
import pandas as pd

# Other packages we wrote
from pydssr.dssr import DSSROutput

# Local imports
from rna_motif_library.basepair import Basepair
from rna_motif_library.chain import get_rna_chains, Chains
from rna_motif_library.motif import (
    Motif,
    get_cached_motifs,
)
from rna_motif_library.motif_factory import HelixFinder
from rna_motif_library.logger import get_logger
from rna_motif_library.pdb_data import (
    PDBStructureData,
    get_pdb_structure_data_for_residues,
    get_basepair_ends_for_strands,
    get_cww_basepairs,
    get_pdb_structure_data,
    get_singlet_pairs,
)
from rna_motif_library.parallel_utils import (
    concat_dataframes_from_files,
    run_w_processes_in_batches,
)
from rna_motif_library.residue import Residue, get_cached_residues
from rna_motif_library.settings import DATA_PATH, RESOURCES_PATH
from rna_motif_library.tranforms import rmsd, superimpose_structures
from rna_motif_library.util import (
    NonRedundantSetParser,
    add_motif_indentifier_columns,
    file_exists_and_has_content,
    get_cached_path,
    get_pdb_ids,
    parse_motif_indentifier,
)
from rna_motif_library.x3dna import X3DNAResidue

log = get_logger("motif_analysis")


class MotifFactoryFromOther:
    """Class for processing motifs and interactions from PDB files"""

    def __init__(
        self,
        pdb_data: PDBStructureData,
    ):
        """
        Initialize the MotifProcessor

        Args:
            count (int): # of PDBs processed (loaded from outside)
            pdb_path (str): path to the source PDB
        """
        self.pdb_id = pdb_data.pdb_id
        self.residues = pdb_data.residues
        self.basepairs = pdb_data.basepairs
        self.chains = pdb_data.chains
        self.cww_basepairs = get_cww_basepairs(pdb_data)
        self.used_names = {}

    def get_motifs_from_dssr(self) -> List[Motif]:
        """
        Process the PDB file and extract motifs and interactions

        Returns:
            motif_list (list): list of motifs
        """
        log.debug(f"{self.pdb_id}")
        new_residues = {
            k: Residue.from_dict(v.to_dict()) for k, v in self.residues.items()
        }
        all_residues = {}
        for res in new_residues.values():
            all_residues[res.get_x3dna_str()] = res

        json_path = os.path.join(DATA_PATH, "dssr_output", f"{self.pdb_id}.json")
        dssr_output = DSSROutput(json_path=json_path)
        dssr_motifs = dssr_output.get_motifs()
        motifs = []
        log.info(f"Processing {len(dssr_motifs)} motifs")
        for m in dssr_motifs:
            residues = self._get_residues_for_motif(m, all_residues)
            if len(residues) == 0:
                continue
            m = self._generate_motif(m.mtype, residues)
            motifs.append(m)
        log.info(f"Final number of motifs: {len(motifs)}")
        return motifs

    def get_motif_from_atlas(self, atlas_mtype, res_strs):
        residues = []
        for res_str in res_strs:
            if res_str in self.residues:
                residues.append(self.residues[res_str])
            else:
                log.warning(f"Residue {res_str} not found in {self.pdb_id}")
        return self._generate_motif(atlas_mtype, residues)

    def from_residues(self, residues: List[Residue]) -> Motif:
        return self._generate_motif(residues)

    def _get_residues_for_motif(
        self, m, all_residues: Dict[str, Residue]
    ) -> List[Residue]:
        residues = []
        for nt in m.nts_long:
            if nt in all_residues:
                residues.append(all_residues[nt])
            else:
                # this might be a parsing issue and might be on my side
                log.warning(f"Residue {nt} not found in {self.pdb_id}")
                return []
        return residues

    def _generate_motif(self, x3dna_mtype: str, residues: List[Residue]) -> Motif:
        # We need to determine the data for the motif and build a class
        # First get the type
        mtype = "UNKNOWN"
        if x3dna_mtype == "SINGLE_STRAND":
            mtype = "SSTRAND"
        elif x3dna_mtype == "HAIRPIN":
            mtype = "HAIRPIN"
        elif x3dna_mtype == "STEM":
            mtype = "HELIX"
        elif x3dna_mtype == "ILOOP":
            mtype = "TWOWAY"
        elif x3dna_mtype == "BULGE":
            mtype = "TWOWAY"
        elif x3dna_mtype == "JUNCTION":
            mtype = "NWAY"
        residue_ids = [res.get_str() for res in residues]
        strands = get_rna_chains(residues)
        end_residue_ids = []
        for s in strands:
            end_residue_ids.append(s[0].get_str())
            end_residue_ids.append(s[-1].get_str())
        basepairs = []
        for bp in self.basepairs:
            if bp.res_1.get_str() in residue_ids and bp.res_2.get_str() in residue_ids:
                basepairs.append(bp)
        end_basepairs = get_basepair_ends_for_strands(strands, self.basepairs)
        sequence = self._find_sequence(strands).replace("&", "-")
        mname = f"{mtype}-{sequence}-{self.pdb_id}"
        if mname in self.used_names:
            self.used_names[mname] += 1
        else:
            self.used_names[mname] = 1
        mname += f"-{self.used_names[mname]}"
        return Motif(
            mname,
            mtype,
            self.pdb_id,
            "",
            sequence,
            strands,
            basepairs,
            end_basepairs,
            [],
        )

    def _find_sequence(self, strands_of_rna: List[List[Residue]]) -> str:
        """Find sequences from found strands of RNA"""
        res_strands = []
        for strand in strands_of_rna:
            res_strand = []
            for residue in strand:
                mol_name = residue.res_id
                if mol_name in ["A", "G", "C", "U"]:
                    res_strand.append(mol_name)
                else:
                    res_strand.append("X")
            strand_sequence = "".join(res_strand)
            res_strands.append(strand_sequence)
        sequence = "&".join(res_strands)
        return sequence


def do_number_of_strands_match_mtype(mtype: str, strands: List[List[Residue]]):
    if mtype == "HAIRPIN":
        return len(strands) == 1
    elif mtype == "HELIX":
        return len(strands) == 2
    elif mtype == "TWOWAY":
        return len(strands) == 2
    elif mtype == "NWAY":
        return len(strands) > 2
    elif mtype == "SSTRAND":
        return len(strands) == 1
    else:
        return True


def do_number_of_basepairs_match_mtype(mtype: str, basepairs: List[Basepair]):
    if mtype == "HAIRPIN":
        return len(basepairs) == 1
    elif mtype == "HELIX":
        return len(basepairs) == 2
    elif mtype == "TWOWAY":
        return len(basepairs) == 2
    elif mtype == "NWAY":
        return len(basepairs) > 2
    elif mtype == "SSTRAND":
        return len(basepairs) == 0
    else:
        return True


# Motifs from Atlas #################################################################


@dataclass
class ResidueId:
    pdb_id: str
    model: str
    chain: str
    residue_type: str
    residue_num: int
    insert_code: str

    @staticmethod
    def _parse_triple_pipe_string(residue_str: str):
        spl_1 = residue_str.split("|||")
        if len(spl_1) != 1:
            insert_code = spl_1[1]
            # dont understand why this is sometimes not an insertion code
            if len(insert_code) > 1:
                insert_code = ""
        else:
            insert_code = ""

    @classmethod
    def from_string(cls, residue_str: str):
        # Split on | delimiter
        pdb_id, model, chain, residue_type, residue_num = residue_str.split("|")[0:5]
        insert_code = cls._parse_triple_pipe_string(residue_str)
        spl_2 = residue_str.split("||")
        if insert_code != "" and len(spl_2) != 1:
            if len(spl_2[1]) > 1:
                insert_code = spl_2[1][-1]
            else:
                insert_code = ""
        else:
            insert_code = ""
        return cls(
            pdb_id=pdb_id,
            model=model,
            chain=chain,
            residue_type=residue_type,
            residue_num=int(residue_num),
            insert_code=insert_code,
        )

    def to_x3dna_residue(self) -> X3DNAResidue:
        return X3DNAResidue(
            chain_id=self.chain,
            res_id=self.residue_type,
            num=self.residue_num,
            ins_code=self.insert_code,
            rtype=self.residue_type,
        )

    def get_str(self):
        return f"{self.chain}-{self.residue_type}-{self.residue_num}-{self.insert_code}"


def parse_atlas_csv(csv_path):
    f = open(csv_path, "r")
    group = None
    mtype = None
    data = []
    if os.path.basename(csv_path).startswith("hl"):
        mtype = "HAIRPIN"
    elif os.path.basename(csv_path).startswith("il"):
        mtype = "TWOWAY"
    elif os.path.basename(csv_path).startswith("j3"):
        mtype = "NWAY"
    else:
        raise ValueError(f"Unknown motif type: {os.path.basename(csv_path)}")
    for line in f:
        if line.startswith(">"):
            group = line.strip()[1:]
            continue
        residue_infos = line.strip().split(",")
        x3dna_res = []
        pdb_id = None
        for residue_info in residue_infos:
            # remove quotes
            residue_info = residue_info[1:-1]
            residue_id = ResidueId.from_string(residue_info)
            x3dna_res.append(residue_id.to_x3dna_residue().get_str())
            pdb_id = residue_id.pdb_id
        if pdb_id == "1GID":
            print(mtype, x3dna_res)
        data.append(
            {
                "pdb_id": pdb_id,
                "group": group,
                "mtype": mtype,
                "residues": x3dna_res,
            }
        )
    return pd.DataFrame(data)


def get_data_from_motif(m: Motif, pdb_id: str, has_singlet_flank: bool):
    return {
        "pdb_id": pdb_id,
        "motif": m.name,
        "mtype": m.mtype,
        "n_strands": len(m.strands),
        "n_basepairs": len(m.basepairs),
        "n_basepair_ends": len(m.basepair_ends),
        "n_residues": len(m.get_residues()),
        "residues": [r.get_str() for r in m.get_residues()],
        "correct_n_strands": do_number_of_strands_match_mtype(m.mtype, m.strands),
        "correct_n_basepairs": do_number_of_basepairs_match_mtype(
            m.mtype, m.basepair_ends
        ),
        "has_singlet_flank": has_singlet_flank,
    }


class MotifSetComparerer:
    def compare_motifs(self, pdb_id, df, motifs):
        path = os.path.join(DATA_PATH, "dataframes", "check_motifs", f"{pdb_id}.csv")
        df_check_motifs = pd.read_csv(path)
        self.low_quality_motifs = {}
        for i, row in df_check_motifs.iterrows():
            if row["contains_helix"] == 0:
                self.low_quality_motifs[row["motif_name"]] = True
            else:
                self.low_quality_motifs[row["motif_name"]] = False
        # Get tertiary contacts info
        in_tc = self._get_tertiary_contacts(pdb_id)
        # Get motif mappings
        res_to_motif_id = get_res_to_motif_mapping(motifs)
        motifs_by_name = {m.name: m for m in motifs}
        seen = []
        # Initialize columns
        other_motifs = self._initialize_other_motifs_df(df)
        # Compare motifs
        other_motifs, seen = self._compare_motifs(other_motifs, motifs, seen)
        # Find overlapping motifs
        other_motifs = self._find_overlapping_motifs(
            other_motifs, res_to_motif_id, motifs_by_name, in_tc
        )
        # Add missing motifs
        other_motifs = self._add_missing_motifs(other_motifs, motifs, seen, pdb_id)
        return other_motifs

    def _get_tertiary_contacts(self, pdb_id):
        """
        Maps motifs in tertiary contacts to their residues
        """
        in_tc = {}
        tert_path = os.path.join(
            DATA_PATH, "dataframes", "tertiary_contacts", f"{pdb_id}.json"
        )
        if file_exists_and_has_content(tert_path):
            df_tert = pd.read_json(tert_path)
        else:
            df_tert = pd.DataFrame()
        for i, row in df_tert.iterrows():
            if row["motif_1"] not in in_tc:
                in_tc[row["motif_1"]] = []
            in_tc[row["motif_1"]].extend(row["motif_1_res"])
            if row["motif_2"] not in in_tc:
                in_tc[row["motif_2"]] = []
            in_tc[row["motif_2"]].extend(row["motif_2_res"])
        return in_tc

    def _initialize_other_motifs_df(self, other_motifs):
        other_motifs = other_motifs.copy()
        other_motifs["in_our_db"] = 0  # Whether the motif was found in our database
        other_motifs["misclassified"] = 0  # Whether the motif was misclassified
        other_motifs["in_other_db"] = 1  # Whether the motif was found in DSSR
        other_motifs["overlapping_motifs"] = [
            [] for _ in range(len(other_motifs))
        ]  # Motifs that overlap with the motif
        other_motifs["contained_in_motifs"] = [
            [] for _ in range(len(other_motifs))
        ]  # Motifs that are contained in the motif
        other_motifs["in_tc"] = 0  # Whether the motif is in tertiary contacts
        other_motifs["in_low_quality_motif"] = (
            0  # Whether the motif is in a low quality motif
        )
        return other_motifs

    def _compare_motifs(self, other_motifs, motifs, seen):
        for i, row in other_motifs.iterrows():
            for m in motifs:
                if len(m.get_residues()) != len(row["residues"]):
                    continue
                res = sorted([r.get_str() for r in m.get_residues()])
                res_dssr = sorted(row["residues"])
                if res != res_dssr:
                    continue
                other_motifs.at[i, "in_our_db"] = 1
                seen.append(m.name)
                if m.mtype != row["mtype"]:
                    other_motifs.at[i, "misclassified"] = 1
                other_motifs.at[i, "overlapping_motifs"].append(m.name)
        return other_motifs, seen

    def _find_overlapping_motifs(
        self, dssr_motifs, res_to_motif_id, motifs_by_name, in_tc
    ):
        for i, row in dssr_motifs.iterrows():
            if row["in_our_db"]:
                continue
            overlapping_motifs = []
            potential_tc_res = []
            for r in row["residues"]:
                if r in res_to_motif_id:
                    m_name = res_to_motif_id[r]
                    if m_name not in overlapping_motifs:
                        overlapping_motifs.append(m_name)
                    if m_name in in_tc:
                        potential_tc_res.extend(in_tc[m_name])
            for r in row["residues"]:
                if r in potential_tc_res:
                    dssr_motifs.at[i, "in_tc"] = 1
            dssr_motifs.at[i, "overlapping_motifs"] = overlapping_motifs
            for m in overlapping_motifs:
                overlap_m = motifs_by_name[m]
                overlap_m_res = [r.get_str() for r in overlap_m.get_residues()]
                if all(r in overlap_m_res for r in row["residues"]):
                    dssr_motifs.at[i, "contained_in_motifs"].append(m)
                    if self.low_quality_motifs[m]:
                        dssr_motifs.at[i, "in_low_quality_motif"] = 1
        return dssr_motifs

    def _add_missing_motifs(self, dssr_motifs, motifs, seen, pdb_id):
        data = []
        for m in motifs:
            if m.name in seen:
                continue
            data.append(
                {
                    "pdb_id": pdb_id,
                    "motif": m.name,
                    "mtype": m.mtype,
                    "n_strands": len(m.strands),
                    "n_basepairs": len(m.basepairs),
                    "n_basepair_ends": len(m.basepair_ends),
                    "n_residues": len(m.get_residues()),
                    "residues": [r.get_str() for r in m.get_residues()],
                    "correct_n_strands": 1,
                    "correct_n_basepairs": 1,
                    "in_other_db": 0,
                    "misclassified": 0,
                    "in_our_db": 1,
                    "overlapping_motifs": [m.name],
                    "has_singlet_flank": 0,
                    "contained_in_motifs": [],
                    "in_tc": 0,
                    "in_low_quality_motif": self.low_quality_motifs[m.name],
                }
            )
        df_missing = pd.DataFrame(data)
        return pd.concat([dssr_motifs, df_missing])


# comparing motifs ################################################################


def get_motifs_included_in_chains(motifs: List[Motif], chain_ids: List[str]):
    keep_motifs = []
    for m in motifs:
        keep = True
        for r in m.get_residues():
            if r.chain_id not in chain_ids:
                keep = False
                break
        if keep:
            keep_motifs.append(m)
    return keep_motifs


def find_duplicate_motifs(motifs, other_motifs):
    """Check for duplicate motifs between two sets of motifs.

    Args:
        motifs: List of reference motifs to compare against
        other_motifs: List of motifs to check for duplicates
    Returns:
        DataFrame containing duplicate information for each motif in other_motifs
    """
    data = []
    used_motifs = []
    for other_motif in other_motifs:
        result = find_best_matching_motif(other_motif, motifs, used_motifs)
        data.append(
            {
                "motif": other_motif.name,
                "repr_motif": result["best_motif"],
                "rmsd": result["best_rmsd"],
                "is_duplicate": result["is_duplicate"],
                "from_repr": True,
            }
        )
        if result["is_duplicate"]:
            used_motifs.append(result["best_motif"])
    return pd.DataFrame(data)


def find_best_matching_motif(query_motif, ref_motifs, used_motifs):
    """Find the best matching reference motif for a query motif.

    Args:
        query_motif: Motif to find match for
        ref_motifs: List of reference motifs to compare against
        used_motifs: List of already matched reference motifs

    Returns:
        Dict containing best matching motif and match statistics
    """
    best_motif = None
    best_rmsd = 1000
    query_coords = query_motif.get_c1prime_coords()

    if len(query_coords) < 2:
        return {"best_motif": None, "best_rmsd": best_rmsd, "is_duplicate": False}

    for ref_motif in ref_motifs:
        # Skip if motif already used or sequences don't match
        if ref_motif.name in used_motifs:
            continue
        if ref_motif.sequence != query_motif.sequence:
            continue
        try:
            ref_coords = ref_motif.get_c1prime_coords()
            if len(ref_coords) != len(query_coords):
                continue
            rotated_coords = superimpose_structures(ref_coords, query_coords)
            rmsd_val = rmsd(query_coords, rotated_coords)
            if rmsd_val < best_rmsd:
                best_rmsd = rmsd_val
                best_motif = ref_motif.name
        except:
            print("issues", ref_motif.name, query_motif.name)
            continue
    # Determine if match is close enough to be a duplicate
    is_duplicate = best_rmsd < 0.20 * len(query_coords)
    return {
        "best_motif": best_motif,
        "best_rmsd": best_rmsd,
        "is_duplicate": is_duplicate,
    }


def create_representative_motif_dataframe(repr_motifs, pdb_id, from_repr=True):
    repr_data = []
    for m in repr_motifs:
        repr_data.append(
            {
                "motif": m.name,
                "repr_motif": None,
                "rmsd": None,
                "is_duplicate": False,
                "repr_pdb": pdb_id,
                "child_pdb": None,
                "from_repr": from_repr,
            }
        )
    return pd.DataFrame(repr_data)


def _process_child_entry(child_entry, repr_motifs):
    """Process a single child entry to find duplicates with representative motifs.

    Args:
        child_entry: Child structure entry to process
        repr_motifs: List of motifs from representative structure

    Returns:
        tuple: (matches_df, unmatched_motifs) where matches_df contains duplicate matches
        and unmatched_motifs contains motifs that didn't match the representative
    """
    # Get motifs from child structure
    try:
        motifs = get_cached_motifs(child_entry.pdb_id)
    except:
        return None, []
    child_motifs = get_motifs_included_in_chains(motifs, child_entry.chain_ids)
    # Find duplicates between representative and child motifs
    duplicates_df = find_duplicate_motifs(repr_motifs, child_motifs)
    if len(duplicates_df) == 0:
        return None, []
    # Track motifs that didn't match representative
    unmatched = duplicates_df.query("rmsd == 1000.0")["motif"].values
    unmatched_set = [m for m in child_motifs if m.name in unmatched]
    # Keep matches with valid RMSD
    matches = duplicates_df.query("rmsd < 1000").copy()
    return matches, unmatched_set


def _align_to_other_entry_members(unmatched_motifs):
    """Compare unmatched motifs against each other to find additional duplicates.

    Args:
        unmatched_motifs: List of lists of unmatched motifs

    Returns:
        list: DataFrames containing duplicate information
    """
    results = []
    for i, motif_set_1 in enumerate(unmatched_motifs):
        if not motif_set_1:
            continue
        pdb_id = parse_motif_indentifier(motif_set_1[0].name)[-1]
        # Compare against all subsequent unmatched sets
        for j, motif_set_2 in enumerate(unmatched_motifs[i + 1 :], i + 1):
            if not motif_set_2:
                continue
            child_pdb_id = parse_motif_indentifier(motif_set_2[0].name)[-1]
            # Find duplicates between unmatched sets
            duplicates = find_duplicate_motifs(motif_set_1, motif_set_2)
            duplicates = duplicates.query("is_duplicate == True")
            duplicates["child_pdb"] = child_pdb_id
            duplicates["repr_pdb"] = pdb_id
            results.append(duplicates)
            # Remove found duplicates from second set
            duplicate_names = duplicates["motif"].values
            unmatched_motifs[j] = [
                m for m in motif_set_2 if m.name not in duplicate_names
            ]
        # Add remaining unmatched motifs from first set
        results.append(
            create_representative_motif_dataframe(motif_set_1, pdb_id, from_repr=False)
        )
    return results


def find_unique_motifs_in_non_redundant_set_entry(args):
    """Find unique motifs in a non-redundant set entry.

    Args:
        args: Tuple containing (set_id, repr_entry, child_entries)
            set_id: ID of the non-redundant set
            repr_entry: Representative structure entry
            child_entries: List of child structure entries

    Returns:
        Path to saved CSV file containing duplicate information, or None if file exists
    """
    set_id, repr_entry, child_entries = args
    # Check if output file already exists
    output_path = os.path.join(
        DATA_PATH, "dataframes", "non_redundant_sets", f"{set_id}.csv"
    )
    if os.path.exists(output_path):
        return None
    # Get motifs from representative structure
    try:
        motifs = get_cached_motifs(repr_entry.pdb_id)
    except:
        print(f"Error getting motifs for {repr_entry.pdb_id}")
        return None
    repr_motifs = get_motifs_included_in_chains(motifs, repr_entry.chain_ids)
    # Initialize results with representative motifs
    results = [
        create_representative_motif_dataframe(
            repr_motifs, repr_entry.pdb_id, from_repr=True
        )
    ]
    unmatched_motifs = []  # Motifs that don't match representative structure
    # Process each child structure
    for child_entry in child_entries:
        matches, unmatched = _process_child_entry(child_entry, repr_motifs)
        if matches is not None:
            matches["child_pdb"] = child_entry.pdb_id
            matches["repr_pdb"] = repr_entry.pdb_id
            results.append(matches)
        if unmatched:
            unmatched_motifs.append(unmatched)
    # Compare unmatched motifs
    results.extend(_align_to_other_entry_members(unmatched_motifs))
    # Combine and save results
    final_df = pd.concat(results).reset_index(drop=True)
    final_df.to_csv(output_path, index=False)

    return output_path


def get_res_to_motif_mapping(motifs):
    res_to_motif_id = {}
    for m in motifs:
        for r in m.get_residues():
            if r.get_str() not in res_to_motif_id:
                res_to_motif_id[r.get_str()] = m.name
            else:
                existing_motif = res_to_motif_id[r.get_str()]
                if existing_motif.startswith("HELIX"):
                    res_to_motif_id[r.get_str()] = m.name
    return res_to_motif_id


def process_pdb_id_for_unique_residues(args):
    """Process a single PDB ID to get unique residues and mapping.

    Args:
        args: Tuple containing (pdb_id, unique_motifs)
            pdb_id: PDB ID to process
            unique_motifs: List of unique motif names

    Returns:
        Tuple containing:
            - Dictionary with PDB ID and residues
            - Dictionary with PDB ID and residue to motif mapping
    """
    pdb_id, unique_motifs = args
    try:
        res_to_motif_id = {}
        motifs = get_cached_motifs(pdb_id)
        res = []
        for m in motifs:
            if m.name not in unique_motifs:
                continue
            for r in m.get_residues():
                if r.get_str() not in res:
                    res.append(r.get_str())
                if r.get_str() not in res_to_motif_id:
                    res_to_motif_id[r.get_str()] = m.name
                else:
                    existing_motif = res_to_motif_id[r.get_str()]
                    if existing_motif.startswith("HELIX"):
                        res_to_motif_id[r.get_str()] = m.name

        return ({"pdb_id": pdb_id, "residues": res}, {"pdb_id": res_to_motif_id})
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


def get_dssr_motifs_for_pdb(pdb_id: str):
    """Process a single PDB ID to get DSSR motifs.

    Args:
        pdb_id: The PDB ID to process

    Returns:
        DataFrame containing motif data or None if processing failed
    """
    path = os.path.join(DATA_PATH, "dataframes", "dssr_motifs", f"{pdb_id}.json")
    if os.path.exists(path):
        return None
    try:
        pdb_data = get_pdb_structure_data(pdb_id)
        dssr_mf = MotifFactoryFromOther(pdb_data)
        motifs = dssr_mf.get_motifs_from_dssr()
        cww_basepairs = get_cww_basepairs(pdb_data)
    except Exception as e:
        print(f"Error getting motifs for {pdb_id}: {e}")
        return None
    data = []
    singlet_pairs_lookup = get_singlet_pairs(cww_basepairs, pdb_data.chains)
    for m in motifs:
        has_singlet_flank = False
        for b in m.basepair_ends:
            key = b.res_1.get_str() + "-" + b.res_2.get_str()
            if key in singlet_pairs_lookup:
                has_singlet_flank = True
                break
        data.append(get_data_from_motif(m, pdb_id, has_singlet_flank))
    df = pd.DataFrame(data)
    df.to_json(path, orient="records")
    return df


def process_pdb_id_for_dssr_comparison(pdb_id: str):
    """Process a single PDB ID to compare DSSR motifs.

    Args:
        pdb_id: The PDB ID to process

    Returns:
        DataFrame containing comparison data
    """
    new_path = os.path.join(
        DATA_PATH, "dataframes", "dssr_motifs_compared", f"{pdb_id}.json"
    )
    if os.path.exists(new_path):
        return pd.read_json(new_path)

    path = os.path.join(DATA_PATH, "dataframes", "dssr_motifs", f"{pdb_id}.json")
    try:
        motifs = get_cached_motifs(pdb_id)
        dssr_motifs = pd.read_json(path)
        comparer = MotifSetComparerer()
        df = comparer.compare_motifs(pdb_id, dssr_motifs, motifs)
        df.to_json(new_path, orient="records")
        return df
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None


def get_altas_motifs_for_pdb(args):
    """Process a single PDB ID to get Atlas motifs.

    Args:
        pdb_id: The PDB ID to process
        group_df: DataFrame containing Atlas motif data for this PDB ID

    Returns:
        DataFrame containing motif data or None if processing failed
    """
    pdb_id, group_df = args
    path = os.path.join(DATA_PATH, "dataframes", "atlas_motifs", f"{pdb_id}.json")
    if os.path.exists(path):
        return pd.read_json(path)
    try:
        pdb_data = get_pdb_structure_data(pdb_id)
        other_mf = MotifFactoryFromOther(pdb_data)
        cww_basepairs = get_cww_basepairs(pdb_data)
    except Exception as e:
        print(f"Error getting motifs for {pdb_id}: {e}")
        return None
    motifs = []
    for i, row in group_df.iterrows():
        motifs.append(other_mf.get_motif_from_atlas(row["mtype"], row["residues"]))

    data = []
    singlet_pairs_lookup = get_singlet_pairs(cww_basepairs, pdb_data.chains)
    for m in motifs:
        has_singlet_flank = False
        for b in m.basepair_ends:
            key = b.res_1.get_str() + "-" + b.res_2.get_str()
            if key in singlet_pairs_lookup:
                has_singlet_flank = True
                break
        data.append(get_data_from_motif(m, pdb_id, has_singlet_flank))
    df = pd.DataFrame(data)
    df.to_json(path, orient="records")
    return df


def process_pdb_id_for_atlas_comparison(pdb_id: str):
    """Process a single PDB ID to compare 3D Atlas motifs.

    Args:
        pdb_id: The PDB ID to process

    Returns:
        DataFrame containing comparison data
    """
    new_path = os.path.join(
        DATA_PATH, "dataframes", "atlas_motifs_compared", f"{pdb_id}.json"
    )
    if os.path.exists(new_path):
        return pd.read_json(new_path)

    path = os.path.join(DATA_PATH, "dataframes", "atlas_motifs", f"{pdb_id}.json")
    try:
        motifs = get_cached_motifs(pdb_id)
        compare_motifs = []
        for m in motifs:
            if m.mtype == "HELIX" or m.mtype == "SSTRAND":
                continue
            if m.mtype == "NWAY":
                if m.num_basepairs() > 3:
                    continue
            compare_motifs.append(m)
        atlas_motifs = pd.read_json(path)
        comparer = MotifSetComparerer()
        df = comparer.compare_motifs(pdb_id, atlas_motifs, compare_motifs)
        df.to_json(new_path, orient="records")
        return df
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None


def do_motifs_share_end_basepairs(motif_1: Motif, motif_2: Motif) -> bool:
    """
    Check if two motifs share any end basepairs.

    Args:
        other (Motif): The other motif to check against

    Returns:
        bool: True if the motifs share any end basepairs, False otherwise
    """
    # Get all residue strings from end basepairs of both motifs
    self_end_residues = set()
    for bp in motif_1.basepair_ends:
        self_end_residues.add(bp.res_1.get_str())
        self_end_residues.add(bp.res_2.get_str())

    other_end_residues = set()
    for bp in motif_2.basepair_ends:
        other_end_residues.add(bp.res_1.get_str())
        other_end_residues.add(bp.res_2.get_str())

    # Check if there's any overlap in the end residues
    return bool(self_end_residues & other_end_residues)


def find_motifs_sharing_basepair(
    basepair: Basepair, motifs: List["Motif"]
) -> List["Motif"]:
    """
    Find all motifs that share the given end basepair.

    Args:
        basepair (Basepair): The basepair to search for
        motifs (List[Motif]): List of motifs to search through

    Returns:
        List[Motif]: List of motifs that share the given basepair
    """
    sharing_motifs = []
    basepair_residues = {basepair.res_1.get_str(), basepair.res_2.get_str()}
    for motif in motifs:
        for end_bp in motif.basepair_ends:
            end_bp_residues = {end_bp.res_1.get_str(), end_bp.res_2.get_str()}
            if basepair_residues == end_bp_residues:
                sharing_motifs.append(motif)
                break
            # dont know if i need this
            end_bp_residues = {end_bp.res_2.get_str(), end_bp.res_1.get_str()}
            if basepair_residues == end_bp_residues:
                sharing_motifs.append(motif)
                break

    return sharing_motifs


def check_motif_is_flanked_by_helices(
    motif: Motif, helices: List[Motif], chains: Chains
) -> bool:
    for bp in motif.basepair_ends:
        res_1, res_2 = chains.get_residues_in_basepair(bp)
        if chains.is_chain_end(res_1) and chains.is_chain_end(res_2):
            continue
        shared_helices = find_motifs_sharing_basepair(bp, helices)
        if len(shared_helices) == 0:
            print(
                f"motif {motif.name} is not flanked by helices {bp.res_1.get_str()}-{bp.res_2.get_str()}"
            )
            return 0
    return 1


def check_motifs_in_pdb(pdb_id: str):
    path = f"data/dataframes/check_motifs/{pdb_id}.csv"
    if os.path.exists(path):
        return
    motifs = get_cached_motifs(pdb_id)
    helices = [m for m in motifs if m.mtype == "HELIX"]
    non_helix_motifs = [m for m in motifs if m.mtype != "HELIX"]
    pdb_data = get_pdb_structure_data(pdb_id)
    cww_basepairs_lookup_min = get_cww_basepairs(
        pdb_data, min_two_hbond_score=0.0, min_three_hbond_score=0.0
    )
    cww_basepairs_lookup = get_cww_basepairs(pdb_data)
    singlet_pairs = get_singlet_pairs(cww_basepairs_lookup, pdb_data.chains)
    data = []
    for m in non_helix_motifs:
        if m.mtype == "UNKNOWN":
            continue
        flanking_helices = 1
        if not check_motif_is_flanked_by_helices(m, helices, pdb_data.chains):
            flanking_helices = 0
        contains_helix = 1
        pdb_data_for_residues = get_pdb_structure_data_for_residues(
            pdb_data, m.get_residues()
        )
        hf = HelixFinder(pdb_data_for_residues, cww_basepairs_lookup_min, [])
        m_helices = hf.get_helices()
        contains_helix = 0
        if len(m_helices) > 0:
            contains_helix = 1
        has_singlet_pair = 0
        has_singlet_pair_end = 0
        for bp in m.basepairs:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in singlet_pairs:
                has_singlet_pair += 1
        for bp in m.basepair_ends:
            key = f"{bp.res_1.get_str()}-{bp.res_2.get_str()}"
            if key in singlet_pairs:
                has_singlet_pair_end += 1
        data.append(
            {
                "pdb_id": pdb_id,
                "motif_name": m.name,
                "motif_type": m.mtype,
                "flanking_helices": flanking_helices,
                "contains_helix": contains_helix,
                "has_singlet_pair": has_singlet_pair,
                "has_singlet_pair_end": has_singlet_pair_end,
            }
        )
    for m in helices:
        other_helices = [h for h in helices if h != m]
        flanking_helices = 0
        for h in other_helices:
            if do_motifs_share_end_basepairs(m, h):
                flanking_helices += 1
        data.append(
            {
                "pdb_id": pdb_id,
                "motif_name": m.name,
                "motif_type": m.mtype,
                "flanking_helices": 0,
                "contains_helix": 0,
                "has_singlet_pair": 0,
                "has_singlet_pair_end": 0,
            }
        )
    df = pd.DataFrame(data)
    df["flanking_helices"] = df["flanking_helices"].astype(int)
    df["contains_helix"] = df["contains_helix"].astype(int)
    df["has_singlet_pair"] = df["has_singlet_pair"].astype(int)
    df["has_singlet_pair_end"] = df["has_singlet_pair_end"].astype(int)
    df.to_csv(path, index=False)


# work functions that are run at cli ###################################################


def split_non_redundant_set(csv_path: str, n_splits: int):
    os.makedirs("splits/non_redundant_set_splits", exist_ok=True)
    df = pd.read_csv("data/csvs/rna_residue_counts.csv")
    rna_count_dict = {}
    for _, row in df.iterrows():
        rna_count_dict[row["pdb_id"]] = row["count"]
    parser = NonRedundantSetParser()
    f = open(csv_path, "r")
    lines = f.readlines()
    f.close()
    sets = parser.parse(csv_path)
    all_args = []
    # Calculate total residues for each set
    set_residues = []
    for i, (set_id, repr_entry, child_entries) in enumerate(sets):
        total_residues = rna_count_dict.get(repr_entry.pdb_id, 0)
        for entry in child_entries:
            total_residues += rna_count_dict.get(entry.pdb_id, 0)
        set_residues.append((set_id, total_residues, lines[i]))
    # Sort by total residues to help with balanced splitting
    set_residues.sort(key=lambda x: x[1])
    # Calculate target residues per split
    total_residues = sum(r[1] for r in set_residues)
    target_per_split = total_residues / n_splits
    # Create splits
    current_split = []
    current_residues = 0
    split_files = []
    count = 0
    for set_data in set_residues:
        set_id, residues, line = set_data
        current_split.append((set_id, residues, line))
        current_residues += residues
        if current_residues + residues > target_per_split and current_split:
            # Write current split to file
            split_file = f"splits/non_redundant_set_splits/split_{len(split_files)}.csv"
            with open(split_file, "w") as f:
                for sid, residues, line in current_split:
                    f.write(line)
                    count += 1
            split_files.append(split_file)
            current_split = []
            current_residues = 0
    # Write the final split if it has any content
    if current_split:
        split_file = f"splits/non_redundant_set_splits/split_{len(split_files)}.csv"
        with open(split_file, "w") as f:
            for sid, residues, line in current_split:
                f.write(line)
                count += 1
        split_files.append(split_file)
    print(f"Created {len(split_files)} splits:")
    for i, split_file in enumerate(split_files):
        print(f"Split {i}: {split_file}")


def get_non_redundant_motifs(csv_path: str, processes: int = 1):
    parser = NonRedundantSetParser()
    sets = parser.parse(csv_path)
    all_args = []
    for set_id, repr_entry, child_entries in sets:
        all_args.append((set_id, repr_entry, child_entries))
    os.makedirs(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets"), exist_ok=True
    )
    run_w_processes_in_batches(
        items=all_args,
        func=find_unique_motifs_in_non_redundant_set_entry,
        processes=processes,
        batch_size=100,
        desc="Processing non-redundant set entries",
    )


def check_motifs(pdb_ids: List[str], processes: int = 1):
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "check_motifs"), exist_ok=True)
    run_w_processes_in_batches(
        items=pdb_ids,
        func=check_motifs_in_pdb,
        processes=processes,
        batch_size=100,
        desc="Checking motifs",
    )


def get_unique_motifs():
    # TODO have summary sub folders for all types of summaries
    os.makedirs(os.path.join(DATA_PATH, "summaries"), exist_ok=True)
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "non_redundant_sets", "*.csv")
    )
    df = concat_dataframes_from_files(csv_files)
    # TODO have this just be the original csv format
    df.rename(
        columns={
            "motif": "motif_id",
            "repr_motif": "non_redundant_motif_id",
            "rmsd": "rmsd_to_non_redundant_motif",
            "child_pdb": "pdb_id",
            "repr_pdb": "non_redundant_motif_pdb_id",
            "from_repr": "is_part_of_non_redundant_pdb_id",
        },
        inplace=True,
    )
    # Fill empty pdb_id with non_redundant_motif_pdb_id
    df.loc[df["pdb_id"].isna(), "pdb_id"] = df.loc[
        df["pdb_id"].isna(), "non_redundant_motif_pdb_id"
    ]
    # Fill empty motif_id with non_redundant_motif_id
    df.loc[df["motif_id"].isna(), "motif_id"] = df.loc[
        df["motif_id"].isna(), "non_redundant_motif_id"
    ]
    df.to_csv(
        os.path.join(DATA_PATH, "summaries", "all_motifs.csv"),
        index=False,
    )
    csv_files = glob.glob(
        os.path.join(DATA_PATH, "dataframes", "check_motifs", "*.csv")
    )
    df_issues = concat_dataframes_from_files(csv_files)
    df_issues = df_issues.drop(columns=["pdb_id"])
    df = df.query("is_duplicate == False").copy()
    df = df[["motif_id"]]
    df.rename(columns={"motif_id": "motif_name"}, inplace=True)
    df = df.merge(df_issues, on="motif_name", how="left")
    path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs.csv")
    df.to_csv(path, index=False)
    df1 = (
        df.query(
            "flanking_helices == 1 and contains_helix == 0 and motif_type != 'HELIX'"
        )
        .copy()
        .reset_index(drop=True)
    )
    df2 = df.query("motif_type == 'HELIX'").copy().reset_index(drop=True)
    df = pd.concat([df1, df2])
    path = os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    df.to_csv(path, index=False)


def get_unique_residues(processes: int = 1):
    pass


# cli ##################################################################################


@click.group()
def cli():
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("splits", type=int)
def run_split_non_redundant_set(csv_path, splits):
    split_non_redundant_set(csv_path, splits)


# Step 1: Get non-redundant motifs
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", type=int, default=1)
def run_get_non_redundant_motifs(csv_path, processes):
    get_non_redundant_motifs(csv_path, processes)


# Step 2: Check motifs
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", type=int, default=1)
def run_check_motifs(csv_path, processes):
    df = pd.read_csv(csv_path)
    pdb_ids = df["pdb_id"].unique()
    check_motifs(pdb_ids, processes)


# Step 3: Get unique motifs
# need to run scripts/check_motifs.py to get details first
@cli.command()
def run_get_unique_motifs():
    get_unique_motifs()


# Step 3: Get unique residues
@cli.command()
@click.option("-p", "--processes", type=int, default=1)
def run_get_unique_residues(processes):
    """Get unique residues from non-redundant motifs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    # Read input data
    df = pd.read_csv(
        os.path.join(DATA_PATH, "summaries", "non_redundant_motifs_no_issues.csv")
    )
    unique_motifs = df["motif_name"].values
    df = add_motif_indentifier_columns(df, "motif_name")
    # Prepare arguments for parallel processing
    pdb_ids = df["pdb_id"].unique()
    args = [(pdb_id, unique_motifs) for pdb_id in pdb_ids]

    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=args,
        func=process_pdb_id_for_unique_residues,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for unique residues",
    )
    # Collect and aggregate results
    data = []
    res_mapping = []
    for result in results:
        if result is not None:
            data.append(result[0])
            res_mapping.append(result[1])

    # Save results
    df = pd.DataFrame(data)
    df.to_json(
        os.path.join(DATA_PATH, "summaries", "unique_residues.json"), orient="records"
    )
    df_res_mapping = pd.DataFrame(res_mapping)
    df_res_mapping.to_json(
        os.path.join(DATA_PATH, "summaries", "res_mapping.json"), orient="records"
    )


# Step 4: Get DSSR motifs
@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def get_dssr_motifs(processes):
    """Get DSSR motifs for all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    pdb_ids = get_pdb_ids()
    os.makedirs(os.path.join(DATA_PATH, "dataframes", "dssr_motifs"), exist_ok=True)
    run_w_processes_in_batches(
        items=pdb_ids,
        func=get_dssr_motifs_for_pdb,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for DSSR motifs",
    )
    df = concat_dataframes_from_files(
        glob.glob(os.path.join(DATA_PATH, "dataframes", "dssr_motifs", "*.json"))
    )
    df.to_json(os.path.join("dssr_motifs.json"), orient="records")


# Step 5: Compare DSSR motifs
@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def compare_dssr_motifs(processes):
    """Compare DSSR motifs across all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    pdb_ids = get_pdb_ids()
    os.makedirs(
        os.path.join(DATA_PATH, "dataframes", "dssr_motifs_compared"), exist_ok=True
    )
    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_id_for_dssr_comparison,
        processes=processes,
        batch_size=200,
        desc="Comparing DSSR motifs",
    )

    # Combine results from all processed DataFrames
    dfs = [df for df in results if not df.empty]
    if dfs:
        df = pd.concat(dfs)
        df.to_json(
            os.path.join("dssr_motifs_compared.json"),
            orient="records",
        )


@cli.command()
@click.argument("csv_paths", type=click.Path(exists=True), nargs=-1)
def get_atlas_motifs_summary(csv_paths):
    # should be hl, il, j3
    dfs = []
    for csv_path in csv_paths:
        df = parse_atlas_csv(csv_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_json(
        os.path.join("atlas_motifs.json"),
        orient="records",
    )


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def get_atlas_motifs(processes):
    """Get Atlas motifs for all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    df = pd.read_json("atlas_motifs.json")
    args = [(pdb_id, group_df) for pdb_id, group_df in df.groupby("pdb_id")]
    results = run_w_processes_in_batches(
        items=args,
        func=get_altas_motifs_for_pdb,
        processes=processes,
        batch_size=100,
        desc="Processing PDB IDs for Atlas motifs",
    )


@cli.command()
@click.option(
    "-p", "--processes", type=int, default=1, help="Number of processes to use"
)
def compare_atlas_motifs(processes):
    """Compare Atlas motifs across all PDB IDs using parallel processing.

    Args:
        processes: Number of processes to use for parallel processing
    """
    pdb_ids = glob.glob(os.path.join(DATA_PATH, "dataframes", "atlas_motifs", "*.json"))
    pdb_ids = [os.path.basename(pdb_id).replace(".json", "") for pdb_id in pdb_ids]
    # Process PDB IDs in parallel
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_pdb_id_for_atlas_comparison,
        processes=processes,
        batch_size=100,
        desc="Comparing Atlas motifs",
    )
    # Combine results from all processed DataFrames
    dfs = [df for df in results if not df.empty]
    if dfs:
        df = pd.concat(dfs)
        df.to_json(
            os.path.join(DATA_PATH, "summaries", "atlas_motifs_compared.json"),
            orient="records",
        )


if __name__ == "__main__":
    cli()
