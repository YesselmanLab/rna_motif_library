import requests
import os
import click
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from rna_motif_library.util import (
    get_pdb_ids,
    ion_list,
    canon_rna_res_list,
    canon_res_list,
    CifParser,
)
from rna_motif_library.residue import (
    get_cached_residues,
    sanitize_x3dna_atom_name,
    Residue,
)
from rna_motif_library.chain import get_cached_protein_chains, get_cached_chains, Chains
from rna_motif_library.logger import setup_logging, get_logger
from rna_motif_library.settings import DATA_PATH, RESOURCES_PATH

log = get_logger("LIGAND")


class PDBQuery:
    BASE_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

    @staticmethod
    def _create_base_request_options() -> Dict:
        return {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
            "paginate": {
                "start": 0,
                "rows": 10000,
            },
        }

    @staticmethod
    def create_noncovalent_ligand_query(comp_id: str) -> Dict:
        return {
            "query": {
                "type": "group",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                            "operator": "exact_match",
                            "value": comp_id,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.type",
                            "operator": "exact_match",
                            "value": "HAS_NO_COVALENT_LINKAGE",
                        },
                    },
                ],
                "logical_operator": "and",
                "label": "nested-attribute",
            },
            "return_type": "entry",
            "request_options": PDBQuery._create_base_request_options(),
        }

    @staticmethod
    def create_covalent_ligand_query(comp_id: str) -> Dict:
        return {
            "query": {
                "type": "group",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                            "operator": "exact_match",
                            "value": comp_id,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.type",
                            "operator": "exact_match",
                            "value": "HAS_COVALENT_LINKAGE",
                        },
                    },
                ],
                "logical_operator": "and",
                "label": "nested-attribute",
            },
            "return_type": "entry",
            "request_options": PDBQuery._create_base_request_options(),
        }

    @staticmethod
    def create_ligand_polymer_query(ligand_id: str) -> Dict:
        return {
            "query": {
                "type": "terminal",
                "label": "text",
                "service": "text",
                "parameters": {
                    "operator": "exact_match",
                    "value": ligand_id,
                    "attribute": "rcsb_polymer_entity_container_identifiers.chem_comp_monomers",
                },
            },
            "return_type": "entry",
            "request_options": PDBQuery._create_base_request_options(),
        }

    @staticmethod
    def submit_query(query: Dict) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(PDBQuery.BASE_URL, json=query)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None

    @staticmethod
    def get_pdb_ids(results: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        if not results:
            return None
        return [hit.get("identifier") for hit in results.get("result_set", [])]


def search_noncovalent_ligand(comp_id: str) -> Optional[Dict[str, Any]]:
    query = PDBQuery.create_noncovalent_ligand_query(comp_id)
    results = PDBQuery.submit_query(query)
    return PDBQuery.get_pdb_ids(results)


def search_covalent_ligand(comp_id: str) -> Optional[Dict[str, Any]]:
    query = PDBQuery.create_covalent_ligand_query(comp_id)
    results = PDBQuery.submit_query(query)
    return PDBQuery.get_pdb_ids(results)


def search_ligand_polymer_instances(ligand_id: str) -> Optional[Dict[str, Any]]:
    query = PDBQuery.create_ligand_polymer_query(ligand_id)
    results = PDBQuery.submit_query(query)
    return PDBQuery.get_pdb_ids(results)


def fetch_molecule_data(molecule_id):
    """
    Fetch detailed molecule data from PDB GraphQL API

    Args:
        molecule_id (str): The PDB molecule ID (e.g., '3V1')

    Returns:
        dict: Parsed JSON response containing molecule data
    """
    url = "https://data.rcsb.org/graphql"

    # GraphQL query
    query = """
    query molecule ($id: String!) {
        chem_comp(comp_id: $id) {
            chem_comp {
                id
                name
                formula
                pdbx_formal_charge
                formula_weight
                type
            }
            pdbx_reference_molecule {
                prd_id
                chem_comp_id
                type
                class
                name
                represent_as
                representative_PDB_id_code
            }
            rcsb_chem_comp_info {
                atom_count
                bond_count
                bond_count_aromatic
                atom_count_chiral
                initial_deposition_date
                revision_date
            }
            rcsb_chem_comp_descriptor {
                InChI
                InChIKey
                SMILES
                SMILES_stereo
            }
            pdbx_reference_entity_poly_seq {
                observed
                mon_id
                num
            }
            pdbx_chem_comp_identifier {
                identifier
                program
            }
            pdbx_chem_comp_descriptor {
                type
                descriptor
                program
                program_version
            }
            rcsb_chem_comp_synonyms {
                name
                type
                provenance_source
            }
            drugbank {
                drugbank_info {
                    drugbank_id
                    cas_number
                    drug_categories
                    mechanism_of_action
                    synonyms
                    name
                    drug_groups
                    description
                    affected_organisms
                    brand_names
                    indication
                    pharmacology
                    atc_codes
                }
                drugbank_target {
                    target_actions
                    name
                    interaction_type
                    seq_one_letter_code
                }
            }
            rcsb_chem_comp_related {
                resource_name
                resource_accession_code
            }
        }
    }
    """

    # Variables for the query
    variables = {"id": molecule_id}

    # Make the request
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


def set_column_to_float(df: pd.DataFrame, column: str):
    df[column] = df[column].astype(float)


def get_residue_from_h_cif(cif_path: str):
    try:
        parser = CifParser()
        df_atoms = parser.parse(cif_path)
    except:
        return None
    res_id = df_atoms.iloc[0]["comp_id"]
    atom_names = [
        sanitize_x3dna_atom_name(name) for name in df_atoms["atom_id"].tolist()
    ]
    try:
        set_column_to_float(df_atoms, "pdbx_model_Cartn_x_ideal")
        set_column_to_float(df_atoms, "pdbx_model_Cartn_y_ideal")
        set_column_to_float(df_atoms, "pdbx_model_Cartn_z_ideal")
        coords = df_atoms[
            [
                "pdbx_model_Cartn_x_ideal",
                "pdbx_model_Cartn_y_ideal",
                "pdbx_model_Cartn_z_ideal",
            ]
        ].values
    except:
        set_column_to_float(df_atoms, "Cartn_x")
        set_column_to_float(df_atoms, "Cartn_y")
        set_column_to_float(df_atoms, "Cartn_z")
        coords = df_atoms[
            [
                "Cartn_x",
                "Cartn_y",
                "Cartn_z",
            ]
        ].values

    return Residue("A", res_id, 1, "", "N/A", atom_names, coords)


def find_connected_atoms(
    atom_coord: np.ndarray,
    all_coords: List[np.ndarray],
    all_names: List[str],
    max_bond_length: float = 1.6,
) -> List[Tuple[str, np.ndarray]]:
    """
    Find atoms that are likely bonded to the given atom based on distance.
    """
    connected = []
    for other_coord, other_name in zip(all_coords, all_names):
        if np.array_equal(atom_coord, other_coord):
            continue
        dist = np.linalg.norm(atom_coord - other_coord)
        if dist <= max_bond_length:
            connected.append((other_name, other_coord))
    return connected


def identify_potential_sites(residue: Residue) -> Tuple[List[str], List[str]]:
    """
    Identifies potential hydrogen bond donors and acceptors in a nucleotide residue
    based on geometric analysis, atom types, and presence of hydrogen atoms.

    Args:
        residue (Residue): Residue object containing atom names and coordinates

    Returns:
        Tuple[List[str], List[str]]: Lists of donor and acceptor atom names
    """
    donors = {}
    acceptors = {}

    # Convert coordinates to numpy arrays
    coords = [np.array(coord) for coord in residue.coords]

    base_indices = [i for i, name in enumerate(residue.atom_names)]

    potential_atoms = [(residue.atom_names[i], coords[i]) for i in base_indices]

    for i, (atom_name, coord) in enumerate(potential_atoms):
        # Only consider N and O atoms as potential donors/acceptors
        if not atom_name.startswith(("N", "O")):
            continue
        # Find connected atoms
        connected = find_connected_atoms(
            coord, [c for _, c in potential_atoms], [n for n, _ in potential_atoms]
        )
        if not connected:
            continue
        # Count number of attached hydrogens
        num_hydrogens = sum(1 for n, _ in connected if n.startswith("H"))

        if atom_name.startswith("N"):
            # handle donors
            if num_hydrogens > 0:
                donors[atom_name] = num_hydrogens
            # excludes NH4+
            if len(connected) != 4:
                acceptors[atom_name] = 1

        elif atom_name.startswith("O"):
            if num_hydrogens > 0:  # O-H group - donor
                donors[atom_name] = num_hydrogens
            acceptors[atom_name] = 2

        elif atom_name.startswith("F"):
            acceptors[atom_name] = 3

    return donors, acceptors


def get_ligand_info_from_pdb():
    molecule_files = glob.glob(os.path.join(DATA_PATH, "residues_w_h_cifs", "*.cif"))
    for molecule_file in molecule_files:
        mol_name = os.path.basename(molecule_file).split(".")[0]
        if mol_name in canon_res_list:
            continue
        json_file = os.path.join(DATA_PATH, "ligand_info", f"{mol_name}.json")
        if os.path.exists(json_file):
            continue
        print(mol_name)
        with open(json_file, "w") as f:
            json.dump(fetch_molecule_data(mol_name), f, indent=2)


def is_amino_acid(res) -> bool:
    """Check if residue is an amino acid based on characteristic atoms.

    Args:
        res: Residue object to check

    Returns:
        bool: True if residue appears to be an amino acid, False otherwise
    """
    required_atoms = {"N", "C"}  # Core peptide backbone atoms
    return all(res.get_atom_coords(atom) is not None for atom in required_atoms)


def is_nucleotide(res) -> bool:
    """Check if residue is a nucleotide based on characteristic atoms.

    Args:
        res: Residue object to check

    Returns:
        bool: True if residue appears to be a nucleotide, False otherwise
    """
    required_atoms = {"O3'", "C4'", "C3'", "C5'"}  # Core nucleotide backbone atoms
    return all(res.get_atom_coords(atom) is not None for atom in required_atoms)


def check_residue_bonds(
    residue: Residue, other_residues: Dict[str, Residue], cutoff: float = 2.0
) -> List[Tuple[str, float]]:
    """Check if any atoms in residue could form bonds with atoms in other residues.

    Args:
        residue: The residue to check for bonds
        other_residues: Dictionary of other residues to check against
        cutoff: Maximum distance in Angstroms for atoms to be considered bonded

    Returns:
        List of tuples containing (residue_id, min_distance) for residues with potential bonds
    """
    bonds = []
    res_com = residue.get_center_of_mass()
    res_str = residue.get_str()

    # First filter residues by center of mass distance
    for other_id, other_res in other_residues.items():
        if other_id == res_str:
            continue
        other_com = other_res.get_center_of_mass()
        com_dist = np.linalg.norm(res_com - other_com)

        # Only check atom distances if centers of mass are within 10A
        if com_dist > 15.0:
            continue
        min_dist = float("inf")

        # Check all atom pairs, ignoring hydrogens
        for atom1, coord1 in zip(residue.atom_names, residue.coords):
            if atom1.startswith("H"):
                continue
            for atom2, coord2 in zip(other_res.atom_names, other_res.coords):
                if atom2.startswith("H"):
                    continue
                dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
                min_dist = min(dist, min_dist)

                # Can break early if we find a bond
                if dist < cutoff:
                    bonds.append((other_id, min_dist))
                    break

            if min_dist < cutoff:
                break

    return bonds


def remove_extra_pdbs(pdb_ids, all_pdb_ids):
    subset = []
    for pdb_id in all_pdb_ids:
        if pdb_id in pdb_ids:
            subset.append(pdb_id)
    return subset


@click.group()
def cli():
    pass


# STEP 1 find all non-canonical residues
@cli.command()
def find_all_potential_ligands():
    setup_logging()
    pdb_ids = get_pdb_ids()
    seen = []
    for pdb_id in pdb_ids:
        residues = get_cached_residues(pdb_id)
        for residue in residues.values():
            if residue.res_id in seen:
                continue
            seen.append(residue.res_id)
    f = open("data/ligands/potential_ligands.csv", "w")
    f.write("ligand_id\n")
    for residue in seen:
        f.write(residue + "\n")
    f.close()


# STEP 2 get all cifs and sdfs for all potential ligands
@cli.command()
def get_ligand_cifs():
    os.makedirs(os.path.join(DATA_PATH, "residues_w_h_cifs"), exist_ok=True)
    df = pd.read_csv("data/ligands/potential_ligands.csv")
    for i, row in df.iterrows():
        ligand_id = row["ligand_id"]
        # Download CIF file for residue type
        cif_url = f"https://files.rcsb.org/ligands/download/{ligand_id}.cif"
        cif_path = os.path.join(DATA_PATH, "residues_w_h_cifs", f"{ligand_id}.cif")
        if not os.path.exists(cif_path):
            print(cif_path)
            os.system(f"wget {cif_url} -O {cif_path}")
        sdf_url = f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf"
        sdf_path = os.path.join(
            DATA_PATH, "residues_w_h_sdfs", f"{ligand_id}_ideal.sdf"
        )
        if not os.path.exists(sdf_path):
            print(sdf_path)
            os.system(f"wget {sdf_url} -O {sdf_path}")


# STEP 3 get hbond donors and acceptors
@cli.command()
def get_hbond_donors_and_acceptors():
    acceptors = {}
    donors = {}
    cif_files = glob.glob(os.path.join(DATA_PATH, "residues_w_h_cifs", "*.cif"))
    for cif_file in cif_files:
        base_name = os.path.basename(cif_file)[:-4]
        # if base_name != "ARG":
        #    continue
        try:
            h_residue = get_residue_from_h_cif(cif_file)
        except:
            print("cannot parse", base_name)
            continue
        if h_residue is None:
            continue
        donors_res, acceptors_res = identify_potential_sites(h_residue)
        if base_name in canon_rna_res_list:
            # these are connection points and dont count as donors
            remove = ["O1P", "O2P", "O3P", "O3'"]
            for r in remove:
                if r in donors_res:
                    del donors_res[r]
        acceptors[base_name] = acceptors_res
        donors[base_name] = donors_res

    with open(os.path.join(RESOURCES_PATH, "hbond_acceptors.json"), "w") as f:
        json.dump(acceptors, f)
    with open(os.path.join(RESOURCES_PATH, "hbond_donors.json"), "w") as f:
        json.dump(donors, f)


# STEP 4 get ligand info
@cli.command()
def get_ligand_info():
    get_ligand_info_from_pdb()
    has_phosphate = {}
    json_files = glob.glob(os.path.join(DATA_PATH, "ligand_info", "*.json"))
    for json_file in json_files:
        mol_name = os.path.basename(json_file).split(".")[0]
        has_phosphate[mol_name] = False
    pdb_ids = get_pdb_ids()
    for pdb_id in pdb_ids:
        residues = get_cached_residues(pdb_id).values()
        for res in residues:
            if res.res_id not in has_phosphate:
                continue
            atom_coords_p = res.get_atom_coords("P")
            atom_coords_pa = res.get_atom_coords("PA")
            if atom_coords_pa is not None or atom_coords_p is not None:
                has_phosphate[res.res_id] = True
            else:
                has_phosphate[res.res_id] = False
    all_data = []
    for json_file in json_files:
        mol_name = os.path.basename(json_file).split(".")[0]
        data = json.load(open(json_file, "r"))["data"]
        try:
            keep_data = {
                "id": mol_name,
                "type": data["chem_comp"]["chem_comp"]["type"],
                "name": data["chem_comp"]["chem_comp"]["name"],
                "formula": data["chem_comp"]["chem_comp"]["formula"],
                "formula_weight": data["chem_comp"]["chem_comp"]["formula_weight"],
                "smiles": data["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES"],
                "has_phosphate": has_phosphate[mol_name],
            }
        except:
            print("error", mol_name)
            continue
        all_data.append(keep_data)
    df = pd.DataFrame(all_data)
    df.to_json(os.path.join(DATA_PATH, "ligands", "ligand_info.json"), orient="records")


# STEP 5 get ligand polymer instances
@cli.command()
def get_ligand_polymer_instances():
    df = pd.read_json(os.path.join(DATA_PATH, "ligands", "ligand_info.json"))
    df["noncovalent_results"] = [[] for _ in range(len(df))]
    df["covalent_results"] = [[] for _ in range(len(df))]
    df["polymer_results"] = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        ligand_id = row["id"]
        noncovalent_results = search_noncovalent_ligand(ligand_id) or []
        covalent_results = search_covalent_ligand(ligand_id) or []
        polymer_results = search_ligand_polymer_instances(ligand_id) or []
        df.at[i, "noncovalent_results"] = noncovalent_results
        df.at[i, "covalent_results"] = covalent_results
        df.at[i, "polymer_results"] = polymer_results
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_filtered.json"),
        orient="records",
    )


# STEP 6 filter to only our pdbs
@cli.command()
def filter_pdbs():
    df = pd.read_json(os.path.join(DATA_PATH, "ligands", "ligand_info_filtered.json"))
    df_ligs = pd.read_csv(os.path.join(DATA_PATH, "ligands", "ligand_instances.csv"))
    df_ligs = df_ligs.query("res_id == 'MLA'")
    pdb_ids = get_pdb_ids()
    for i, row in df.iterrows():
        noncovalent_results = row["noncovalent_results"]
        covalent_results = row["covalent_results"]
        polymer_results = row["polymer_results"]
        df.at[i, "noncovalent_results"] = remove_extra_pdbs(
            pdb_ids, noncovalent_results
        )
        df.at[i, "covalent_results"] = remove_extra_pdbs(pdb_ids, covalent_results)
        df.at[i, "polymer_results"] = remove_extra_pdbs(pdb_ids, polymer_results)
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_final.json"),
        orient="records",
    )


# STEP 7 get ligand instances
@cli.command()
def get_ligand_instances():
    df = pd.read_json(os.path.join(DATA_PATH, "ligands", "ligand_info_filtered.json"))
    exclude = (
        canon_res_list
        + ion_list
        + ["HOH"]
        + ["UNK", "UNX", "N", "DN"]  # unknown residues
    )
    pdb_ids = get_pdb_ids()
    data = []
    for pdb_id in pdb_ids:
        pchains = Chains(get_cached_protein_chains(pdb_id))
        rchains = Chains(get_cached_chains(pdb_id))
        residues = get_cached_residues(pdb_id)
        for res in residues.values():
            if res.res_id in exclude:
                continue
            is_nuc = is_nucleotide(res)
            is_aa = is_amino_acid(res)
            if is_nuc:
                c = rchains.get_chain_for_residue(res)
                if c is None:
                    continue
                if len(c) > 1:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "RNA",
                            is_nuc,
                            is_aa,
                        ]
                    )
                else:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "SMALL-MOLECULE",
                            is_nuc,
                            is_aa,
                        ]
                    )
            elif is_aa:
                c = pchains.get_chain_for_residue(res)
                if len(c) > 1:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "PROTEIN",
                            is_nuc,
                            is_aa,
                        ]
                    )
                else:
                    data.append(
                        [
                            pdb_id,
                            res.res_id,
                            res.get_str(),
                            "SMALL-MOLECULE",
                            is_nuc,
                            is_aa,
                        ]
                    )
            else:
                data.append(
                    [
                        pdb_id,
                        res.res_id,
                        res.get_str(),
                        "SMALL-MOLECULE",
                        is_nuc,
                        is_aa,
                    ]
                )
    df = pd.DataFrame(
        data, columns=["pdb_id", "res_id", "res_str", "type", "is_nuc", "is_aa"]
    )
    path = os.path.join(DATA_PATH, "ligands", "ligand_instances.csv")
    df.to_csv(path, index=False)


# STEP 8 get ligand instances with bonds
@cli.command()
def get_ligand_instances_with_bonds():
    df = pd.read_csv(os.path.join(DATA_PATH, "ligands", "ligand_instances.csv"))
    df["bonded_residues"] = [[] for _ in range(len(df))]
    df = df.query("type == 'SMALL-MOLECULE'")
    exclude = ion_list + ["HOH"]
    for i, g in df.groupby("pdb_id"):
        residues = get_cached_residues(i)
        keep_residues = {}
        for id, res in residues.items():
            if res.res_id in exclude:
                continue
            keep_residues[id] = res
        for index, row in g.iterrows():
            res = residues[row["res_str"]]
            bonds = check_residue_bonds(res, keep_residues)
            df.at[index, "bonded_residues"] = bonds
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_instances_w_bonds.json"),
        orient="records",
    )


# STEP 9 get final ligand instances
@cli.command()
def filter_ligands():
    df = pd.read_json(os.path.join(DATA_PATH, "ligands", "ligand_info_final.json"))
    df_solvent = pd.read_csv("solvent_and_buffers.csv")
    df_likely_polymer = pd.read_csv("likely_polymer.csv")
    df_likely_ligand = pd.read_csv("likely_ligands.csv")
    exclude = (
        df_solvent["id"].tolist()
        + df_likely_polymer["id"].tolist()
        + df_likely_ligand["id"].tolist()
        + canon_res_list
        + ion_list
        + ["HOH"]
        + ["UNK", "UNX", "N", "DN"]  # unknown residues
    )
    df = df[~df["id"].isin(exclude)]
    df = df.sort_values("formula_weight", ascending=True)
    df.to_json("sorted_ligand_info.json", orient="records")
    # df = df.query("formula_weight < 200")
    print(
        "id\tname\tformula\tformula_weight\tnum_noncovalent\tnum_covalent\tnum_polymer"
    )
    for i, row in df.iterrows():
        num_noncovalent = len(row["noncovalent_results"])
        num_covalent = len(row["covalent_results"])
        num_polymer = len(row["polymer_results"])
        if num_noncovalent > 0 and num_covalent == 0 and num_polymer == 0:
            print(
                f"{row['id']}\t{row['name']}\t{row['formula']}\t{row['formula_weight']}\t{num_noncovalent}\t{num_covalent}\t{num_polymer}"
            )
    exit()

    # greater than 700 is a ligand
    df = df.query("600 < formula_weight < 700")
    for i, row in df.iterrows():
        print(f"{row['id']}\t{row['name']}\t{row['formula']}\t{row['formula_weight']}")


# STEP 10 get final ligand instances
@cli.command()
def get_final_ligand_instances():
    df = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "ligand_instances_w_bonds.json")
    )
    df_lig = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "ligand_info_complete_final.json")
    )
    polymer_type = {
        "DNA linking": "NON-CANONICAL NA",
        "RNA linking": "NON-CANONICAL NA",
        "L-peptide linking": "NON-CANONICAL AA",
        "D-peptide linking": "NON-CANONICAL AA",
        "peptide-like": "NON-CANONICAL AA",
        "peptide linking": "NON-CANONICAL AA",
        "peptide-like-D": "NON-CANONICAL AA",
        "L-RNA linking": "NON-CANONICAL NA",
        "D-RNA linking": "NON-CANONICAL NA",
        "RNA OH 3 prime terminus": "NON-CANONICAL NA",
        "RNA OH 5 prime terminus": "NON-CANONICAL NA",
        "DNA OH 3 prime terminus": "NON-CANONICAL NA",
        "DNA OH 5 prime terminus": "NON-CANONICAL NA",
        "L-peptide NH3 amino terminus": "NON-CANONICAL AA",
    }
    exclude = ["UNK", "UNL"]
    df = df[~df["res_id"].isin(exclude)]
    lig_type = {}
    for i, row in df_lig.iterrows():
        if row["assigned_ligand"]:
            lig_type[row["id"]] = "LIGAND"
        elif row["assigned_solvent"]:
            lig_type[row["id"]] = "SOLVENT"
        elif row["assigned_polymer"]:
            if row["pdb_type"] in polymer_type:
                lig_type[row["id"]] = polymer_type[row["pdb_type"]]
            else:
                lig_type[row["id"]] = "OTHER-POLYMER"
        else:
            lig_type[row["id"]] = "UNKNOWN"

    df["final_type"] = ""
    for i, row in df.iterrows():
        cur_lig_type = lig_type[row["res_id"]]
        # there is a clear type already assigned for this ID
        if cur_lig_type != "UNKNOWN":
            df.at[i, "final_type"] = cur_lig_type
            continue
        lig_row = df_lig[df_lig["id"] == row["res_id"]].iloc[0]
        # if there are no bonded residues, it is a ligand
        if len(row["bonded_residues"]) == 0:
            df.at[i, "final_type"] = "LIGAND"
            continue
        # if there is bonded residues, it is a polymer
        df.at[i, "final_type"] = "POLYMER"
    df.to_json(
        os.path.join(DATA_PATH, "ligands", "ligand_instances_final.json"),
        orient="records",
    )


# STEP 11 assign identies to all non canonical residues
@cli.command()
def assign_final_indenties():
    df_sm = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "ligand_instances_final.json")
    )
    data = []
    d = {}
    for i, row in df_sm.iterrows():
        d[row["res_str"] + row["pdb_id"]] = row["final_type"]
    path = os.path.join(DATA_PATH, "ligands", "ligand_instances.csv")
    df = pd.read_csv(path)
    data = []
    dna = ["DC", "DA", "DT", "DG"]
    exclude = ["UNK", "UNL"]
    df = df[~df["res_id"].isin(exclude)]
    for i, row in df.iterrows():
        key = row["res_str"] + row["pdb_id"]
        if key in d:
            data.append([row["res_id"], row["res_str"], row["pdb_id"], d[key]])
        elif row["res_id"] in dna:
            data.append([row["res_id"], row["res_str"], row["pdb_id"], "DNA"])
        elif row["type"] == "RNA":
            data.append(
                [row["res_id"], row["res_str"], row["pdb_id"], "NON-CANONICAL NA"]
            )
        elif row["type"] == "PROTEIN":
            data.append(
                [row["res_id"], row["res_str"], row["pdb_id"], "NON-CANONICAL AA"]
            )
        else:
            print(row)
            exit()
    df = pd.DataFrame(data, columns=["res_id", "res_str", "pdb_id", "type"])
    path = os.path.join(DATA_PATH, "ligands", "non_standard_res_identities.csv")
    df.to_csv(path, index=False)
    single_type_data = []
    multi_type_data = []
    for res_id, g in df.groupby("res_id"):
        types = g["type"].unique()
        if len(types) == 1:
            single_type_data.append([res_id, types[0]])
            continue
        else:
            type_counts = g["type"].value_counts()
            most_common_type = type_counts.index[0]
            single_type_data.append([res_id, most_common_type])
            # Add all types as exceptions to multi_type_data
            for type_name in types:
                if type_name != most_common_type:
                    g_filtered = g[g["type"] == type_name]
                    multi_type_data.extend(g_filtered.to_dict(orient="records"))

    single_type_df = pd.DataFrame(single_type_data, columns=["res_id", "type"])
    multi_type_df = pd.DataFrame(multi_type_data)
    single_type_df.to_csv(
        os.path.join(DATA_PATH, "ligands", "single_type_res_identities.csv"),
        index=False,
    )
    multi_type_df.to_csv(
        os.path.join(DATA_PATH, "ligands", "multi_type_res_identities.csv"),
        index=False,
    )


if __name__ == "__main__":
    cli()
