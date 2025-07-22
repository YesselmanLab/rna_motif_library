import requests
import shutil
import os
import click
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Callable
import concurrent.futures
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBD,
    CalcNumHBA,
    CalcNumAromaticRings,
    CalcNumRings,
)

from rna_motif_library.util import (
    get_pdb_ids,
    ion_list,
    canon_rna_res_list,
    canon_res_list,
    CifParser,
)
from rna_motif_library.parallel_utils import (
    run_w_processes_in_batches,
    run_w_threads_in_batches,
)
from rna_motif_library.residue import (
    get_cached_residues,
    sanitize_x3dna_atom_name,
    residues_to_cif_file,
    Residue,
)
from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.chain import get_cached_protein_chains, get_cached_chains, Chains
from rna_motif_library.logger import setup_logging, get_logger
from rna_motif_library.settings import DATA_PATH, RESOURCES_PATH, VERSION
from rna_motif_library.tranforms import pymol_align, rmsd

log = get_logger("LIGAND")

LIGAND_DATA_PATH = os.path.join(DATA_PATH, "ligands")

ONE_MINUTE = 60
MAX_REQUESTS_PER_MINUTE = 10000


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_with_retry(url):
    return requests.get(url, timeout=30)


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
            response = requests.post(PDBQuery.BASE_URL, json=query, timeout=30)
            # Check if the response status is successful
            response.raise_for_status()

            # Check if the response is not empty
            if not response.text:
                print("Received empty response from PDB API")
                return None

            # Try to parse JSON
            try:
                return response.json()
            except json.JSONDecodeError as json_err:
                print(f"Invalid JSON response: {json_err}")
                print(
                    f"Response content: {response.text[:100]}..."
                )  # Print first 100 chars
                return None

        except requests.exceptions.Timeout:
            print("Request timed out. The PDB server may be busy.")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:100]}...")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
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
    """Fetch detailed molecule data from PDB GraphQL API for all potential ligands."""
    # Get all CIF files to process
    molecule_files = glob.glob(
        os.path.join(LIGAND_DATA_PATH, "residues_w_h_cifs", "*.cif")
    )
    mol_names = [os.path.basename(f).split(".")[0] for f in molecule_files]

    # Filter out molecules that are already processed or are canonical residues
    mol_names = [name for name in mol_names if name not in canon_rna_res_list]
    mol_names = [
        name
        for name in mol_names
        if not os.path.exists(
            os.path.join(LIGAND_DATA_PATH, "ligand_info", f"{name}.json")
        )
    ]

    # Process molecules in parallel using threading (I/O bound)
    results = run_w_threads_in_batches(
        items=mol_names,
        func=fetch_molecule_data,
        threads=30,  # Use more threads since this is I/O bound
        batch_size=100,
        desc="Fetching molecule data",
    )

    # Save results
    for mol_name, data in zip(mol_names, results):
        if data is not None:
            with open(
                os.path.join(LIGAND_DATA_PATH, "ligand_info", f"{mol_name}.json"), "w"
            ) as f:
                json.dump(data, f, indent=2)


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
                # Use np.subtract and np.dot for faster squared distance calculation, avoid sqrt unless needed
                diff = np.subtract(coord1, coord2)
                dist = np.dot(diff, diff) ** 0.5
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


def read_sdf_file(file_path):
    """Reads an SDF file and returns a list of RDKit molecule objects.

    Args:
        file_path: The path to the SDF file.

    Returns:
        A list of RDKit molecule objects, or an empty list if an error occurs.
    """
    try:
        suppl = Chem.SDMolSupplier(file_path)
        molecules = [mol for mol in suppl if mol is not None]
        return molecules
    except Exception as e:
        print(f"Error reading SDF file: {e}")
        return []


def assign_ligand_type_features(df):
    """
    Assign ligand type features to each row in the dataframe
    """
    df["aromatic_rings"] = 0
    df["h_acceptors"] = 0
    df["h_donors"] = 0
    df["rings"] = 0
    count = 0
    for i, row in df.iterrows():
        try:
            mol = read_sdf_file(
                os.path.join(
                    LIGAND_DATA_PATH,
                    "residues_w_h_sdfs",
                    f"{row['id']}_ideal.sdf",
                )
            )[0]
            df.at[i, "aromatic_rings"] = CalcNumAromaticRings(mol)
            df.at[i, "rings"] = CalcNumRings(mol)
            df.at[i, "h_acceptors"] = CalcNumHBA(mol)
            df.at[i, "h_donors"] = CalcNumHBD(mol)
        except Exception as e:
            print(f"Error reading SDF file: {e}")
            print("Attempting to generate 3D structure from SMILES...")
            try:
                mol = Chem.MolFromSmiles(row["smiles"])
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                df.at[i, "aromatic_rings"] = CalcNumAromaticRings(mol)
                df.at[i, "rings"] = CalcNumRings(mol)
                df.at[i, "h_acceptors"] = CalcNumHBA(mol)
                df.at[i, "h_donors"] = CalcNumHBD(mol)
            except Exception as e2:
                print(f"Error generating 3D structure: {e2}")
                df.at[i, "aromatic_rings"] = -1
                df.at[i, "h_acceptors"] = -1
                df.at[i, "h_donors"] = -1
                df.at[i, "rings"] = -1
                count += 1
    print(f"Failed to generate 3D structure for {count} rows")
    df.to_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_features.csv"),
        index=False,
    )


def assign_new_ligand_instances(df):
    df_sm = pd.read_json(
        os.path.join(DATA_PATH, "ligands", "summary", "ligand_info_final.json")
    )
    df_bonded = pd.read_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances_w_bonds.json")
    )
    lig_type = {}
    for i, row in df_sm.iterrows():
        lig_type[row["id"]] = row["type"]

    for i, row in df.iterrows():
        if row["res_id"] not in lig_type:
            print("missing ligand type", row["res_id"])
            continue
        cur_lig_type = lig_type[row["res_id"]]
        if cur_lig_type != "UNKNOWN":
            df.at[i, "final_type"] = lig_type[row["res_id"]]
        # there is a clear type already assigned for this ID
        if row["final_type"] != "":
            continue
        bonded_info = df_bonded[df_bonded["res_id"] == df_bonded["res_id"]]
        if len(bonded_info) == 0:
            row["final_type"] = "UNKNOWN"
            continue
        lig_row = bonded_info.iloc[0]
        # if there are no bonded residues, it is a ligand
        if len(lig_row["bonded_residues"]) == 0:
            df.at[i, "final_type"] = "LIGAND"
            continue
        # if there is bonded residues, it is a polymer
        df.at[i, "final_type"] = "POLYMER"


def generate_res_motif_mapping(motifs):
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


# functiosn to run with parallel or threads ###########################################


def find_potential_ligands_in_pdb(pdb_id):
    path = os.path.join(LIGAND_DATA_PATH, "potential_ligand_ids", f"{pdb_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        residues = get_cached_residues(pdb_id)
        seen = set()  # Use set for faster lookups
        for residue in residues.values():
            seen.add(residue.res_id)

        df = pd.DataFrame({"ligand_id": list(seen)})
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        log.error(f"Error processing {pdb_id}: {e}")
        return pd.DataFrame({"ligand_id": []})  # Return empty DataFrame on error


def download_ligand_files(ligand_id):
    """Download CIF and SDF files for a single ligand."""
    try:
        # Download CIF file
        cif_url = f"https://files.rcsb.org/ligands/download/{ligand_id}.cif"
        cif_path = os.path.join(
            LIGAND_DATA_PATH, "residues_w_h_cifs", f"{ligand_id}.cif"
        )
        if not os.path.exists(cif_path):
            response = download_with_retry(cif_url)
            response.raise_for_status()
            with open(cif_path, "wb") as f:
                f.write(response.content)

        # Download SDF file
        sdf_url = f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf"
        sdf_path = os.path.join(
            LIGAND_DATA_PATH, "residues_w_h_sdfs", f"{ligand_id}_ideal.sdf"
        )
        if not os.path.exists(sdf_path):
            response = download_with_retry(sdf_url)
            response.raise_for_status()
            with open(sdf_path, "wb") as f:
                f.write(response.content)

        return ligand_id
    except Exception as e:
        log.error(f"Error downloading files for {ligand_id}: {e}")
        return None


def get_hbond_donors_and_acceptors_for_cif(cif_file):
    """Process a single CIF file to extract donor and acceptor information."""
    base_name = os.path.basename(cif_file)[:-4]
    try:
        h_residue = get_residue_from_h_cif(cif_file)
    except:
        print("cannot parse", base_name)
        return base_name, (None, None)
    if h_residue is None:
        return base_name, (None, None)

    donors_res, acceptors_res = identify_potential_sites(h_residue)
    if base_name in canon_rna_res_list:
        # these are connection points and dont count as donors
        remove = ["O1P", "O2P", "O3P", "O3'"]
        for r in remove:
            if r in donors_res:
                del donors_res[r]
    return base_name, (donors_res, acceptors_res)


def check_phosphate_for_pdb(pdb_id):
    """Check phosphate status for all residues in a PDB file."""
    try:
        residues = get_cached_residues(pdb_id).values()
    except:
        print("missing residues", pdb_id)
        return {}

    phosphate_status = {}
    for res in residues:
        if res.res_id not in phosphate_status:
            atom_coords_p = res.get_atom_coords("P")
            atom_coords_pa = res.get_atom_coords("PA")
            phosphate_status[res.res_id] = (
                atom_coords_pa is not None or atom_coords_p is not None
            )
    return phosphate_status


def process_json_file(json_file):
    """Process a single JSON file to extract ligand information."""
    mol_name = os.path.basename(json_file).split(".")[0]
    try:
        data = json.load(open(json_file, "r"))["data"]
        return {
            "id": mol_name,
            "type": data["chem_comp"]["chem_comp"]["type"],
            "name": data["chem_comp"]["chem_comp"]["name"],
            "formula": data["chem_comp"]["chem_comp"]["formula"],
            "formula_weight": data["chem_comp"]["chem_comp"]["formula_weight"],
            "smiles": data["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES"],
        }
    except Exception as e:
        print(f"error processing {mol_name}: {e}")
        return None


def process_ligand_polymer_info(row):
    """Process a single ligand to get its polymer information.

    Args:
        row: DataFrame row containing ligand information

    Returns:
        dict: Dictionary containing ligand polymer information
    """
    ligand_id = row["id"]
    pdb_ids = get_pdb_ids()

    # Check if JSON file already exists
    json_path = os.path.join(
        LIGAND_DATA_PATH, "ligand_polymer_info", f"{ligand_id}.json"
    )
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error reading existing JSON file for {ligand_id}: {e}")

    try:
        noncovalent_results = search_noncovalent_ligand(ligand_id) or []
        covalent_results = search_covalent_ligand(ligand_id) or []
        polymer_results = search_ligand_polymer_instances(ligand_id) or []

        noncovalent_results = remove_extra_pdbs(pdb_ids, noncovalent_results)
        covalent_results = remove_extra_pdbs(pdb_ids, covalent_results)
        polymer_results = remove_extra_pdbs(pdb_ids, polymer_results)

        result = {
            "id": ligand_id,
            "noncovalent_results": noncovalent_results,
            "covalent_results": covalent_results,
            "polymer_results": polymer_results,
        }

        # Save results to JSON file
        with open(json_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        log.error(f"Error processing ligand {ligand_id}: {e}")
        return None
    return result


def process_single_pdb_ligand_instances(pdb_id):
    """Process ligand instances for a single PDB ID.

    Args:
        pdb_id (str): The PDB ID to process

    Returns:
        pd.DataFrame: DataFrame containing ligand instances for this PDB
    """
    exclude = (
        canon_rna_res_list
        + ion_list
        + ["HOH"]
        + ["UNK", "UNX", "N", "DN"]  # unknown residues
    )

    path = os.path.join(LIGAND_DATA_PATH, "ligand_instances", f"{pdb_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    data = []
    try:
        pchains = Chains(get_cached_protein_chains(pdb_id))
        rchains = Chains(get_cached_chains(pdb_id))
        residues = get_cached_residues(pdb_id)
    except:
        log.error(f"missing residues {pdb_id}")
        return pd.DataFrame(
            columns=["pdb_id", "res_id", "res_str", "type", "is_nuc", "is_aa"]
        )

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
    df.to_csv(path, index=False)
    return df


def process_single_pdb_ligand_bonds(csv_file):
    """Process ligand bonds for a single PDB file.

    Args:
        csv_file (str): Path to the CSV file containing ligand instances

    Returns:
        pd.DataFrame: DataFrame containing ligand instances with bond information
    """
    pdb_id = os.path.basename(csv_file).split(".")[0]
    exclude = ion_list + ["HOH"]

    # Skip if already processed
    if os.path.exists(
        os.path.join(LIGAND_DATA_PATH, "ligand_instances_w_bonds", f"{pdb_id}.json")
    ):
        return pd.read_json(
            os.path.join(LIGAND_DATA_PATH, "ligand_instances_w_bonds", f"{pdb_id}.json")
        )

    try:
        df = pd.read_csv(csv_file)
        df = df.query("type == 'SMALL-MOLECULE'").copy()
        df["bonded_residues"] = [[] for _ in range(len(df))]

        if len(df) == 0:
            path = os.path.join(
                LIGAND_DATA_PATH, "ligand_instances_w_bonds", f"{pdb_id}.json"
            )
            df.to_json(path, orient="records")
            return df

        residues = get_cached_residues(pdb_id)

        # Filter out excluded residues
        keep_residues = {
            id: res for id, res in residues.items() if res.res_id not in exclude
        }

        # Process each ligand
        for index, row in df.iterrows():
            res = residues[row["res_str"]]
            bonds = check_residue_bonds(res, keep_residues)
            df.at[index, "bonded_residues"] = bonds

        # Save individual PDB results
        path = os.path.join(
            LIGAND_DATA_PATH, "ligand_instances_w_bonds", f"{pdb_id}.json"
        )
        df.to_json(path, orient="records")
        return df

    except Exception as e:
        log.error(f"Error processing {pdb_id}: {e}")
        return None


def process_single_pdb_interactions(pdb_id):
    """Process hydrogen bond interactions for a single PDB structure.

    Args:
        pdb_id (str): PDB ID to process

    Returns:
        list: List of interaction data dictionaries
    """
    data = []
    if not os.path.exists(
        os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_id}.csv")
    ):
        return data

    try:
        df = pd.read_csv(
            os.path.join(DATA_PATH, "dataframes", "hbonds", f"{pdb_id}.csv")
        )
        df = df.query("res_type_2 == 'LIGAND'")

        for ligand_res, g in df.groupby("res_2"):
            hbonds = []
            for i, row in g.iterrows():
                hbonds.append(
                    {
                        "res_1": row["res_1"],
                        "atom_1": row["atom_1"],
                        "atom_2": row["atom_2"],
                        "score": row["score"],
                    }
                )
            split_res_id = ligand_res.split("-")
            res_id = split_res_id[0]
            data.append(
                {
                    "ligand_res": ligand_res,
                    "ligand_id": res_id,
                    "interacting_residues": g["res_1"].unique(),
                    "hbonds": hbonds,
                    "num_hbonds": len(g),
                    "hbond_score": g["score"].sum(),
                    "pdb_id": pdb_id,
                }
            )
    except Exception as e:
        log.error(f"Error processing {pdb_id}: {e}")

    return data


# run in main cli #####################################################################


def generate_ligand_info(pdb_ids, processes, overwrite):
    # find all potential ligands #######################################################
    output_path = os.path.join(LIGAND_DATA_PATH, "summary", "potential_ligands.csv")
    if os.path.exists(output_path) and not overwrite:
        log.info("potential ligands already found, skipping...")
    else:
        _find_all_potential_ligands(pdb_ids, processes)
    df = pd.read_csv(output_path)
    ligand_ids = df["ligand_id"].tolist()
    _download_ligand_cif_and_sdf_files(ligand_ids, 30)


def _find_all_potential_ligands(pdb_ids, processes):
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "potential_ligand_ids"), exist_ok=True)
    all_dfs = run_w_processes_in_batches(
        pdb_ids, find_potential_ligands_in_pdb, processes, batch_size=200
    )
    df = pd.concat(all_dfs)
    df = df.drop_duplicates(subset=["ligand_id"])
    # Save results
    output_path = os.path.join(LIGAND_DATA_PATH, "summary", "potential_ligands.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    pass


def _download_ligand_cif_and_sdf_files(ligand_ids, threads):
    # Create necessary directories
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "residues_w_h_cifs"), exist_ok=True)
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "residues_w_h_sdfs"), exist_ok=True)

    # Read ligand IDs
    df = pd.read_csv(os.path.join(LIGAND_DATA_PATH, "summary", "potential_ligands.csv"))
    ligand_ids = df["ligand_id"].tolist()

    # Use thread pool to download files
    results = run_w_threads_in_batches(
        items=ligand_ids,
        func=download_ligand_files,
        threads=threads,
        batch_size=100,  # Process 100 ligands at a time
        desc="Downloading ligand files",
    )

    # Count successes and failures
    successful_downloads = len(results)
    failed_downloads = len(ligand_ids) - successful_downloads
    log.info(
        f"Download complete. Successful: {successful_downloads}, Failed: {failed_downloads}"
    )


# cli ##################################################################################


@click.group()
def cli():
    pass


# STEP 1 find all non-canonical residues
@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def find_all_potential_ligands(csv_path, processes):
    setup_logging()
    pdb_ids = pd.read_csv(csv_path)["pdb_id"].tolist()
    _find_all_potential_ligands(pdb_ids, processes)


# STEP 2 get all cifs and sdfs for all potential ligands
@cli.command()
@click.option(
    "-t", "--threads", default=30, help="Number of threads to use for downloading."
)
def get_ligand_cifs(threads):
    """Download CIF and SDF files for all potential ligands using multiple threads."""
    setup_logging()
    df = pd.read_csv(os.path.join(LIGAND_DATA_PATH, "summary", "potential_ligands.csv"))
    ligand_ids = df["ligand_id"].tolist()
    _download_ligand_cif_and_sdf_files(ligand_ids, threads)


@cli.command()
@click.option("-p", "--processes", default=4, help="Number of processes to use.")
def get_hbond_donors_and_acceptors(processes):
    """Process CIF files in parallel to extract donor and acceptor information."""
    setup_logging()

    # Create necessary directories
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "residues_w_h_cifs"), exist_ok=True)
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "residues_w_h_sdfs"), exist_ok=True)

    # Get all CIF files
    cif_files = glob.glob(os.path.join(LIGAND_DATA_PATH, "residues_w_h_cifs", "*.cif"))

    # Process CIF files in parallel using batched processing
    results = run_w_processes_in_batches(
        items=cif_files,
        func=get_hbond_donors_and_acceptors_for_cif,
        processes=processes,
        batch_size=50,  # Process 50 files at a time
        desc="Processing CIF files",
    )

    # Organize results into donors and acceptors dictionaries
    acceptors = {}
    donors = {}
    for result in results:
        if result is not None:  # Skip None results
            base_name, (donors_res, acceptors_res) = result
            if base_name is not None:
                acceptors[base_name] = acceptors_res
                donors[base_name] = donors_res

    # Save results
    with open(os.path.join(RESOURCES_PATH, "hbond_acceptors.json"), "w") as f:
        json.dump(acceptors, f)
    with open(os.path.join(RESOURCES_PATH, "hbond_donors.json"), "w") as f:
        json.dump(donors, f)


@cli.command()
@click.option("-p", "--processes", default=4, help="Number of processes to use.")
def get_ligand_info(processes):
    """Process ligand information in parallel."""
    setup_logging()
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "ligand_info"), exist_ok=True)

    # First get ligand info from PDB
    get_ligand_info_from_pdb()

    # Get all JSON files
    json_files = glob.glob(os.path.join(LIGAND_DATA_PATH, "ligand_info", "*.json"))
    mol_names = [os.path.basename(f).split(".")[0] for f in json_files]

    # Initialize phosphate status
    has_phosphate = {mol_name: False for mol_name in mol_names}

    # Check phosphate status in parallel using process pool
    pdb_ids = get_pdb_ids()
    phosphate_results = run_w_processes_in_batches(
        items=pdb_ids,
        func=check_phosphate_for_pdb,
        processes=processes,
        batch_size=100,
        desc="Checking phosphate status",
    )

    # Process phosphate results
    for phosphate_status in phosphate_results:
        if phosphate_status is not None:
            for res_id, has_p in phosphate_status.items():
                if res_id in has_phosphate:
                    has_phosphate[res_id] = has_phosphate[res_id] or has_p

    # Process JSON files in parallel using process pool
    json_results = run_w_processes_in_batches(
        items=json_files,
        func=process_json_file,
        processes=processes,
        batch_size=100,
        desc="Processing JSON files",
    )

    # Process JSON results and add phosphate status
    all_data = []
    for result in json_results:
        if result is not None:
            mol_name = result["id"]
            result["has_phosphate"] = has_phosphate[mol_name]
            all_data.append(result)

    # Create and save DataFrame
    df = pd.DataFrame(all_data)
    df.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info.json"), orient="records"
    )


# STEP 5 get ligand polymer instances
@cli.command()
@click.option("-p", "--processes", default=4, help="Number of processes to use.")
def get_ligand_polymer_instances(processes):
    """Get ligand polymer instances using parallel processing."""
    setup_logging()
    df = pd.read_json(os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info.json"))
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "ligand_polymer_info"), exist_ok=True)

    results = run_w_processes_in_batches(
        items=df.to_dict(orient="records"),
        func=process_ligand_polymer_info,
        processes=processes,
        desc="Processing ligand polymer information",
    )
    all_data = []
    for result in results:
        if result is not None:
            all_data.append(result)

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.rename(columns={"id": "res_id"}, inplace=True)
    df.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_polymer_info.json"),
        orient="records",
    )


# STEP 6 get ligand instances
@cli.command()
@click.option("-p", "--processes", default=1, help="Number of processes to use.")
def get_ligand_instances(processes):
    """Get ligand instances using parallel processing."""
    setup_logging()
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(LIGAND_DATA_PATH, "ligand_instances"), exist_ok=True)
    # Get all PDB IDs to process
    pdb_ids = get_pdb_ids()

    # Process PDBs in parallel
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_single_pdb_ligand_instances,
        processes=processes,
        batch_size=100,
        desc="Processing ligand instances",
    )
    # Combine results
    dfs = [df for df in results if not df.empty]
    if not dfs:
        log.error("No valid results found")
        return
    df = pd.concat(dfs)
    df = df.query("type == 'SMALL-MOLECULE'")
    # Save final results
    df.to_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances.csv"), index=False
    )


# STEP 7 get ligand instances with bonds
@cli.command()
@click.option("-p", "--processes", default=4, help="Number of processes to use.")
def get_ligand_instances_with_bonds(processes):
    """Get ligand instances with bonds using parallel processing."""
    setup_logging()
    # Create output directory if it doesn't exist
    os.makedirs(
        os.path.join(LIGAND_DATA_PATH, "ligand_instances_w_bonds"), exist_ok=True
    )
    # Get all CSV files to process
    csv_files = glob.glob(os.path.join(LIGAND_DATA_PATH, "ligand_instances", "*.csv"))
    # Process files in parallel
    results = run_w_processes_in_batches(
        items=csv_files,
        func=process_single_pdb_ligand_bonds,
        processes=processes,
        batch_size=100,
        desc="Processing ligand bonds",
    )
    # Combine results
    dfs = [df for df in results if df is not None]
    if not dfs:
        log.error("No valid results found")
        return
    df = pd.concat(dfs)
    # Save final combined results
    df.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances_w_bonds.json"),
        orient="records",
    )


# STEP 8 get ligand features
@cli.command()
def get_ligand_features():
    df = pd.read_csv(os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances.csv"))
    df_sm = pd.DataFrame({"id": df["res_id"].unique()})
    assign_ligand_type_features(df_sm)


# STEP 9 generate final ligand info
@cli.command()
def generate_final_ligand_info():
    df = pd.read_json(os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info.json"))
    df["pdb_type"] = df["type"]
    df_poly = pd.read_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_polymer_info.json")
    )
    df = pd.merge(df, df_poly, left_on="id", right_on="res_id", how="left")
    df.drop(columns=["res_id"], inplace=True)
    df_features = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_features.csv")
    )
    df_features = df_features[
        ["id", "h_acceptors", "h_donors", "aromatic_rings", "rings"]
    ]
    df = pd.merge(df, df_features, on="id", how="left")
    df_solvent = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "solvent_and_buffers.csv")
    )
    df["assigned_solvent"] = df["id"].isin(df_solvent["id"].values)
    df_ligands = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "ligands.csv")
    )
    df["assigned_ligand"] = df["id"].isin(df_ligands["id"].values)
    df_poly = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "manual", "polymers.csv")
    )
    df["assigned_polymer"] = df["id"].isin(df_poly["id"].values)

    # assign final types
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
    df["type"] = ""
    for i, row in df.iterrows():
        if row["assigned_ligand"]:
            df.at[i, "type"] = "LIGAND"
        elif row["assigned_solvent"]:
            df.at[i, "type"] = "SOLVENT"
        elif row["assigned_polymer"]:
            if row["pdb_type"] in polymer_type:
                df.at[i, "type"] = polymer_type[row["pdb_type"]]
            else:
                df.at[i, "type"] = "OTHER-POLYMER"
        else:
            df.at[i, "type"] = "UNKNOWN"

    df.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info_final.json"),
        orient="records",
    )


# TODO need to make sure there is a current version or this wont work
# STEP 10 assign identies to all non canonical residues
@cli.command()
def assign_final_indenties():
    df_prev = pd.read_json(
        os.path.join(
            LIGAND_DATA_PATH,
            "summary",
            "versions",
            f"v{VERSION}",
            "ligand_instances.json",
        )
    )
    prev_assignments = {}
    for i, row in df_prev.iterrows():
        prev_assignments[row["res_id"] + row["pdb_id"]] = row["final_type"]
    df = pd.read_csv(os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances.csv"))
    df["final_type"] = ""
    for i, row in df.iterrows():
        key = row["res_id"] + row["pdb_id"]
        if key in prev_assignments:
            df.at[i, "final_type"] = prev_assignments[key]
    assign_new_ligand_instances(df)
    df.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances_final.json"),
        orient="records",
    )


# STEP 11 create new version
@cli.command()
def create_new_version():
    NEW_VERSION = VERSION + 1
    PATH = os.path.join(LIGAND_DATA_PATH, "summary", "versions", f"v{NEW_VERSION}")
    os.makedirs(PATH, exist_ok=True)
    df = pd.read_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_instances_final.json")
    )
    df.to_json(os.path.join(PATH, "ligand_instances.json"), orient="records")
    df = pd.read_json(os.path.join(PATH, "ligand_instances.json"))
    shutil.copy(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info_final.json"),
        os.path.join(PATH, "ligand_info.json"),
    )
    single_type_data = []
    multi_type_data = []
    for res_id, g in df.groupby("res_id"):
        types = g["final_type"].unique()
        if len(types) == 1:
            single_type_data.append([res_id, types[0]])
            continue
        else:
            type_counts = g["final_type"].value_counts()
            most_common_type = type_counts.index[0]
            single_type_data.append([res_id, most_common_type])
            # Add all types as exceptions to multi_type_data
            for type_name in types:
                if type_name != most_common_type:
                    g_filtered = g[g["final_type"] == type_name]
                    multi_type_data.extend(g_filtered.to_dict(orient="records"))

    single_type_df = pd.DataFrame(single_type_data, columns=["res_id", "type"])
    multi_type_df = pd.DataFrame(multi_type_data)
    single_type_df.to_csv(
        os.path.join(PATH, "single_type_res_identities.csv"),
        index=False,
    )
    multi_type_df.to_csv(
        os.path.join(PATH, "multi_type_res_identities.csv"),
        index=False,
    )


# STEP 11 get final analysis summaries
@cli.command()
@click.option("-p", "--processes", default=4, help="Number of processes to use.")
def generate_interaction_summary(processes):
    """Generate summary of RNA-ligand interactions using parallel processing."""
    setup_logging()

    # Get all PDB IDs to process
    pdb_ids = get_pdb_ids()

    # Process PDBs in parallel using batched processing
    results = run_w_processes_in_batches(
        items=pdb_ids,
        func=process_single_pdb_interactions,
        processes=processes,
        batch_size=100,
        desc="Processing RNA-ligand interactions",
    )

    # Combine results from all processes
    all_data = []
    for result in results:
        if result is not None:
            all_data.extend(result)

    # Create final dataframe and sort by hbond score
    df = pd.DataFrame(all_data)
    df.sort_values(by="hbond_score", ascending=False, inplace=True)
    log.info(f"Processed {len(df)} total interactions")

    # Save results
    df.to_json("rna_ligand_interactions.json", orient="records")


@cli.command()
def find_unique_ligand_interactions():
    df = pd.read_json("rna_ligand_interactions.json")
    df["duplicate"] = -1
    dup_count = 0
    for pdb_id, g in df.groupby("pdb_id"):
        if len(g) == 1:
            continue
        try:
            residues = get_cached_residues(pdb_id)
        except:
            print("missing residues", pdb_id)
            continue
        all_res_objs = []
        pos = []
        # Keep track of duplicates
        duplicate_groups = []
        current_group = []

        # Store residue objects and their row indices
        for i, row in g.iterrows():
            res_1 = row["ligand_res"]
            res_objs = [residues[res_1]] + [
                residues[r] for r in row["interacting_residues"]
            ]
            all_res_objs.append((res_objs, i))  # Store tuple of res_objs and index
            pos.append(i)

        # Group res_objs by length
        length_groups = {}
        for res_objs, idx in all_res_objs:
            length = len(res_objs)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((res_objs, idx))  # Store tuple

        # Compare groups of same length
        for length, group in length_groups.items():
            if len(group) < 2:  # Need at least 2 to compare
                continue
            # Compare each pair in the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    coords1 = []
                    for res in group[i][0]:  # group[i][0] contains res_objs
                        coords1.extend(res.coords)
                    coords1 = np.array(coords1)
                    coords2 = []
                    for res in group[j][0]:  # group[j][0] contains res_objs
                        coords2.extend(res.coords)
                    coords2 = np.array(coords2)
                    if len(coords1) != len(coords2):
                        continue

                    # Superimpose and get RMSD
                    aligned_coords, final_rmsd, stats = pymol_align(coords1, coords2)
                    if final_rmsd > len(group[i][0]) * 0.2:
                        continue
                    dup_count += 1
                    idx1, idx2 = group[i][1], group[j][1]  # Get original row indices
                    # Add both indices to current group if not already in a group
                    if not any(idx1 in g or idx2 in g for g in duplicate_groups):
                        current_group = [idx1, idx2]
                        duplicate_groups.append(current_group)
                    # Add index to existing group if other index is already in a group
                    else:
                        for g in duplicate_groups:
                            if idx1 in g and idx2 not in g:
                                g.append(idx2)
                            elif idx2 in g and idx1 not in g:
                                g.append(idx1)

        # Set duplicate flag to first member in each group
        for group in duplicate_groups:
            for idx in group[1:]:  # Skip first member
                df.at[idx, "duplicate"] = group[
                    0
                ]  # Set duplicate to first member's index
        for group in duplicate_groups:
            print(f"Group: {group}")

    df.to_json("rna_ligand_interactions_w_duplicates.json", orient="records")


@cli.command()
def get_final_summaries():
    df = pd.read_json("rna_ligand_interactions_w_duplicates.json")
    res_mapping = {}
    count = 0
    df["interacting_motifs"] = [[] for _ in range(len(df))]
    df["num_motifs"] = [0 for _ in range(len(df))]
    for pdb_id, g in df.groupby("pdb_id"):
        print(pdb_id)
        motifs = get_cached_motifs(pdb_id)
        res_mapping[pdb_id] = generate_res_motif_mapping(motifs)
        for i, row in g.iterrows():
            interacting_motifs = []
            for res in row["interacting_residues"]:
                if res not in res_mapping[pdb_id]:
                    print("missing", res)
                if res_mapping[pdb_id][res] not in interacting_motifs:
                    interacting_motifs.append(res_mapping[pdb_id][res])
            df.at[i, "interacting_motifs"] = interacting_motifs
            df.at[i, "num_motifs"] = len(interacting_motifs)
    df.to_json("rna_ligand_interactions_w_motifs.json", orient="records")


@cli.command()
def get_motif_summary():
    RELEASE_PATH = os.path.join("release", "ligand_interactions")
    os.makedirs(RELEASE_PATH, exist_ok=True)
    df = pd.read_json("rna_ligand_interactions_w_motifs.json")
    data = []
    for pdb_id, g in df.groupby("pdb_id"):
        motifs = get_cached_motifs(pdb_id)
        all_residues = get_cached_residues(pdb_id)
        motif_by_name = {m.name: m for m in motifs}
        for i, row in g.iterrows():
            for m_name in row["interacting_motifs"]:
                m = motif_by_name[m_name]
                atom_names = []
                coords = []
                residues = []
                for r in m.get_residues():
                    atom_names.append(r.atom_names)
                    coords.append(r.coords)
                    residues.append(r.get_str())
                is_duplicate = row["duplicate"] != -1
                data.append(
                    {
                        "pdb_id": pdb_id,
                        "ligand_res_id": row["ligand_res"],
                        "motif_id": m.name,
                        "motif_type": m.mtype,
                        "motif_topology": m.size,
                        "motif_sequence": m.sequence,
                        "residues": residues,
                        "num_residues": len(residues),
                        "atom_names": atom_names,
                        "coords": coords,
                        "ligand_coords": [all_residues[row["ligand_res"]].coords],
                        "ligand_atom_names": [
                            all_residues[row["ligand_res"]].atom_names
                        ],
                        "is_duplicate": is_duplicate,
                    }
                )
    df = pd.DataFrame(data)
    df.to_json(
        os.path.join(RELEASE_PATH, "all_motif_ligand_interactions_w_coords.json"),
        orient="records",
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(RELEASE_PATH, "all_motif_ligand_interactions_w_coords.json")
        )
    )
    df = df.query("is_duplicate == 0")
    df.to_json(
        os.path.join(
            RELEASE_PATH, "non_redundant_motif_ligand_interactions_w_coords.json"
        ),
        orient="records",
    )
    os.system(
        "gzip -9 -f {}".format(
            os.path.join(
                RELEASE_PATH, "non_redundant_motif_ligand_interactions_w_coords.json"
            )
        )
    )


if __name__ == "__main__":
    cli()
