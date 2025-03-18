import requests
import json
import glob
import os
import pandas as pd

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.residue import get_cached_residues
from rna_motif_library.util import canon_res_list, get_pdb_ids


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


def get_ligand_info_from_pdb():
    molecule_files = glob.glob(os.path.join(DATA_PATH, "residues_w_h_cifs", "*.cif"))
    for molecule_file in molecule_files:
        mol_name = os.path.basename(molecule_file).split(".")[0]
        if mol_name in canon_res_list:
            continue
        print(mol_name)
        json_file = os.path.join(DATA_PATH, "ligand_info", f"{mol_name}.json")
        with open(json_file, "w") as f:
            json.dump(fetch_molecule_data(mol_name), f, indent=2)


# Example usage
if __name__ == "__main__":
    json_files = glob.glob(os.path.join(DATA_PATH, "ligand_info", "*.json"))
    has_phosphate = {}
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
        keep_data = {
            "id": mol_name,
            "type": data["chem_comp"]["chem_comp"]["type"],
            "name": data["chem_comp"]["chem_comp"]["name"],
            "formula": data["chem_comp"]["chem_comp"]["formula"],
            "formula_weight": data["chem_comp"]["chem_comp"]["formula_weight"],
            "smiles": data["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES"],
            "has_phosphate": has_phosphate[mol_name],
        }
        all_data.append(keep_data)
    df = pd.DataFrame(all_data)
    df.to_json("ligand_info.json", orient="records")
