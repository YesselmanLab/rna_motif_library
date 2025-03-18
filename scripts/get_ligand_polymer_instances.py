import requests
import json
import pandas as pd
from typing import Dict, Optional, Any, List


class PDBQuery:
    BASE_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

    @staticmethod
    def _create_base_request_options() -> Dict:
        return {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
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


def search_ligand_polymer_instances(ligand_id: str) -> Optional[Dict[str, Any]]:
    query = PDBQuery.create_ligand_polymer_query(ligand_id)
    results = PDBQuery.submit_query(query)
    return PDBQuery.get_pdb_ids(results)


def filter_non_redundant_pdb_ids(
    pdb_ids: Optional[List[str]], non_redundant_ids: List[str]
) -> Optional[List[str]]:
    if pdb_ids is None:
        return None
    filtered_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in non_redundant_ids]
    return filtered_ids if filtered_ids else None


if __name__ == "__main__":
    df = pd.read_json("ligand_info.json")
    df_non_redundant = pd.read_csv("data/csvs/non_redundant_set.csv")
    non_redundant_ids = df_non_redundant["pdb_id"].to_list()
    df["noncovalent_results"] = None
    df["polymer_results"] = None
    for i, row in df.iterrows():
        ligand_id = row["id"]
        noncovalent_results = search_noncovalent_ligand(ligand_id)
        polymer_results = search_ligand_polymer_instances(ligand_id)
        filtered_noncovalent_results = filter_non_redundant_pdb_ids(
            noncovalent_results, non_redundant_ids
        )
        filtered_polymer_results = filter_non_redundant_pdb_ids(
            polymer_results, non_redundant_ids
        )
        df.at[i, "noncovalent_results"] = filtered_noncovalent_results
        df.at[i, "polymer_results"] = filtered_polymer_results
    df.to_json("ligand_info_filtered.json", orient="records")
