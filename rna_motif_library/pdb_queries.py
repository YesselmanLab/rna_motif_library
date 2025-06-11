import requests
import json
import sys
import time
import concurrent.futures
from typing import List, Dict

from rna_motif_library.logger import get_logger

log = get_logger("pdb-queries")


def get_structure_resolutions(pdb_ids, batch_size=100):
    """
    Fetch resolution data for a batch of PDB structures using the RCSB GraphQL API.

    Args:
        pdb_ids: List of PDB IDs to fetch data for
        batch_size: Number of structures to fetch per API request

    Returns:
        List of tuples containing (pdb_id, resolution) pairs
    """
    structures = []

    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i : i + batch_size]
        log.info(
            f"Fetching resolutions for structures {i+1}-{i+len(batch)} of {len(pdb_ids)}..."
        )

        # Construct URL for batch
        ids_param = ",".join(batch)
        url = f"https://data.rcsb.org/graphql"

        # GraphQL query for resolution data
        query = """
        query($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            rcsb_entry_info {
              resolution_combined
            }
          }
        }
        """

        try:
            response = requests.post(
                url, json={"query": query, "variables": {"ids": batch}}
            )
            response.raise_for_status()
            data = response.json()

            # Extract resolution for each structure
            for entry in data.get("data", {}).get("entries", []):
                pdb_id = entry["rcsb_id"]
                try:
                    resolution = entry["rcsb_entry_info"]["resolution_combined"][0]
                    structures.append((pdb_id, resolution))
                except (KeyError, IndexError, TypeError):
                    log.warning(f"Could not get resolution for {pdb_id}")

            # Small delay to avoid overwhelming the server
            time.sleep(0.1)

        except Exception as e:
            log.error(f"Error fetching batch {i+1}-{i+len(batch)}: {str(e)}")

    return structures


def get_rna_structures(resolution_cutoff=3.5):
    """
    Search RCSB PDB for RNA structures with resolution better than specified cutoff.

    Args:
        resolution_cutoff: Maximum resolution in Angstroms (default 3.5)

    Returns:
        List of (pdb_id, resolution) tuples sorted by resolution
    """

    # Base URL for RCSB Search API
    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Construct query for RNA structures with resolution cutoff
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "negation": False,
                        "value": resolution_cutoff,
                    },
                },
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "entity_poly.rcsb_entity_polymer_type",
                                "operator": "exact_match",
                                "value": "RNA",
                            },
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid",
                                "operator": "greater",
                                "value": 0,
                            },
                        },
                    ],
                },
            ],
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry",
    }

    try:
        # Send POST request
        log.info("Sending request to RCSB PDB API...")
        response = requests.post(base_url, json=query)
        response.raise_for_status()

        # Parse response
        data = response.json()
        result_set = data.get("result_set", [])

        if not result_set:
            log.warning("No structures found matching criteria")
            return []

        log.info(f"Found {len(result_set)} structures matching criteria")

        # Extract PDB IDs
        pdb_ids = [result["identifier"] for result in result_set]

        # Fetch resolution data for all structures
        structures = get_structure_resolutions(pdb_ids)

        # Sort by resolution
        structures.sort(key=lambda x: x[1])

        return structures

    except requests.exceptions.RequestException as e:
        log.error(f"Error accessing RCSB API: {e}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            log.error(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        log.error(f"Error type: {type(e).__name__}")
        return []


def get_pdb_title(pdb_id):
    """
    Query the RCSB PDB API and extract the title from the primary citation.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            citations = data.get("citation", [])
            for c in citations:
                if c.get("rcsb_is_primary") == "Y":
                    return c.get("title", "").lower()
            return ""
        else:
            log.error(f"PDB ID {pdb_id}: HTTP {response.status_code}")
            return ""
    except Exception as e:
        log.error(f"PDB ID {pdb_id}: {e}")
        return ""


def get_pdb_titles_batch(pdb_ids: List[str], max_workers: int = 10) -> Dict[str, str]:
    """
    Get titles for multiple PDB IDs using parallel processing.
    
    Args:
        pdb_ids: List of PDB IDs to fetch titles for
        max_workers: Maximum number of concurrent threads (default: 10)
        
    Returns:
        Dictionary mapping PDB IDs to their titles
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping of future to pdb_id
        future_to_pdb = {executor.submit(get_pdb_title, pdb_id): pdb_id for pdb_id in pdb_ids}
        
        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                title = future.result()
                results[pdb_id] = title
            except Exception as e:
                log.error(f"Error processing {pdb_id}: {e}")
                results[pdb_id] = ""
    
    return results
