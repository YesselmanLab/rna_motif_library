import requests
import json
import sys
import time


def fetch_structure_data(pdb_ids, batch_size=100):
    """Fetch data for a batch of PDB structures"""
    structures = []

    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i : i + batch_size]
        print(
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
                    print(f"Warning: Could not get resolution for {pdb_id}")

            # Small delay to avoid overwhelming the server
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching batch {i+1}-{i+len(batch)}: {str(e)}")

    return structures


def search_rna_structures(resolution_cutoff=3.5):
    """
    Search PDB for RNA structures with resolution better than specified cutoff.
    Returns structures sorted by resolution.
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
                        "value": resolution_cutoff
                    }
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
                                "value": "RNA"
                            }
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid",
                                "operator": "greater",
                                "value": 0
                            }
                        }
                    ]
                }
            ]
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry"
    }

    try:
        # Send POST request
        print("Sending request to RCSB PDB API...")
        response = requests.post(base_url, json=query)
        response.raise_for_status()

        # Parse response
        data = response.json()
        result_set = data.get("result_set", [])

        if not result_set:
            print("No structures found matching criteria")
            return []

        print(f"Found {len(result_set)} structures matching criteria")

        # Extract PDB IDs
        pdb_ids = [result["identifier"] for result in result_set]

        # Fetch resolution data for all structures
        structures = fetch_structure_data(pdb_ids)

        # Sort by resolution
        structures.sort(key=lambda x: x[1])

        return structures

    except requests.exceptions.RequestException as e:
        print(f"Error accessing RCSB API: {e}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print("Response content:", e.response.text)
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        return []


def main():
    # Search for RNA structures
    print("Searching for RNA structures...")
    structures = search_rna_structures(3.51)

    if not structures:
        print("No structures found or an error occurred.")
        sys.exit(1)

    # Print results
    print(
        f"\nFound {len(structures)} RNA-containing structures with resolution < 3.5 Å"
    )
    print("\nTop 10 structures by resolution:")
    print("PDB ID  Resolution (Å)")
    print("-" * 20)
    for pdb_id, resolution in structures[:10]:
        print(f"{pdb_id}    {resolution:.2f}")

    # Save all results to file
    output_file = "rna_structures.txt"
    try:
        with open(output_file, "w") as f:
            f.write("PDB_ID,Resolution\n")
            for pdb_id, resolution in structures:
                f.write(f"{pdb_id},{resolution:.2f}\n")
        print(f"\nAll results saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    main()
