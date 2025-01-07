import json
import click
import os

from rna_motif_library.motif import DATA_PATH
from rna_motif_library.motif import get_motifs_from_json, get_hbonds_and_basepairs


@click.command()
@click.argument("pdb_code")
def main(pdb_code):
    os.makedirs("hairpins", exist_ok=True)
    json_path = os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_code}.json")
    motifs = get_motifs_from_json(json_path)
    # _, basepairs = get_hbonds_and_basepairs(pdb_code)
    count = 0
    for motif in motifs:
        if "HAIRPIN" not in motif.name:
            continue
        has_helix = False
        end_basepair = motif.basepair_ends[0]
        shared_motif = None
        for m in motifs:
            if m.name == motif.name:
                continue
            if end_basepair in m.basepair_ends:
                if "HELIX" in m.name:
                    has_helix = True
                    break
                shared_motif = m
        if not has_helix:
            os.makedirs(f"hairpins/{count}", exist_ok=True)
            motif.to_cif(f"hairpins/{count}/{motif.name}.cif")
            if shared_motif is not None:
                shared_motif.to_cif(f"hairpins/{count}/{shared_motif.name}.cif")
            count += 1


if __name__ == "__main__":
    main()
