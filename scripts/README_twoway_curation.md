# TWOWAY Motif Curation Pipeline

## Overview

The curation pipeline reduces ~16,800 TWOWAY motifs to ~456 representative motifs through sequential filtering. Every motif is tracked with its rejection reason and representative assignment.

## Pipeline

```bash
# Step 1: Filter and curate motifs
python scripts/filter_twoway_motifs.py \
    -o data/summaries/motifs/curated_twoway.csv \
    --tracking-output data/summaries/motifs/twoway_tracking.csv

# Step 2 (optional): Validate structures with ViennaRNA
python scripts/validate_twoway_structures.py \
    -i data/summaries/motifs/curated_twoway.csv \
    -o data/summaries/motifs/validated_twoway.csv
```

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `curated_twoway.csv` | ~456 | Kept representative motifs |
| `twoway_tracking.csv` | ~16,806 | All motifs with status, rejection reason, and representative |

## Rejection Reasons

| Reason | Count | Description |
|--------|-------|-------------|
| `too_many_residues` | ~1,592 | More than 15 residues |
| `sequence_dedup` | ~11,038 | Duplicate sequence within same topology (lower resolution) |
| `topology_bp_sig_dedup` | ~1,666 | Duplicate topology+basepair signature (lower resolution) |
| `topology_cap` | ~255 | Exceeded per-topology cap of 15 (selected by RMSD diversity) |
| `no_bp_topology_cap` | ~1,799 | Exceeded per-topology cap of 3 for motifs with no cross-strand basepairs |

## Viewing Clusters in PyMOL

### View a representative with all its rejected members

Find a representative motif ID from the curated CSV, then view everything that mapped to it:

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.csv \
    -r TWOWAY-1-1-AGC-GUU-4V9F-2 \
    -o cluster_view

pymol cluster_view/view.pml
```

This loads the representative (green) and all rejected motifs aligned by C1' superposition, colored by rejection reason and grouped for easy toggling.

### Filter by rejection reason

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.csv \
    -r TWOWAY-1-1-AGC-GUU-4V9F-2 \
    -o cluster_view \
    --reason sequence_dedup
```

### View all kept motifs for a topology

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.csv \
    --topology 3-3 \
    -o topo_view

pymol topo_view/view.pml
```

All kept motifs for that topology are aligned to the highest-resolution one (green). Useful for inspecting structural diversity within a topology.

### Limit cluster size

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.csv \
    -r TWOWAY-1-1-AGC-GUU-4V9F-2 \
    -o cluster_view \
    --max-members 20
```

### PyMOL tips

In the generated `view.pml`:
- The representative is colored **green**
- Rejected motifs are colored by rejection reason and placed in groups (`grp_sequence_dedup`, etc.)
- Toggle groups on/off: `disable grp_sequence_dedup` / `enable grp_sequence_dedup`
- Each motif has a comment with its metadata (sequence, topology, basepairs, resolution, etc.)

## CSV Column Reference

### Curated CSV (21 columns)

| Column | Example | Description |
|--------|---------|-------------|
| `motif_id` | `TWOWAY-1-1-AGC-GUU-4V9F-2` | Unique motif identifier |
| `pdb_id` | `4V9F` | Source PDB structure |
| `motif_sequence` | `AGC-GUU` | Full sequence (strands separated by `-`) |
| `motif_topology` | `3-3` | Strand lengths |
| `strand1_sequence` | `AGC` | 5' strand sequence |
| `strand2_sequence` | `GUU` | 3' strand sequence |
| `is_bulge` | `True` | One strand has length 0 |
| `bp_sig` | `AG_cWW` | Cross-strand basepair signature |
| `basepairs` | `[["AG","cWW"]]` | Cross-strand basepairs as JSON |
| `closing_basepairs` | `[["CG","cWW"],["GU","cWW"]]` | 5' and 3' closing basepairs as JSON |
| `resolution` | `2.1` | PDB resolution in angstroms |
| `num_residues` | `6` | Total residue count |
| `num_non_canonical_basepairs` | `1` | Non-cWW basepairs in motif |
| `num_hbonds` | `8` | Total hydrogen bonds |
| `num_protein_hbonds` | `0` | Protein-RNA hydrogen bonds |
| `num_ligand_hbonds` | `0` | Ligand-RNA hydrogen bonds |
| `in_tertiary_contact` | `0` | Whether motif participates in tertiary contact |
| `num_tertiary_contacts` | `0` | Number of tertiary contacts |
| `has_non_canonical_residue` | `0` | Contains modified nucleotides |
| `has_non_canonical_basepair_flank` | `0` | Flanking basepair involves modified nucleotide |
| `is_isolatable` | `1` | Can be extracted as independent structural unit |

### Tracking CSV (24 columns)

All columns from the curated CSV plus:

| Column | Example | Description |
|--------|---------|-------------|
| `status` | `kept` / `rejected` | Whether the motif was kept |
| `rejection_reason` | `sequence_dedup` | Why it was rejected (empty if kept) |
| `representative_motif_id` | `TWOWAY-1-1-AGC-GUU-4V9F-2` | The kept motif this maps to (empty if kept or too_many_residues) |
