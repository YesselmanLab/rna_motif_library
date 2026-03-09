# TWOWAY Motif Curation Pipeline

## Overview

The curation pipeline reduces ~16,800 TWOWAY motifs from the non-redundant structure set down to ~359 representative motifs through sequential filtering. Every motif is tracked with its rejection reason and representative assignment.

## Algorithm

The pipeline applies the following steps in order. At each step, rejected motifs are recorded with a reason and mapped to their nearest kept representative.

### Step 1a: Isolatable filter

Removes motifs that cannot be extracted as independent structural units. A motif is "isolatable" if it is not heavily entangled with protein contacts, tertiary contacts, or other structural dependencies that would make it non-functional in isolation. This is the largest single filter, removing ~82% of all TWOWAY motifs.

- **Cutoff**: `is_isolatable == 1` (required)
- **Rejection reason**: `not_isolatable`

### Step 1b: Max residue filter

Removes very large motifs that are unlikely to be useful as modular building blocks.

- **Cutoff**: `num_residues <= 15`
- **Rejection reason**: `too_many_residues`

### Step 2: Sequence deduplication

Groups motifs by their full sequence (including closing basepair residues) and keeps only the one with the best (lowest) crystallographic resolution. Duplicate sequences from different PDB structures are redundant.

- **Keeps**: best resolution per unique `motif_sequence`
- **Rejection reason**: `sequence_dedup`

### Step 2.5: A/C internal residue filter

Requires that every motif has at least one adenine (A) or cytosine (C) among its internal (unpaired) residues. Motifs with only G/U in the loop are less useful for library design since A and C provide more chemical diversity and are more commonly found in functional internal loops and bulges.

Internal residues are determined by stripping the first and last nucleotide from each strand (the closing basepair residues). For example, in `CUUUA-UGAAG` (topology 3-3), the internal residues are `UUU` and `GAA` — this passes because `A` is present.

- **Cutoff**: at least one A or C in internal positions
- **Rejection reason**: `no_ac_internal`

### Step 2.6: ViennaRNA secondary structure validation

Each remaining motif is embedded in a hairpin test construct (5 bp random Watson-Crick helices flanking the motif, capped with a GAAA tetraloop) and folded with ViennaRNA. The predicted structure is compared to the expected structure (closing pairs paired, internal residues unpaired) to compute an accuracy score.

This accuracy is not used as a hard filter but as a **sorting preference**: motifs with <=6 residues and accuracy >= 0.9 are given priority in all subsequent selection steps. This ensures that small, well-folding motifs are favored when choosing representatives.

- **Cutoff**: no hard cutoff; used for sorting priority
- **Priority**: `num_residues <= 6` AND `accuracy >= 0.9` → priority 0 (best)
- **Helix length**: 5 bp
- **Random seed**: 42

### Step 3: Split into with-basepair and no-basepair populations

Motifs are split based on whether they have any basepairs between nucleotides on opposite strands of the internal loop. Only basepairs with a hydrogen bond score >= 0.5 are counted — this threshold filters out weak or marginal interactions so that only confident basepair assignments influence the classification. A motif like a single-nucleotide bulge with no cross-strand interactions goes into the "no-basepair" population, while an internal loop with a sheared G-A pair (cSH) goes into "with-basepair."

The two populations are handled differently because motifs with internal basepairs have more structural diversity (different basepair types and geometries) that needs to be preserved.

- **Basepair hydrogen bond score threshold**: `>= 0.5`

### Step 4: With-basepair — topology + basepair signature dedup

Within the with-basepair population, motifs are grouped by (topology, basepair_signature). The basepair signature is the sorted, comma-joined Leontis-Westhof types of all cross-strand basepairs (e.g., `cWW,tSH`). Only the best-resolution motif per group is kept.

Sorting within groups prioritizes:
1. Small motifs with good ViennaRNA accuracy (priority 0)
2. Higher A/C count in internal positions
3. Better resolution
4. Motif ID (deterministic tiebreaker)

- **Rejection reason**: `topology_bp_sig_dedup`

### Step 5: With-basepair — topology cap via RMSD diversity (disabled by default)

This step is **disabled by default** because the topology + basepair signature dedup in Step 4 already keeps exactly one motif per unique (topology, basepair_signature) combination, which preserves all structurally distinct motif types. Applying a per-topology cap on top of this would discard motifs with unique basepair signatures.

If enabled via `--topo-cap N`, a farthest-point greedy algorithm selects the maximally diverse subset based on pairwise C1' RMSD:

1. Compute all pairwise RMSDs (C1' superposition) within the topology
2. Start with the best-resolution motif
3. Iteratively add the motif that is farthest from all already-selected motifs
4. Stop when the cap is reached

Rejected motifs are mapped to their nearest selected representative.

- **Default**: disabled (`--topo-cap` not set)
- **RMSD computation**: C1' atom superposition
- **Workers**: 20 parallel processes
- **Rejection reason**: `topology_cap`

### Step 6: No-basepair — topology cap

For motifs without cross-strand basepairs (bulges and simple internal loops), a smaller per-topology cap is applied since these motifs have less structural diversity.

Sorting prioritizes:
1. Small motifs with good ViennaRNA accuracy
2. A/C bulge residues (for bulge motifs: all-A/C internal residues preferred)
3. Higher A/C count in internal positions
4. Better resolution

- **Cutoff**: `no_bp_per_topo = 3` motifs per topology
- **Rejection reason**: `no_bp_topology_cap`

### Post-processing: Extended sequence and structure

For all kept motifs, the pipeline loads the 3D structural data to find the flanking helix basepair immediately outside the motif. This adds one additional basepair context on each side, producing:

- `sequence`: the motif sequence plus the flanking helix pair (e.g., `CGCGG-CCCG`)
- `structure`: dot-bracket with both the motif closing pairs and the helix pair (e.g., `((.((&))))`)

The original motif-only sequence is preserved in `motif_sequence`.

## Filtering Summary (current run)

| Step | Removed | Remaining | Reason |
|------|---------|-----------|--------|
| Start | — | 16,806 | |
| Isolatable filter | 13,799 | 3,007 | `not_isolatable` |
| Max residues (<=15) | 103 | 2,904 | `too_many_residues` |
| Sequence dedup | 1,866 | 1,038 | `sequence_dedup` |
| A/C internal filter | 164 | 874 | `no_ac_internal` |
| ViennaRNA accuracy | 0 | 874 | (sorting only) |
| Split: with-bp / no-bp | — | 571 / 303 | |
| Topology+bp_sig dedup | 277 | 294 | `topology_bp_sig_dedup` |
| Topology cap | 0 | 294 | `topology_cap` (disabled) |
| No-bp topology cap (3) | 238 | 65 | `no_bp_topology_cap` |
| **Final** | | **359** | |

## Usage

```bash
# Step 1: Filter and curate motifs
python scripts/filter_twoway_motifs.py \
    -o data/summaries/motifs/curated_twoway.json \
    --tracking-output data/summaries/motifs/twoway_tracking.json

# Step 2 (optional): Validate structures with ViennaRNA (standalone)
python scripts/validate_twoway_structures.py \
    -i data/summaries/motifs/curated_twoway.json \
    -o data/summaries/motifs/validated_twoway.json
```

### Custom parameters

```bash
python scripts/filter_twoway_motifs.py \
    -o curated.json \
    --tracking-output tracking.json \
    --max-residues 12 \
    --no-bp-per-topo 5 \
    --bp-score-threshold 0.5 \
    --topo-cap 20 \
    --workers 10 \
    -v
```

## Output Files

| File | Records | Description |
|------|---------|-------------|
| `curated_twoway.json` | ~359 | Kept representative motifs |
| `twoway_tracking.json` | ~16,806 | All motifs with status, rejection reason, and representative |

## Viewing Clusters in PyMOL

### View a representative with all its rejected members

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.json \
    -r TWOWAY-1-1-GAC-GCC-7ECN-1 \
    -o cluster_view

pymol cluster_view/view.pml
```

### Filter by rejection reason

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.json \
    -r TWOWAY-1-1-GAC-GCC-7ECN-1 \
    -o cluster_view \
    --reason sequence_dedup
```

### View all kept motifs for a topology

```bash
python scripts/view_twoway_clusters.py \
    -t data/summaries/motifs/twoway_tracking.json \
    --topology 3-3 \
    -o topo_view

pymol topo_view/view.pml
```

## Column Reference

### Curated JSON

| Column | Example | Description |
|--------|---------|-------------|
| `motif_id` | `TWOWAY-1-1-GAC-GCC-7ECN-1` | Unique motif identifier |
| `pdb_id` | `4V9F` | Source PDB structure |
| `sequence` | `CAGCG-CGUUC` | Extended sequence with flanking helix pair |
| `structure` | `((.((&)))` | Dot-bracket for extended sequence |
| `motif_sequence` | `AGC-GUU` | Motif-only sequence (strands separated by `-`) |
| `motif_topology` | `1-1` | Internal loop strand lengths |
| `strand1_sequence` | `AGC` | 5' strand sequence |
| `strand2_sequence` | `GUU` | 3' strand sequence |
| `is_bulge` | `true` | One strand has 0 internal residues |
| `bp_sig` | `cWW` | Cross-strand basepair signature |
| `basepairs` | `[["AG","cWW"]]` | Cross-strand basepairs |
| `resolution` | `2.1` | PDB resolution in angstroms |
| `accuracy` | `1.0` | ViennaRNA prediction accuracy (0.0-1.0) |
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

### Tracking JSON

All columns from the curated JSON plus:

| Column | Example | Description |
|--------|---------|-------------|
| `status` | `kept` / `rejected` | Whether the motif was kept |
| `rejection_reason` | `sequence_dedup` | Why it was rejected (empty if kept) |
| `representative_motif_id` | `TWOWAY-1-1-GAC-GCC-7ECN-1` | The kept motif this maps to |
