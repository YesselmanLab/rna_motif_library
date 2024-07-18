# RNA Motif Library

A Python package for RNA Motif Library creation.<br>
preprint: (insert paper link here)

## Data Download

A default CSV (`nrlist_3.262_3.5A.csv`) is in the directory `data/csvs`.<br>
Make sure to download the most recent data (3.5 Å resolution):<br>
http://rna.bgsu.edu/rna3dhub/nrlist

In the directory `data/csvs`, delete the default CSV file and replace with your download.<br>
Delete the directory `data/pdbs`, it was used to create the results of the paper.<br>
Optionally you can run the program with these files to reproduce the results.<br>

## Installation

```bash
# Create and activate new conda environment
conda create --name rna_motif_env python=3.8
conda activate rna_motif_env

# Clone the repository and navigate to project directory
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Install the package
pip install .

# Verify installation
python -m pip list

```

## Creating the library

```bash
# Make sure you put the CSV in the right place or you will get errors
# Note, despite the use of "PDB" in language, all files are actually ".cif", not ".pdb"

# To create the library first we need to download the PDBs specified in the CSV
python rna_motif_library/cli.py download_cifs --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 15 minutes for around 2000 .cifs

# After downloading we need to process with DSSR
python rna_motif_library/cli.py process_dssr --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 40 minutes for around 2000 .cifs

# After processing with DSSR we need to process with SNAP
python rna_motif_library/cli.py process_snap --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 12 hours for around 2000 .cifs

# After processing with SNAP we need to generate motif files
python rna_motif_library/cli.py generate_motifs
# No threading for this one
# Estimated time: 72 hours for around 2000 .cifs

# After generating motifs we find tertiary contacts
python rna_motif_library/cli.py find_tertiary_contacts
# No threading for this one
# Estimated time: 24-36 hours for around 2000 .cifs

```

When finished, you will see several new directories. <br>
`data/motifs` - motifs found in the non-redundant set go here, categorized by type, size, and sequence
`data/interactions` - individual residues which hydrogen-bond with each other go here, classified by which residues are
interacting
`data/tertiary_contacts` - tertiary contacts found go here, classified by what two types of motifs are in the contact
`data/out_csvs` - CSVs with further data go here

Note: folders in `data/motifs` named `nways` refer to n-way junctions (2ways, 3ways, etc)

## Figure generation

The figures used were generated whilst running `update_library.py` using the default CSV inside the
directory `data/csvs`.<br>
For further details, check out `figure_plotting.py`.

Figures 2 and 3 are PNGs; they are in the project directory.<br>
Figure 4 also consists of PNGs, however, every interaction/atom combination gets its own figure.<br>
These figures can be found in the directory `heatmaps`.<br>
Data for each respective figure is broken down in CSV files, which are in `heatmap_data`.<br>

## Other functions

If you are interested in only a certain number of PDBs, you can run the following:

```bash
# Make sure to delete the directories "motifs", "interactions", "tertiary_contacts", "heatmaps", and "heatmap_data" if you've run the full code already
python cli.py generate_motifs --limit 8
# Replace "8" with your desired number
```

If you are interested in a specific PDB, you can run the following:

```bash
# Make sure to delete the directories "motifs", "interactions", "tertiary_contacts", "heatmaps", and "heatmap_data" if you've run the full code already
# Make sure your file is within the nonredundant set
# Look for "PDB_name.json" and "PDB_name.out" in /dssr_output and /snap_output
python cli.py generate_motifs --pdb 3R9X
# Replace "3R9X" with your desired PDB
```

## Error handling

You may get a very unusual error involving DSSR (or other aspects of the program) that I have yet to discover.<br>
In that case, remove the offending `.cif`, `.json`, and `.out` from `data/pdbs`, `data/dssr_output` , and `data/snap_output`
and `data/snap_output`, before running the `generate_motifs` command again.<br>
This will remove the offending PDB from the end data set.<br>
If such errors do come up, contact us at (email), and send the traceback in a .txt file, along with the files you removed.

## CSV documentation

When running is finished you may see a number of new CSVs with data inside the package directory.<br>
Here I will describe the most important CSVs.

interactions.csv - shows the size of each motif (in nucleotides) and number of each type (base:base/sugar/phos/aa/etc)
of interaction within a motif

- columns: (name,type,size,base:base,base:sugar,base:phos,sugar:base,sugar:sugar,sugar:phos,phos:base,phos:sugar,phos:
  phos,base:aa,sugar:aa,phos:aa)
- name: name of motif
- type: type of motif (SSTRAND/HELIX/HAIRPIN/NWAY/TWOWAY)
- size: number of nucleotides in the motif
- base:base: # of base-base interactions involving the motif
- base:sugar: # of base-sugar interactions involving the motif
- all further columns in the CSV follow a similar pattern

interactions_detailed.csv - shows detailed information about each interaction found/listed in interactions.csv (though
not by name)

- columns: (name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2)
- name: name of motif
- res_1: residue #1 in the interaction
- res_2: residue #2 in the interaction
- res_1_name: residue type (what base/amino acid is it? A/U/C/G/LEU/etc)
- res_2_name: same as res_1_name
- atom_1: the exact atom inside res_1 interacting with atom_2
- atom_2: the exact atom inside res_2 interacting with atom_1
- distance: distance between atom_1 and atom_2
- angle: dihedral angle between residues at atom_1 and atom_2, along with the closest atoms they are connected to on
  their respective residues
- nt_1: is res_1 an amino acid or nucleotide (nt/aa)
- nt_2: is res_2 an amino acid or nucleotide (nt/aa)

unique_tert_contacts.csv - shows detailed information about tertiary contacts

- columns: (seq_1,seq_2,motif_1,motif_2,type_1,type_2,res_1,res_2,count)
- seq_1: sequence of motif_1
- seq_2: sequence of motif_2
- motif_1: name of motif_1
- motif_2: name of motif_2
- type_1: type of motif_1 (hairpin, helix, 2way/nway junction, single strand)
- type_2: type of motif_2 (hairpin, helix, 2way/nway junction, single strand)
- res_1: was used in intermediate processes, irrelevant here; is the name of one of the residues in motif_1 involved in
  the tertiary contact
- res_2: was used in intermediate processes, irrelevant here; is the name of one of the residues in motif_2 involved in
  the tertiary contact
- count: number of interactions (h-bonds) between motif_1 and motif_2 in the tertiary contact

twoway_motif_list.csv - shows surface level information about two-way junctions and is used to make a heatmap showing
sizes of two-way junctions

- columns: (motif_name, motif_type, nucleotides_in_strand_1, nucleotides_in_strand_2, bridging_nts_0, bridging_nts_1)
- motif_name: name of the two-way junction
- motif_type: original classification of junction; irrelevant
- nucleotides_in_strand_1: nucleotides in strand 1 of junction; retrieved from raw counting of strand NTs
- nucleotides_in_strand_2: nucleotides in strand 2 of junction; retrieved from raw counting of strand NTs
- bridging_nts_0: nucleotides in strand 1 of junction - 2; used to classify junction size
- bridging_nts_1: nucleotides in strand 2 of junction - 2; used to classify junction size




