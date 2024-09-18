# RNA Motif Library

A Python package for RNA Motif Library creation.<br>
preprint: (insert paper link here)

## Data Download

A default CSV (`nrlist_3.262_3.5A.csv`) is in the directory `data/csvs`.<br>
Make sure to download the most recent data (3.5 Å resolution):<br>
http://rna.bgsu.edu/rna3dhub/nrlist

In the directory `data/csvs`, delete the default CSV file and replace with your download.<br>

## Installation

```bash
# Clone the repository and navigate to project directory
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Create and activate new conda environment
conda create --name rna_motif_env python=3.8
conda activate rna_motif_env

# Install the package
pip install .

```

## Creating the library


### download CIF files

```bash
# Make sure you put the downloaded CSV in the right place or you will get errors
# Put the CSV where the default CSV is and delete the default
# Note: despite the use of "PDB" in language, all files are actually ".cif", not ".pdb"
# ALWAYS CLEAR THE data/out_csvs DIRECTORY BEFORE RUNNING THE SCRIPT (if it exists)! Move the data somewhere else if you want to keep it.
# If this folder is not empty (its nonexistence is OK), there will be problems!

# To create the library first we need to download the PDBs specified in the CSV
python rna_motif_library/cli.py download-cifs --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 15 minutes for around 2000 .cifs
# Expect a progress bar when it's working

# After downloading we need to process with DSSR
python rna_motif_library/cli.py process-dssr --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 90 minutes for around 2000 .cifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will consist of the path to the PDB/CIF files

# After processing with DSSR we need to process with SNAP
python rna_motif_library/cli.py process-snap --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 9 hours for around 2000 .cifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will consist of the path + other information on nucleotides/etc

# After processing with SNAP we need to generate motif files
python rna_motif_library/cli.py generate-motifs --threads 8
# Replace "8" with the number of CPU cores you want to use
# Estimated time: 72 hours for around 2000 .cifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will display the names of the motifs being processed

# After generating motifs we find tertiary contacts
python rna_motif_library/cli.py load-tertiary-contacts
# No threading for this one
# Estimated time: 18 hours for around 2000 .cifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will display which motifs' hydrogen bonding it's looking at

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
# Make sure to first delete the directories "data/motifs", "data/interactions", "data/tertiary_contacts", and "data/out_csvs" first so data doesn't overlap
python cli.py generate-motifs --limit 8
# Replace "8" with your desired number
# This will always run certain files first; the order is not random, but fixed every time
```

If you are interested in a specific PDB, you can run the following:

```bash
# Make sure to delete the directories "motifs", "interactions", "tertiary_contacts", "heatmaps", and "heatmap_data" if you've run the full code already
# Make sure your file is within the nonredundant set
# Look for "PDB_name.json" and "PDB_name.out" in /dssr_output and /snap_output
python cli.py generate-motifs --pdb 3R9X
# Replace "3R9X" with your desired PDB
```

## Error handling

You may get a very unusual error involving DSSR (or other aspects of the program) that I have yet to discover.<br>
In that case, remove the offending `.cif`, `.json`, and `.out` from `data/pdbs`, `data/dssr_output` , and `data/snap_output`
and `data/snap_output`, before running the `generate_motifs` command again.<br>
This will remove the offending PDB from the end data set.<br>
I have added automated error handling for this, but if it doesn't work (ends up in an infinite loop), or other errors come up, contact us at (email), and send the traceback in a .txt file, along with the files you removed.

## CSV documentation

When running is finished you may see a number of new CSVs with data inside the package directory.<br>
Here we will describe the most important CSVs.

TODO update this section post-refactor