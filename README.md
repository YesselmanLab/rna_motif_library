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

# reduce for adding hydrogens to residues
conda install reduce -c bioconda



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
python rna_motif_library/cli.py generate-motifs
# Estimated time: 5 days for around 2000 .cifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will display the names of the motifs being processed


