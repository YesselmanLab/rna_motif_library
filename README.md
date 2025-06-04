# RNA Motif Library

A Python package for RNA Motif Library creation.<br>
preprint: (insert paper link here)

## Download the current library



## Understanding the json files

### Understanding the motif indentifiers

All motifs are indentified as strings as `mtype-msize-msequence-pdb_id`. This is a unique format that may be visually unappealing but is easy to work with and is easy to parse. 

`mtype` is the motif type (like HAIRPIN, HELIX, etc).
`msize` is the motif size (like 1, 2, 3, etc).
`msequence` is the motif sequence (like A, C, G, U, etc).
`pdb_id` is the PDB ID (like 1GID, 1GID, etc).

### Understanding the residue indentifiers

All residues are indentified as strings as `chain_id-res_id-res_num-ins_code`. This is a unique format that may be visually unappealing but is easy to work with and is easy to parse. These will all match the CIF files downloaded from the PDB.

`chain_id` is the chain identifier (auth_asym_id in CIF format).
`res_id` is the residue identifier (auth_comp_id in CIF format).
`res_num` is the residue number (auth_seq_id in CIF format).
`ins_code` is the insertion code (pdbx_PDB_ins_code in CIF format), most of the time this is empty.

Although trival to parse there is a function in `util.py` named `parse_residue_identifier` that can be used to parse the residue identifier into a dictionary.


## Generating the database 

```bash
# STEP 1: Get the latest RNA 3D Hub release
python rna_motif_library/setup_database.py get-atlas-release

# STEP 2: Generate a list of RNA PDBs with resolution better than 3.5Å
python rna_motif_library/setup_database.py get-all-rna-pdbs

# STEP 3: Download the PDBs
python rna_motif_library/setup_database.py download-cifs data/csvs/rna_structures.csv

# STEP 4: Process the PDBs with DSSR
python rna_motif_library/setup_database.py generate-dssr-outputs data/csvs/rna_structures.csv



```



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
# need DSSR installed which is not included here 
python rna_motif_library/cli.py process-dssr --threads 8
# Replace "8" with the number of CPU cores you want to use
# There will be visual feedback in the terminal window if it's working properly
# Feedback will consist of the path to the PDB/CIF files

# After processing with DSSR we need to process with SNAP
python rna_motif_library/cli.py process-snap --threads 8
# Replace "8" with the number of CPU cores you want to use
# There will be visual feedback in the terminal window if it's working properly
# Feedback will consist of the path + other information on nucleotides/etc

# After processing with SNAP we need to generate motif files
python rna_motif_library/cli.py generate-motifs
# There will be visual feedback in the terminal window if it's working properly
# Feedback will display the names of the motifs being processed

# ligand stuff 
python rna_motif_library/ligand.py find-all-potential-ligands
python rna_motif_library/ligand.py get-ligand-cifs
python rna_motif_library/ligand.py get-hbond-donors-and-acceptors
python rna_motif_library/ligand.py get-ligand-info
python rna_motif_library/ligand.py get-ligand-polymer-instances
