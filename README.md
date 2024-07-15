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
# Create and activate new conda environment
conda create --name rna_motif_env python=3.8
conda activate rna_motif_env

# Clone the repository and navigate to project directory
git clone https://github.com/YesselmanLab/rna_motif_library.git
cd rna_motif_library

# Install the package
pip install .

```

## Creating the library

```bash
# Make sure you put the CSV in the right place or you will get errors
# Note, despite the use of "PDB" in language, all files are actually ".cif", not ".pdb"

# Navigate to directory
cd rna_motif_library

# To create the library first we need to download the PDBs specified in the CSV
python cli.py download_cifs --threads 8
# Replace "8" with the number of CPU cores you want to use

# After downloading we need to process with DSSR
python cli.py process_dssr --threads 8
# Replace "8" with the number of CPU cores you want to use

# After processing with DSSR we need to process with SNAP
python cli.py process_snap --threads 8
# Replace "8" with the number of CPU cores you want to use

# After processing with SNAP we need to generate motif files
python cli.py generate_motifs
# No threading for this one

# After generating motifs we find tertiary contacts
python cli.py find_tertiary_contacts
# No threading for this one

# Finally, make the figures
python cli.py make_figures
# No threading for this one

```

When finished, you will see several new directories, CSVs, and figures in the project directory. <br>
`motifs` - motifs found in the non-redundant set go here, categorized by type, size, and sequence
`interactions` - individual residues which hydrogen-bond with each other go here, classified by which residues are interacting
`tertiary_contacts` - tertiary contacts found go here, classified by what two types of motifs are in the contact

## Figure generation

The figures used were generated whilst running `update_library.py` using the default CSV inside the directory `data/csvs`.<br>
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
python cli.py generate_motifs --PDB 3R9X
# Replace "3R9X" with your desired PDB
```

```

