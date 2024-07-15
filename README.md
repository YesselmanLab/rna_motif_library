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

## Error handling

I've found some bugs relating to bugged files that can be fixed by rerunning some code.
If you run into such problems, I have some remedies down below:<br>

```bash

# No error handling yet





### This error handling is old, I'll remove it if the new version doesn't work
# One such error is a JSON decode error
# You might run into this while generating motifs
# It will generate some error message that says something along the lines of "expected a bracket"
# You didn't do anything wrong, DSSR may have just bugged out whilst generating the JSON files
# To fix this, find the DSSR JSON outputs at data/dssr_output and delete the erroring JSON, then run the following again:
python cli.py process_dssr --threads 8
# Replace "8" with the number of CPU cores you want to use
# This should re-create the bugged file
# Next you want to resume generating the motifs from where it errored
# You should save the generated data first to save from being overwritten, so copy and paste the following files to another place in your computer:
# interactions.csv
# interactions_detailed.csv
# motif_residues_list.csv 
# Next run the command:
python cli.py generate_motifs --errored_count 8
# Replace "8" with the number printed on the left just before the error in a line that looks like this:
# 59, /path/to/your/rna_motif_library/data/pdbs/5M3H.cif, 5M3H
# ^ If this errored, you would enter 59 for the argument
# If this happens multiple times, save all versions of the CSV files
# When it is done running, copy/paste the CSV data from the saved files into the newly generated CSV


# Another such error is a blank or otherwise bugged .out file
# In this case just run the SNAP command again:
python cli.py process_snap --threads 8
# Replace "8" with the number of CPU cores you want to use
# This should fix any empty/bugged .out files


```

