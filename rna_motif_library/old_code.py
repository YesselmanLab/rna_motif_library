# this is a repository of old code I didn't want to delete
# delete later if found unnecessary

"""# delete specified directories if they exist
def safe_delete_dir(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception as e:
            print(f"Error deleting directory '{directory_path}': {e}")"""

canon_amino_acid_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# amino acid/canonical residue dictionary
canon_res_dict = {
    'A': 'Adenine',
    'ALA': 'Alanine',
    'ARG': 'Arginine',
    'ASN': 'Asparagine',
    'ASP': 'Aspartic Acid',
    'CYS': 'Cysteine',
    'C': 'Cytosine',
    'G': 'Guanine',
    'GLN': 'Glutamine',
    'GLU': 'Glutamic Acid',
    'GLY': 'Glycine',
    'HIS': 'Histidine',
    'ILE': 'Isoleucine',
    'LEU': 'Leucine',
    'LYS': 'Lysine',
    'MET': 'Methionine',
    'PHE': 'Phenylalanine',
    'PRO': 'Proline',
    'SER': 'Serine',
    'THR': 'Threonine',
    'TRP': 'Tryptophan',
    'TYR': 'Tyrosine',
    'U': 'Uracil',
    'VAL': 'Valine'
}

# questions
# about dihedral angles:
# does directionality matter for our purposes, as in, should I pay attention to +/- 180 degrees whist recording interaction angles?
# I don't think it does, as directionality depends purely on which atom you take first while calculating
# I'm more or less consistent about this, I think as long as we mention it in the methods it should be fine

# done:

# fix misclassification of motif types while finding tertiary contacts
# fix classification of n-way junctions so that it refers to the number of nucleotides in each strand
# fix duplicate motifs showing up in data that is plotted in tertiary contact plots
# put a gap between the bars on graphs and fix graph styles
# check if DA/C/U/G are numerous enough to include as sometimes they are incorrectly counted as nucleosides (they are not; over 36 PDBs I found around 30/5000)
# fix interactions.csv so it includes ALL interactions and properly counts them
# increase text sizes on all figures
# add a column to unique_tert_contacts.csv that records base/sugar/phos data (which part of things are contacts coming from)
# add figure that describes types of tertiary contacts by base/sugar/phos
# deleted "questionable" and "unknown" donAcc type h-bonds
# merged dist-angle data so reverse orders are the same (might need to minus angle by 180 for those) (this one might need a bit of work)
# multithreading (undone, this is slower than single-threading holy fuck)
# added single strands and their respective graphs; make sure tert contacts between others and SSTRAND works

# fix these:
# nothing to fix for now

# NOT CODE THINGS just things to remember
# get new real data and analyze
# meet with joe to discuss this thing again soon

# old code
"""def __generate_motif_files():
    # defines directories
    pdb_dir = settings.LIB_PATH + "/data/pdbs/"
    # rnp_dir = settings.LIB_PATH + "/data/snap_output"
    # grabs all the stuff
    pdbs = glob.glob(pdb_dir + "/*.cif")  # PDBs
    # rnps = glob.glob(rnp_dir + "/*.out")  # RNP interactions
    # creates directories
    dirs = [
        "motifs",
        "motif_interactions",
    ]
    for d in dirs:
        __safe_mkdir(d)
    motif_dir = "motifs/nways/all"
    hbond_vals = [
        "base:base",
        "base:sugar",
        "base:phos",
        "sugar:base",
        "sugar:sugar",
        "sugar:phos",
        "phos:base",
        "phos:sugar",
        "phos:phos",
        "base:aa",
        "sugar:aa",
        "phos:aa",
    ]
    # opens the file where information about nucleotide interactions are stored
    f = open("interactions.csv", "w")
    f.write("name,type,size")
    # writes to the CSV information about nucleotide interactions
    f.write(",".join(hbond_vals) + "\n")
    count = 0
    # CSV about ind. interactions
    f_inter = open("interactions_detailed.csv", "w")
    f_inter.write(
        "name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2" + "\n")

    # CSV listing all th residues present in a given motif
    f_residues = open("motif_residues_list.csv", "w")
    f_residues.write("motif_name,residues" + "\n")

    # CSV for twoway motifs
    f_twoways = open("twoway_motif_list.csv", "w")
    f_twoways.write(
        "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1" + "\n")

    # writes motif/motif interaction information to PDB files
    for pdb_path in pdbs:
        name = pdb_path.split("/")[-1][:-4]
        count += 1
        print(count, pdb_path, name)

        # debug, here we define which exact pdb to run (if we need to for whatever reason)
        # if pdb_path == "/Users/jyesselm/PycharmProjects/rna_motif_library/data/pdbs/7PKQ.cif": # change the part before .cif
        s = os.path.getsize(pdb_path)
        json_path = settings.LIB_PATH + "/data/dssr_output/" + name + ".json"
        # if s < 100000000:  # size-limit on PDB; enable if machine runs out of RAM

        # get RNP interactions
        rnp_out_path = settings.LIB_PATH + "/data/snap_output/" + name + ".out"
        # is a list of snap.RNPInteraction objects
        rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)

        # prepare list for RNP data
        rnp_data = []

        for interaction in rnp_interactions:
            # print(interaction)
            atom1, res1 = interaction.nt_atom.split("@")
            atom2, res2 = interaction.aa_atom.split("@")

            # list format: (res1, res2, atom1, atom2, distance)
            rnp_interaction_tuple = (res1, res2, atom1, atom2, str(interaction.dist))
            # rnp_data should be imported into interactions
            rnp_data.append(rnp_interaction_tuple)

        pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
        (
            motifs,
            motif_hbonds,
            motif_interactions, hbonds_in_motif
        ) = dssr.get_motifs_from_structure(json_path)

        # hbonds_in_motif is a list, describing all the chain.res ids with hbonds
        # some are counted twice so we need to purify to make it unique
        # RNP data from snap is injected here
        hbonds_in_motif.extend(rnp_data)
        unique_inter_motifs = list(set(hbonds_in_motif))

        # counting total motifs present in PDB
        dssr.total_motifs_count = len(motifs)
        # process each motif present in PDB
        for m in motifs:
            print(m.name)
            spl = m.name.split(".")  # this is the filename
            # don't run if these aren't in the motif name
            if not (spl[0] == "TWOWAY" or spl[0] == "NWAY" or spl[0] == "HAIRPIN" or spl[
                0] == "HELIX"):
                continue

            # Writing to interactions.csv
            f.write(m.name + "," + spl[0] + "," + str(len(m.nts_long)) + ",")

            # counting of # of hbond interactions (-base:base)
            if m.name not in motif_hbonds:
                vals = ["0" for _ in hbond_vals]
            else:
                vals = [str(motif_hbonds[m.name][x]) for x in hbond_vals]

            f.write(",".join(vals) + "\n")
            # if there are no interactions with the motif then it skips and avoids a crash
            try:
                interactions = motif_interactions[m.name]
            except KeyError:
                interactions = None  # this means that no interactions are found between the motif and other
            # Writing the residues AND interactions to the CIF files
            dssr.write_res_coords_to_pdb(
                m.nts_long, interactions, pdb_model,
                motif_dir + "/" + m.name, unique_inter_motifs, f_inter, f_residues, f_twoways
            )
        # count motifs found
        # get the PDB name
        ...
        spl_pdbs = pdb_path.split("/")
        pdb_name = spl_pdbs[6].split(".")[0]
        print("Motifs removed:")
        print(dssr.removed_motifs_count)

        if dssr.total_motifs_count != 0:
            print("Percentage removed:")
            percentage_removed = (dssr.removed_motifs_count / dssr.total_motifs_count) * 100
            print(percentage_removed)
        ...

    f.close()
    f_inter.close()
    f_twoways.close()"""

# new code; multi threaded; multi threads per PDB
"""def __generate_motif_files():
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))
    dirs = ["motifs", "motif_interactions"]
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = os.path.join("motifs", "nways", "all")
    hbond_vals = [
        "base:base", "base:sugar", "base:phos", "sugar:base", "sugar:sugar", "sugar:phos",
        "phos:base", "phos:sugar", "phos:phos", "base:aa", "sugar:aa", "phos:aa"
    ]

    # opens the file where information about nucleotide interactions are stored
    with open("interactions.csv", "w") as f:
        f.write("name,type,size," + ",".join(hbond_vals) + "\n")
        # CSV about individual interactions
        with open("interactions_detailed.csv", "w") as f_inter, \
                open("motif_residues_list.csv", "w") as f_residues, \
                open("twoway_motif_list.csv", "w") as f_twoways:

            f_inter.write("name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2,type_1,type_2\n")
            f_residues.write("motif_name,residues\n")
            f_twoways.write(
                "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1\n")

            # Use ThreadPoolExecutor for multithreading
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                count = 0
                for pdb_path in pdbs:
                    count += 1
                    future = executor.submit(process_pdb, pdb_path, motif_dir, hbond_vals, count)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        results = future.result()
                        for row, nts_long, interactions, unique_inter_motifs, pdb_model in results:
                            f.write(",".join(row) + "\n")
                            dssr.write_res_coords_to_pdb(nts_long, interactions, pdb_model,
                                                         os.path.join(motif_dir, row[0]),
                                                         unique_inter_motifs, f_inter, f_residues, f_twoways)
                    except Exception as exc:
                        print(f'Exception processing {future}: {exc}')"""

# new code; multi threads per motif
"""def __generate_motif_files():
    # defines directories
    pdb_dir = os.path.join(settings.LIB_PATH, "data/pdbs/")
    # grabs all the stuff
    pdbs = glob.glob(os.path.join(pdb_dir, "*.cif"))  # PDBs
    # creates directories
    dirs = ["motifs", "motif_interactions"]
    for d in dirs:
        __safe_mkdir(d)

    motif_dir = os.path.join("motifs", "nways", "all")
    hbond_vals = [
        "base:base", "base:sugar", "base:phos", "sugar:base", "sugar:sugar", "sugar:phos",
        "phos:base", "phos:sugar", "phos:phos", "base:aa", "sugar:aa", "phos:aa"
    ]

    # opens the file where information about nucleotide interactions are stored
    with open("interactions.csv", "w") as f:
        f.write("name,type,size," + ",".join(hbond_vals) + "\n")
        # CSV about individual interactions
        with open("interactions_detailed.csv", "w") as f_inter, \
                open("motif_residues_list.csv", "w") as f_residues, \
                open("twoway_motif_list.csv", "w") as f_twoways:

            f_inter.write("name,res_1,res_2,res_1_name,res_2_name,atom_1,atom_2,distance,angle,nt_1,nt_2\n")
            f_residues.write("motif_name,residues\n")
            f_twoways.write(
                "motif_name,motif_type,nucleotides_in_strand_1,nucleotides_in_strand_2,bridging_nts_0,bridging_nts_1\n")

            count = 0
            for pdb_path in pdbs:
                name = os.path.basename(pdb_path).replace(".cif", "")
                count += 1
                print(count, pdb_path, name)

                s = os.path.getsize(pdb_path)
                json_path = os.path.join(settings.LIB_PATH, "data/dssr_output", name + ".json")

                rnp_out_path = os.path.join(settings.LIB_PATH, "data/snap_output", name + ".out")
                rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)
                rnp_data = [(interaction.nt_atom.split("@")[1], interaction.aa_atom.split("@")[1],
                             interaction.nt_atom.split("@")[0], interaction.aa_atom.split("@")[0],
                             str(interaction.dist)) for interaction in rnp_interactions]

                pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
                motifs, motif_hbonds, motif_interactions, hbonds_in_motif = dssr.get_motifs_from_structure(json_path)

                hbonds_in_motif.extend(rnp_data)
                unique_inter_motifs = list(set(hbonds_in_motif))

                dssr.total_motifs_count = len(motifs)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for m in motifs:
                        spl = m.name.split(".")
                        if spl[0] not in ["TWOWAY", "NWAY", "HAIRPIN", "HELIX", "SSTRAND"]:
                            continue

                        future = executor.submit(process_motif, m, motif_hbonds, motif_interactions,
                                                 unique_inter_motifs, hbond_vals, pdb_model, motif_dir, f_inter,
                                                 f_residues, f_twoways)
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            row = future.result()
                            f.write(",".join(row) + "\n")
                        except Exception as exc:
                            print(f'Exception processing motif: {exc}')"""

"""def process_motif(m, motif_hbonds, motif_interactions, unique_inter_motifs, hbond_vals, pdb_model, motif_dir, f_inter,
                  f_residues, f_twoways):
    print(m.name)
    row = [m.name, m.name.split(".")[0], str(len(m.nts_long))]
    vals = [str(motif_hbonds[m.name][x]) if m.name in motif_hbonds else "0" for x in hbond_vals]
    row.extend(vals)

    interactions = motif_interactions.get(m.name, None)
    dssr.write_res_coords_to_pdb(m.nts_long, interactions, pdb_model,
                                 os.path.join(motif_dir, m.name),
                                 unique_inter_motifs, f_inter, f_residues, f_twoways)

    return row


def process_pdb(pdb_path, motif_dir, hbond_vals, count):
    name = os.path.basename(pdb_path).replace(".cif", "")
    print(count, pdb_path, name)

    json_path = os.path.join(settings.LIB_PATH, "data/dssr_output", name + ".json")

    rnp_out_path = os.path.join(settings.LIB_PATH, "data/snap_output", name + ".out")
    rnp_interactions = snap.get_rnp_interactions(out_file=rnp_out_path)

    rnp_data = [(interaction.nt_atom.split("@")[1], interaction.aa_atom.split("@")[1],
                 interaction.nt_atom.split("@")[0], interaction.aa_atom.split("@")[0],
                 str(interaction.dist), interaction.type.split(":")[0], interaction.type.split(":")[1]) for interaction in rnp_interactions]

    # rnp_data is the same format as other interactions

    pdb_model = PandasMmcifOverride().read_mmcif(path=pdb_path)
    motifs, motif_hbonds, motif_interactions, hbonds_in_motif = dssr.get_motifs_from_structure(json_path)

    hbonds_in_motif.extend(rnp_data)
    unique_inter_motifs = list(set(hbonds_in_motif))

    print(unique_inter_motifs)
    exit(0)

    results = []

    for m in motifs:
        print(m.name)
        spl = m.name.split(".")
        if spl[0] not in ["TWOWAY", "NWAY", "HAIRPIN", "HELIX", "SSTRAND"]:
            continue

        row = [m.name, spl[0], str(len(m.nts_long))]
        vals = [str(motif_hbonds[m.name][x]) if m.name in motif_hbonds else "0" for x in hbond_vals]
        row.extend(vals)

        # Process each motif concurrently
        f_inter = open("interactions_detailed.csv", "a")  # Append mode for concurrent writing
        f_residues = open("motif_residues_list.csv", "a")  # Append mode for concurrent writing
        f_twoways = open("twoway_motif_list.csv", "a")  # Append mode for concurrent writing

        interactions = motif_interactions.get(m.name, None)
        dssr.write_res_coords_to_pdb(m.nts_long, interactions, pdb_model,
                                     os.path.join(motif_dir, m.name),
                                     unique_inter_motifs, f_inter, f_residues, f_twoways)

        f_inter.close()
        f_residues.close()
        f_twoways.close()

        results.append((row, m.nts_long, interactions, unique_inter_motifs, pdb_model))

    return results"""

# Finally, for each group, make heatmaps of (distance,angle)
"""    for group in grouped_hbond_df:
    group_name = group[0]
    type_1 = str(group_name[0])
    type_2 = str(group_name[1])
    atom_1 = str(group_name[2])
    atom_2 = str(group_name[3])

    print(f"Processing {type_1}-{type_2} {atom_1}-{atom_2}")
    hbonds = group[1]
    hbonds_subset = hbonds[['distance', 'angle']]
    hbonds_subset = hbonds_subset.reset_index(drop=True)

    if (
            len(hbonds_subset) >= 100):  # & (len(hbonds_subset) <= 400): this limit existed before size limit was removed
        # Set global font size
        plt.rc('font', size=14)  # Adjust the font size as needed

        distance_bins = [i / 10 for i in range(20, 41)]  # Bins from 0 to 4 in increments of 0.1
        angle_bins = [i for i in range(0, 181, 10)]  # Bins from 0 to 180 in increments of 10

        hbonds_subset['distance_bin'] = pd.cut(hbonds_subset['distance'], bins=distance_bins)
        hbonds_subset['angle_bin'] = pd.cut(hbonds_subset['angle'], bins=angle_bins)

        heatmap_data = hbonds_subset.groupby(['angle_bin', 'distance_bin']).size().unstack(fill_value=0)

        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap_data, cmap='gray_r', xticklabels=1, yticklabels=range(0, 181, 10), square=True)

        plt.xticks(np.arange(len(distance_bins)) + 0.5, [f'{bin_val:.1f}' for bin_val in distance_bins], rotation=0)
        plt.yticks(np.arange(len(angle_bins)) + 0.5, angle_bins, rotation=0)

        plt.xlabel("Distance (angstroms)")
        plt.ylabel("Angle (degrees)")
        map_name = type_1 + "-" + type_2 + " " + atom_1 + "-" + atom_2
        plt.title(map_name + " H-bond heatmap")

        if len(type_1) == 1 and len(type_2) == 1:
            map_dir = "heatmaps/RNA-RNA"
        else:
            map_dir = "heatmaps/RNA-PROT"

        __safe_mkdir(map_dir)

        map_dir = map_dir + "/" + map_name
        plt.savefig(f"{map_dir}.png", dpi=250)
        plt.close()

        heatmap_csv_path = "heatmap_data"
        __safe_mkdir(heatmap_csv_path)

        heat_data_csv_path = heatmap_csv_path + "/" + map_name + ".csv"
        hbonds.to_csv(heat_data_csv_path, index=False)

        heatmap_res_names.append(map_name)
        heatmap_atom_names.append(len(hbonds_subset))

        # Insert the code for the 2D histogram here
        plt.figure(figsize=(10, 8))
        plt.hist2d(hbonds_subset['distance'], hbonds_subset['angle'], bins=[distance_bins, angle_bins],
                   cmap='gray_r')
        plt.xlabel("Distance (angstroms)")
        plt.ylabel("Angle (degrees)")
        plt.colorbar(label='Frequency')
        map_name = type_1 + "-" + type_2 + " " + atom_1 + "-" + atom_2
        plt.title(map_name + " H-bond heatmap")

        if len(type_1) == 1 and len(type_2) == 1:
            map_dir = "heatmaps/RNA-RNA"
        else:
            map_dir = "heatmaps/RNA-PROT"

        __safe_mkdir(map_dir)
        map_dir = map_dir + "/" + map_name
        # Save the 2D histogram as a PNG file
        plt.savefig(f"{map_dir}.png", dpi=250)
        # Sometimes the terminal might kill the process
        # if that happens lower the DPI setting above

        plt.close()  # Close the plot to prevent overlapping plots

        # Also print a CSV of the appropriate data
        heatmap_csv_path = "heatmap_data"
        __safe_mkdir(heatmap_csv_path)

        # set name
        heat_data_csv_path = heatmap_csv_path + "/" + map_name + ".csv"

        # print data
        hbonds.to_csv(heat_data_csv_path, index=False)

        # need to print a histogram of the number of data points in each heatmap
        # so need to collect this data first

        heatmap_res_names.append(map_name)
        heatmap_atom_names.append(len(hbonds_subset))
    else:
        print(f"Skipping {type_1}-{type_2} {atom_1}-{atom_2} due to insufficient or too many data points.")
"""

# after collecting data make the final histogram of all the data in heatmaps
# first compile the list into a df
# gonna comment this out because this is for debug purposes
# histo_df = pd.DataFrame({"heatmap": heatmap_res_names, "count": heatmap_atom_names})

# plot histogram
# plt.hist(histo_df.iloc[:, 1], bins=400)  # Adjust the number of bins as needed

# set labels
# plt.xlabel('# of datapoints inside a heatmap')
# plt.ylabel('# of heatmaps with X datapoints')
# plt.title('1d_histogram')

# set y-axis limit
# plt.ylim(0, max(df.iloc[:, 1]) * 0.9)

# plt.savefig('1d_histo.png')
# plt.close()

# old code I didn't want to fully delete

# constructs PDB DF
"""
pdb_columns = ['group_PDB', 'id', 'label_atom_id', 'label_comp_id',
                       'auth_asym_id', 'auth_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                       'occupancy', 'B_iso_or_equiv', 'type_symbol']
pdb_df = df[pdb_columns]
pdb_df_list.append(pdb_df)
"""

# separates CIF and PDB files after all is said and done
"""
def cif_pdb_sort(directory):
    # Create a copy of the directory with "_PDB" suffix
    directory_copy = directory + '_PDB'
    shutil.copytree(directory, directory_copy)

    # Iterate over the files in the copied directory
    for root, dirs, files in os.walk(directory_copy):
        for file in files:
            if file.endswith('.cif'):
                # Construct the file path
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)

    print(f".cif files deleted from {directory_copy}")

    # Iterate over the files in the original directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb'):
                # Construct the file path
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)

    print(f".pdb files deleted from {directory}")"""

# takes data from a dataframe and writes it to a PDB (deprecated)
"""
def dataframe_to_pdb(df, file_path):
    with open(file_path, 'w') as f:
        for row in df.itertuples(index=False):
            f.write("{:<5}{:>6}  {:<3} {:>3}{:>2}  {:>2}     {:>7} {:>7} {:>7}   {:>3} {:>3}         {:>3}\n".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                row[10], row[11]))
"""

# old code, restore if broken
"""def extract_continuous_chains(pair_list):
    chains = []

    for pair in pair_list:
        matched_chain = None

        for chain in chains:
            if chain[-1][1] == pair[0] or chain[0][0] == pair[1]:
                matched_chain = chain
                break
            elif chain[-1][1] == pair[1] or chain[0][0] == pair[0]:
                matched_chain = chain
                break

        if matched_chain:
            matched_chain.append(pair)

        else:
            chains.append([pair])

    return chains
"""

"""def connect_continuous_chains(chains):
    connected_chains = []

    for chain in chains:
        connected = False
        for current_chain in connected_chains:
            # chain is appended to current_chain

            if current_chain[-1][1] == chain[0][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][0] == chain[-1][1]:
                current_chain.insert(0, chain)
                connected = True
                break
            elif current_chain[-1][0] == chain[-1][1]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][1] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[0][0] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[-1][0] == chain[0][1]:
                current_chain.extend(chain)
                connected = True
                break
            elif current_chain[-1][1] == chain[-1][0]:
                current_chain.extend(chain)
                connected = True
                break

        if not connected:
            connected_chains.append(chain)

    return connected_chains
"""

"""def refine_continuous_chains(input_lists):
    merged = []

    def merge_lists(list1, list2):
        for sub_list1 in list1:
            for sub_list2 in list2:
                if any(item1 in sub_list2 for item1 in sub_list1):
                    list1.extend(sub_list2)
                    list2.clear()
                    return list1

    for i in range(len(input_lists)):
        current_list = input_lists[i]
        for j in range(i + 1, len(input_lists)):
            item = input_lists[j]
            if merge_lists(current_list, item):
                break

        if current_list:  # Check if current_list is not empty
            merged.append(current_list)

    return merged"""

"""def write_res_coords_to_pdb(nts, interactions, pdb_model, pdb_path, motif_bond_list, csv_file, residue_csv_list,
                            twoway_csv):
    # directory setup for later
    dir = pdb_path.split("/")
    sub_dir = dir[3].split(".")
    motif_name = dir[3]
    # motif extraction
    nt_list = []
    # list of residues
    res = []
    # convert the MMCIF to a dictionary, and the resulting dictionary to a Dataframe
    model_df_first = pdb_model.df
    # df to CSV for debug
    model_df_first.to_csv("model_df.csv", index=False)
    # keep only needed DF columns so further functions don't error
    columns_to_keep = ['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                       'label_alt_id', 'label_comp_id', 'label_asym_id',
                       'label_entity_id', 'label_seq_id',
                       'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                       'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge',
                       'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                       'auth_atom_id', 'pdbx_PDB_model_num']

    # keeps necessary columns from PDB file
    model_df = model_df_first[columns_to_keep]
    # model_df.to_csv("model_df.csv", index=False)
    # extracts identification data from nucleotide list
    for nt in nts:
        # r = DSSRRes(nt)
        # splits nucleotide names (chain_id, type + res_id)
        nt_spl = nt.split(".")
        # purify IDs; the first one is the chain_id, the second the res_id
        chain_id = nt_spl[0]
        residue_id = extract_longest_numeric_sequence(nt_spl[1])
        # if nt_spl[1] contains a '/' split; sometimes it's weird so this is here
        if "/" in nt_spl[1]:
            sub_spl = nt_spl[1].split("/")
            residue_id = sub_spl[1]
        # define nucleotide ID
        new_nt = chain_id + "." + residue_id
        # add it to the list of nucleotides being processed
        nt_list.append(new_nt)
    # sorts nucleotide list for further processing
    nucleotide_list_sorted, chain_list_sorted = group_residues_by_chain(
        nt_list)  # nt_list_sorted is a list of lists

    # this list is for strand-counting purposes; will help when determining N-way jcts
    list_of_chains = []
    # extraction of residues into dataframes
    for chain_number, residue_list in zip(chain_list_sorted, nucleotide_list_sorted):
        for residue in residue_list:
            # Find residue in the PDB model, first it picks the chain
            chain_res = model_df[model_df['auth_asym_id'].astype(str) == str(chain_number)]
            res_subset = chain_res[
                chain_res['auth_seq_id'].astype(str) == str(residue)]  # then it find the atoms
            res.append(res_subset)  # "res" is a list with all the residue DFs inside
        list_of_chains.append(res)

    df_list = []  # List to store the DataFrames for each line (type = 'list')
    res = remove_empty_dataframes(res)  # delete blank space
    for r in res:
        # Data reprocessing stuff, this loop is moving it into a DF
        lines = r.to_string(index=False, header=False).split('\n')
        for line in lines:
            values = line.split()  # (type 'values' = list)
            df = pd.DataFrame([values],
                              columns=['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                                       'label_alt_id', 'label_comp_id', 'label_asym_id',
                                       'label_entity_id', 'label_seq_id',
                                       'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                                       'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge',
                                       'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                                       'auth_atom_id', 'pdbx_PDB_model_num'])
            df_list.append(df)

    if df_list:  # i.e. if there are things inside df_list:
        # Concatenate all DFs into a single DF
        result_df = pd.concat(df_list, axis=0, ignore_index=True)

        # this is for NWAY/2WAY jcts
        if (sub_dir[0] == "NWAY") or (sub_dir[0] == "TWOWAY"):
            basepair_ends, motif_name = count_strands(result_df, motif_name=motif_name,
                                                      twoway_jct_csv=twoway_csv)  # you need a master DF of
            # residues here
            # Write # of BP ends to the motif name (labeling of n-way junction)
            if not (basepair_ends == 1):
                motif_name_spl = motif_name.split(".")  # size reclassification using data from count_connections
                new_path = dir[0] + "/" + str(basepair_ends) + "ways" + "/" + dir[2] + "/" + motif_name_spl[2] + "/" + \
                           sub_dir[3]
                name_path = new_path + "/" + motif_name
                # writing the file to its place
                make_dir(new_path)
            else:
                # if only 1 basepair end it should be reclassified as a hairpin
                sub_dir[0] = "HAIRPIN"
        if sub_dir[0] == "HAIRPIN":
            # hairpins classified by the # of looped nucleotides at the top of the pin
            # two NTs in a hairpin are always canonical pairs so just: (len nts - 2)
            hairpin_bridge_length = len(nts) - 2
            sub_dir[2] = str(hairpin_bridge_length)
            motif_name = '.'.join(sub_dir)
            if hairpin_bridge_length >= 3:
                # after classification into tri/tetra/etc
                hairpin_path = dir[0] + "/hairpins/" + str(hairpin_bridge_length)
                make_dir(hairpin_path)
                name_path = hairpin_path + "/" + motif_name
            else:
                sub_dir[0] = "SSTRAND"
        if sub_dir[0] == "HELIX":
            # helices should be classified into folders by their # of basepairs
            # this should be very simple as the lengths are given in the motif names
            # also classify further by the sequence composition, this is also given in motif name
            helix_count = str(sub_dir[2])
            helix_comp = str(sub_dir[3])
            # after classification put em in the folders
            helix_path = dir[0] + "/helices/" + helix_count + "/" + helix_comp
            make_dir(helix_path)
            name_path = helix_path + "/" + motif_name

        if sub_dir[0] != "SSTRAND":  # if the motif type is NOT a single strand
            # all results will use this, but the specific paths are changed above depending on what the motif is
            dataframe_to_cif(df=result_df, file_path=f"{name_path}.cif", motif_name=motif_name)

    # print list of residues in motif to CSV
    residue_csv_list.write(motif_name + ',' + ','.join(nts) + '\n')
    # VERY important for later when we need to find what residues belong where

    # TODO replace with RNP interactions obtained from snap
    # if there are interactions, do this:
    if interactions is not None:
        # remove duplicate amino acids (otherwise it breaks)
        # TODO inject data from snap here if need be
        interactions_filtered = remove_duplicate_residues_in_chain(interactions)
        ...
        # interaction processing list initialization
        inter_list = []
        inter_res = []

        # for each protein in the list of proteins
        for inter in interactions_filtered:
            inter_spl = inter.split(".")
            # purify IDs
            inter_chain_id = inter_spl[0]  # get the chain ID of the protein
            inter_protein_id = extract_longest_numeric_sequence(inter_spl[1])  # get the residue ID of the protein
            # sometimes there is a slash for some reason
            if "/" in inter_spl[1]:
                sub_spl = inter_spl[1].split("/")
                inter_protein_id = sub_spl[1]

            # define new protein ID
            new_inter = inter_chain_id + "." + inter_protein_id
            # add it to the list of new proteins
            inter_list.append(new_inter)

            # new protein ID; making a list (chain, res)
            inter_id = new_inter.split(".")
            # First find the right chain
            inter_chain = model_df[
                model_df['auth_asym_id'].astype(str) == inter_id[0]]
            # Then find the right atoms
            inter_res_subset = inter_chain[inter_chain['auth_seq_id'].astype(str) == str(inter_id[1])]
            # "inter_res" is a list with all the needed dataframes inside it (of the individual atoms of the 
            # appropriate residues)
            inter_res.append(inter_res_subset)

        inter_df_list = []  # List to store the DataFrames for each line (type = 'list')
        inter_res = remove_empty_dataframes(inter_res)

        for inter in inter_res:
            # Data reprocessing stuff, this loop is moving it into a DF
            lines = inter.to_string(index=False, header=False).split('\n')
            for line in lines:
                values = line.split()  # (type 'values' = list)
                inter_df = pd.DataFrame([values],
                                        columns=['group_PDB', 'id', 'type_symbol', 'label_atom_id',
                                                 'label_alt_id', 'label_comp_id', 'label_asym_id',
                                                 'label_entity_id', 'label_seq_id',
                                                 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y',
                                                 'Cartn_z',
                                                 'occupancy', 'B_iso_or_equiv',
                                                 'pdbx_formal_charge',
                                                 'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                                                 'auth_atom_id', 'pdbx_PDB_model_num'])
                inter_df_list.append(inter_df)
        if df_list and inter_df_list:
            # concatenate proteins with RNA
            result_inter_df = pd.concat(inter_df_list, axis=0, ignore_index=True)  # interactions
            # result_inter_df.to_csv("inter.csv", index=False)
            total_result_df = pd.concat([result_df, result_inter_df], ignore_index=True)

            # for JCTs
            if ((sub_dir[0] == "NWAY") or (sub_dir[0] == "TWOWAY")):
                # set a path for the interactions
                inter_new_path = "motif_interactions/" + str(basepair_ends) + "ways/" + dir[
                    2] + "/" + \
                                 sub_dir[2] + "/" + sub_dir[3]
                inter_name_path = inter_new_path + "/" + motif_name + ".inter"
                make_dir(inter_new_path)
            # for hairpins
            if (sub_dir[0] == "HAIRPIN"):
                inter_hairpin_path = "motif_interactions/hairpins"
                inter_name_path = inter_hairpin_path + "/" + motif_name + ".inter"
                make_dir(inter_hairpin_path)
            # for helices
            if (sub_dir[0] == "HELIX"):
                inter_helix_path = "motif_interactions/helices/" + str(helix_count) + "/" + str(
                    helix_comp)
                inter_name_path = inter_helix_path + "/" + motif_name + ".inter"
                make_dir(inter_helix_path)

            if (sub_dir[0] != "SSTRAND"):
                # writes interactions to CIF
                dataframe_to_cif(df=total_result_df, file_path=f"{inter_name_path}.cif", motif_name=motif_name)
        ...
        # extracting individual interactions:
        extract_individual_interactions(interactions_filtered, motif_bond_list, model_df, motif_name,
                                        csv_file)
"""

# unrefactored code
"""def get_lib_path():
    file_path = os.path.realpath(__file__)
    spl = file_path.split("/")
    base_dir = "/".join(spl[:-2])
    return base_dir

def get_os():
    OS = None
    if platform.system() == 'Linux':
        OS = 'linux'
    elif platform.system() == 'Darwin':
        OS = 'osx'
    else:
        raise SystemError(platform.system() + " is not supported currently")
    return OS

def get_query_term(json_query_path):
    with open(json_query_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


LIB_PATH = get_lib_path()
UNITTEST_PATH = LIB_PATH + "/test/"
RESOURCES_PATH = LIB_PATH + "/rna_motif_library/resources/"
DSSR_EXE = RESOURCES_PATH + "snap/%s/x3dna-dssr " % (get_os())

QUERY_TERM = get_query_term(json_query_path=get_lib_path() + "/rna_motif_library/json_query.json")
"""