//
// Created by Joe Yesselman on 1/12/23.
//

#include <filesystem>
#include "nmr.h"
#include "update.h"
#include "../base/settings.h"

void safe_mkdir_nmr(const std::__fs::filesystem::path& dir) {
    if (std::__fs::filesystem::is_directory(dir)) {
        return;
    }
    std::__fs::filesystem::create_directory(dir);
}

void download_cif_files_nmr(DataFrame df) {
    String pdb_dir = LIB_PATH + "/data/pdbs/";
    int count = 0;
    for (int i = 0; i < df.rows(); i++) {
        auto row = df.get_row(i);
        Strings spl = df.split_row("represent", '|');
        String pdb_name = spl[0];
        String out_path = pdb_dir + pdb_name + ".cif";
        String path = "https://files.rcsb.org/download/" + pdb_name + ".cif";
        if (file_exists(out_path.c_str())) {
            count++;
            // cout << pdb_name << " ALREADY DOWNLOADED!" << endl;
            continue;
        }
        else {
            std::cout << pdb_name << " DOWNLOADING" << std::endl;
        }
        download_file(path, out_path);
    }
    std::cout << count << " pdbs already downloaded!" << std::endl;
}

void get_dssr_files_nmr() {

}

void get_snap_files_nmr() {

}

void generate_motif_files_nmr() {

}