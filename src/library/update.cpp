//
// Created by Joe Yesselman on 1/9/23.
//

#include "update.h"
#include "../base/settings.h"
#include "../snap/snap_functions.h"
#include <iosfwd>
#include <fstream>
#include <dirent.h>
#include <unistd.h>

bool file_exists(const String& fileName) {
    std::ifstream f(fileName.c_str());
    return f.good();
}

size_t write_data(void *ptr, size_t size, size_t nmemb, std::stringstream *data) {
    data->write((char *) ptr, size * nmemb);
    return size * nmemb;
}

void download_file(const String &url, const String &file_name) {
    std::ifstream  src(url, std::ios::binary);
    std::ofstream  dst(file_name,   std::ios::binary);

    dst << src.rdbuf();
}

void safe_mkdir(const std::__fs::filesystem::path& dir) {
    if (std::__fs::filesystem::is_directory(dir)) {
        return;
    }
    std::__fs::filesystem::create_directory(dir);
}

void download_cif_files(DataFrame df) {
    String pdb_dir = LIB_PATH + "/data/pdbs/";
    int count = 0;
    for (int i = 0; i < df.rows(); i++) {
        auto row = df.get_row(i);
        Strings spl = df.split_row("represent", '|');
        String pdb_name = spl[0];
        String out_path = pdb_dir + pdb_name + ".cif";
        String path = "https://files.rcsb.org/download/" + pdb_name + ".cif";
        if (file_exists(out_path)) {
            count++;
            std::cout << pdb_name << " ALREADY DOWNLOADED!" << std::endl;
            continue;
        }
        else {
            std::cout << pdb_name << " DOWNLOADING" << std::endl;
        }
        download_file(path, out_path);
    }
    std::cout << count << " pdbs already downloaded!" << std::endl;
}

void get_dssr_files() {
    String pdb_dir = LIB_PATH + "/data/pdbs/";
    String dssr_path = DSSR_EXE;
    String out_path = LIB_PATH + "/data/dssr_output";

    DIR *dir;
    struct dirent *ent;
    dir = opendir(pdb_dir.c_str());
    int count = 0;
    while ((ent = readdir (dir)) != nullptr) {
        String file_name(ent->d_name);
        if(file_name.find(".cif") != String::npos){
            String pdb_path = pdb_dir + "/" + file_name;
            std::ifstream pdb_file(pdb_path, std::ios::binary | std::ios::ate);
            std::streamsize size = pdb_file.tellg();
            pdb_file.close();
            std::cout << pdb_path << " " << size << std::endl;
            // if (size > 10000000) {
            //     continue;
            // }
            Strings parts = split(pdb_path, '/');
            String name = parts[parts.size() - 1].substr(0, file_name.length() - 4);
            String out_file_path = out_path + "/" + name + ".out";
            if (access(out_file_path.c_str(), F_OK) != -1) {
                count += 1;
                continue;
            }
            // pydssr.dssr.write_dssr_json_output_to_file(
            //     dssr_path, pdb_path, out_file_path
            // );
        }
    }
    closedir (dir);
}

/*
Strings get_snap_files() {
    String pdb_dir = LIB_PATH + "/data/pdbs/";
    String out_path = LIB_PATH + "/data/snap_output";
    Strings pdbs;
    DIR* dirp = opendir(pdb_dir.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        String file_name = dp->d_name;
        if (file_name.find(".cif") != String::npos) {
            pdbs.push_back(pdb_dir + file_name);
        }
    }
    closedir(dirp);
    int count = 0;
    for (const auto& pdb_path : pdbs) {
        std::ifstream in(pdb_path, std::ifstream::ate | std::ifstream::binary);
        int s = in.tellg();
        in.close();
        // if (s > 10000000) {
        //    continue;
        // }
        std::cout << count << " " << pdb_path << std::endl;
        String name = pdb_path.substr(pdb_path.find_last_of("/") + 1, pdb_path.find_last_of(".") - pdb_path.find_last_of("/") - 1);
        String out_file = out_path + "/" + name + ".out";
        std::ifstream f(out_file.c_str());
        if (f.good()) {
            count++;
            continue;
        }
        std::cout << pdb_path << std::endl;
        generate_out_file(pdb_path, out_file);
    }
    return pdbs;
}

 */

/*
void generate_motif_files() {
    std::string pdb_dir = LIB_PATH + "/data/pdbs/";
    std::vector<std::string> pdbs;
    DIR* dirp = opendir(pdb_dir.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::string file_name = dp->d_name;
        if (file_name.find(".cif") != std::string::npos) {
            pdbs.push_back(pdb_dir + file_name);
        }
    }
    closedir(dirp);
    int count = 0;
    std::vector<std::string> dirs = {
            "motifs",
            "motif_interactions",
            "motifs/twoways",
            "motifs/nways",
            "motif_interactions/twoways",
            "motif_interactions/twoways/all",
            "motif_interactions/nways",
            "motif_interactions/nways/all",
            "motifs/twoways/all",
            "motifs/nways/all",
    };
    for (auto d : dirs) {
        safe_mkdir(d);
    }

    std::string motif_dir = "motifs/twoways/all";
    std::string interactions_dir = "motif_interactions/twoways/all";
    std::vector<std::string> hbond_vals = {
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
    };
    std::ofstream f("interactions.csv");
    f << "name,type,size";
    for (auto val : hbond_vals) {
        f << "," << val;
    }
    f << "\n";
    for (auto pdb_path : pdbs) {
        std::ifstream in(pdb_path, std::ifstream::ate | std::ifstream::binary);
        int s = in.tellg();
        in.close();
        std::string name = pdb_path.substr(pdb_path.find_last_of("/") + 1, pdb_path.find_last_of(".") - pdb_path.find_last_of("/") - 1);
        std::string json_path = LIB_PATH + "/data/dssr_output/" + name + ".out";
        // if (s > 10000000) {
        //    continue;
        // }
        std::cout << count << " " << pdb_path << std::endl;
        count++;
        try {
            //pdb_model = atomium.open(pdb_path);
        } catch (...) {
            continue;
        }
        // (motifs, motif_hbonds, motif_interactions) = dssr.get_motifs_from_structure(json_path);
        for (auto m : motifs) {
            std::cout << m.name << std::endl;
            std::vector<std::string> spl = __split(m.name, '.');
            if (!(spl[0] == "TWOWAY" || spl[0] == "NWAY")) {
                continue;
            }
            try {
                //dssr.write_res_coords_to_pdb(m.nts_long, pdb_model, motif_dir + "/" + m.name);
            } catch (...) {
                continue;
            }
            f << m.name << "," << spl[0] << "," << m.nts_long.size() << ",";
            std::vector<std::string> vals;
            if (m.name not in motif_hbonds) {
                vals = std::vector<std::string>(hbond_vals.size(), "0");
            } else {
                for (auto x : hbond_vals) {
                    vals.push_back(std::to_string(motif_hbonds[m.name][x]));
                }
            }
            f << join(vals, ',') << "\n";

            //if (m.name in motif_interactions) {
            //    try {
            //        dssr.write_res_coords_to_pdb(
            //            m.nts_long + motif_interactions[m.name],
            //            pdb_model,
            //            interactions_dir + "/" + m.name + ".inter",
            //        );
            //    } catch (...) {
            //    }
            //}

        }
    }
    f.close();
}
 */
