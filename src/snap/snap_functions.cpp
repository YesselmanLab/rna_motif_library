//
// Created by Joe Yesselman on 1/6/23.
//

#include <iosfwd>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sstream>
#include <fstream>

#include "../base/settings.h"
#include "snap_functions.h"

// commented out lines are functions that need to be implemented, don't delete anything yet

/*
void generate_out_file(const String &pdb_path, const String &out_path) {
    String dssr_exe = DSSR_EXE;
    String cmd = dssr_exe + String(" snap -i=") + pdb_path + String(" -o=") + out_path;
    int status = system(cmd.c_str());
    if (status != 0) {
        std::cerr << "Error: unable to run command '" << cmd << "'" << std::endl;
        return;
    }
    String files[] = {"dssr-2ndstrs.bpseq", "dssr-2ndstrs.ct", "dssr-2ndstrs.dbn", "dssr-atom2bases.pdb",
                           "dssr-stacks.pdb", "dssr-torsions.txt"};
    for (const String &f: files) {
        if (unlink(f.c_str()) != 0) {
            std::cerr << "Error: unable to delete file '" << f << "'" << std::endl;
        }
    }
}

std::vector<RNPInteraction> get_rnp_interactions(const String &pdb_path = "", const String &out_file = "") {
    if (pdb_path.empty() && out_file.empty()) {
        throw std::invalid_argument("must supply either a pdb or out file");
    }

    String out_file_path;
    if (!pdb_path.empty()) {
        generate_out_file(pdb_path);
        out_file_path = "test.out";
    } else {
        out_file_path = out_file;
    }

    std::ifstream f(out_file_path);
    std::stringstream buffer;
    buffer << f.rdbuf();
    String k = buffer.str();
    Strings spl = split_by_long(k, "List");

    std::vector<RNPInteraction> interactions;
    for (const auto &s: spl) {
        if (s.find("H-bonds") == String::npos) {
            continue;
        }
        Strings lines = split(s, '\n');
        lines.erase(lines.begin());
        lines.erase(lines.begin());
        for (const auto &l: lines) {
            Strings i_spl = split(l, ' ');
            if (i_spl.size() < 4) {
                continue;
            }
            interactions.emplace_back(RNPInteraction(i_spl[2], i_spl[3], i_spl[4], i_spl[5]));
        }
    }
    return interactions;
}
 */
