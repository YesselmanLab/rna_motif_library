//
// Created by Joe Yesselman on 12/21/22.
//
#include <fstream>
#include <iostream>
//#include "dssr/DSSRRes.h"
//#include "snap/RNPInteraction.h"

/// @brief - workflow
// - get PDBs from online (3.0 angstrom resolution), pass them as input
// - process with dssr.py
// - process with snap.py
// - that should spit out a CSV file with a shitload of data

int main(int argc, char*argv[]) {
    std::cout << "Retrieving PDB..." << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input_file" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file) {
        std::cerr << "Error: unable to open input file " << argv[1] << std::endl;
        return 1;
    }

    // read file contents and process here

    return 0;
}