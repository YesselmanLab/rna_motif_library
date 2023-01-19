//
// Created by Joe Yesselman on 1/12/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_NMR_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_NMR_H

#include "../base/DataFrame.h"

void safe_mkdir_nmr(const std::__fs::filesystem::path& dir);

void download_cif_files_nmr(DataFrame df);

void get_dssr_files_nmr();

void get_snap_files_nmr();

void generate_motif_files_nmr();










#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_NMR_H
