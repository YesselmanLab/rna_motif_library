//
// Created by Joe Yesselman on 1/9/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_UPDATE_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_UPDATE_H

#include <filesystem>
#include "../base/DataFrame.h"


bool file_exists(const String& fileName);

size_t write_data(void *ptr, size_t size, size_t nmemb, std::stringstream *data);

void safe_mkdir(const std::__fs::filesystem::path& dir);

void download_file(const String &url, const String &file_name);

void download_cif_files(DataFrame df);

void get_dssr_files();

//Strings get_snap_files();

//void generate_motif_files();



#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_UPDATE_H
