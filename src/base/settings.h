//
// Created by Joe Yesselman on 1/9/23.
//
// C++ implementation of update_library.py
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_SETTINGS_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_SETTINGS_H

#include "../base/string_operations.h"

String get_lib_path();

String get_os();

static String LIB_PATH = get_lib_path();
static String UNITTEST_PATH = LIB_PATH + "/test/";
static String RESOURCES_PATH = LIB_PATH + "/rna_motif_library_py/resources/";
static String DSSR_EXE = RESOURCES_PATH + "snap/" + get_os() + "/x3dna-dssr";

#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_SETTINGS_H
