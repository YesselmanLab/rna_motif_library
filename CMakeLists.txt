cmake_minimum_required(VERSION 3.22)
project(ref_resources/rna_motif_library_py)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wno-c++11-extensions -std=gnu++1y -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O4")

include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/)
#include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/atom)
include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/base)
include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/dssr)
include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/snap)
include_directories(/Users/jyesselm/CLionProjects/rna_motif_library/src/library)

add_executable(rna_motif_library
        src/main.cpp
        src/base/settings.cpp src/base/DataFrame.cpp src/base/string_operations.cpp
        src/dssr/DSSRRes.cpp src/dssr/dssr.cpp
        src/snap/RNPInteraction.cpp src/snap/snap_functions.cpp
        src/library/update.cpp src/library/nmr.cpp)
