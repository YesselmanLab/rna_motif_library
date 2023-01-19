//
// Created by Joe Yesselman on 1/4/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_STRINGS_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_STRINGS_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>


typedef std::string String;
typedef std::vector<std::string> Strings;
typedef int Int;
typedef std::vector<double> Vector;
typedef char Char;
typedef std::vector<char> Chars;

Strings split(const String &s, char delimiter);

Strings split_by_long(const String& s, const String& delimiter);

String join(const Strings& vec, char delimiter);











#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_STRINGS_H
