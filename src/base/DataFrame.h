//
// Created by Joe Yesselman on 1/10/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DATAFRAME_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DATAFRAME_H

#include <vector>
#include <string>
#include <fstream>
#include "string_operations.h"

class DataFrame {

public:
    // default constructor
    DataFrame() = default;

    // constructor taking column_names and data as arguments
    DataFrame(Strings column_names, std::vector<Strings> data)
            : _column_names(column_names), _data(data) {}

    // constructor taking a path to a csv file and reading the data
    DataFrame(const std::string &file_path) {
        std::ifstream file(file_path);
        String line;
        std::getline(file, line);
        _column_names = split(line, ',');
        while (std::getline(file, line)) {
            _data.push_back(split(line, ','));
        }
    }

    // Copy constructor
    DataFrame(const DataFrame &other)
            : _column_names(other._column_names), _data(other._data) {
    }

    // Move constructor
    DataFrame(DataFrame &&other) noexcept
            : _column_names(std::move(other._column_names)), _data(std::move(other._data)) {
    }

    // Copy assignment operator
    DataFrame &operator=(const DataFrame &other);

    // Move assignment operator
    DataFrame &operator=(DataFrame &&other) noexcept;

    // Destructor
    ~DataFrame() = default;

    // getting number of rows
    int rows() const;

    // getting number of columns
    int cols() const;

    // function to get specific row
    Strings get_row(int index) const;

    // function to get specific column
    Strings get_column(int index) const;

    // function to get specific column by name
    Strings get_column(String column_name) const;

    Strings split_row(const String& row_name, char delimiter);

private:
    Strings _column_names;
    std::vector<Strings> _data;

};

#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DATAFRAME_H
