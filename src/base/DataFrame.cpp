//
// Created by Joe Yesselman on 1/10/23.
//

#include "DataFrame.h"

int DataFrame::rows() const {
    return _data.size();
}

int DataFrame::cols() const {
    return _column_names.size();
}

Strings DataFrame::get_row(int index) const {
    return _data[index];
}

Strings DataFrame::get_column(int index) const {
    Strings column_data;
    for (const auto &row: _data) {
        column_data.push_back(row[index]);
    }
    return column_data;
}

Strings DataFrame::get_column(String column_name) const {
    int col_index = 0;
    for (int i = 0; i < _column_names.size(); i++) {
        if (_column_names[i] == column_name) {
            col_index = i;
            break;
        }
    }
    Strings column_data;
    for (const auto &row: _data) {
        column_data.push_back(row[col_index]);
    }
    return column_data;
}

Strings DataFrame::split_row(const String &row_name, char delimiter) {
    // Iterate through the rows to find the specified row
    int counter = 0;
    for (const auto &row : _data) {
        if (row[counter] == row_name) {
            // Split the row using the delimiter and return the result
            return split(row[counter], delimiter);
        }
        counter++;
    }
    // If the specified row is not found, return an empty Strings object
    return {};
}

DataFrame &DataFrame::operator=(DataFrame &&other) noexcept {
    if (this != &other) {
        _column_names = std::move(other._column_names);
        _data = std::move(other._data);
    }
    return *this;
}

DataFrame &DataFrame::operator=(const DataFrame &other) {
    if (this != &other) {
        _column_names = other._column_names;
        _data = other._data;
    }
    return *this;
}
