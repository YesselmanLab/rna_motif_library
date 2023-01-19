//
// Created by Joe Yesselman on 1/19/23.
//

#include "string_operations.h"

Strings split(const String &s, char delimiter) {
    Strings tokens;
    String token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

Strings split_by_long(const String &s, const String &delimiter) {
    Strings tokens;
    size_t start = 0;
    size_t end = 0;
    while ((end = s.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(s.substr(start, end - start));
        start = end + delimiter.length();
    }
    return tokens;
}

String join(const Strings &vec, char delimiter) {
    String result;
    for (unsigned int i = 0; i < vec.size(); i++) {
        result += vec[i];
        if (i != vec.size() - 1) {
            result += delimiter;
        }
    }
    return result;
}
