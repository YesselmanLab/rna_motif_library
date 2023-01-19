//
// Created by Joe Yesselman on 1/9/23.
//
#include "settings.h"
#include <sys/utsname.h>
#include <algorithm>
#include <unistd.h>

String get_lib_path() {
    char file_path[PATH_MAX];
    char link_path[PATH_MAX];
    std::string base_dir;
    int cnt = readlink("/proc/self/exe", link_path, PATH_MAX);
    if (cnt < 0 || cnt >= PATH_MAX) {
        throw std::runtime_error("Cannot get the real path");
    }
    else {
        link_path[cnt] = '\0';
        std::string file_path_str(link_path);
        std::vector<std::string> splitted_str = split(file_path_str, '/');
        std::vector<std::string> splitted_path;
        for (int i = 0; i < splitted_str.size() - 2; i++) {
            splitted_path.push_back(splitted_str[i]);
        }
        base_dir = join(splitted_path, '/');
    }
    return base_dir;
}

String get_os() {
    struct utsname unameData{};
    uname(&unameData);
    String os_name(unameData.sysname);
    String OS;
    if (os_name == "Linux"){
        OS = "linux";
    }
    else if (os_name == "Darwin"){
        OS = "osx";
    }
    else {
        throw std::runtime_error(os_name + " is not supported currently");
    }
    return OS;
}