//
// Created by Joe Yesselman on 1/4/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DSSRRES_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DSSRRES_H

#include "../base/string_operations.h"

class DSSRRes {

public:
    DSSRRes(const String in_s) {
        Strings spl = split(in_s, '^');
        String s1 = spl[0];
        std::istringstream iss(s1);
        String chain_id;
        std::getline(iss, chain_id, '.');
        String res_id;
        std::getline(iss, res_id);
        int cur_num = -1;
        for (int i = 0; i < res_id.size(); i++) {
            try {
                cur_num = std::stoi(res_id.substr(i));
                break;
            } catch (...) {
                continue;
            }
        }
        _num = cur_num;
        _chain_id = chain_id;
        _res_id = res_id.substr(0, res_id.size() - 1);
    }

    inline int num() const { return _num; }

    inline String chain_id() const { return _chain_id; }

    inline String res_id() const { return _res_id; }

private:
    int _num;
    String _chain_id;
    String _res_id;

};


#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_DSSRRES_H
