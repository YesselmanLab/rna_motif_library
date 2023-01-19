//
// Created by Joe Yesselman on 1/5/23.
//
#ifndef REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_RNPINTERACTION_H
#define REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_RNPINTERACTION_H

#include <utility>
#include "../base/settings.h"

class RNPInteraction {

public:
    RNPInteraction(const std::string &nt_atom, std::string aa_atom, std::string dist, std::string type)
            : _nt_atom(nt_atom), _aa_atom(std::move(aa_atom)), _dist(dist), _type(std::move(type)) {
        std::vector<std::string> spl = split(nt_atom, '@');
        _nt_res = spl[1];
    }

    String nt_atom() const { return _nt_atom; }

    String aa_atom() const { return _aa_atom; }

    String dist() const { return _dist; }

    String type() const { return _type; }

    String nt_res() const { return _nt_res; }

private:

    String _nt_atom;
    String _aa_atom;
    String _dist;
    String _type;
    String _nt_res;

};




#endif //REF_RESOURCES_RNA_MOTIF_LIBRARY_PY_RNPINTERACTION_H
