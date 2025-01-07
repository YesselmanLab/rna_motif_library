atom_renames = {
    "OP1": "O1P",
    "OP2": "O2P",
    "OP3": "O3P",
}


def main():
    f = open("missing_atoms.txt", "r")
    missing_atoms = []
    for line in f:
        spl = line.strip().split(",")
        spl = [x.strip() for x in spl]
        missing_atoms.append(spl)

    not_solved = []
    for missing in missing_atoms:
        _, atom_name, res_name, _ = missing
        if atom_name.find(".") != -1:
            continue
        if atom_name in atom_renames:
            continue
        if res_name.endswith("-") or res_name.endswith("/"):
            continue
        not_solved.append(missing)
    print(not_solved)


if __name__ == "__main__":
    main()
