import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcNumAromaticRings

def read_sdf_file(file_path):
    """Reads an SDF file and returns a list of RDKit molecule objects.

    Args:
        file_path: The path to the SDF file.

    Returns:
        A list of RDKit molecule objects, or an empty list if an error occurs.
    """
    try:
        suppl = Chem.SDMolSupplier(file_path)
        molecules = [mol for mol in suppl if mol is not None]
        return molecules
    except Exception as e:
        print(f"Error reading SDF file: {e}")
        return []



def main():
    """
    main function for script
    """
    mol = read_sdf_file("/Users/jyesselman2/Downloads/ZZW_ideal.sdf")[0]
    # Get number of aromatic rings
    # Get number of aromatic rings by finding SSSR and checking aromaticity
    print(f"Number of aromatic rings: {CalcNumAromaticRings(mol)}")
    print(f"Number of H-bond donors: {CalcNumHBD(mol)}")
    print(f"Number of H-bond acceptors: {CalcNumHBA(mol)}")
    

if __name__ == '__main__':
    main()
