import os
import platform


def get_lib_path() -> str:
    """Gets the base directory path for the library.

    Returns:
        The base directory path of the library.
    """
    return os.getcwd()



def get_os() -> str:
    """Determines the operating system type that is currently being used.

    Returns:
        A string representing the operating system type ('linux' or 'osx').

    Raises:
        SystemError: If the operating system is neither Linux nor Darwin (macOS).
    """
    system = platform.system()
    if system == "Darwin":
        return "osx"
    else:
        raise SystemError(f"{system} is not supported currently")


# Define library paths using the functions defined above.
LIB_PATH: str = get_lib_path()
UNITTEST_PATH: str = os.path.join(LIB_PATH, "rna_motif_library/")
RESOURCES_PATH: str = os.path.join(LIB_PATH, "rna_motif_library/resources/")
DSSR_EXE: str = os.path.join(RESOURCES_PATH, f"snap/{get_os()}/x3dna-dssr ")