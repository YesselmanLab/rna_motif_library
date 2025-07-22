import os
import platform
import sys
from pathlib import Path


class PathManager:
    """Manages paths for the application, handling different execution contexts."""

    def __init__(self, project_name="rna_motif_library"):
        """
        Initialize the PathManager.

        Args:
            project_name (str): Name of the project directory
        """
        self.project_name = project_name
        self._initialize_paths()

    def _initialize_paths(self):
        """Set up all paths based on execution context."""
        # Determine if we're in a notebook
        self.in_notebook = self._is_notebook()

        # Get the project root directory (works in multiple contexts)
        self.project_root = self._find_project_root()

        # Define all other paths relative to project root
        self.data_path = os.path.join(self.project_root, "data")
        self.resources_path = os.path.join(
            self.project_root, "rna_motif_library/resources"
        )
        self.unittest_path = os.path.join(self.project_root, "tests")

        # OS-specific paths
        self.dssr_exe = os.path.join(
            self.resources_path, f"dssr/{self.get_os()}/x3dna-dssr"
        )

    def _is_notebook(self):
        """Check if code is running in a Jupyter notebook."""
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type
        except NameError:
            return False  # Standard Python interpreter

    def _find_project_root(self):
        """
        Find the project root directory regardless of where the code is executed from.
        This looks for a marker file/directory (like .git, pyproject.toml, or your project name)
        and uses that to determine the root.
        """
        # Strategy 1: Try using the module path if this code is in a package
        try:
            import rna_motif_library

            return os.path.dirname(
                os.path.dirname(os.path.abspath(rna_motif_library.__file__))
            )
        except (ImportError, AttributeError):
            pass

        # Strategy 2: Look for marker files/directories by walking up from current dir
        current_dir = Path(os.getcwd())
        markers = [".git", "pyproject.toml", "setup.py", self.project_name]

        # Walk up directory tree looking for markers
        while current_dir != current_dir.parent:
            for marker in markers:
                if (current_dir / marker).exists():
                    return str(current_dir)
            current_dir = current_dir.parent

        # Strategy 3: Fall back to directory containing the executed script
        return os.path.dirname(os.path.abspath(sys.argv[0]))

    def get_os(self):
        """
        Determine the operating system type.

        Returns:
            str: Operating system identifier ('linux' or 'osx')

        Raises:
            SystemError: If the operating system is not supported
        """
        system = platform.system()
        if system == "Darwin":
            return "osx"
        elif system == "Linux":
            return "linux"
        else:
            raise SystemError(f"{system} is not supported currently")

    def ensure_paths_exist(self):
        """Create directories if they don't exist."""
        paths = [self.data_path, self.resources_path, self.unittest_path]
        for path in paths:
            os.makedirs(path, exist_ok=True)


# Create a singleton instance
paths = PathManager()

# Constants for backward compatibility
LIB_PATH = paths.project_root
DATA_PATH = paths.data_path
UNITTEST_PATH = paths.unittest_path
RESOURCES_PATH = paths.resources_path
DSSR_EXE = paths.dssr_exe
VERSION = 3
