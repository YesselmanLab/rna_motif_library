"""
Interactive PyMOL viewer for RNA motifs.

This script provides an interactive CLI tool for viewing RNA motifs in PyMOL.
It uses PyMOL's XML-RPC server for persistent session control.

Usage:
    # Start PyMOL with XML-RPC server first:
    pymol -R

    # Then run the viewer:
    python scripts/pymol_motif_viewer.py -c twoway_flanking_basepairs.csv

    # In the interactive REPL:
    > 0          # Load first motif by index
    > 5          # Load 6th motif by index
    > TWOWAY-5-4-AAAAGUC-GUCGCU-2O3Y-1  # Load by name
    > clear      # Clear all objects
    > next       # Load next motif
    > prev       # Load previous motif
    > quit       # Exit viewer
"""

import os
import subprocess
import sys
import tempfile
import time
import xmlrpc.client
from typing import Optional

import click
import pandas as pd

from rna_motif_library.logger import get_logger
from rna_motif_library.motif import Motif, get_cached_motifs

log = get_logger("pymol_viewer")


class PyMOLSession:
    """Manages connection to PyMOL's XML-RPC server."""

    def __init__(self, host: str = "localhost", port: int = 9123):
        """
        Initialize PyMOL session connector.

        Args:
            host: PyMOL server host
            port: PyMOL server port (default 9123)
        """
        self.host = host
        self.port = port
        self.server = None
        self._connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to PyMOL XML-RPC server.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        try:
            url = f"http://{self.host}:{self.port}"
            self.server = xmlrpc.client.ServerProxy(url)
            # Test connection with a simple command
            self.server.do("print('PyMOL connected')")
            self._connected = True
            log.info(f"Connected to PyMOL at {url}")
            return True
        except Exception as e:
            log.error(f"Failed to connect to PyMOL: {e}")
            log.info("Make sure PyMOL is running with XML-RPC server: pymol -R")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """Check if connected to PyMOL."""
        return self._connected

    def execute(self, cmd: str) -> None:
        """
        Execute a PyMOL command.

        Args:
            cmd: PyMOL command string
        """
        if not self._connected:
            log.error("Not connected to PyMOL")
            return
        try:
            self.server.do(cmd)
        except Exception as e:
            log.error(f"PyMOL command failed: {e}")

    def load_cif(self, path: str, name: str) -> None:
        """
        Load a CIF file into PyMOL.

        Args:
            path: Path to CIF file
            name: Object name in PyMOL
        """
        self.execute(f"load {path}, {name}")

    def clear_all(self) -> None:
        """Clear all objects from PyMOL."""
        self.execute("delete all")

    def zoom(self, selection: str = "all") -> None:
        """Zoom to selection."""
        self.execute(f"zoom {selection}")

    def show_sticks(self, selection: str = "all") -> None:
        """Show sticks representation."""
        self.execute(f"show sticks, {selection}")

    def color_by_chain(self) -> None:
        """Color by chain."""
        self.execute("util.cbc")


class MotifViewer:
    """Interactive viewer for RNA motifs."""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize motif viewer.

        Args:
            csv_path: Optional path to CSV with motif data
        """
        self.pymol = PyMOLSession()
        self.csv_path = csv_path
        self.df = None
        self.current_index = -1
        self.temp_dir = tempfile.mkdtemp()
        self._motif_cache = {}

    def load_csv(self, csv_path: str) -> bool:
        """
        Load CSV file with motif data.

        Args:
            csv_path: Path to CSV file

        Returns:
            True if loaded successfully
        """
        try:
            self.df = pd.read_csv(csv_path)
            self.csv_path = csv_path
            log.info(f"Loaded {len(self.df)} motifs from {csv_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load CSV: {e}")
            return False

    def _get_motif(self, motif_name: str, pdb_id: str) -> Optional[Motif]:
        """
        Get a motif from cache or load from disk.

        Args:
            motif_name: Name of the motif
            pdb_id: PDB ID

        Returns:
            Motif object or None
        """
        cache_key = f"{pdb_id}:{motif_name}"
        if cache_key in self._motif_cache:
            return self._motif_cache[cache_key]

        try:
            motifs = get_cached_motifs(pdb_id)
            for m in motifs:
                key = f"{pdb_id}:{m.name}"
                self._motif_cache[key] = m
                if m.name == motif_name:
                    return m
        except Exception as e:
            log.error(f"Failed to load motifs for {pdb_id}: {e}")
        return None

    def load_motif_by_index(self, index: int) -> bool:
        """
        Load a motif by row index.

        Args:
            index: Row index in the CSV

        Returns:
            True if loaded successfully
        """
        if self.df is None:
            log.error("No CSV loaded")
            return False

        if index < 0 or index >= len(self.df):
            log.error(f"Index {index} out of range (0-{len(self.df)-1})")
            return False

        row = self.df.iloc[index]
        motif_name = row["motif_name"]
        pdb_id = row["pdb_id"]

        return self._load_motif(motif_name, pdb_id, index)

    def load_motif_by_name(self, name: str) -> bool:
        """
        Load a motif by name.

        Args:
            name: Motif name

        Returns:
            True if loaded successfully
        """
        if self.df is None:
            log.error("No CSV loaded")
            return False

        matches = self.df[self.df["motif_name"] == name]
        if len(matches) == 0:
            log.error(f"Motif '{name}' not found")
            return False

        row = matches.iloc[0]
        index = matches.index[0]
        pdb_id = row["pdb_id"]

        return self._load_motif(name, pdb_id, index)

    def _load_motif(self, motif_name: str, pdb_id: str, index: int) -> bool:
        """
        Internal method to load a motif into PyMOL.

        Args:
            motif_name: Name of the motif
            pdb_id: PDB ID
            index: Row index

        Returns:
            True if loaded successfully
        """
        if not self.pymol.is_connected():
            log.error("Not connected to PyMOL")
            return False

        motif = self._get_motif(motif_name, pdb_id)
        if motif is None:
            log.error(f"Failed to get motif {motif_name}")
            return False

        # Generate temp CIF file
        cif_path = os.path.join(self.temp_dir, f"{motif_name}.cif")
        try:
            motif.to_cif(cif_path)
        except Exception as e:
            log.error(f"Failed to write CIF: {e}")
            return False

        # Clear previous objects before loading new one
        self.pymol.clear_all()

        # Load into PyMOL
        obj_name = motif_name.replace("-", "_")
        self.pymol.load_cif(cif_path, obj_name)
        self.pymol.show_sticks(obj_name)
        self.pymol.color_by_chain()
        self.pymol.zoom(obj_name)

        self.current_index = index
        log.info(f"Loaded motif {motif_name} (index {index})")

        # Print info from CSV if available
        if self.df is not None and index < len(self.df):
            row = self.df.iloc[index]
            if "sequence" in row:
                print(f"  Sequence: {row['sequence']}")
            if "structure" in row:
                print(f"  Structure: {row['structure']}")
            if "extended_sequence" in row:
                print(f"  Extended: {row['extended_sequence']}")
            if "flanking_bp_type_5p" in row and row["flanking_bp_type_5p"]:
                print(f"  5' BP: {row['flanking_bp_type_5p']}")
            if "flanking_bp_type_3p" in row and row["flanking_bp_type_3p"]:
                print(f"  3' BP: {row['flanking_bp_type_3p']}")

        return True

    def next_motif(self) -> bool:
        """Load the next motif."""
        if self.df is None:
            log.error("No CSV loaded")
            return False
        new_index = self.current_index + 1
        if new_index >= len(self.df):
            log.info("Already at last motif")
            return False
        return self.load_motif_by_index(new_index)

    def prev_motif(self) -> bool:
        """Load the previous motif."""
        if self.df is None:
            log.error("No CSV loaded")
            return False
        new_index = self.current_index - 1
        if new_index < 0:
            log.info("Already at first motif")
            return False
        return self.load_motif_by_index(new_index)

    def clear(self) -> None:
        """Clear all objects from PyMOL."""
        self.pymol.clear_all()
        log.info("Cleared all objects")

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass


@click.command()
@click.option(
    "-c",
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with motif data",
)
@click.option(
    "--host",
    default="localhost",
    help="PyMOL server host",
)
@click.option(
    "--port",
    default=9123,
    type=int,
    help="PyMOL server port",
)
def viewer(csv_path, host, port):
    """Interactive PyMOL viewer for RNA motifs.

    Start PyMOL with XML-RPC server first: pymol -R

    Commands:
        <number>  - Load motif by index
        <name>    - Load motif by name
        clear     - Clear all objects
        next      - Load next motif
        prev      - Load previous motif
        info      - Show current motif info
        list      - List first 20 motifs
        quit      - Exit viewer
    """
    mv = MotifViewer()

    # Set PyMOL connection parameters
    mv.pymol.host = host
    mv.pymol.port = port

    # Load CSV
    if not mv.load_csv(csv_path):
        sys.exit(1)

    # Connect to PyMOL
    print("\nConnecting to PyMOL...")
    print("Make sure PyMOL is running with: pymol -R")
    if not mv.pymol.connect():
        print("\nFailed to connect to PyMOL. Please ensure:")
        print("  1. PyMOL is installed")
        print("  2. PyMOL is running with XML-RPC server: pymol -R")
        sys.exit(1)

    print(f"\nLoaded {len(mv.df)} motifs from {csv_path}")
    print("\nCommands:")
    print("  <number>  - Load motif by index (0 to {})".format(len(mv.df) - 1))
    print("  <name>    - Load motif by name")
    print("  clear     - Clear all objects")
    print("  next      - Load next motif")
    print("  prev      - Load previous motif")
    print("  info      - Show current motif info")
    print("  list      - List first 20 motifs")
    print("  quit      - Exit viewer")
    print()

    # Interactive loop
    try:
        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                break

            if not cmd:
                continue

            if cmd.lower() == "quit" or cmd.lower() == "exit" or cmd.lower() == "q":
                break
            elif cmd.lower() == "clear":
                mv.clear()
            elif cmd.lower() == "next" or cmd.lower() == "n":
                mv.next_motif()
            elif cmd.lower() == "prev" or cmd.lower() == "p":
                mv.prev_motif()
            elif cmd.lower() == "info":
                if mv.current_index >= 0:
                    row = mv.df.iloc[mv.current_index]
                    print(f"Current: {row['motif_name']} (index {mv.current_index})")
                    for col in mv.df.columns:
                        print(f"  {col}: {row[col]}")
                else:
                    print("No motif loaded")
            elif cmd.lower() == "list":
                for i, row in mv.df.head(20).iterrows():
                    print(f"  {i}: {row['motif_name']}")
                if len(mv.df) > 20:
                    print(f"  ... and {len(mv.df) - 20} more")
            elif cmd.isdigit():
                index = int(cmd)
                mv.load_motif_by_index(index)
            else:
                # Try to load by name
                mv.load_motif_by_name(cmd)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        mv.cleanup()

    print("Goodbye!")


if __name__ == "__main__":
    viewer()
