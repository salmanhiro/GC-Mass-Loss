"""GC parameters and potential utilities for globular cluster simulations."""

from __future__ import annotations
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from astropy.table import QTable
import agama

# Resolve the bundled data table relative to this file so the default path
# works regardless of the current working directory.
_DEFAULT_TABLE = str(
    Path(__file__).parents[2] / "data" / "mw_gc_parameters_orbital_structural_time.ecsv"
)


def _normalize(name: str) -> str:
    """Lower-case and strip non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# GC parameters & potentials

@dataclass
class GCParams:
    """Helper to load the GC parameter table and select a cluster row."""
    table_path: str = _DEFAULT_TABLE

    def get_row(self, cluster_name: str) -> QTable:
        tab = QTable.read(self.table_path)
        sel = tab[tab["Cluster"] == cluster_name]
        if len(sel) == 0:
            raise ValueError(f"Cluster '{cluster_name}' not found in: {self.table_path}")
        return sel[0:1]  # keep as 1-row table for convenience

    def get_all_cluster_names(self) -> list[str]:
        tab = QTable.read(self.table_path)
        return list(tab["Cluster"])

    def find_cluster(self, query: str, n: int = 1, cutoff: float = 0.6) -> list[str]:
        """Return up to *n* cluster names that best match *query*.

        Matching is case-insensitive and ignores spaces, underscores, and
        hyphens, so ``"NGC 4590"``, ``"ngc4590"``, and ``"NGC_4590"`` all
        resolve to the same entry.  If no close match is found, an empty
        list is returned.

        Parameters
        ----------
        query : str
            Free-form cluster name to search for.
        n : int
            Maximum number of matches to return (default 1).
        cutoff : float
            Similarity threshold in [0, 1] passed to
            ``difflib.get_close_matches`` (default 0.6).

        Returns
        -------
        list[str]
            Matching cluster names from the catalogue, best match first.
        """
        names = self.get_all_cluster_names()
        norm_query = _normalize(query)
        norm_names = [_normalize(n_) for n_ in names]

        # Exact match on normalised name takes priority
        for raw, norm in zip(names, norm_names):
            if norm == norm_query:
                return [str(raw)]

        # Fuzzy fallback via difflib
        norm_to_raw = {norm: raw for norm, raw in zip(norm_names, names)}
        close = difflib.get_close_matches(norm_query, norm_names, n=n, cutoff=cutoff)
        return [str(norm_to_raw[c]) for c in close]


class PotentialFactory:
    """Create host & satellite potentials, and ensure Agama unit setup."""
    def __init__(self, potentials_dir: str = "potentials"):
        self.potentials_dir = potentials_dir
        self._ensure_units()

    @staticmethod
    def _ensure_units():
        # Matches your original: code assumes ini files are already in these units
        agama.setUnits(length=1, velocity=1, mass=1)

    def host(self, potential_name: str) -> agama.GalaPotential:
        ini = Path(self.potentials_dir) / f"{potential_name}.ini"
        if not ini.is_file():
            raise FileNotFoundError(f"Host potential ini not found: {ini}")
        return agama.GalaPotential(str(ini))

    @staticmethod
    def satellite_plummer(mass: float, rhm: float) -> agama.GalaPotential:
        # scaleRadius expects same length units as Agama config (here unitless, consistent with setUnits above)
        return agama.GalaPotential(type="Plummer", mass=mass, scaleRadius=rhm)

    @staticmethod
    def satellite_king(mass: float, W0: float) -> agama.GalaPotential:
        return agama.GalaPotential(type="King", mass=mass, W0=W0)
