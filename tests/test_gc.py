"""Tests for the streamcutter.dynamics module."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Stub out agama before importing dynamics, since agama is not installed in CI
# ---------------------------------------------------------------------------
_agama_stub = types.ModuleType("agama")
_agama_stub.setUnits = MagicMock()
_agama_stub.GalaPotential = MagicMock()
sys.modules["agama"] = _agama_stub

from streamcutter.dynamics import GCParams, PotentialFactory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(tmp_path, clusters):
    """Write a minimal ecsv table with a 'Cluster' column and return its path."""
    from astropy.table import QTable
    tab = QTable({"Cluster": clusters})
    p = tmp_path / "mw_gc_parameters_orbital_structural_time.ecsv"
    tab.write(str(p), format="ascii.ecsv", overwrite=True)
    return str(p)


# ---------------------------------------------------------------------------
# GCParams tests
# ---------------------------------------------------------------------------

class TestGCParams:
    def test_default_table_path(self):
        from streamcutter.dynamics import _DEFAULT_TABLE
        gcp = GCParams()
        assert gcp.table_path == _DEFAULT_TABLE
        assert gcp.table_path.endswith("mw_gc_parameters_orbital_structural_time.ecsv")

    def test_custom_table_path(self):
        gcp = GCParams(table_path="some/other/path.ecsv")
        assert gcp.table_path == "some/other/path.ecsv"

    def test_get_all_cluster_names(self, tmp_path):
        clusters = ["NGC104", "NGC5139", "NGC6752"]
        path = _make_table(tmp_path, clusters)
        gcp = GCParams(table_path=path)
        assert gcp.get_all_cluster_names() == clusters

    def test_get_row_returns_single_row(self, tmp_path):
        clusters = ["NGC104", "NGC5139"]
        path = _make_table(tmp_path, clusters)
        gcp = GCParams(table_path=path)
        row = gcp.get_row("NGC104")
        assert len(row) == 1
        assert row["Cluster"][0] == "NGC104"

    def test_get_row_unknown_cluster_raises(self, tmp_path):
        path = _make_table(tmp_path, ["NGC104"])
        gcp = GCParams(table_path=path)
        with pytest.raises(ValueError, match="not found"):
            gcp.get_row("Unknown")


class TestFindCluster:
    """Tests for GCParams.find_cluster (fuzzy name search)."""

    def setup_method(self, _method):
        # Use the real bundled table for these tests
        self.gcp = GCParams()

    def test_exact_name(self):
        assert self.gcp.find_cluster("NGC_4590") == ["NGC_4590"]

    def test_space_instead_of_underscore(self):
        assert self.gcp.find_cluster("NGC 4590") == ["NGC_4590"]

    def test_no_separator(self):
        assert self.gcp.find_cluster("ngc4590") == ["NGC_4590"]

    def test_case_insensitive(self):
        assert self.gcp.find_cluster("ngc_4590") == ["NGC_4590"]

    def test_no_match_returns_empty(self):
        assert self.gcp.find_cluster("XXXXXXXXXXX") == []

    def test_n_greater_than_one(self):
        results = self.gcp.find_cluster("NGC", n=3, cutoff=0.5)
        assert isinstance(results, list)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# PotentialFactory tests
# ---------------------------------------------------------------------------

class TestPotentialFactory:
    def test_init_calls_set_units(self):
        _agama_stub.setUnits.reset_mock()
        PotentialFactory()
        _agama_stub.setUnits.assert_called_once_with(length=1, velocity=1, mass=1)

    def test_custom_potentials_dir(self):
        pf = PotentialFactory(potentials_dir="my_pots")
        assert pf.potentials_dir == "my_pots"

    def test_host_raises_if_ini_missing(self, tmp_path):
        pf = PotentialFactory(potentials_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            pf.host("MWPotential2014")

    def test_host_calls_gala_potential(self, tmp_path):
        ini = tmp_path / "MWPotential2014.ini"
        ini.write_text("[potential]\ntype=NFW\n")
        _agama_stub.GalaPotential.reset_mock()
        pf = PotentialFactory(potentials_dir=str(tmp_path))
        pf.host("MWPotential2014")
        _agama_stub.GalaPotential.assert_called_once_with(str(ini))

    def test_satellite_plummer(self):
        _agama_stub.GalaPotential.reset_mock()
        PotentialFactory.satellite_plummer(mass=1e5, rhm=0.005)
        _agama_stub.GalaPotential.assert_called_once_with(
            type="Plummer", mass=1e5, scaleRadius=0.005
        )

    def test_satellite_king(self):
        _agama_stub.GalaPotential.reset_mock()
        PotentialFactory.satellite_king(mass=1e5, W0=5.0)
        _agama_stub.GalaPotential.assert_called_once_with(
            type="King", mass=1e5, W0=5.0
        )
