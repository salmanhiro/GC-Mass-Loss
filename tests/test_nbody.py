"""Tests for the streamcutter.nbody module."""

import sys
import types
from unittest.mock import MagicMock, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out agama, pyfalcon, and scipy before importing nbody.
# ---------------------------------------------------------------------------
_agama_stub = types.ModuleType("agama")
_agama_stub.G = 1.0
_agama_stub.setUnits = MagicMock()

# agama.Potential mock: support density/force/potential queries and totalMass
_pot_mock = MagicMock()
_pot_mock.density.return_value = 1.0
_pot_mock.force.return_value = np.array([-1.0, 0.0, 0.0])  # Vc²=r at r=1
_pot_mock.potential.return_value = -1.0
_pot_mock.totalMass.return_value = 1e5
_agama_stub.Potential = MagicMock(return_value=_pot_mock)
_agama_stub.GalaPotential = MagicMock(return_value=_pot_mock)

# GalaxyModel / DF stubs for make_satellite_ics sampling
_df_mock    = MagicMock()
_gm_mock    = MagicMock()
_N_SAT      = 10
_xv_sample  = np.zeros((_N_SAT, 6))
_mass_sample = np.ones(_N_SAT) * 100.0
_gm_mock.sample.return_value = (_xv_sample.copy(), _mass_sample.copy())
_agama_stub.DistributionFunction  = MagicMock(return_value=_df_mock)
_agama_stub.GalaxyModel           = MagicMock(return_value=_gm_mock)

sys.modules["agama"] = _agama_stub

_pyfalcon_stub = types.ModuleType("pyfalcon")
sys.modules.setdefault("pyfalcon", _pyfalcon_stub)

# scipy: stub only the special sub-module used by dynfricAccel
import scipy.special as _real_scipy_special  # noqa: E402
_scipy_special_stub = MagicMock(spec=_real_scipy_special)
_scipy_special_stub.erf.return_value  = 0.5
sys.modules["scipy.special"] = _scipy_special_stub

# Remove any cached nbody import so it picks up the stubs above
sys.modules.pop("streamcutter.nbody", None)

from streamcutter.nbody import (  # noqa: E402
    dynfricAccel,
    king_rt_over_scaleRadius,
    tidal_radius,
    make_satellite_ics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pot():
    """Return a fresh mock potential that behaves like a simple isothermal sphere."""
    p = MagicMock()
    p.density.return_value   = 1.0
    p.force.return_value     = np.array([-1.0, 0.0, 0.0])
    p.potential.return_value = -1.0
    p.totalMass.return_value = 1e5
    return p


# ---------------------------------------------------------------------------
# dynfricAccel tests
# ---------------------------------------------------------------------------

class TestDynfricAccel:
    def test_zero_velocity_returns_zero_vector(self):
        pot = _make_pot()
        sigma = lambda r: 100.0
        result = dynfricAccel(pot, sigma, np.array([1.0, 0.0, 0.0]),
                              np.array([0.0, 0.0, 0.0]), 1e5)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_returns_3d_vector(self):
        pot = _make_pot()
        sigma = lambda r: 50.0
        result = dynfricAccel(pot, sigma,
                              np.array([8.0, 0.0, 0.0]),
                              np.array([0.0, 220.0, 0.0]), 1e4)
        assert result.shape == (3,)

    def test_friction_opposes_velocity(self):
        """The friction acceleration must point anti-parallel to velocity."""
        pot = _make_pot()
        sigma = lambda r: 50.0
        vel = np.array([0.0, 220.0, 0.0])
        result = dynfricAccel(pot, sigma, np.array([8.0, 0.0, 0.0]), vel, 1e4)
        # dot product with velocity direction should be negative
        assert np.dot(result, vel) < 0


# ---------------------------------------------------------------------------
# tidal_radius tests
# ---------------------------------------------------------------------------

class TestTidalRadius:
    def test_returns_positive_float(self):
        pot = _make_pot()
        rt = tidal_radius(pot, r=8.0, msat=1e4)
        assert isinstance(rt, float)
        assert rt > 0

    def test_scales_with_mass(self):
        """Larger satellite mass → larger tidal radius."""
        pot = _make_pot()
        rt_small = tidal_radius(pot, r=8.0, msat=1e3)
        rt_large = tidal_radius(pot, r=8.0, msat=1e5)
        assert rt_large > rt_small


# ---------------------------------------------------------------------------
# make_satellite_ics tests
# ---------------------------------------------------------------------------

class TestMakeSatelliteIcs:
    """Tests for make_satellite_ics using mocked agama."""

    _center0 = np.array([7.1, 0.22, 15.74, -49.28, -144.14, -10.2])

    def _run(self, satellite_type="king"):
        pot = _make_pot()
        return make_satellite_ics(
            ft=0.95,
            seed=0,
            M_SAT=1e4,
            Nbody=_N_SAT,
            pot_host=pot,
            r_center0=self._center0,
            KING_W0=5.0,
            KING_TRUNC=1.0,
            RT_OVER_R0=5.0,
            satellite_type=satellite_type,
        )

    def test_returns_six_items(self):
        result = self._run()
        assert len(result) == 6

    def test_f_xv_shape(self):
        f_xv, *_ = self._run()
        assert f_xv.shape == (_N_SAT, 6)

    def test_mass_array_shape(self):
        _, mass, *_ = self._run()
        assert mass.shape == (_N_SAT,)

    def test_initmass_is_float(self):
        _, _, initmass, *_ = self._run()
        assert isinstance(initmass, float)

    def test_particles_shifted_to_center(self):
        """Sampled particles should be shifted by r_center0."""
        f_xv, *_ = self._run()
        # _xv_sample is all zeros, so after shift f_xv ≈ _center0 (broadcast)
        np.testing.assert_allclose(f_xv[0], self._center0, atol=1e-10)

    def test_plummer_type_accepted(self):
        """satellite_type='plummer' must not raise."""
        result = self._run(satellite_type="plummer")
        assert len(result) == 6

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite_type"):
            self._run(satellite_type="nfw")

    def test_r_out_positive(self):
        *_, r_out, r_tidal, r0 = self._run()
        assert r_out > 0

    def test_r0_positive(self):
        *_, r_tidal, r0 = self._run()
        assert r0 > 0
