"""Tests for the streamcutter.stream_generator module."""

import sys
import types
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out agama and pyfalcon before importing stream_generator,
# since neither is installed in CI.  We always overwrite sys.modules["agama"]
# so that stream_generator.agama is bound to *our* stub regardless of which
# test file was imported first.
# ---------------------------------------------------------------------------
_agama_stub = types.ModuleType("agama")
_agama_stub.G = 1.0
_agama_stub.orbit = MagicMock()
_agama_stub.Potential = MagicMock()
sys.modules["agama"] = _agama_stub

# Force re-import of stream_generator so it binds to the stub above.
sys.modules.pop("streamcutter.stream_generator", None)

_pyfalcon_stub = types.ModuleType("pyfalcon")
sys.modules.setdefault("pyfalcon", _pyfalcon_stub)

from streamcutter.stream_generator import create_mock_stream_fardal15  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_PARTICLES = 6        # must be even; trajsize = N_PARTICLES // 2 = 3
_N_STEPS = _N_PARTICLES // 2  # = 3  (trajectory steps for the satellite orbit)
_N_ICS = 2 * _N_STEPS         # = 6  (pairs of seed particles)


def _make_orbit_return(n_steps=_N_STEPS):
    """Fake return value for the first agama.orbit call (satellite orbit)."""
    time_sat = np.linspace(0.0, 1.0, n_steps)
    orbit_sat = np.ones((n_steps, 6))
    return time_sat, orbit_sat


def _make_stream_orbit_return(n_ics=_N_ICS):
    """
    Fake return value for the second agama.orbit call (stream integration).

    agama.orbit called with multiple ICs and trajsize=1 returns an object
    array of shape (n_ics, 2) where [:,1] holds the final-position arrays.
    """
    result = np.empty((n_ics, 2), dtype=object)
    for i in range(n_ics):
        result[i, 0] = np.array([0.0])             # times array (length 1)
        result[i, 1] = np.ones((1, 6)) * (i + 1)  # trajectory (1 step × 6D)
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestCreateStream:
    """Tests for create_mock_stream_fardal15 using monkey-patched agama.orbit / agama.Potential."""

    def setup_method(self, _method):
        _agama_stub.orbit.reset_mock()
        _agama_stub.Potential.reset_mock()

    def _setup_side_effects(self, n_steps=_N_STEPS, n_ics=_N_ICS):
        _agama_stub.orbit.side_effect = [
            _make_orbit_return(n_steps),
            _make_stream_orbit_return(n_ics),
        ]

    def _default_args(self, n_steps=_N_STEPS):
        rng = MagicMock()
        pot_host = MagicMock()
        posvel_sat = np.zeros(6)
        mass_sat = 1e5
        ic_return = np.ones((2 * n_steps, 6))
        create_ic_method = MagicMock(return_value=ic_return)
        return rng, pot_host, posvel_sat, mass_sat, create_ic_method

    # ------------------------------------------------------------------
    # Basic call structure
    # ------------------------------------------------------------------

    def test_positive_time_no_reversal(self):
        """With positive time_total, time_sat is in ascending order."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        time_sat, _, _, _ = create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        assert time_sat[0] <= time_sat[-1]

    def test_negative_time_causes_reversal(self):
        """With negative time_total, time_sat is reversed (descending)."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        time_sat, _, _, _ = create_mock_stream_fardal15(
            create_ic_method, rng, -1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        # linspace(0,1,3) reversed → [1.0, 0.5, 0.0]
        assert time_sat[0] >= time_sat[-1]

    # ------------------------------------------------------------------
    # agama.orbit first call (satellite orbit)
    # ------------------------------------------------------------------

    def test_first_orbit_call_uses_correct_potential_and_ic(self):
        """agama.orbit is called with pot_host and posvel_sat for the satellite orbit."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        first_call = _agama_stub.orbit.call_args_list[0]
        assert first_call.kwargs["potential"] is pot_host
        assert np.array_equal(first_call.kwargs["ic"], posvel_sat)

    def test_first_orbit_call_trajsize(self):
        """agama.orbit trajsize equals num_particles // 2."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        first_call = _agama_stub.orbit.call_args_list[0]
        assert first_call.kwargs["trajsize"] == _N_PARTICLES // 2

    def test_first_orbit_call_time(self):
        """agama.orbit time argument matches time_total."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()
        time_total = 2.5

        create_mock_stream_fardal15(
            create_ic_method, rng, time_total, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        first_call = _agama_stub.orbit.call_args_list[0]
        assert first_call.kwargs["time"] == time_total

    # ------------------------------------------------------------------
    # create_ic_method
    # ------------------------------------------------------------------

    def test_create_ic_method_called_once(self):
        """create_ic_method is called exactly once."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        create_ic_method.assert_called_once()

    def test_create_ic_method_receives_rng_and_pot_host(self):
        """create_ic_method receives rng and pot_host as positional args."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        call_args = create_ic_method.call_args
        assert call_args.args[0] is rng
        assert call_args.args[1] is pot_host

    def test_create_ic_method_receives_mass_sat(self):
        """create_ic_method receives mass_sat as a positional arg."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        call_args = create_ic_method.call_args
        assert call_args.args[3] == mass_sat

    def test_extra_kwargs_forwarded_to_create_ic_method(self):
        """Extra kwargs are forwarded to create_ic_method."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat,
            gala_modified=True
        )

        call_args = create_ic_method.call_args
        assert call_args.kwargs.get("gala_modified") is True

    # ------------------------------------------------------------------
    # Potential handling
    # ------------------------------------------------------------------

    def test_no_pot_sat_does_not_create_agama_potential(self):
        """When pot_sat is None, agama.Potential is never called."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat,
            pot_sat=None
        )

        _agama_stub.Potential.assert_not_called()

    def test_with_pot_sat_creates_combined_potential(self):
        """When pot_sat is provided, agama.Potential is called twice."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()
        pot_sat = MagicMock()

        create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat,
            pot_sat=pot_sat
        )

        # First call: agama.Potential(potential=pot_sat, center=traj)
        # Second call: agama.Potential(pot_host, pot_traj)
        assert _agama_stub.Potential.call_count == 2

    # ------------------------------------------------------------------
    # Return values
    # ------------------------------------------------------------------

    def test_return_value_shapes(self):
        """create_mock_stream_fardal15 returns arrays of the expected shapes."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()

        time_sat, orbit_sat, xv_stream, ic_stream = create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        assert time_sat.shape == (_N_STEPS,)
        assert orbit_sat.shape == (_N_STEPS, 6)
        assert ic_stream.shape == (_N_ICS, 6)
        assert xv_stream.shape == (_N_ICS, 6)

    def test_ic_stream_is_create_ic_method_return(self):
        """ic_stream in the return tuple is exactly what create_ic_method returned."""
        self._setup_side_effects()
        rng, pot_host, posvel_sat, mass_sat, create_ic_method = self._default_args()
        expected_ic = np.ones((_N_ICS, 6)) * 42.0
        create_ic_method.return_value = expected_ic

        _, _, _, ic_stream = create_mock_stream_fardal15(
            create_ic_method, rng, 1.0, _N_PARTICLES, pot_host, posvel_sat, mass_sat
        )

        assert np.array_equal(ic_stream, expected_ic)
