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


# ---------------------------------------------------------------------------
# integrate_orbit and create_mock_stream_nbody tests
# ---------------------------------------------------------------------------

from streamcutter.stream_generator import (  # noqa: E402
    integrate_orbit,
    create_mock_stream_nbody,
)

_N_ORBIT_STEPS = 8


class TestIntegrateOrbit:
    """Tests for integrate_orbit."""

    def _make_orbit_rv(self, n=_N_ORBIT_STEPS):
        times = np.linspace(0.0, -1.0, n)
        traj  = np.ones((n, 6)) * 0.5
        return times, traj

    def _setup(self, n=_N_ORBIT_STEPS):
        _agama_stub.orbit.reset_mock()
        _agama_stub.orbit.side_effect = None
        _agama_stub.orbit.return_value = self._make_orbit_rv(n)

    def test_returns_tuple_of_two(self):
        self._setup()
        pot   = MagicMock()
        times, traj = integrate_orbit(pot, np.zeros(6), -1.0, _N_ORBIT_STEPS)
        assert times.shape == (_N_ORBIT_STEPS,)
        assert traj.shape  == (_N_ORBIT_STEPS, 6)

    def test_calls_agama_orbit_once(self):
        self._setup()
        integrate_orbit(MagicMock(), np.zeros(6), -1.0, _N_ORBIT_STEPS)
        _agama_stub.orbit.assert_called_once()

    def test_passes_potential_and_ic(self):
        self._setup()
        pot    = MagicMock()
        posvel = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        integrate_orbit(pot, posvel, -1.0, _N_ORBIT_STEPS)
        call_kwargs = _agama_stub.orbit.call_args[1]
        assert call_kwargs["potential"] is pot
        np.testing.assert_array_equal(call_kwargs["ic"], posvel)

    def test_trajsize_matches_num_steps(self):
        self._setup()
        integrate_orbit(MagicMock(), np.zeros(6), -1.0, _N_ORBIT_STEPS)
        call_kwargs = _agama_stub.orbit.call_args[1]
        assert call_kwargs["trajsize"] == _N_ORBIT_STEPS


class TestCreateMockStreamNbody:
    """Tests for create_mock_stream_nbody."""

    _N = 4   # small even number for num_particles
    _KING_W0 = 3.0
    _SIGMA = 20.0

    def _make_orbit_rv(self, n=10):
        times = np.linspace(0.0, -1.0, n)
        traj  = np.ones((n, 6)) * 0.5
        return times, traj

    def _setup(self):
        _agama_stub.orbit.reset_mock()
        _agama_stub.orbit.side_effect = None
        _agama_stub.orbit.return_value = self._make_orbit_rv()
        _pyfalcon_stub.gravity = MagicMock(
            return_value=(np.zeros((self._N, 3)), np.zeros(self._N))
        )

    def _make_nbody_mocks(self, mock_dynfric, mock_make_ics, mock_krt, mock_tr, pot_host):
        mock_dynfric.return_value = np.zeros(3)
        mock_krt.return_value = 3.0
        mock_tr.return_value = 0.5
        mock_make_ics.return_value = (
            np.zeros((self._N, 6)),   # f_xv_ic
            np.ones(self._N) * 1.0,  # mass
            float(self._N),           # initmass
            1.0, 0.5, 0.1,           # r_out, r_tidal_a, r0
        )
        pot_host.force.return_value = np.zeros((self._N, 3))

    @patch("streamcutter.stream_generator.tidal_radius")
    @patch("streamcutter.stream_generator.king_rt_over_scaleRadius")
    @patch("streamcutter.stream_generator.make_satellite_ics")
    @patch("streamcutter.stream_generator.dynfricAccel")
    def test_returns_tuple_of_four(self, mock_dynfric, mock_make_ics, mock_krt, mock_tr):
        self._setup()
        pot_host = MagicMock()
        self._make_nbody_mocks(mock_dynfric, mock_make_ics, mock_krt, mock_tr, pot_host)
        rng = np.random.default_rng(0)
        result = create_mock_stream_nbody(
            rng, -1.0, self._N, pot_host, np.zeros(6), 1e4, self._KING_W0, self._SIGMA
        )
        assert len(result) == 4

    @patch("streamcutter.stream_generator.tidal_radius")
    @patch("streamcutter.stream_generator.king_rt_over_scaleRadius")
    @patch("streamcutter.stream_generator.make_satellite_ics")
    @patch("streamcutter.stream_generator.dynfricAccel")
    def test_orbit_arrays_have_correct_ndim(self, mock_dynfric, mock_make_ics, mock_krt, mock_tr):
        self._setup()
        pot_host = MagicMock()
        self._make_nbody_mocks(mock_dynfric, mock_make_ics, mock_krt, mock_tr, pot_host)
        rng = np.random.default_rng(0)
        times, orbit, *_ = create_mock_stream_nbody(
            rng, -1.0, self._N, pot_host, np.zeros(6), 1e4, self._KING_W0, self._SIGMA
        )
        assert times.ndim == 1
        assert orbit.ndim == 2
        assert orbit.shape[1] == 6

    @patch("streamcutter.stream_generator.tidal_radius")
    @patch("streamcutter.stream_generator.king_rt_over_scaleRadius")
    @patch("streamcutter.stream_generator.make_satellite_ics")
    @patch("streamcutter.stream_generator.dynfricAccel")
    def test_uses_integrate_orbit_internally(self, mock_dynfric, mock_make_ics, mock_krt, mock_tr):
        """create_mock_stream_nbody must delegate to agama.orbit for integration."""
        self._setup()
        pot_host = MagicMock()
        self._make_nbody_mocks(mock_dynfric, mock_make_ics, mock_krt, mock_tr, pot_host)
        rng = np.random.default_rng(0)
        create_mock_stream_nbody(
            rng, -1.0, self._N, pot_host, np.zeros(6), 1e4, self._KING_W0, self._SIGMA
        )
        _agama_stub.orbit.assert_called()


# ---------------------------------------------------------------------------
# Coordinate transform tests (get_observed_coords / get_galactocentric_coords)
# ---------------------------------------------------------------------------

# These functions depend only on astropy (installed in CI); no mocking needed.
from streamcutter.coordinate import (  # noqa: E402
    get_observed_coords,
    get_galactocentric_coords,
)

# A simple reference point near the Galactic Center that has a known
# heliocentric direction and easy-to-check round-trip properties.
_XV_GC = np.array([[8.0, 0.0, 0.0, 0.0, 220.0, 0.0]])   # one row, (N,6)



class TestGetObservedCoords:
    """Tests for get_observed_coords."""

    def test_returns_six_arrays(self):
        result = get_observed_coords(_XV_GC)
        assert len(result) == 6

    def test_shapes_match_input(self):
        n = 5
        xv = np.tile(_XV_GC, (n, 1))
        ra, dec, vlos, pmra, pmde, dist = get_observed_coords(xv)
        for arr in (ra, dec, vlos, pmra, pmde, dist):
            assert arr.shape == (n,)

    def test_ra_in_range(self):
        ra, *_ = get_observed_coords(_XV_GC)
        assert np.all((ra >= 0) & (ra < 360))

    def test_dec_in_range(self):
        _, dec, *_ = get_observed_coords(_XV_GC)
        assert np.all((dec >= -90) & (dec <= 90))

    def test_distance_positive(self):
        *_, dist = get_observed_coords(_XV_GC)
        assert np.all(dist > 0)

    def test_accepts_list_input(self):
        """get_observed_coords should accept nested lists (not just ndarray)."""
        xv_list = _XV_GC.tolist()
        result = get_observed_coords(xv_list)
        assert len(result) == 6


class TestGetGalactocentricCoords:
    """Tests for get_galactocentric_coords."""

    def test_returns_array_shape(self):
        ra       = np.array([266.4])
        dec      = np.array([-28.9])
        distance = np.array([8.5])
        vlos     = np.array([0.0])
        pmra     = np.array([0.0])
        pmdec    = np.array([0.0])
        xv = get_galactocentric_coords(ra, dec, distance, vlos, pmra, pmdec)
        assert xv.shape == (1, 6)

    def test_returns_six_columns(self):
        xv = get_galactocentric_coords(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]),
        )
        assert xv.shape[1] == 6


class TestCoordRoundTrip:
    """Round-trip: Galactocentric → observed → Galactocentric."""

    _tol_pos = 1e-6   # kpc
    _tol_vel = 1e-6   # km/s

    def _roundtrip(self, xv_in):
        ra, dec, vlos, pmra, pmde, dist = get_observed_coords(xv_in)
        xv_out = get_galactocentric_coords(ra, dec, dist, vlos, pmra, pmde)
        return xv_out

    def test_single_particle_position(self):
        xv_out = self._roundtrip(_XV_GC)
        np.testing.assert_allclose(xv_out[:, :3], _XV_GC[:, :3], atol=self._tol_pos)

    def test_single_particle_velocity(self):
        xv_out = self._roundtrip(_XV_GC)
        np.testing.assert_allclose(xv_out[:, 3:], _XV_GC[:, 3:], atol=self._tol_vel)

    def test_multiple_particles(self):
        rng = np.random.default_rng(0)
        # Random positions within ~20 kpc, velocities within ±300 km/s
        xv_in = np.column_stack([
            rng.uniform(-20, 20, 10),
            rng.uniform(-20, 20, 10),
            rng.uniform(-20, 20, 10),
            rng.uniform(-300, 300, 10),
            rng.uniform(-300, 300, 10),
            rng.uniform(-300, 300, 10),
        ])
        xv_out = self._roundtrip(xv_in)
        np.testing.assert_allclose(xv_out, xv_in, atol=self._tol_pos)
