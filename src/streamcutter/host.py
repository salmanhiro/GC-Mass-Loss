"""Galactic host potential utilities."""

import numpy as np
import agama

agama.setUnits(length=1, velocity=1, mass=1)  # 1 kpc, 1 km/s, 1 Msun

def build_host(pot_ini):
    """Build a galactic host potential and velocity dispersion profile.

    Parameters
    ----------
    pot_ini : str
        Path to an agama potential ini file (e.g. configs/MWPotential2014.ini).

    Returns
    -------
    pot_host : agama.Potential
        The host galaxy potential.
    sigma : callable
        A function sigma(r) returning the 1-D velocity dispersion at radius r.
    """
    pot_host = agama.Potential(pot_ini)
    df_host = agama.DistributionFunction(type="quasispherical", potential=pot_host)

    # sigma(r) spline for DF (Chandrasekhar)
    grid_r = np.logspace(-1, 2, 16)
    grid_sig = agama.GalaxyModel(pot_host, df_host).moments(
        np.column_stack((grid_r, grid_r * 0, grid_r * 0)),
        dens=False, vel=False, vel2=True
    )[:, 0] ** 0.5

    logspl = agama.Spline(np.log(grid_r), np.log(grid_sig))

    def sigma(r):
        return np.exp(logspl(np.log(r)))

    return pot_host, sigma
