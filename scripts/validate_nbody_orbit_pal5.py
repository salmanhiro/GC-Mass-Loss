"""Validate backward orbit integration for Pal 5 using create_mock_stream_nbody.

Usage
-----
    pip install -e .
    python scripts/validate_nbody_orbit_pal5.py

This script:
  1. Loads Pal 5 parameters from the bundled GC catalogue.
  2. Builds the MWPotential2014 host potential and a King satellite potential.
  3. Calls create_mock_stream_nbody which internally runs integrate_orbit
     backward for 3 orbital periods and returns the satellite orbit.
  4. Plots and saves the backward orbit to validate it before running the
     full N-body stream simulation.
"""

from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import agama

from streamcutter.dynamics import GCParams, PotentialFactory
from streamcutter.nbody import king_rt_over_scaleRadius
from streamcutter.stream_generator import create_mock_stream_nbody

agama.setUnits(length=1, velocity=1, mass=1)

_GYR_PER_AGAMA_TU = 0.9778

N_ORBITS      = 3
NUM_PARTICLES = 500   # used for time resolution inside create_mock_stream_nbody
KING_W0       = 5.0
KING_TRUNC    = 1.0
RNG_SEED      = 42


def main():
    # ------------------------------------------------------------------
    # 1. Load Pal 5 parameters
    # ------------------------------------------------------------------
    gcp     = GCParams()
    matches = gcp.find_cluster("Pal 5")
    if not matches:
        raise RuntimeError("Pal 5 not found in the GC catalogue.")
    row = gcp.get_row(matches[0])

    posvel_sat = np.array([
        row["X_gc"][0].to_value(u.kpc),
        row["Y_gc"][0].to_value(u.kpc),
        row["Z_gc"][0].to_value(u.kpc),
        row["Vx_gc"][0].to_value(u.km/u.s),
        row["Vy_gc"][0].to_value(u.km/u.s),
        row["Vz_gc"][0].to_value(u.km/u.s),
    ])
    mass_sat         = row["Mass"][0].to_value(u.solMass)
    rhm_kpc          = row["rh,m"][0].to_value(u.pc) * 1e-3
    orbit_period_myr = row["orbit_period_max"][0].to_value(u.Myr)
    orbit_period     = orbit_period_myr * 1e-3 / _GYR_PER_AGAMA_TU

    print(f"Cluster: {matches[0]}")
    print(f"  posvel       : {posvel_sat}")
    print(f"  mass         : {mass_sat:.3e} Msun")
    print(f"  rh,m         : {rhm_kpc*1e3:.2f} pc")
    print(f"  orbit period : {orbit_period_myr:.1f} Myr  ({orbit_period:.3f} Agama TU)")

    # ------------------------------------------------------------------
    # 2. Build host and satellite potentials
    # ------------------------------------------------------------------
    configs_dir = Path(__file__).parents[1] / "configs"
    pot_host    = PotentialFactory(potentials_dir=str(configs_dir)).host("MWPotential2014")

    rt_over_r0 = king_rt_over_scaleRadius(W0=KING_W0, trunc=KING_TRUNC)
    pot_sat = agama.Potential(
        type="King",
        mass=mass_sat,
        scaleRadius=rhm_kpc,
        W0=float(KING_W0),
        trunc=float(KING_TRUNC),
    )

    # ------------------------------------------------------------------
    # 3. Run create_mock_stream_nbody (backward integration only for now)
    # ------------------------------------------------------------------
    rng        = np.random.default_rng(seed=RNG_SEED)
    time_total = -N_ORBITS * orbit_period

    print(f"\nRunning create_mock_stream_nbody for {N_ORBITS} orbits "
          f"(time_total = {time_total:.3f} Agama TU, {NUM_PARTICLES} particles) ...")

    time_sat, orbit_sat = create_mock_stream_nbody(
        rng, time_total, NUM_PARTICLES, pot_host, pot_sat, posvel_sat, mass_sat
    )

    print(f"  Orbit shape  : {orbit_sat.shape}")
    print(f"  t range      : [{time_sat[0]:.3f}, {time_sat[-1]:.3f}] Agama TU")

    # ------------------------------------------------------------------
    # 4. Plot backward orbit
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Pal 5 backward orbit — {N_ORBITS} orbits via create_mock_stream_nbody",
        fontsize=13,
    )

    for ax, (xi, yi, lx, ly) in zip(axes, [
        (0, 1, "x [kpc]", "y [kpc]"),
        (0, 2, "x [kpc]", "z [kpc]"),
        (1, 2, "y [kpc]", "z [kpc]"),
    ]):
        ax.plot(orbit_sat[:, xi], orbit_sat[:, yi],
                lw=0.7, alpha=0.8, color="steelblue")
        ax.scatter(posvel_sat[xi], posvel_sat[yi],
                   s=80, color="red", marker="*", zorder=5, label="Pal 5 (t=0)")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    axes[0].set_title("Galactocentric x–y")
    axes[1].set_title("Galactocentric x–z")
    axes[2].set_title("Galactocentric y–z")

    fig.tight_layout()
    out_path = Path(__file__).parents[1] / "results" / "pal5_orbit_nbody.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
