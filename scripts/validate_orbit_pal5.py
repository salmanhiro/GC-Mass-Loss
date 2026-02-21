"""Validate backward orbit integration for Pal 5 using integrate_orbit.

Usage
-----
    pip install -e .
    python scripts/validate_orbit_pal5.py

This script:
  1. Loads Pal 5 parameters from the bundled GC catalogue.
  2. Builds the MWPotential2014 host potential.
  3. Integrates the Pal 5 orbit backward for 3 orbital periods using
     integrate_orbit from stream_generator.py.
  4. Plots and saves the orbit in Galactocentric x-y, x-z, and y-z projections.
"""

from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import agama

from streamcutter.dynamics import GCParams, PotentialFactory
from streamcutter.stream_generator import integrate_orbit

agama.setUnits(length=1, velocity=1, mass=1)

_GYR_PER_AGAMA_TU = 0.9778   # 1 Agama TU ≈ 0.9778 Gyr

N_ORBITS   = 3
NUM_STEPS  = 500


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
    orbit_period_myr = row["orbit_period_max"][0].to_value(u.Myr)
    orbit_period     = orbit_period_myr * 1e-3 / _GYR_PER_AGAMA_TU

    print(f"Cluster: {matches[0]}")
    print(f"  posvel       : {posvel_sat}")
    print(f"  orbit period : {orbit_period_myr:.1f} Myr  ({orbit_period:.3f} Agama TU)")

    # ------------------------------------------------------------------
    # 2. Build host potential
    # ------------------------------------------------------------------
    configs_dir = Path(__file__).parents[1] / "configs"
    pot_host    = PotentialFactory(potentials_dir=str(configs_dir)).host("MWPotential2014")

    # ------------------------------------------------------------------
    # 3. Integrate orbit backward for N_ORBITS periods
    # ------------------------------------------------------------------
    time_total = -N_ORBITS * orbit_period   # negative → backward
    print(f"\nIntegrating orbit backward for {N_ORBITS} orbits "
          f"(time_total = {time_total:.3f} Agama TU, {NUM_STEPS} steps) ...")

    times, orbit = integrate_orbit(pot_host, posvel_sat, time_total, NUM_STEPS)

    print(f"  Orbit shape  : {orbit.shape}")
    print(f"  t range      : [{times[0]:.3f}, {times[-1]:.3f}] Agama TU")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Pal 5 backward orbit — {N_ORBITS} orbits via integrate_orbit", fontsize=13)

    for ax, (xi, yi, lx, ly) in zip(axes, [
        (0, 1, "x [kpc]", "y [kpc]"),
        (0, 2, "x [kpc]", "z [kpc]"),
        (1, 2, "y [kpc]", "z [kpc]"),
    ]):
        ax.plot(orbit[:, xi], orbit[:, yi], lw=0.7, alpha=0.8, color="steelblue")
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
    out_path = Path(__file__).parents[1] / "results" / "pal5_orbit_integrate_orbit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
