"""Simulate tidal disruption of Pal 5 using create_mock_stream_fardal15.

Usage
-----
    pip install -e .
    python scripts/create_stream_pal5.py

This script:
  1. Loads Pal 5 parameters from the bundled GC catalogue.
  2. Builds the MWPotential2014 host potential.
  3. Uses create_mock_stream_fardal15 (Fardal+15 particle-spray method) to simulate
     3 orbits of tidal disruption.
  4. Prints the resulting 6-D phase-space coordinates of the stream
     particles.
"""

from pathlib import Path

import numpy as np
import agama

from streamcutter.dynamics import GCParams
from streamcutter.stream_generator import (
    create_mock_stream_fardal15,
    create_initial_condition_fardal15,
    get_observed_coords,
)

# Agama unit system: 1 kpc, 1 km/s, 1 Msun
# The corresponding time unit is 1 kpc / (1 km/s) ≈ 0.9778 Gyr.
agama.setUnits(length=1, velocity=1, mass=1)

# Conversion factor: Myr → Agama time units
# 1 Agama TU ≈ 0.9778 Gyr  →  1 Myr = 1e-3 Gyr / 0.9778 (Gyr / Agama TU)
_GYR_PER_AGAMA_TU = 0.9778


def main():
    # ------------------------------------------------------------------
    # 1. Load Pal 5 parameters from the GC catalogue
    # ------------------------------------------------------------------
    gcp = GCParams()
    matches = gcp.find_cluster("Pal 5")
    if not matches:
        raise RuntimeError("Pal 5 not found in the GC catalogue.")
    cluster_name = matches[0]
    row = gcp.get_row(cluster_name)

    print(f"Cluster: {cluster_name}")

    # 6-D phase-space coordinates (kpc and km/s)
    posvel_sat = np.array([
        float(row["X_gc"][0]),
        float(row["Y_gc"][0]),
        float(row["Z_gc"][0]),
        float(row["Vx_gc"][0]),
        float(row["Vy_gc"][0]),
        float(row["Vz_gc"][0]),
    ])

    mass_sat = float(row["Mass"][0])           # [Msun]
    rhm_kpc  = float(row["rh,m"][0]) * 1e-3   # half-mass radius [pc → kpc]

    # Orbital period in Agama time units
    orbit_period_myr = float(row["orbit_period_max"][0])   # [Myr]
    orbit_period     = orbit_period_myr * 1e-3 / _GYR_PER_AGAMA_TU  # [Agama TU]

    print(f"  posvel       : {posvel_sat}")
    print(f"  mass         : {mass_sat:.3e} Msun")
    print(f"  rh,m         : {float(row['rh,m'][0]):.2f} pc")
    print(f"  orbit period : {orbit_period_myr:.1f} Myr  ({orbit_period:.3f} Agama TU)")

    # ------------------------------------------------------------------
    # 2. Build the host potential
    # ------------------------------------------------------------------
    pot_ini  = str(Path(__file__).parents[1] / "configs" / "MWPotential2014.ini")
    pot_host = agama.Potential(pot_ini)

    # ------------------------------------------------------------------
    # 3. Satellite potential (Plummer sphere)
    # ------------------------------------------------------------------
    pot_sat = agama.Potential(type="Plummer", mass=mass_sat, scaleRadius=rhm_kpc)

    # ------------------------------------------------------------------
    # 4. Simulate tidal disruption for 3 orbits (backward integration)
    # ------------------------------------------------------------------
    n_orbits      = 3
    time_total    = -n_orbits * orbit_period   # negative → integrate backward
    num_particles = 1000
    rng           = np.random.default_rng(seed=42)

    print(
        f"\nRunning create_mock_stream_fardal15 for {n_orbits} orbits "
        f"(time_total = {time_total:.3f} Agama TU, "
        f"{num_particles} particles) ..."
    )

    time_sat, orbit_sat, xv_stream, ic_stream = create_mock_stream_fardal15(
        create_initial_condition_fardal15,
        rng,
        time_total,
        num_particles,
        pot_host,
        posvel_sat,
        mass_sat,
        pot_sat=pot_sat,
    )

    # ------------------------------------------------------------------
    # 5. Report results — 6-D phase space of the stream
    # ------------------------------------------------------------------
    print(f"\nSatellite orbit : {orbit_sat.shape[0]} steps")
    print(f"Stream particles: {xv_stream.shape[0]}")
    print("\n6-D phase-space of first 5 stream particles [kpc, km/s]:")
    print(f"{'x':>10} {'y':>10} {'z':>10} {'vx':>10} {'vy':>10} {'vz':>10}")
    for ps in xv_stream[:5]:
        print(" ".join(f"{v:>10.4f}" for v in ps))

    # ------------------------------------------------------------------
    # 6. Convert to observed (heliocentric) coordinates
    # ------------------------------------------------------------------
    ra, dec, vlos, pmra, pmde, dist = get_observed_coords(xv_stream)
    print("\nObserved coords of first 5 stream particles:")
    print(f"{'RA (deg)':>10} {'Dec (deg)':>10} {'dist (kpc)':>11} "
          f"{'vlos (km/s)':>12} {'pmRA (mas/yr)':>14} {'pmDec (mas/yr)':>15}")
    for i in range(min(5, len(ra))):
        print(
            f"{ra[i]:>10.4f} {dec[i]:>10.4f} {dist[i]:>11.4f} "
            f"{vlos[i]:>12.4f} {pmra[i]:>14.4f} {pmde[i]:>15.4f}"
        )


if __name__ == "__main__":
    main()
