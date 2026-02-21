"""Example: generate N-body initial conditions for Pal 5.

Usage
-----
    pip install -e .
    python scripts/run_nbody_pal5.py

This script:
  1. Loads Pal 5 parameters from the bundled GC catalogue.
  2. Builds the MWPotential2014 host potential.
  3. Calls make_satellite_ics to sample N-body particles from a King
     potential whose scale is set by Pal 5's tidal radius.
  4. Prints a summary of the resulting particle set.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import agama

from streamcutter.coordinate import get_galactocentric_coords
from streamcutter.dynamics import GCParams, PotentialFactory
from streamcutter.nbody import king_rt_over_scaleRadius, make_satellite_ics

agama.setUnits(length=1, velocity=1, mass=1)

# --- King profile parameters (customisable) ---
KING_W0    = 5.0   # dimensionless central potential
KING_TRUNC = 1.0   # truncation parameter
FT         = 0.95  # fraction of tidal radius → truncation radius

# --- Simulation parameters ---
N_PARTICLES = 1000
RNG_SEED    = 42


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

    ra_deg   = row["RA"][0].to_value(u.deg)
    dec_deg  = row["DEC"][0].to_value(u.deg)
    dist_kpc = row["Rsun"][0].to_value(u.kpc)
    pmra     = row["mualpha"][0].to_value(u.mas/u.yr)
    pmdec    = row["mu_delta"][0].to_value(u.mas/u.yr)
    rv       = row["<RV>"][0].to_value(u.km/u.s)
    mass_sat = row["Mass"][0].to_value(u.solMass)

    # Convert observed → Galactocentric 6-D
    xv_sat = get_galactocentric_coords(
        ra_deg, dec_deg, dist_kpc, rv, pmra, pmdec
    )   # shape (1, 6)
    r_center0 = np.zeros(6)
    r_center0[:] = xv_sat[0]

    print(f"  Galactocentric posvel : {r_center0}")
    print(f"  mass                  : {mass_sat:.3e} Msun")

    # ------------------------------------------------------------------
    # 2. Build the host potential
    # ------------------------------------------------------------------
    configs = Path(__file__).parents[1] / "configs"
    pot_ini  = str(configs / "MWPotential2014.ini")
    pot_host = PotentialFactory(potentials_dir=str(configs)).host("MWPotential2014")

    # ------------------------------------------------------------------
    # 3. Compute King scale-radius ratio (rt / r0) for the chosen W0/trunc
    # ------------------------------------------------------------------
    rt_over_r0 = king_rt_over_scaleRadius(W0=KING_W0, trunc=KING_TRUNC)
    print(f"  King W0={KING_W0}, trunc={KING_TRUNC} → rt/r0 = {rt_over_r0:.4f}")

    # ------------------------------------------------------------------
    # 4. Sample N-body initial conditions
    # ------------------------------------------------------------------
    print(f"\nSampling {N_PARTICLES} N-body particles (King, ft={FT}) ...")

    f_xv, mass, initmass, r_out, r_tidal, r0 = make_satellite_ics(
        ft=FT,
        seed=RNG_SEED,
        M_SAT=mass_sat,
        Nbody=N_PARTICLES,
        pot_host=pot_host,
        r_center0=r_center0,
        KING_W0=KING_W0,
        KING_TRUNC=KING_TRUNC,
        RT_OVER_R0=rt_over_r0,
        satellite_type="king",
    )

    # ------------------------------------------------------------------
    # 5. Report results
    # ------------------------------------------------------------------
    print(f"\n  Tidal radius   : {r_tidal:.4f} kpc")
    print(f"  Truncation r   : {r_out:.4f} kpc  (ft * r_tidal)")
    print(f"  King r0        : {r0:.4f} kpc")
    print(f"  initmass       : {initmass:.3e} Msun")
    print(f"  Particles      : {f_xv.shape[0]}")
    print("\nFirst 5 particles [kpc, km/s]:")
    print(f"{'x':>10} {'y':>10} {'z':>10} {'vx':>10} {'vy':>10} {'vz':>10}")
    for ps in f_xv[:5]:
        print(" ".join(f"{v:>10.4f}" for v in ps))

    # TODO: The Nbody simulation

if __name__ == "__main__":
    main()
