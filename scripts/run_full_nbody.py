"""Example: run a full N-body stream simulation.

Usage
-----
    pip install -e .
    python scripts/run_full_nbody.py

This script demonstrates importing streamcutter and building the host potential.
Replace the placeholder agama call with your actual simulation logic.
"""

from streamcutter.host import build_host

POT_INI = "../configs/MWPotential2014.ini"

def main():
    print("Building host potential from:", POT_INI)
    pot_host, sigma = build_host(POT_INI)
    print("Host potential built successfully.")
    print(f"  sigma(8 kpc) = {sigma(8.0):.2f} km/s")


if __name__ == "__main__":
    main()
