"""Example: use GCParams to query the MW globular cluster table.

Usage
-----
    pip install -e .
    python scripts/gc_example.py

This script demonstrates how to load the bundled GC parameter table and
retrieve data for a specific cluster using the GCParams helper.
"""

from streamcutter.gc import GCParams


def main():
    gcp = GCParams()  # uses the bundled data/mw_gc_parameters_orbital_structural_time.ecsv

    names = gcp.get_all_cluster_names()
    print(f"Catalogue contains {len(names)} clusters.")
    print(f"First five: {names[:5]}")

    cluster = names[0]
    row = gcp.get_row(cluster)
    print(f"\nParameters for {cluster}:")
    for col in ("Mass", "rh,m", "R_GC_orb", "lg(Trh)"):
        if col in row.colnames:
            print(f"  {col:12s} = {row[col][0]}")


if __name__ == "__main__":
    main()
