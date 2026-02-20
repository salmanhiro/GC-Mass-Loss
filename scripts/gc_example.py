"""Example: use GCParams to query the MW globular cluster table.

Usage
--
    pip install -e .
    python scripts/gc_example.py

This script demonstrates how to load the bundled GC parameter table and
retrieve data for a specific cluster using the GCParams helper, including
fuzzy name matching.
"""

from streamcutter.dynamics import GCParams


def main():
    gcp = GCParams()  # uses the bundled data/mw_gc_parameters_orbital_structural_time.ecsv

    names = gcp.get_all_cluster_names()
    print(f"Catalogue contains {len(names)} clusters.")
    print(f"First five: {names[:5]}")

    #  exact lookup by catalogue name 
    cluster = names[0]
    row = gcp.get_row(cluster)
    print(f"\nParameters for {cluster}:")
    for col in ("Mass", "rh,m", "R_GC_orb", "lg(Trh)"):
        if col in row.colnames:
            print(f"  {col:12s} = {row[col][0]}")

    #  fuzzy lookup: "NGC 4590" resolves to "NGC_4590" in the catalogue 
    query = "NGC 4590"
    matches = gcp.find_cluster(query)
    if matches:
        matched_name = matches[0]
        print(f"\nFuzzy search '{query}' -> '{matched_name}'")
        row2 = gcp.get_row(matched_name)
        for col in ("Mass", "rh,m", "R_GC_orb", "lg(Trh)"):
            if col in row2.colnames:
                print(f"  {col:12s} = {row2[col][0]}")
    else:
        print(f"\nNo match found for '{query}'")


if __name__ == "__main__":
    main()
