"""Coordinate transformation utilities for stream simulations.

Provides two functions for converting between Galactocentric Cartesian
phase-space coordinates and heliocentric observables (ICRS).
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord


def get_observed_coords(xv):
    """
    Convert Galactocentric Cartesian phase-space coordinates to heliocentric
    observables.

    Parameters
    ----------
    xv : array-like of shape (N, 6)
        Galactocentric Cartesian coordinates. Columns are
        [x (kpc), y (kpc), z (kpc), vx (km/s), vy (km/s), vz (km/s)].

    Returns
    -------
    ra : ndarray
        Right Ascension in degrees.
    dec : ndarray
        Declination in degrees.
    vlos : ndarray
        Line-of-sight velocity in km/s.
    pmra : ndarray
        Proper motion in RA (with cos δ factor) in mas/yr.
    pmde : ndarray
        Proper motion in Dec in mas/yr.
    dist : ndarray
        Heliocentric distance in kpc.
    """
    xv = np.asarray(xv, dtype=float)

    x, y, z    = xv[:, 0] * u.kpc,      xv[:, 1] * u.kpc,      xv[:, 2] * u.kpc
    vx, vy, vz = xv[:, 3] * (u.km/u.s), xv[:, 4] * (u.km/u.s), xv[:, 5] * (u.km/u.s)

    gc_frame = Galactocentric()
    c = SkyCoord(
        x=x, y=y, z=z,
        v_x=vx, v_y=vy, v_z=vz,
        frame=gc_frame,
    ).icrs

    ra   = c.ra.to_value(u.deg)
    dec  = c.dec.to_value(u.deg)
    dist = c.distance.to_value(u.kpc)
    vlos = c.radial_velocity.to_value(u.km/u.s)
    pmra = c.pm_ra_cosdec.to_value(u.mas/u.yr)
    pmde = c.pm_dec.to_value(u.mas/u.yr)

    return ra, dec, vlos, pmra, pmde, dist


def get_galactocentric_coords(
    ra_deg,
    dec_deg,
    distance_kpc,
    vlos_kms,
    pmra_masyr,
    pmdec_masyr,
):
    """
    Convert heliocentric observables to Galactocentric Cartesian coordinates.

    Parameters
    ----------
    ra_deg, dec_deg : float or array
        Right Ascension and Declination in degrees.
    distance_kpc : float or array
        Heliocentric distance in kpc.
    vlos_kms : float or array
        Line-of-sight velocity in km/s.
    pmra_masyr : float or array
        Proper motion in RA (with cos δ factor) in mas/yr.
    pmdec_masyr : float or array
        Proper motion in Dec in mas/yr.

    Returns
    -------
    xv : ndarray of shape (N, 6)
        Galactocentric Cartesian coordinates
        [x (kpc), y (kpc), z (kpc), vx (km/s), vy (km/s), vz (km/s)].
    """
    icrs = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        distance=distance_kpc * u.kpc,
        pm_ra_cosdec=pmra_masyr * u.mas/u.yr,
        pm_dec=pmdec_masyr * u.mas/u.yr,
        radial_velocity=vlos_kms * u.km/u.s,
    )

    gc_frame = Galactocentric()
    gc = icrs.transform_to(gc_frame)

    x  = gc.cartesian.x.to(u.kpc).value
    y  = gc.cartesian.y.to(u.kpc).value
    z  = gc.cartesian.z.to(u.kpc).value
    vx = gc.velocity.d_x.to(u.km/u.s).value
    vy = gc.velocity.d_y.to(u.km/u.s).value
    vz = gc.velocity.d_z.to(u.km/u.s).value

    return np.column_stack([x, y, z, vx, vy, vz])
