#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for computing spectral waveform corrections.

@author: Stephen Hicks, Imperial College London.
@date: July 2020.
"""


def vel2dens(vp):
    """
    Coefficients from Nafe-Drake curve (Brocher, 2005 BSSA) for Vp < 8.5 km/s.

    Parameters
    ----------
    vp : float
        P-wave velocity in km/s.

    Returns
    -------
    rho : float
        Density in kg.m^-3

    """
    rho = ((1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3
            - 0.0043 * vp**4 + 0.00011 * vp**5) * 1000.0)
    return(rho)


def amplitude_correction(mod_name, Δ, z_src, phase):
    """
    Compute amplitude correction.

    Parameters
    ----------
    mod_name : str
        Name of velocity model for TauP (e.g. "iasp91").
    Δ : float
        Epicentral distance (degrees).
    z_src : float
        Source depth in km.
    phase : str
        Name of phase "P", "S".

    Returns
    -------
    amom : float
        Correction factor.

    """
    import os
    import numpy as np
    import obspy.taup

    # Set constants
    Δ_inc = 0.1  # Increment in degrees for del to calculate dihddel
    rad_E = 6371.0  # Earth radius
    # End of constants

    # Make array of phases containing down-going and up-going legs
    # (see TauP manual)
    phases = [phase[0], phase[0].lower()]

    # 1. Get velocities at receiver and source depth ########################
    # Find velocity at source depth. Read in table, then find nearest vel.
    # Find receiver velocity, too.

    # ObsPy path that contains velocity model in tvel format
    vel_mod = ("{:}/{:}.tvel".format(
        os.path.join(
            os.path.dirname(os.path.abspath(obspy.taup.__file__)), "data"),
        mod_name))

    # Import table and find velocities
    vel_table = [
        (float(lyr.split()[0]), float(lyr.split()[1]), float(lyr.split()[2]))
        for n, lyr in enumerate(open(vel_mod)) if n > 1]
    vp_s = [layer for n, layer in enumerate(vel_table) if layer[0] <= z_src
            and vel_table[n+1][0] > z_src and n < len(vel_table)-1][0][1]
    vs_s = [layer for n, layer in enumerate(vel_table) if layer[0] <= z_src
            and vel_table[n+1][0] > z_src and n < len(vel_table)-1][0][2]
    vp_r = vel_table[0][1]
    vs_r = vel_table[0][2]

    # 2. Compute density at source and receiver
    rho_s = vel2dens(vp_s)
    rho_r = vel2dens(vp_r)

    # Now do geometric spreading correction ################################
    # Get first arrival from TauP for incidence angle and ray param
    taup_model = obspy.taup.TauPyModel(model=mod_name)
    arrivals = taup_model.get_travel_times(source_depth_in_km=z_src,
                                          distance_in_degree=Δ,
                                          phase_list=phases)
    if len(arrivals) == 0:
        return 0.0

    # Estimate dih/dΔ (dihdel) - change of takeoff angle with distance
    if Δ - Δ_inc >= 0:
        Δ_1 = Δ - Δ_inc
    else:
        Δ_1 = 0.0
    ang_toff_1 = taup_model.get_travel_times(
        source_depth_in_km=z_src, distance_in_degree=Δ_1,
        phase_list=phases)[0].takeoff_angle
    Δ_2 = Δ + Δ_inc
    ang_toff_2 = taup_model.get_travel_times(
        source_depth_in_km=z_src, distance_in_degree=Δ_2,
        phase_list=phases)[0].takeoff_angle
    dihddel = np.abs(
        np.deg2rad(ang_toff_2 - ang_toff_1) / np.deg2rad(Δ_2 - Δ_1))

    # Compute angles and slowness. i_0 = incidence angle; i_h = take-off angle
    i_0 = np.deg2rad(arrival.incident_angle)
    i_h = np.deg2rad(arrival.takeoff_angle)

    # Select velocity to use
    if phase == "P":
        v_source = vp_s
        v_rec = vp_r
    elif phase == "S":
        v_source = vs_s
        v_rec = vs_r

    # 1. Calculate geometric spreading correction.
    # Equation 8.70 of Lay and Wallace.
    g = np.sqrt(
        (rho_s * v_source * np.sin(i_h) * dihddel) /
        (rho_r * v_rec * np.sin(np.deg2rad(Δ)) * np.cos(i_0)))

    # 2. Now do free surface correction ######################################
    # Aki & Richards; Kennett (1991, GJI = EQ15); Yoshimoto et al (1997, PEPI)
    if phase == "P":
        p = arrival.ray_param / rad_E   # Convert s/rad-> s/km . 1 rad = rad_E
        q_p0 = np.sqrt(vp_r**-2 - p**2)   # Vertical P-wave slowness
        q_s0 = np.sqrt(vs_r**-2 - p**2)   # Vertical S-wave slowness
        # Calculate the denominator for the slowness-dependent quantities
        # Also known as the Rayleigh function (Cerveny)
        rayleighf = (q_s0**2 - p**2)**2 + 4 * p**2 * q_p0 * q_s0
        rpz = 2 * vp_r * q_p0 * (q_s0**2 - p**2) / (vs_r**2) / rayleighf
        U = rpz
#    elif phase == "SV":
#         rst = 2 * q_s0 * (q_s0**2 - p**2) / vs_r / rayleighf
#         U = rst
    elif phase == "S":  # Use SH as we are using transverse component
        U = 2.0

    # 3. Now compute the RMS (spherical average) Radiation pattern
    if phase == "P":
        radp = np.sqrt(4.0 / 15.0)
    elif phase == "S":
        radp = np.sqrt(2.0 / 5.0)

    # 5. Now combine into single correction (units: m)
    amom = (g * radp * U) / (4 * np.pi * rho_s * (rad_E * 1000) *
                             (v_source*1000)**3)
    return amom
