#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main t* inversion routines for different cases of P- and S-waves.

@author: Stephen Hicks, Imperial College London.
@date: July 2020.
"""

import numpy as np
from util import fitting, invert_tstar


def PS_case(a_event, fc, phase, α, icase, min_fit, res_all, misfit_all, allt):
    """
    

    Parameters
    ----------
    a_event : class
        Event object.
    fc_p : float
        P-wave corner frequency.
    phase : str
        Seismic phase label.
    α : float
        Frequency dependent term.
    icase : int
        Iterative stage.
    min_fit : float
        Minimum fit to use phase.

    Returns
    -------
    a_event : class
        Event object.

    """
    constrainMoS = 0
    (data, model, residu, lnMo, tstar, g, Ginv, vardat,
     lnnnomenerr, estdataerr, tstarerr, L2P) = invert_tstar(
             a_event, fc, phase, α, constrainMoS, icase=icase)
    # Add residual and misfit to totals for all events
    res_all = res_all + residu**2 / np.sum(data[:, 0]**2)
    misfit_all = misfit_all + residu / np.sum(data[:, 0])
    allt = allt + 1

    if phase == "P" and icase == 2:
        arrivals = a_event.p_arrivals_LQ
    elif phase == "P" and icase == 3:
        arrivals = a_event.p_arrivals_LQ_fitting
    elif phase == "S" and icase == 2:
        arrivals = a_event.s_arrivals_LQ
    elif phase == "S" and icase == 3:
        arrivals = a_event.s_arrivals_LQ_fitting

    # compute t* & t*  errors for each station, make array of arrivals with
    # excellent fitting
    for nnn, arr in enumerate(arrivals):
        arr.tstar = tstar[nnn]
        ndat = len(arrivals[nnn].aspectrum.freq_good)
        dat = data[0: ndat+1]
        est = estdataerr[0: ndat+1]
        var = np.linalg.norm(dat-est)**2 / (ndat-2)
        arr.misfit = np.sqrt(var*(ndat-2)) / ndat
        arr.err = np.sqrt(var*Ginv.diagonal()[nnn+1])
        arr.fit = fitting(arr, lnMo, fc, α)
        arr.tstar_pathave = (arr.tstar / (arr.time - a_event.origins[0].time))
        if icase == 2:
            if arr.fit >= min_fit:
                if phase == "P":
                    a_event.p_arrivals_LQ_fitting.append(arr)
                elif phase == "S":
                    a_event.s_arrivals_LQ_fitting.append(arr)
        elif icase == 3:
            if phase == "P":
                a_event.p_arrivals_LQ_fitting[nnn] = arr
            elif phase == "S":
                a_event.s_arrivals_LQ_fitting[nnn] = arr
    if phase == "P":
        a_event.Mw_p = 2/3 * np.log10(np.exp(lnMo)*1e7) - 10.73  # Mw
    elif phase == "S":
        a_event.Mw_s = 2/3 * np.log10(np.exp(lnMo)*1e7) - 10.73  # Mw
    return a_event, res_all, misfit_all, allt
