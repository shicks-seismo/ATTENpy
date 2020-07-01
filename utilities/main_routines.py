#!/usr/bin/env python
"""
AttenPy main routines
"""
import numpy as np
from util import fitting, invert_tstar


def P_case2(a_event, fc_p, phase, alpha_p, icase, alpha,
            min_fit_P):
    """
    docstring
    """
    constrainMoS = 0
    (data, _, _, _, a_event.lnMo_p, tstar, _, Ginv, vardat,
     lnnnomenerr, estdataerr, tstarerr, L2P) = invert_tstar(
         a_event, fc_p, phase, alpha, alpha_p, constrainMoS, icase=icase)
    # compute t* & t*  errors for each station, make array of arrivals with
    # excellent fitting
    for nnn in range(0, len(a_event.p_arrivals_LQ)):
        a_event.p_arrivals_LQ[nnn].tstar = tstar[nnn]
        ndat = len(a_event.p_arrivals_LQ[nnn].aspectrum.freq_good)
        dat = data[0: ndat+1]
        est = estdataerr[0: ndat+1]
        var = np.linalg.norm(dat-est)**2 / (ndat-2)
        a_event.p_arrivals_LQ[nnn].misfit = np.sqrt(var*(ndat-2)) / ndat
        a_event.p_arrivals_LQ[nnn].err = [np.sqrt(var*Ginv.diagonal()[nnn+1])]
        a_event.p_arrivals_LQ[nnn].fit = fitting(
            a_event.p_arrivals_LQ[nnn], a_event.lnMo_p, a_event.fc_p, alpha_p)
        if a_event.p_arrivals_LQ[nnn].fit >= min_fit_P:
            a_event.p_arrivals_LQ_fitting.append(a_event.p_arrivals_LQ[nnn])

    return a_event


def P_case3(a_event, fc_p, phase, alpha_p, icase, alpha,
            min_fit_P, P_res_all, P_misfit_all, alltP):
    """
    docstring
    """
    constrainMoS = 0
    (data, ialco, model, residu, a_event.lnMo_p, tstar, g, Ginv, vardat,
     lnnnomenerr, estdataerr, tstarerr, L2P) = invert_tstar(
        a_event, fc_p, phase, alpha, alpha_p, constrainMoS, icase=icase)

    # Add residual and misfit to totals for all events
    P_res_all = P_res_all + residu**2 / np.sum(data[:, 0]**2)
    P_misfit_all = P_misfit_all + residu / np.sum(data[:, 0])
    alltP = alltP + 1

    # Estimate t* errors for each station
    for nnn in range(0, len(a_event.p_arrivals_LQ_fitting)):
        ndat = len(a_event.p_arrivals_LQ_fitting[nnn].aspectrum.freq_good)
        dat = data[0: ndat+1]
        est = estdataerr[0: ndat+1]
        var = np.linalg.norm(dat-est)**2 / (ndat-2)
        a_event.p_arrivals_LQ_fitting[nnn].tstar = tstar[nnn]
        a_event.p_arrivals_LQ_fitting[nnn].misfit = np.sqrt(var*ndat-2) / ndat
        a_event.p_arrivals_LQ_fitting[nnn].err = [
            np.sqrt(var*Ginv.diagonal()[nnn+1])]
        a_event.p_arrivals_LQ_fitting[nnn].fit = fitting(
            a_event.p_arrivals_LQ_fitting[nnn],
            a_event.lnMo_p, a_event.fc_p, alpha_p)

    # Compute path-averaged t* for each station
        a_event.p_arrivals_LQ_fitting[nnn].tstar_pathave = (
            tstar[nnn] / (a_event.p_arrivals_LQ_fitting[nnn].time
                          - a_event.origins[0].time))

    # Compute moment magnitude
    a_event.Mw_p = 2/3 * np.log10(np.exp(a_event.lnMo_p)*1e7) - 10.73

    return a_event, P_res_all, P_misfit_all, alltP


def S_case2(a_event, fc_s, phase, alpha_p, icase, alpha,
            min_fit_S, constrainMoS):
    """
    docstring
    """
    (data, ialco, model, residu, a_event.lnMo_s, tstar, g, Ginv, vardat,
     lnnnomenerr, estdataerr, tstarerr, L2P) = invert_tstar(
        a_event, fc_s, phase, alpha, alpha_p, constrainMoS, icase=icase)
    # compute t* & t*  errors for each station, make array of arrivals with
    # excellent fitting
    for nnn in range(0, len(a_event.s_arrivals_LQ)):
        a_event.s_arrivals_LQ[nnn].tstar = tstar[nnn]
        ndat = len(a_event.s_arrivals_LQ[nnn].aspectrum.freq_good)
        dat = data[0: ndat+1]
        est = estdataerr[0: ndat+1]
        var = np.linalg.norm(dat-est)**2 / (ndat-2)
        a_event.s_arrivals_LQ[nnn].misfit = np.sqrt(var*ndat-2) / ndat
        a_event.s_arrivals_LQ[nnn].err = [np.sqrt(var*Ginv.diagonal()[nnn+1])]
        a_event.s_arrivals_LQ[nnn].fit = fitting(
            a_event.s_arrivals_LQ[nnn], a_event.lnMo_s, a_event.fc_s, alpha_p)
        if a_event.s_arrivals_LQ[nnn].fit >= min_fit_S:
            a_event.s_arrivals_LQ_fitting.append(a_event.s_arrivals_LQ[nnn])

    return a_event


def S_case3(a_event, fc_s, phase, alpha_p, icase, alpha,
            min_fit_S, S_res_all, S_misfit_all, alltS, constrainMoS):
    """
    docstring
    """
    (data, ialco, model, residu, a_event.lnMo_s, tstar, g, Ginv, vardat,
     lnnnomenerr, estdataerr, tstarerr, L2P) = invert_tstar(
        a_event, fc_s, phase, alpha, alpha_p, constrainMoS, icase=icase)

    # Add residual and misfit to totals for all events
    S_res_all = S_res_all + residu**2 / np.sum(data[:, 0]**2)
    S_misfit_all = S_misfit_all + residu / np.sum(data[:, 0])
    alltS = alltS + 1

    # Estimate t* errors for each station
    for nnn in range(0, len(a_event.s_arrivals_LQ_fitting)):
        ndat = len(a_event.s_arrivals_LQ_fitting[nnn].aspectrum.freq_good)
        dat = data[0: ndat+1]
        est = estdataerr[0: ndat+1]
        var = np.linalg.norm(dat-est)**2 / (ndat-2)
        a_event.s_arrivals_LQ_fitting[nnn].tstar = tstar[nnn]
        a_event.s_arrivals_LQ_fitting[nnn].misfit = np.sqrt(var*ndat-2) / ndat
        a_event.s_arrivals_LQ_fitting[nnn].err = [
            np.sqrt(var*Ginv.diagonal()[nnn+1])]
        a_event.s_arrivals_LQ_fitting[nnn].fit = fitting(
            a_event.s_arrivals_LQ_fitting[nnn],
            a_event.lnMo_s, a_event.fc_s, alpha_p)

    # Compute path-averaged t* for each station
        a_event.s_arrivals_LQ_fitting[nnn].tstar_pathave = (
            tstar[nnn] / (a_event.s_arrivals_LQ_fitting[nnn].time
                          - a_event.origins[0].time))

    # Compute moment magnitude
    a_event.Mw_s = 2/3 * np.log10(np.exp(a_event.lnMo_s)*1e7) - 10.73

    return a_event, S_res_all, S_misfit_all, alltS
