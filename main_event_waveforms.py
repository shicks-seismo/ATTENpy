#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATTENpy main script.py

Compute t* attenuation from amplitude spectra for P-waves, and optionally,
S-waves. Inverts for a single seismic moment and corner frequency for each
earthquake. t* is calculated for each source-station path.
Requirements:
    - ObsPy.
    - MTSpec (http://krischer.github.io/mtspec/).
    - NumPy.
    - SciPy.
@author: Stephen Hicks, Imperial College London.
@date: July 2020.
"""

import os
import sys
from main_routines import PS_case
from util import (calc_spec, plotsumm, invert_tstar, get_fc_range, Mw_to_M0,
                  plot_corner_freq, plot_corner_freq_v_tstar, comp_syn_spec)
from objects import Atten_DB, AEvent, Aorigin, Aarrival, Adata, Aspectrum
from obspy import read_inventory, read
from obspy.geodetics.base import gps2dist_azimuth
from amplitude_corrections import amplitude_correction
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def main_event_waveforms(cfg, iter, cat):
    """
    Attenuation computation routine.

    This outputs files to working directory.

    Parameters
    ----------
    cfg : class
        Configuration parameters.
    wdir : str
        Working directory for process.
    cat : obspy.core.event.Catalog object
        Input catalogue.
    n_proc : int
        Number of cores to use.

    Returns
    -------
    None.

    """
    wdir = "{:02g}".format(iter)

    # Initialise database (class structure)
    atten_db = Atten_DB()

    # Create log file
    log = open("logs/log{:}.txt".format(wdir), "w")

#    for n_evt, event in enumerate(tqdm(cat, total=len(cat), position=iter, desc="Proc.: {:2g}".format(iter), leave=True)):
    if cfg.inv.remove_stacorr:
        with open(cfg.inv.stacorr_file, "rb") as f:
            p_sta_corr, s_sta_corr = pickle.load(f) 

    for n_evt, event in enumerate(cat):
        arrivals_used = []
        origin = event.preferred_origin()
        ev_id = event.event_descriptions[0].text
        log.write("Working on event: {:}\n".format(ev_id))
        log.flush()

        # Skip event if either no directory of waveforms, gap too large, or not
        # enough arrivals, or magnitude to small
        if not os.path.exists("{:}/{:}".format(
                cfg.dat.waveform_dir, ev_id)):
            print("No waveform directory")
            #print("{:}/{:}".format(
            #    cfg.dat.waveform_dir, ev_id))
            # print("Skipping event - no waveform directory")
            continue
        sys.stdout.flush()

        # Get magnitude
        mag = event.preferred_magnitude()
        a_event = AEvent(origin_id=ev_id,
                         mag=mag.mag, magnitude_type=mag.magnitude_type)
        log.write("{:} = {:3.2f}\n".format(
            a_event.magnitude_type, a_event.mag))
        depth_km = origin.depth / 1000
        if os.path.exists("output/figures/{:}".format(ev_id)):
            continue   # Skiup event if duplicate
        os.makedirs("output/figures/{:}".format(ev_id))

        # Set depth to zero if above surface
        if depth_km < 0:
            depth_km = 0
            log.write("Air quake - setting depth = 0 km\n")

        # Find beta at source for stress calculation
        a_origin = Aorigin(time=origin.time, latitude=origin.latitude,
                           longitude=origin.longitude, depth_km=depth_km,
                           beta_src=cfg.earth_mod.β)
        a_event.origins.append(a_origin)
        log.write("S-wave velocity at source = {:}\n".format(cfg.earth_mod.β))

        # Loop over arrivals
        log.write("Looping over picked arrivals, importing data and "
                  "computing spectra\n")
        for arrival in origin.arrivals:
            log.flush()
            if arrival.phase not in cfg.inv.phases:
                log.write("{:} - arrival not in "
                          "tstar_cfg.phases\n".format(arrival))
                continue

            # Find corresponding pick
            found_pick = False
            for pick in event.picks:
                if (pick.resource_id == arrival.pick_id):
                    _pick = pick
                    found_pick = True
                    break
            if found_pick is False:
                _pick = []
                continue
            stapha = "{:}-{:}".format(arrival.phase,
                                      _pick.waveform_id.station_code)
            # Skip any duplicate arrivals
            if stapha in arrivals_used:
                continue
            else:
                arrivals_used.append(stapha)

            # Get trace information
            network = _pick.waveform_id.network_code
            station = _pick.waveform_id.station_code
            if station in cfg.dat.sta_exclude:
                continue
            #if network == "TR":
            #    print("TR", station)
            #    continue
#            if network == "XX":
#                continue
            location = _pick.waveform_id.location_code
            channel = _pick.waveform_id.channel_code
#            if channel[0:1] == "E":  # Skip short-period stations
#                continue
            inv = read_inventory("{}/{}.{}.xml".format(
                cfg.dat.metadata_dir, network, station))
            inv = inv.select(station=station, location=location)
            if len(inv) == 0:
                log.write(
                    "ERROR: No inventory info found or station outside AoI"
                    " for {:}.{:}.{:}.{:} - skipping\n"
                    .format(network, station, location, channel))
                continue
#            if mag.mag > 4.5 and arrival.distance < 1.0:
#            if arrival.distance < 2:
                # Exclude stations with near-field effects
#                continue

            # Calculate back-azimuth
            baz = (gps2dist_azimuth(inv[0][0].latitude, inv[0][0].longitude,
                                    a_origin.latitude, a_origin.longitude))[1]
            if cfg.inv.remove_stacorr:
                if arrival.phase == "P":
                    try:
                        stacorr = p_sta_corr[station]
                    except:
                        stacorr = [[], []]
                elif arrival.phase == "S":
                    try:
                        stacorr = s_sta_corr[station]
                    except:
                        stacorr = [[], []]
            else:
                stacorr = [[], []]
            # Store arrival information and waveform correction
            a_arrival = Aarrival(network=network, station=station,
                                 channel=channel,
                                 station_lat=inv[0][0].latitude,
                                 station_lon=inv[0][0].longitude,
                                 station_ele=inv[0][0].elevation,
                                 back_azimuth=baz, phase=arrival.phase,
                                 time=_pick.time,
                                 correction=amplitude_correction(
                                     cfg.earth_mod.mod_name, arrival.distance,
                                     a_origin.depth_km, arrival.phase),
                                 stacorrf = stacorr[0],
                                 stacorr = stacorr[1])

            # Skip arrival if no take-off angle found
            if a_arrival.correction == 0.0 or a_arrival.correction == [] or np.isnan(a_arrival.correction):
                log.write("Take-off angle not found for "
                          "dist = {:}deg, depth={:}km, "
                          "phase={:}\n".format(arrival.distance,
                                               a_origin.depth_km,
                                               arrival.phase))
                continue

            # Remove water-column multiples from window
            # TODO: This might need removing (attenuaiton of water~0)
            # if a_arrival.phase == "P" and a_arrival.station_ele < 0:
            # mult_tt = (-2 * a_arrival.station_ele) / 1500
            # if wl_p - prewin_p >= mult_tt:
            # wl_p = np.around(2 * mult_tt * 0.95, 0) / 2

            # Rotate horizontal components 1,2 -> N,E -> R,T
            # Take the transverse component for S-waves
            # Signal is generally higher for transverse (Roth et al., 1999)
            # Shawn Wei selects horizontal comp with largest amplitude

            if a_arrival.phase == "P":
                channel_read = "Z"
            elif a_arrival.phase == "S":
                channel_read = "T"

            # Read in velocity waveforms
            log.write("Reading in waveforms")
            wave_file = ("{0:}/{1:}/{1:}.{2:}.{3:}.*{4:}.msd".format(
                cfg.dat.waveform_dir, a_event.origin_id,
                a_arrival.network, a_arrival.station, channel_read))
            try:
                vel_instcorr = read(wave_file, format="MSEED")
            except Exception:
                log.write("Could not read in velocity waveform file: {:}\n"
                          .format(wave_file))
                continue
            # If multiple channels exist, then prioritise higher-rates
            if len(vel_instcorr) > 1:
                channels = [tr.stats.channel for tr in vel_instcorr]
                if [i for i in channels if i.startswith("H")]:
                    vel_instcorr = vel_instcorr.select("H*")
                elif [i for i in channels if i.startswith("B")]:
                    vel_instcorr = vel_instcorr.select("B*")
                else:
                    vel_instcorr = vel_instcorr.select("E*")

            dis_instcorr = vel_instcorr.copy().integrate()

            # Define window times
            if a_arrival.phase == "P":
                a_arrival.sig_win = [_pick.time - cfg.inv.prewin_p,
                                     _pick.time + cfg.inv.wl_p
                                     - cfg.inv.prewin_p]
                a_arrival.noise_win = [_pick.time - cfg.inv.prewin_p
                                       - cfg.inv.wl_p,
                                       _pick.time - cfg.inv.prewin_p]
                snrcrt1 = cfg.inv.snrcrtp1
                snrcrt2 = cfg.inv.snrcrtp2

            elif a_arrival.phase == "S":
                a_arrival.sig_win = [_pick.time - cfg.inv.prewin_s,
                                     _pick.time + cfg.inv.wl_s -
                                     cfg.inv.prewin_s]
                a_arrival.noise_win = [_pick.time  - 0.5 - cfg.inv.prewin_s -
                                       cfg.inv.wl_s, _pick.time - 0.5 -
                                       cfg.inv.prewin_s]
                snrcrt1 = cfg.inv.snrcrts1
                snrcrt2 = cfg.inv.snrcrts2

            # Windowing. Noise and signal should have same length.
            log.write("Windowing waveforms\n")
            signal_st = (vel_instcorr.slice(
                a_arrival.sig_win[0], a_arrival.sig_win[1])
                .detrend(type='demean'))
            noise_st = (vel_instcorr.slice(
                a_arrival.noise_win[0], a_arrival.noise_win[1])
                .detrend(type='demean'))
            data = Adata(vel_corr=vel_instcorr, dis_corr=dis_instcorr,
                         signal_vel_corr=signal_st, noise_vel_corr=noise_st)
            if (len(signal_st) == 0 or len(noise_st) == 0
                    or np.max(np.abs(signal_st[0].data)) == 0.0
                    or np.max(np.abs(noise_st[0].data)) == 0.0):
                continue
            a_arrival.data.append(data)
            

            # Calcuate high_qualty signal spectra for fc and α inversion
            log.write("Calculating high_qualty signal spectra for fc "
                      "inversion\n")
            icase = 1
            aspectrum = Aspectrum()
            (aspectrum.freq_sig, aspectrum.sig_full_dis,
             aspectrum.noise_dis, aspectrum.SNR, aspectrum.freq_good,
             aspectrum.sig_good_dis, aspectrum.fr_good, good_bool) =\
                calc_spec(data.signal_vel_corr, data.noise_vel_corr,
                          snrcrt1, cfg.inv.lincor, a_event.origin_id)
            if len(aspectrum.freq_sig) != len(aspectrum.sig_full_dis):
                good_bool = False
            if good_bool is True:
                HQ_arrival = a_arrival
                HQ_arrival.aspectrum = aspectrum
                if arrival.phase == "P":
                    a_event.p_arrivals_HQ.append(HQ_arrival)
                elif arrival.phase == "S":
                    a_event.s_arrivals_HQ.append(HQ_arrival)

            # Calculate low quality signal spectra for t* inversion.
            aspectrum = Aspectrum()
            (aspectrum.freq_sig, aspectrum.sig_full_dis,
             aspectrum.noise_dis, aspectrum.SNR, aspectrum.freq_good,
             aspectrum.sig_good_dis, aspectrum.fr_good, good_bool) =\
                calc_spec(data.signal_vel_corr, data.noise_vel_corr,
                          snrcrt2, cfg.inv.lincor, a_event.origin_id)
            if good_bool is True:
                LQ_arrival = a_arrival
                LQ_arrival.aspectrum = aspectrum
                if arrival.phase == "P":
                    a_event.p_arrivals_LQ.append(LQ_arrival)
                elif arrival.phase == "S":
                    a_event.s_arrivals_LQ.append(LQ_arrival)
            # Skip to next event if no arrivals (with found waveforms)
        if len(a_event.p_arrivals_HQ) < cfg.inv.min_arr_fc:
            log.write("Not enough HQ P arrivals - skipping\n")
            atten_db.Aevents.append(a_event)
            continue

        # Approximate corner frequency based on magnitude
        # Make list containing range of corner frequencies to account for
        # uncertainities in M and beta (Pozgay et al., 2009)
        # Mw_est = Mw_scaling(Mw_scale_type, a_event.mag_mb)
        mag_uncertainty = mag.mag_errors["uncertainty"]
        Mw_est_m = (a_event.mag ) * 1.048 - 0.418  # Lidner vs Bie relationship
        Mw_est_p = (a_event.mag ) * 1.048 - 0.418  # Lidner vs Bie relationship
        Mo_Nm_m = Mw_to_M0(Mw_est_m)
        Mo_Nm_p = Mw_to_M0(Mw_est_p)
        fc_range = get_fc_range(cfg.inv.Δσ_r, Mo_Nm_m, Mo_Nm_p, a_origin.beta_src)

        cont = False
        # 1. Calculate best corner frequency for P- and S-wave (case = 1)
        for phase in cfg.inv.phases:
            log.write("Calculating corner frequency for phase: {:}\n"
                      .format(phase))
            if phase == "P":
                if len(a_event.p_arrivals_HQ) < cfg.inv.min_arr_fc:
                    log.write("Not enough HQ P-arrivals for fc - skipping\n")
                    continue
            # If not enough S-waves to find fc, then fc_s = fc_p / 1.5
            # (Pozgay 2009; Madariaga, 1976)
            if phase == "S":
                if (len(a_event.s_arrivals_HQ) < cfg.inv.min_arr_fc):
                    log.write("Not enough HQ S-arrivals for corner freq\n")
                    skip_S = True
                    continue
                else:
                    skip_S = False

            if phase == "P":
                n_arr = len(a_event.p_arrivals_HQ)
                arrivals = a_event.p_arrivals_HQ
                constrainMoS = 0
            elif phase == "S":
                n_arr = len(a_event.s_arrivals_HQ)
                arrivals = a_event.s_arrivals_HQ
                if cfg.inv.constrainMoS == 0:
                    constrainMoS = 0
                elif cfg.inv.constrainMoS == 1:
                    constrainMoS = 1
            if phase == "P" or (phase == "S" and cfg.inv.constr_fcs == 0): 
                tsfc = np.zeros((len(fc_range), n_arr))
                for ifc, fcr in enumerate(fc_range):
                    (_, _, _, lnmomen, tstar, G, Ginv, vardat,
                     lnmomenErr, estdataErr, tstarerr, L2P) = invert_tstar(
                        a_event, fcr, phase, cfg.inv.α, constrainMoS,
                         icase=1)
                    tsfc[ifc, :] = tstar
                    if ifc == 0:
                        result = np.array([[fc_range[ifc], lnmomen, L2P,
                                            vardat, lnmomenErr, cfg.inv.α]])
                    else:
                        result = np.vstack((result, np.array(
                            [[fc_range[ifc], lnmomen, L2P, vardat, lnmomenErr,
                              cfg.inv.α]])))
                L2all = result[:, 2].tolist()
                bestresult = result[L2all.index(min(L2all))]
                magall = result[:, 1].tolist()
                bestmag = magall[np.argmin(L2all)]
                a_event.Mw_p = 2/3 * np.log10(np.exp(bestmag)*1e7) - 10.73
                
                if cfg.plt.summ_case1:
                    for arrival in a_event.p_arrivals_HQ:
                        plotsumm(a_event, arrival, snrcrt1, icase,
                                 cfg.inv.α, cfg.plt.show)

                # Plot L2P & Mw Vs. corner freqency
                if cfg.plt.l2p_fck:
                    plot_corner_freq(result, L2all, bestresult, phase, a_event,
                                     cfg.plt.show)
                if cfg.plt.fc_tstar:
                    plot_corner_freq_v_tstar(bestresult, phase, arrivals, tsfc,
                                             fc_range, a_event, cfg.plt.show)

            # Skip event if fc is at lower of upper limit of fc_range
            if phase == "P":
                fr_good_all_low_min = np.min([a.aspectrum.freq_good[0] for a in a_event.p_arrivals_HQ])
                fr_good_all_high_max = np.max([a.aspectrum.freq_good[-1] for a in a_event.p_arrivals_HQ])
                if (bestresult[0] == max(fc_range) or bestresult[0] ==
                        min(fc_range) or bestresult[0] < fr_good_all_low_min or 
                        bestresult[0] > fr_good_all_high_max):
                    log.write("Warning: best P-wave fc is at limit of fcrange."
                              "Or out of good frequency range"
                              " Skipping event\n")
                    #atten_db.Aevents.append(a_event)
                    cont = True
                    continue
                a_event.fc_p = bestresult[0]
                log.write("p-wave fc = {:5.2f}\n".format(a_event.fc_p))
                log.write("best mw = {:3.2f}\n".format(
                    2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73))

            elif phase == "S":
                if cfg.inv.constr_fcs != 0:
                    skip_S = False
                    if cfg.inv.constr_fcs == 1:
                        fcps = 1
                    elif cfg.inv.constr_fcs == 2:
                        fcps = 1.5
                    a_event.fc_s = a_event.fc_p / fcps
                elif cfg.inv.constr_fcs == 0:
                    fr_good_all_low_min = np.max([a.aspectrum.freq_good[0] for a in a_event.s_arrivals_HQ])
                    fr_good_all_high_max = np.max([a.aspectrum.freq_good[-1] for a in a_event.s_arrivals_HQ])
                    if (bestresult[0] == max(fc_range) or bestresult[0] ==
                            min(fc_range) or bestresult[0] < fr_good_all_low_min or
                            bestresult[0] > fr_good_all_high_max):
                        log.write("Warning: best S-wave fc is at limit of fcrange."
                          "Or out of good frequency range"
                          " Skipping event\n")
                        skip_S = True
                    else:
                        a_event.fc_s = float(bestresult[0])
                        log.write("s-wave corner freq = {:5.2f}\n"
                                  .format(a_event.fc_s))
                        log.write("best mw = {:3.2f}\n".format(
                            2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73))
                        skip_S = False

        if cont is True:
            atten_db.Aevents.append(a_event)
            continue

        # 2. INVERT t*(P) WITH BEST fc AND alpha (case = 2) ###################
        # Inversion
        phase = "P"
        icase = 2
        log.write("Inverting t*(P) - case 2\n")
        if len(a_event.p_arrivals_LQ) > 0 and phase in cfg.inv.phases:
            a_event = PS_case(a_event, a_event.fc_p, phase, cfg.inv.α,
                              icase, cfg.inv.min_fit_p, 0)
        else:
            log.write("Not enough LQ P-arrivals - skipping event\n")
            atten_db.Aevents.append(a_event)
            continue

        if cfg.plt.summ_case2:
            for arrival in a_event.p_arrivals_LQ:
                plotsumm(a_event, arrival, cfg.inv.snrcrtp2, icase, cfg.inv.α,
                         cfg.plt.show)

        # 3. Now invert t*(P) and Mo again with arrivals of misfit < 1
        phase = "P"
        icase = 3
        log.write("Inverting t*(P) - case 3\n")
        if phase in cfg.inv.phases and len(a_event.p_arrivals_LQ_fitting) > 2:
            a_event = PS_case(a_event, a_event.fc_p, phase, cfg.inv.α, icase,
                              cfg.inv.min_fit_p, 0)
            if cfg.plt.summ_case3:
                for arrival in a_event.p_arrivals_LQ_fitting:
                    plotsumm(a_event, arrival, cfg.inv.snrcrtp2, icase,
                             cfg.inv.α, cfg.plt.show
                             )
        # 4. Invert t*(S) with best fc and alpha (case=2) ####################
        if "S" in cfg.inv.phases and skip_S is False:
            phase = "S"
            icase = 2
            log.write("Inverting t*(S) - case 2\n")
            if len(a_event.s_arrivals_LQ) > 3 and phase in cfg.inv.phases:
                a_event = PS_case(a_event, a_event.fc_s, phase, cfg.inv.α,
                                  icase, cfg.inv.min_fit_s, cfg.inv.constrainMoS)
                if cfg.plt.summ_case2:
                    for arrival in a_event.s_arrivals_LQ:
                        plotsumm(a_event, arrival, cfg.tstar.snrcrts2,
                                 icase, cfg.inv.α, cfg.plt.show)

            # If not enough arrivals, set parameters to default (0)
            else:
                log.write("Not enough LQ S-wave arrivals\n")
                atten_db.Aevents.append(a_event)
                continue

            # 5. Now invert t*(S) and Mo again with arrivals of misfit<=1
            phase = "S"
            icase = 3
            log.write("Inverting t*(S) - case 3\n")
            if phase in cfg.inv.phases and len(a_event.s_arrivals_LQ_fitting) > 2:
                a_event = PS_case(a_event, a_event.fc_s, phase, cfg.inv.α,
                                  icase, cfg.inv.min_fit_s, cfg.inv.constrainMoS)
                if cfg.plt.summ_case3:
                    for arrival in a_event.s_arrivals_LQ_fitting:
                        plotsumm(a_event, arrival, cfg.inv.snrcrts2, icase,
                                 cfg.inv.α, cfg.plt.show)
                # Compute residual
                for n_arr, arrival in enumerate(a_event.s_arrivals_LQ_fitting):
                    lnmo = np.log(Mw_to_M0(a_event.Mw_s))
                    synspec = comp_syn_spec(arrival, lnmo, arrival.aspectrum, cfg.inv.α, a_event.fc_s)
                    a_event.s_arrivals_LQ_fitting[n_arr].resid = [arrival.aspectrum.freq_sig, 
                                                                  arrival.aspectrum.sig_full_dis - synspec]

        # Append event to attenuation databasoe
        atten_db.Aevents.append(a_event)
    log.close()
    return(atten_db)
