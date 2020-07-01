#!/usr/bin/env python
"""
AttenPy
Compute t* attenuation from amplitude spectra for P-waves, and optionally,
S-waves. Inverts for a single seismic moment and corner frequency for each
earthquake. t* is calculated for each source-station path.

Requirements:
    - ObsPy
    - MTSpec (http://krischer.github.io/mtspec/)
    - NumPy
    - SciPy
"""
import os
import shutil
import sys
from main_routines import P_case2, P_case3, S_case2, S_case3
from util import (calc_spec, plotsumm, invert_tstar,
                  get_Svel_source, get_fc_range, Mw_to_M0,
                  plot_corner_freq, plot_corner_freq_v_tstar)
from object_classes import Atten_DB, AEvent, Aorigin, Aarrival, Adata, Aspectrum
from obspy import read_inventory, read
from obspy.geodetics.base import gps2dist_azimuth
from amplitude_corrections import amplitude_correction
import numpy as np
from scipy.interpolate import interp1d


def main_event_waveforms(earthmodel_cfg, in_data_cfg, output_cfg,
                         plot_cfg, tstar_cfg, work_dir, cat, alpha_init,
                         const_alpha, alpha_depth, n_proc):
    """
    docstring
    """

    # Initialise database (class structure)
    atten_db = Atten_DB()

    # SAVE RESIDUAL AND MISFIT OVER ALL EVENTS
    p_resall = 0
    p_misfitall = 0
    p_allt = 0
    s_resall = 0
    s_misfitall = 0
    s_allt = 0

    # Remove existing figure directories
    if os.path.isdir("{:}".format(work_dir)):
        shutil.rmtree("{:}".format(work_dir, ignore_errors=True))
    os.makedirs("{:}/figures".format(work_dir))
    os.makedirs("{:}/events".format(work_dir))
    os.makedirs("{:}/events/P".format(work_dir))
    os.makedirs("{:}/events/S".format(work_dir))

    # Create log file
    log = open("{:}/log.txt".format(work_dir), "w")

    # Prepare output files
    evt_out = open(
        "{:}/{:}".format(work_dir, output_cfg.event_out), "w")
    arr_p_out = open(
        "{:}/{:}".format(work_dir, output_cfg.arrivals_p_out), "w")
    arr_s_out = open(
        "{:}/{:}".format(work_dir, output_cfg.arrivals_s_out), "w")
    fit_p_out = open("{:}/{:}".format(work_dir, output_cfg.fits_p_out), "w")
    fit_s_out = open("{:}/{:}".format(work_dir, output_cfg.fits_s_out), "w")

    # Interpolate alpha vs depth
    if const_alpha == 0:
        alpha_depth_interp = interp1d(alpha_depth[0], alpha_depth[1])

    for n_evt, event in enumerate(cat):
        arrivals_used = []
        origin = event.preferred_origin()
        ev_id = event.event_descriptions[0].text
        log.write("Working on event: {:}\n".format(ev_id))
        log.flush()
        print("P{:} - Event {:}/{:}".format(n_proc, n_evt+1, len(cat)))

        # Skip event if either no directory of waveforms, gap too large, or not
        # enough arrivals, or magnitude to small
        if not os.path.exists("{:}/{:}/{:}".format(
                in_data_cfg.root_path, in_data_cfg.waveform_dir, ev_id)):
            print("{:}/{:}/{:}".format(
                in_data_cfg.root_path, in_data_cfg.waveform_dir, ev_id))
            print("Skipping event - no waveform directory")
            continue
        sys.stdout.flush()

        # Get magnitude
        mag = event.preferred_magnitude()
        a_event = AEvent(origin_id=ev_id,
                         mag=mag.mag, magnitude_type=mag.magnitude_type)
        log.write("{:} = {:3.2f}\n".format(
            a_event.magnitude_type, a_event.mag))
        depth_km = origin.depth / 1000

        # Set depth to zero if above surface
        if depth_km < 0:
            depth_km = 0
            log.write("Air quake - setting depth = 0 km\n")

        # Choose appropriate alpha
        if const_alpha == 1:
            alpha = [alpha_init]
        elif const_alpha == 0:
            alpha = [alpha_depth_interp(depth_km).tolist()]
            log.write("Depth = {:}km, alpha = {:}\n".format(depth_km, alpha))

        # Find beta at source for stress calculation
        if earthmodel_cfg.beta_src_const == 0:
            beta_src = get_Svel_source(earthmodel_cfg.mod_name, depth_km)
        elif earthmodel_cfg.beta_src_const == 1:
            beta_src = earthmodel_cfg.beta_const
        a_origin = Aorigin(time=origin.time, latitude=origin.latitude,
                           longitude=origin.longitude, depth_km=depth_km,
                           beta_src=beta_src)
        a_event.origins.append(a_origin)
        log.write("S-wave velocity at source = {:}\n".format(beta_src))

        # Loop over arrivals
        log.write("Looping over picked arrivals, importing data and "
                  "computing spectra\n")
        for arrival in origin.arrivals:
            log.flush()
            if arrival.phase not in tstar_cfg.phases:
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
            if station == "TRNT":
                continue
            location = _pick.waveform_id.location_code
            channel = _pick.waveform_id.channel_code
            inv = read_inventory("{0:}/{1:}/{2:}.{3:}.xml".format(
                in_data_cfg.root_path, in_data_cfg.metadata_dir, network,
                station))
            inv = inv.select(station=station, location=location,
					minlatitude=in_data_cfg.min_lat, maxlatitude=in_data_cfg.max_lat)
            if len(inv) == 0:
                log.write(
                    "ERROR: No inventory info found or station outside AoI"
					" for {:}.{:}.{:}.{:} - skipping\n"
                    .format(network, station, location, channel))
                continue

            # Calculate back-azimuth
            baz = (gps2dist_azimuth(inv[0][0].latitude, inv[0][0].longitude,
                                    a_origin.latitude, a_origin.longitude))[1]

            # Store arrival information and waveform correction
            a_arrival = Aarrival(network=network, station=station,
                                 channel=channel,
                                 station_lat=inv[0][0].latitude,
                                 station_lon=inv[0][0].longitude,
                                 station_ele=inv[0][0].elevation,
                                 back_azimuth=baz, phase=arrival.phase,
                                 time=_pick.time,
                                 correction=waveform_corrections(
                                     earthmodel_cfg.mod_name,
                                     arrival.distance, a_origin.depth_km,
                                     arrival.phase))

            # Skip arrival if no take-off angle found
            if a_arrival.correction == 0.0 or a_arrival.correction == []:
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
            wave_file = ("{0:}/{1:}/{2:}/{2:}.{3:}.{4:}.*{5:}.msd".format(
                in_data_cfg.root_path, in_data_cfg.waveform_dir,
                a_event.origin_id, a_arrival.network, a_arrival.station,
                channel_read))
            try:
                vel_instcorr = read(wave_file, format="MSEED")
            except Exception as e:
                log.write("Could not read in velocity waveform file: {:}\n"
                          .format(wave_file))
                continue

            # Read in displacement waveforms
            wave_file = ("{0:}/{1:}/{2:}/{2:}.{3:}.{4:}.*{5:}_DIS.msd".format(
                in_data_cfg.root_path, in_data_cfg.waveform_dir,
                a_event.origin_id, a_arrival.network, a_arrival.station,
                channel_read))
            try:
                dis_instcorr = read(wave_file, format="MSEED")
            except Exception as e:
                log.write("Could not read in disp. waveform file: {:}\n"
                          .format(wave_file))
                dis_instcorr = []

            # Define window times
            if a_arrival.phase == "P":
                a_arrival.sig_win = [_pick.time - tstar_cfg.prewin_p,
                                     _pick.time + tstar_cfg.wl_p
                                     - tstar_cfg.prewin_p]
                a_arrival.noise_win = [_pick.time - tstar_cfg.prewin_p
                                       - tstar_cfg.wl_p,
                                       _pick.time - tstar_cfg.prewin_p]
                snrcrt1 = tstar_cfg.snrcrt_p1
                snrcrt2 = tstar_cfg.snrcrt_p2

            elif a_arrival.phase == "S":
                a_arrival.sig_win = [_pick.time - tstar_cfg.prewin_s,
                                     _pick.time + tstar_cfg.wl_s -
                                     tstar_cfg.prewin_s]
                a_arrival.noise_win = [_pick.time - tstar_cfg.prewin_s -
                                       tstar_cfg.wl_s, _pick.time -
                                       tstar_cfg.prewin_s]
                snrcrt1 = tstar_cfg.snrcrt_s1
                snrcrt2 = tstar_cfg.snrcrt_s2

            # Windowing. Noise and signal should have same length.
            log.write("Windowing waveforms\n")
            signal_st = (vel_instcorr.slice(
                a_arrival.sig_win[0], a_arrival.sig_win[1])
                .detrend(type='demean'))
            noise_st = (vel_instcorr.slice(
                a_arrival.noise_win[0], a_arrival.noise_win[1])
                .detrend(type='demean'))
            data = Adata(vel_corr=vel_instcorr, dis_corr=dis_instcorr,
                         signal_vel_corr=signal_st,
                         noise_vel_corr=noise_st)
            a_arrival.data.append(data)

            # Calcuate high_qualty signal spectra for fc and alpha inversion
            log.write("Calculating high_qualty signal spectra for fc "
                      "inversion\n")
            icase = 1
            aspectrum = Aspectrum()
            (aspectrum.freq_sig, aspectrum.sig_full_dis, aspectrum.noise_dis,
             aspectrum.SNR, aspectrum.freq_good, aspectrum.sig_good_dis,
             aspectrum.fr_good, good_bool) = calc_spec(
                 data.signal_vel_corr, data.noise_vel_corr, snrcrt1,
                 tstar_cfg.lincor, tstar_cfg.nyquist_frac)
            if good_bool is True:
                HQ_arrival = a_arrival
                HQ_arrival.aspectrum = aspectrum
                if arrival.phase == "P":
                    a_event.p_arrivals_HQ.append(HQ_arrival)
                elif arrival.phase == "S":
                    a_event.s_arrivals_HQ.append(HQ_arrival)
                if plot_cfg.plot_summ_case1 == 1:
                    plotsumm(a_event, HQ_arrival, snrcrt1, icase, alpha[0],
                             work_dir)

            # Calculate low quality signal spectra for t* inversion.
            aspectrum = Aspectrum()
            (aspectrum.freq_sig, aspectrum.sig_full_dis, aspectrum.noise_dis,
             aspectrum.SNR, aspectrum.freq_good, aspectrum.sig_good_dis,
             aspectrum.fr_good, good_bool) = calc_spec(
                 data.signal_vel_corr, data.noise_vel_corr, snrcrt2,
                 tstar_cfg.lincor, tstar_cfg.nyquist_frac)
            if good_bool is True:
                LQ_arrival = a_arrival
                LQ_arrival.aspectrum = aspectrum
                if arrival.phase == "P":
                    a_event.p_arrivals_LQ.append(LQ_arrival)
                elif arrival.phase == "S":
                    a_event.s_arrivals_LQ.append(LQ_arrival)

        # Skip to next event if no arrivals (with found waveforms)
        if len(a_event.p_arrivals_HQ) < tstar_cfg.min_arr_fc:
            log.write("Not enough HQ P arrivals - skipping\n")
            continue

        # Approximate corner frequency based on magnitude
        # Make list containing range of corner frequencies to account for
        # uncertainities in M and beta (Pozgay et al., 2009)
        # Mw_est = Mw_scaling(Mw_scale_type, a_event.mag_mb)
        Mw_est = a_event.mag
        log.write("Mw_est = {:3.2f}\n".format(Mw_est))
        Mo_Nm = Mw_to_M0(Mw_est)
        fc_range = get_fc_range(tstar_cfg.d_stress, Mo_Nm, a_origin.beta_src)

        cont = False
        # 1. Calculate best corner frequency for P- and S-wave (case = 1)
        for phase in tstar_cfg.phases:
            log.write("Calculating corner frequency for phase: {:}\n"
                      .format(phase))
            if phase == "P":
                if len(a_event.p_arrivals_HQ) < tstar_cfg.min_arr_fc:
                    log.write("Not enough HQ P-arrivals for fc - skipping\n")
                    continue
            # If not enough S-waves to find fc, then fc_s = fc_p / 1.5
            # (Pozgay 2009; Madariaga, 1976)
            if phase == "S":
                if (len(a_event.s_arrivals_HQ) < tstar_cfg.min_arr_fc or
                        tstar_cfg.constrainfcs == 1):
                    log.write("Not enough HQ S-arrivals for corner freq\n")
                    log.write("Constraining fc_s\n")
                    log.write("Using default fc_p / fc_s ratio = {:}\n"
                              .format(tstar_cfg.fcps))
                    a_event.fc_s = a_event.fc_p / tstar_cfg.fcps
                    continue

            if phase == "P":
                n_arr = len(a_event.p_arrivals_HQ)
                arrivals = a_event.p_arrivals_HQ
            elif phase == "S":
                n_arr = len(a_event.s_arrivals_HQ)
                arrivals = a_event.s_arrivals_HQ
            tsfc = np.zeros((len(fc_range), n_arr))
            for ialco, alph in enumerate(alpha):
                for ifc, fcr in enumerate(fc_range):
                    (_, _, _, _, lnmomen, tstar, G, Ginv, vardat,
                     lnmomenErr, estdataErr, tstarerr, L2P) = invert_tstar(
                         a_event, fcr, phase, alpha, alph, constrainMoS=0,
                        icase=1)
                    tsfc[ifc, :] = tstar
                    if ifc == 0:
                        result = np.array([[fc_range[ifc], lnmomen, L2P,
                                            vardat, lnmomenErr, alpha[ialco]]])
                    result = np.vstack((result, np.array(
                        [[fc_range[ifc], lnmomen, L2P, vardat, lnmomenErr,
                          alpha[ialco]]])))
            L2all = result[:, 2].tolist()
            bestresult = result[L2all.index(min(L2all))]

            # Plot L2P & Mw Vs. corner freqency
            if plot_cfg.plt_l2p_fck == 1:
                plot_corner_freq(result, L2all, bestresult, phase, a_event,
                                 work_dir)
            if plot_cfg.plt_fc_tstar == 1:
                plot_corner_freq_v_tstar(bestresult, phase, arrivals, tsfc,
                                         fc_range, a_event, work_dir)

            # Skip event if fc is at lower of upper limit of fc_range
            if phase == "P":
                if (bestresult[0] == max(fc_range) or bestresult[0] ==
                        min(fc_range)):
                    log.write("Warning: best P-wave fc is at limit of fcrange."
                              " Skipping event\n")
                    cont = True
                    continue
                a_event.fc_p = bestresult[0]
                log.write("p-wave corner freq = {:5.2f}\n"
                          .format(a_event.fc_p))
                log.write("best mw = {:3.2f}\n".format(
                    2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73))

            elif phase == "S":
                if tstar_cfg.constrainfcs == 2:
                    if ((a_event.fc_s < a_event.fc_p / tstar_cfg.fcps)
                            or bestresult[0] == max(fc_range) or
                            bestresult[0] == min(fc_range)):
                        log.write(
                            "Calculated fc_s < fc_p / fcps - Using ratio\n")
                        a_event.fc_s = a_event.fc_p / tstar_cfg.fcps
                    else:
                        a_event.fc_s = float(bestresult[0])
                elif tstar_cfg.constrainfcs == 0:
                    a_event.fc_s = float(bestresult[0])

                log.write("s-wave corner freq = {:5.2f}\n"
                          .format(a_event.fc_s))
                log.write("best mw = {:3.2f}\n".format(
                    2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73))

        if cont is True:
            continue

        # 2. INVERT t*(P) WITH BEST fc AND alpha (case = 2) ###################
        # Inversion
        phase = "P"
        icase = 2
        log.write("Inverting t*(P) - case 2\n")
        if len(a_event.p_arrivals_LQ) > 0 and phase in tstar_cfg.phases:
            a_event = P_case2(a_event, a_event.fc_p, phase, alpha[0],
                              icase, alpha, tstar_cfg.minfit_p)
        else:
            log.write("Not enough LQ P-arrivals - skipping event\n")
            continue

        if plot_cfg.plot_summ_case2 == 1:
            for arrival in a_event.p_arrivals_LQ:
                plotsumm(a_event, arrival, tstar_cfg.snrcrt_p2, icase,
                         alpha[0], work_dir)

        # 3. Now invert t*(P) and Mo again with arrivals of misfit < 1
        phase = "P"
        icase = 3
        log.write("Inverting t*(P) - case 3\n")
        if (len(a_event.p_arrivals_LQ_fitting) > 4 and phase in
                tstar_cfg.phases):
            a_event, p_resall, p_misfitall, p_allt\
                    = P_case3(a_event, a_event.fc_p, phase, alpha[0],
                              icase, alpha, tstar_cfg.minfit_p,
                              p_resall, p_misfitall, p_allt)
            if plot_cfg.plot_summ_case3 == 1:
                for arrival in a_event.p_arrivals_LQ_fitting:
                    plotsumm(a_event, arrival, tstar_cfg.snrcrt_p2, icase,
                             alpha[0], work_dir)
        else:
            log.write("Not enough good fitting P-wave records - n = {:}\n"
                      .format(len(a_event.p_arrivals_LQ_fitting)))
            continue

        # 4. Invert t*(S) with best fc and alpha (case=2) ####################
        if "S" in tstar_cfg.phases:
            phase = "S"
            icase = 2
            log.write("Inverting t*(S) - case 2\n")
            if len(a_event.s_arrivals_LQ) > 3 and phase in tstar_cfg.phases:
                a_event = S_case2(a_event, a_event.fc_s, phase, alpha[0],
                                  icase, alpha, tstar_cfg.minfit_s,
                                  tstar_cfg.constrainmos)
                if plot_cfg.plot_summ_case2 == 1:
                    for arrival in a_event.s_arrivals_LQ:
                        plotsumm(a_event, arrival, tstar_cfg.snrcrt_s2,
                                 icase, alpha[0], work_dir)

            # If not enough arrivals, set parameters to default (0)
            else:
                log.write("Not enough LQ S-wave arrivals\n")

            # 5. Now invert t*(S) and Mo again with arrivals of misfit<=1
            phase = "S"
            icase = 3
            log.write("Inverting t*(S) - case 3\n")
            if (len(a_event.s_arrivals_LQ_fitting) > 4 and phase in
                    tstar_cfg.phases):
                a_event, s_resall, s_misfitall, s_allt\
                    = S_case3(a_event, a_event.fc_s, phase, alpha[0],
                              icase, alpha, tstar_cfg.minfit_s,
                              s_resall, s_misfitall, s_allt,
                              tstar_cfg.constrainmos)
                if plot_cfg.plot_summ_case3 == 1:
                    for arrival in a_event.s_arrivals_LQ_fitting:
                        plotsumm(a_event, arrival, tstar_cfg.snrcrt_s2, icase,
                                 alpha[0], work_dir)
            else:
                log.write("Not enough good S-wave records\n")

        # Append event to attenuation database
        atten_db.Aevents.append(a_event)

        # Write event details to file
        origin = a_event.origins[0]
        evt_out = open("{:}/{:}".format(work_dir, output_cfg.event_out), "a")
        evt_out.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:9s} {:3.2f} {:8.2f} "
                      "{:8.2f} {:8.2f} {:8.2f} {:3.2f} {:3g} {:3g} {:3g} "
                      "{:3g} {:3g} {:3g}\n".format(
                          a_event.origin_id, origin.time.datetime,
                          origin.latitude, origin.longitude, origin.depth_km,
                          a_event.magnitude_type, a_event.mag,
                          a_event.Mw_p, a_event.Mw_s,
                          a_event.fc_p, a_event.fc_s, alpha[0],
                          len(a_event.p_arrivals_HQ),
                          len(a_event.s_arrivals_HQ),
                          len(a_event.p_arrivals_LQ),
                          len(a_event.s_arrivals_LQ),
                          len(a_event.p_arrivals_LQ_fitting),
                          len(a_event.s_arrivals_LQ_fitting)))
        evt_out.close()

        # Write arrival details tosevent file
        w = open("{:}/events/P/{:}.tstar".format(work_dir, a_event.origin_id),
                 "w")
        w.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:3.2f} {:5.2f}\n".format(
            a_event.origin_id, origin.time.datetime, origin.latitude,
            origin.longitude, origin.depth_km, a_event.Mw_p, a_event.fc_p))
        for arrival in a_event.p_arrivals_LQ_fitting:
            if arrival.tstar > 0.0:
                w.write("{:} {:6.2f} {:6.4f} {:4.3f}\n".format(
                    arrival.station, arrival.time-origin.time, arrival.tstar,
                    arrival.fit))
        w.close()

        if "S" in tstar_cfg.phases:
            w = open("{:}/events/S/{:}.tstar".format(work_dir,
                                                     a_event.origin_id), "w")
            w.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:3.2f} {:5.2f}\n".format(
                a_event.origin_id, origin.time.datetime, origin.latitude,
                origin.longitude, origin.depth_km, a_event.Mw_p, a_event.fc_s))
            for arrival in a_event.s_arrivals_LQ_fitting:
                if arrival.tstar > 0.0:
                    w.write("{:} {:6.2f} {:6.4f} {:4.3f}\n".format(
                        arrival.station, arrival.time-origin.time, arrival.tstar,
                        arrival.fit))
            w.close()

        # Write arrival details to file
        for arrival in a_event.p_arrivals_LQ_fitting:
            arr_p_out = open("{:}/{:}".format(work_dir,
                                              output_cfg.arrivals_p_out), "a")
            arr_p_out.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:} {:8.4f} "
                            "{:8.4f} {:5.1f} {:6.4f} {:6.4f} {:4.2f}\n"
                            .format(
                                a_event.origin_id, origin.time.datetime,
                                origin.latitude, origin.longitude,
                                origin.depth_km, arrival.station,
                                arrival.station_lat, arrival.station_lon,
                                arrival.station_ele, arrival.tstar,
                                arrival.tstar_pathave, arrival.fit))
            arr_p_out.close()
        for arrival in a_event.s_arrivals_LQ_fitting:
            arr_s_out = open("{:}/{:}".format(work_dir,
                                              output_cfg.arrivals_s_out), "a")
            arr_s_out.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:} {:8.4f} "
                            "{:8.4f} {:5.1f} {:6.4f} {:6.4f} {:4.2f}\n"
                            .format(
                                a_event.origin_id, origin.time.datetime,
                                origin.latitude, origin.longitude,
                                origin.depth_km, arrival.station,
                                arrival.station_lat, arrival.station_lon,
                                arrival.station_ele, arrival.tstar,
                                arrival.tstar_pathave, arrival.fit))
            arr_s_out.close()

    # Write misfits
    fit_p_out.write("{:} {:} {:} {:}\n".format(
        alpha[0], p_resall, p_misfitall, p_allt))
    if "S" in tstar_cfg.phases:
        fit_s_out.write("{:} {:} {:} {:}\n".format(
            alpha[0], s_resall, s_misfitall, s_allt))
    fit_p_out.close()
    fit_s_out.close()

    log.close()
