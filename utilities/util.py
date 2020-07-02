#!/usr/bin/env python
"""
Various routines for AttenPy
"""

import os
import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr
import obspy
from obspy import read
from mtspec.multitaper import mtspec
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib

font = {'family': 'Arial',
        'weight': 'normal',
        'size':    7}
matplotlib.rc('font', **font)


def read_data_sds(sds_path, origin, pick, channel, prewin_P, WL_P):
    mseed_file_path = ('{0:}/{1:}/*/{2:}/{3:}.D/*.{2:}.*.{3}.D.{1:}.{4:03d}'
                       .format(sds_path, pick.time.year,
                               pick.waveform_id.station_code,
                               channel, pick.time.julday))
    try:
        st = read(mseed_file_path, format="MSEED")
    except:
        st = []
        return st

    # Check that noise window doesn't pass into previous day
    if (pick.time-prewin_P-WL_P).julday < pick.time.julday:
        newday = pick.time - prewin_P - WL_P
        mseed_file_path = (
            '{0:}/{1:}/*/{2:}/{3:}.D/*.{2:}.*.{3}.D.{1:}.{4:03d}'
            .format(sds_path, newday.year, pick.waveform_id.station_code,
                    channel, newday.julday))
        try:
            st += read(mseed_file_path, format="MSEED")
        except:
            print("Cannot read {:}.{:}.{:}-{:}.{:}".format(
                pick.waveform_id.network_code,
                pick.waveform_id.station_code,
                channel, newday.year, newday.julday))

    # Check that signal window doesn't go into next day
    elif (pick.time-prewin_P+WL_P).julday > pick.time.julday:
        newday = pick.time - prewin_P + WL_P
        mseed_file_path = (
            '{0:}/{1:}/*/{2:}/{3:}.D/*.{2:}.*.{3}.D.{1:}.{4:03d}'
            .format(sds_path, newday.year, pick.waveform_id.station_code,
                    channel, newday.julday))
        try:
            st += read(mseed_file_path, format="MSEED")
        except:
            print("Cannot read {:}.{:}.{:}-{:}.{:}".format(
                pick.waveform_id.network_code,
                pick.waveform_id.station_code,
                channel, newday.year, newday.julday))

        st.merge(fill_value=0)
    return st


def get_station_locations(station_file):
    """
    Get station location information from space delimited file of network,
    station code, lat, long, elevation.
    """
    f = open(station_file, 'r')
    STATIONS = []
    for line in f:
        station = {}
        station['network'] = line.split()[0]
        station['station'] = line.split()[1]
        station['lat'] = float(line.split()[2])
        station['lon'] = float(line.split()[3])
        station['elev'] = float(line.split()[4])
        STATIONS.append(station)
    f.close()
    return STATIONS


def calc_spec(P_signal, P_noise, snrcrt, linresid):
    """
    Calculate multi-taper spectrum
    """
    nft = 1024
    npi = 3.0
    smlen = 11
    spec_sig_vel, freq_sig = mtspec(data=P_signal[0].data,
                                    delta=P_signal[0].stats.delta,
                                    time_bandwidth=npi, nfft=nft,
                                    number_of_tapers=int(npi*2-1))
    spec_noise_vel, freq_noise = mtspec(data=P_noise[0].data,
                                        delta=P_noise[0].stats.delta,
                                        time_bandwidth=npi, nfft=nft,
                                        number_of_tapers=int(npi*2-1))
    spec_sig_vel = spec_sig_vel[1:]
    spec_noise_vel = spec_noise_vel[1:]
    freq_all = freq_sig[1:]
    spec_sig_disp = np.sqrt(spec_sig_vel) / (2*np.pi*freq_all)
    spec_noise_disp = np.sqrt(spec_noise_vel) / (2*np.pi*freq_all)

    SNR = spec_sig_disp / spec_noise_disp
    if smlen > 0:
        SNR_smooth = smooth(SNR, smlen)

    # Set maximum frequency of spectrum  0.9 is the Nyquist fraction
    f_max = 0.90 * P_signal[0].stats.sampling_rate / 2

    (begind, endind, frminp, frmaxp, frangep) = longest_segment(
        SNR_smooth, snrcrt, freq_sig, f_max)
    fr_good = [frminp, frmaxp]

    if frangep < snrcrt[1] or frminp > 4:
        goodP = False
        freq_px = []
        spec_px = []
    else:
        goodP = True

        spec_px = spec_sig_disp[begind:endind]
        freq_px = freq_sig[begind:endind]

        coeffp = np.polyfit(freq_px, np.log(spec_px), 1)
        residp = pearsonr(freq_px, np.log(spec_px))[0]
        if coeffp[0] < 0 and abs(residp) >= linresid[0]:
            goodP = True
        else:
            goodP = False

    return(freq_all, spec_sig_disp, spec_noise_disp, SNR_smooth,
           freq_px, spec_px, fr_good, goodP)


def filter_cat(cat, cfg):
    """
    Sort and filter event catalog.

    Parameters
    ----------
    cat : obspy.core.event.catalog
        Unfiltered catalog.
    cfg : obj
        Input data configuration options.

    Returns
    -------
    cat_final : obspy.core.event.catalog
        Filtered catalog.

    """
    from obspy.core.event import Catalog
    cat = Catalog(sorted(cat, key=lambda x: x.preferred_origin().time))
    cat = cat.filter("magnitude > {:}".format(cfg.min_mag),
                     "azimuthal_gap < {:}".format(cfg.max_gap))
    cat_final = Catalog()
    for evt in cat:
        orig = evt.preferred_origin()
        if (len(orig.arrivals) > 5
                and evt.event_descriptions[0].text not in cfg.sta_blacklist):
            cat_final.append(evt)
    return cat_final


def get_fc_range(d_stress_minmax, Mo_Nm, src_beta):
    """
    Get range of corner frequencies from min and max stress drops
    References for fc equation:
        -  Boore, 1983;
        -  Boore and Atkinson, 1987
        -  Lam et al., 2000
        -  Hough, 1996, Tectonophysics

    Note the extra *100 factor in fc equation.
    This is because stress drop is in Mpa instead of Pa.
    = m^1s^1[10^6m^-1kg^1s^-2]^1/3[m^2kg^1s^-2]^-1/3
    = 10^2 s^-1
    Note that other uses of this equation use stress drop in bars and moment in
    dyne-cm.

    Inputs:
        - d_stress_minmax: len=2 list of min and max stress drop [MPa]
        - Mo_Nm: Best guess of Moment [Newton-metres]
        - src_beta: S-wave velocity at location of source [m/s]
    Outputs:
        - fc_range: list containing corner frequency trial points
    """
    fc_minmax = []
    for d_stress in d_stress_minmax:
        fc = (0.49 * ((d_stress / Mo_Nm) ** (1.0 / 3.0)) * src_beta * 100)
        fc_minmax.append(fc)
    if fc_minmax[0] < 1 and fc_minmax[1] <= 1.1:
        fc_range = np.arange(fc_minmax[0], fc_minmax[1], 0.02)
    elif fc_minmax[0] < 1 and fc_minmax[1] > 1.1:
        fc_range = np.hstack((np.arange(fc_minmax[0], 1.09, 0.02),
                              np.arange(1.1, fc_minmax[1], 0.1)))
    else:
        fc_range = np.arange(fc_minmax[0], fc_minmax[1], 0.1)

    return fc_range


def Mw_to_M0(Mw):
    """
    Compute moment from moment magnitude
    """
    M0_dynecm = 10**(1.5 * (Mw + 10.7))
    M0_Nm = M0_dynecm * 1e-7
    return M0_Nm


def longest_segment(snr, _snrcrt, freq_sig, maxf, minf=0.05):
    """
    Find longest segment of spectra with SNR > SNRCRT
    """
    # Find length of spectrum with given frequency range
    lenspec = len([ifreq for ifreq in freq_sig
                   if (ifreq < maxf and ifreq >= minf)])

    # Find starting and end point indices of freqency array
    ind1 = int(min(np.nonzero(freq_sig >= minf)[0]))
    ind2 = int(max(np.nonzero(freq_sig < maxf)[0]))

    snrcrt = _snrcrt[0]
    w = 0
    m = []
    bindex = []
    eindex = []

    # Loop over each frequency point in SNR spectrum
    for kk in range(ind1+1, lenspec):

        # only first > crt
        if snr[kk] < snrcrt and snr[kk-1] >= snrcrt and kk == 1:
            m.append(w)
            bindex.append(kk-1)
            eindex.append(kk-1)
            w = 0

        # at first and continuously > crt
        elif snr[kk] >= snrcrt and snr[kk-1] >= snrcrt and kk == 1:
            w = w + 2

        # begin of continuously > crt
        elif (snr[kk] >= snrcrt and snr[kk-1] < snrcrt and kk >= 1
              and kk < (lenspec-1)):
            w = w + 1

        # continuously >= crt
        elif (snr[kk] >= snrcrt and snr[kk-1] >= snrcrt and kk > 1
              and kk < (lenspec-1)):
            w = w + 1

        # end of continuously > crt
        elif (snr[kk] < snrcrt and snr[kk-1] >= snrcrt and kk > 1
              and kk <= (lenspec-1)):
            m.append(w)
            bindex.append(kk-w)
            eindex.append(kk-1)
            w = 0

        # continuously < crt
        elif (snr[kk] < snrcrt and snr[kk-1] < snrcrt and kk >= 1
              and kk <= (lenspec-1)):
            w = 0

        # at last and continuously > crt
        elif snr[kk] >= snrcrt and snr[kk] >= snrcrt and kk == (lenspec-1):
            w = w + 1
            m.append(w)
            bindex.append(kk-w+1)
            eindex.append(kk)

        # only last > crt
        elif snr[kk] >= snrcrt and snr[kk] < snrcrt and kk == (lenspec-1):
            w = 1
            m.append(w)
            bindex.append(kk-w+1)
            eindex.append(kk)

    if len(m) == 0:
        frange = 0
        frmin = 6
        frmax = 6
        begind = 0
        endind = 0
        return begind, endind, frmin, frmax, frange

    # Now find the longest sesgent
    longest = m.index(max(m))
    frmin = freq_sig[bindex[longest]]
    frmax = freq_sig[eindex[longest]]
    frange = frmax - frmin

    # Favour the 2nd longest segment if freq < 4 Hz and longer than 1/4 of
    # longest
    if len(m) >= 2:
        for mind in list(reversed(range(len(m)))):
            mii = mind - len(m)
            longest2 = m.index(sorted(m)[mii])
            frmin2 = freq_sig[bindex[longest2]]
            if frmin2 <= 2.0:
                frmax2 = freq_sig[eindex[longest2]]
                frange2 = frmax2 - frmin2
                if (frmin2 < frmin and (4 * frange2) > frange and frange2 >
                        _snrcrt[1]):
                    frmin = frmin2
                    frmax = frmax2
                    frange = frange2
                    longest = longest2
                    break
    begind = bindex[longest]
    endind = eindex[longest]

    # Extend frequency band to low SNR
    if _snrcrt[2] < _snrcrt[0]:
        if begind > ind1 + 1:
            while (snr[begind-1] < snr[begind] and snr[begind-1] > _snrcrt[2]
                   and begind-1 > ind1 + 1):
                begind = begind-1
        if endind < ind2 - 1:
            while (snr[endind+1] < snr[endind] and snr[endind+1] > _snrcrt[2]
                   and endind+1 < ind2 - 1):
                endind = endind + 1
        frmin = freq_sig[begind]
        frmax = freq_sig[endind]
        frange = frmax - frmin
    return begind, endind, frmin, frmax, frange


def Mw_scaling(scale_type, M_in):
    """
    Types available:
    - "Mb_ISC_Das" -  Das et al. (2011), Nat. Hazards
    """

    if scale_type == "Mb_ISC_Das":
        a = 1.54
        b = -2.54
    Mw = a * M_in + b

    return Mw


def plotsumm(event, arrival, snrcrt, icase, alpha_, show):
    from matplotlib.gridspec import GridSpec
    Tpre = 2

    origin = event.origins[0]
    spectrum = arrival.aspectrum
    if arrival.phase == "P":
        fc = event.fc_p
        Mw = event.Mw_p
        lnmo = event.lnMo_p
    elif arrival.phase == "S":
        fc = event.fc_s
        Mw = event.Mw_s
        lnmo = event.lnMo_s

    fig = plt.figure(figsize=(9, 5.5))
    gs = GridSpec(2, 4)
    network = arrival.data[0].vel_corr[0].stats.network
    station = arrival.data[0].vel_corr[0].stats.station
    channel = arrival.data[0].vel_corr[0].stats.channel
    plt.suptitle("Origin time: {:}  $M_L$ = {:3.1f}\n"
                 "NET.STA.CHA: {:}.{:}.{:}\n{:}-wave\n{:}".format(
        event.origins[0].time.isoformat(), event.mag, network, station, channel, arrival.phase, arrival.fit))
    # Corrected, unfiltered velocity
    #ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
    #trim = arrival.data[0].vel_corr[0].slice(arrival.noise_win[0]-Tpre,
    #                                         arrival.sig_win[1]+Tpre)
    #ax1.plot(trim.times(reftime=origin.time), trim.data/1e-9, "b-")
    #i, j = ax1.get_ylim()
    #ax1.vlines(arrival.time-origin.time, i, j, label="Onset")
    #ax1.axvspan(arrival.sig_win[0]-origin.time,
    #            arrival.sig_win[1]-origin.time,
    #            label="Signal window", alpha=0.5, color="red")
    #ax1.axvspan(arrival.noise_win[0]-origin.time,
    #            arrival.noise_win[1]-origin.time,
    #            label="Noise window", alpha=0.5, color="green")
    #ax1.text(0.02, 0.95, "Instrument-corrected, unfiltered "
    #                     "velocity waveform",
    #         transform=ax1.transAxes,
    #         bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    #ax1.set_xlabel("Time relative to origin (s)")
    #ax1.set_ylabel("Velocity (nm/s)")
    #ax1.legend(loc='lower left')

    # Corrected, filtered displacement
    ax2 = fig.add_subplot(gs[0,0:3])
    if len(arrival.data[0].dis_corr) != 0:
        trim = (arrival.data[0].dis_corr[0].filter(
            'bandpass', freqmin=spectrum.fr_good[0],
            freqmax=spectrum.fr_good[1]).detrend(type="demean")
            .slice(arrival.noise_win[0]-Tpre, arrival.sig_win[1]+Tpre))
        ax2.plot(trim.times(reftime=origin.time), trim.data/1e-9, "b-", lw=1)
    i, j = ax2.get_ylim()
    ax2.vlines(arrival.time-origin.time, i, j, linestyle="--",
               label="Picked onset")
    ax2.axvspan(arrival.sig_win[0]-origin.time,
                arrival.sig_win[1]-origin.time,
                label="Signal window", alpha=0.25, color="red")
    ax2.axvspan(arrival.noise_win[0]-origin.time,
                arrival.noise_win[1]-origin.time,
                label="Noise window", alpha=0.25, color="green")
    ax2.text(0.02, 0.9, "a) Instrument-corrected, filtered "
                         "displacement waveform", fontsize=8,
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    ax2.set_xlabel("Time relative to origin (s)")
    ax2.set_ylabel("Displacement (nm)")
    ax2.legend(loc='lower left')

    # SNR vs. freq
    #ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=3)
    #ax3.plot(spectrum.freq_sig, 20*np.log10(spectrum.SNR))
    #ax3.set_ylabel("20log10(SNR) (dB)")
    #ax3.set_xlabel("Frequency (Hz)")
    #i, j = ax3.get_xlim()
#    ax3.hlines(20*np.log10(snrcrt[0]), i, j, label="HQ threshold",
#               linestyle="--", color="red")
 #              linestyle="--", color="blue")
#    #ax3.hlines(20*np.log10(snrcrt[0]), i, j, label="LQ threshold",
 #   ax3.set_xlim(0, 25)
 #   ax3.text(0.02, 0.95, "Signal-to-noise ratio", transform=ax3.transAxes,
 #            bbox=dict(boxstyle="round", fc="w", alpha=0.8))
 #   ax3.legend(loc='upper right')

    # Spectra
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.grid()
    ax4.semilogy(spectrum.freq_sig, spectrum.sig_full_dis/1e-9, label="Signal",
                 color="red")
    ax4.semilogy(spectrum.freq_sig, spectrum.noise_dis/1e-9, label="Noise",
                 color="green")
    #ax4.semilogy(spectrum.freq_good, spectrum.sig_good_dis/1e-9,
    #             label="Good signal", linestyle="--")
    if icase > 1:
        synspec = (arrival.correction * np.exp(lnmo)
                   * np.exp(-np.pi * spectrum.freq_sig
                            * (spectrum.freq_sig**(-alpha_))*arrival.tstar) /
                   (1+(spectrum.freq_sig/fc)**2))
        ax4.semilogy(spectrum.freq_sig, synspec/1e-9,
                     label="Synthetic fit", linestyle="--", zorder=20)
    ax4.set_ylabel('Displacement (nm)')
    ax4.set_xlabel('Frequency (Hz)')
    i, j = ax4.get_ylim()
    ax4.vlines(spectrum.fr_good, i, j, label="Signal-to-noise ratio limits",
               linestyle="dotted")
    ax4.set_xlim(0, 24)
    if arrival.phase == "P":
        ax4.set_ylim(10**-2, 10**3.5)
    elif arrival.phase == "S":
        ax4.set_ylim(10**-2, 10**5)
    ax4.legend(loc='upper right')
    ax4.text(0.5, 0.9, "$t^*_{:}$ = {:.3f}$\pm${:.3f}\n1000/Q = {:3.0f}".format(
        arrival.phase, arrival.tstar, arrival.err, 1000*arrival.tstar_pathave), transform=ax4.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8), va="top")
    ax4.text(0.02, 0.9, "b) Amplitude spectrum", transform=ax4.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8), fontsize=8)

    # Map
    ax5 = fig.add_subplot(gs[:, 3])
    ax5.set_title("c) Event to station path", fontsize=8)
    map = Basemap(projection='mill',
                  llcrnrlon=-63.5, llcrnrlat=9, urcrnrlon=-58,
                  urcrnrlat=19, resolution="i")
    map.shadedrelief()
    map.drawcoastlines()
    map.readshapefile("/Users/sph1r17/Downloads/PB2002_boundaries", color="k",
                       name='tectonic_plates', drawbounds=True)
    map.drawparallels(np.arange(-90,90, 2), labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180,180,2),labels=[0,0,0,1])
    x_sta, y_sta = map(arrival.station_lon, arrival.station_lat)
    ax5.scatter(x_sta, y_sta, s=50, marker="^", linewidth=0.5, zorder=10,
             edgecolor="k", c="white", label="Station")
    x_evt, y_evt = map(origin.longitude, origin.latitude)
    ax5.scatter(x_evt, y_evt, s=50, marker="*", linewidth=0.5, zorder=10,
             edgecolor="k", c="red", label="Epicentre")
    ax5.text(x_evt*1.10, y_evt, "Z = {:3.0f} km".format(origin.depth_km))
    map.plot([x_sta, x_evt], [y_sta, y_evt])
    ax5.legend(loc="lower right")
    

    #ax6 = plt.subplot2grid((4, 4), (3, 3), rowspan=1)
    #ax6.text(0, 0.98, "fc = {:3.1f} Hz\n{:} = {:2.1f}\n"
    #                  "Mw = {:3.1f}\nt* = {:5.3f}\nfit = {:4.2f}"
    #         .format(fc, event.magnitude_type, event.mag, event.Mw_p,
    #                 arrival.tstar, arrival.fit),
    #         transform=ax6.transAxes,
 #   ax6.axis("off")
#    #         bbox=dict(boxstyle="round", fc="w", alpha=0.8))

    if show:
        plt.show()
    else:
        plt.savefig("output/figures/{:}/{:}.{:}.case{:}.png".format(
            str(event.origin_id).split('/')[-1],
            arrival.station, arrival.phase, icase), transparent=True, dpi=300)
        plt.gcf().subplots_adjust(bottom=0.25)
        fig.clf()


def plot_corner_freq(result, L2all, bestresult, phase, event, show):
    """
    Plot norm vs corner frequency and moment vs corner frequency
    """
    #plt.clf()
    fig = plt.figure(1, figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(result[:, 0], L2all, 'b*-')
    ax1.plot(bestresult[0], min(L2all), 'r^', ms=10)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_xlabel('Corner Frequency (Hz)')
    ax1.set_ylabel('L2 Norm')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(result[:, 0], 2/3 * np.log10(np.exp(result[:, 1])*1e7)-10.73,
             'b*-')
    ax2.plot(bestresult[0], 2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73,
             'r^', ms=10)
    ax2.set_xlabel('Corner Frequency (Hz)')
    ax2.set_ylabel('Mw')
    if show:
        plt.show()
    else:
        plt.savefig("output/figures/{:}/Fc_norm-{:}.png".format(
            str(event.origin_id).split('/')[-1], phase))


def plot_corner_freq_v_tstar(bestresult, phase, arrivals, tsfc, fcrange,
                             event, show):
    """
    Plot t* vs corner frequency for each station
    """
    for n, sta in enumerate(arrivals):
        plt.clf()
        fig = plt.figure(2)
        if np.mean(tsfc[:, n]) == 0.0:
            continue
        tspert = (tsfc[:, n]-np.mean(tsfc[:, n]))/np.mean(tsfc[:, n])*100
        plt.plot(fcrange, tspert, 'b*-')
        plt.plot(bestresult[0], tspert[fcrange == bestresult[0]], 'r^', ms=10)
        plt.xlabel('{:}-wave corner frequency (Hz)'.format(phase))
        plt.ylabel('t* perturbation (%)')
        if show:
            plt.show()
        else:
            plt.savefig("output/figures/{:}/Fc-tstar-{:}-{:}.jpg".format(
                str(event.origin_id).split('/')[-1],
                phase, arrivals[n].station))


def get_Svel_source(mod_name, z_src):
    """
    Get S-wave velocity at source from 1-D velocity model

    Inputs:
        - mod_name = name of velocity model (e.g. iasp91)
        - z_src = source depth (km)
    Output parameters:
        - beta_src = S-wave velocity at source (m/s)
    """

    # ObsPy path that contains velocity model in tvel format
    vel_mod = ("{:}/{:}.tvel".format(
        os.path.join(
            os.path.dirname(os.path.abspath(obspy.taup.__file__)), "data"),
        mod_name))

    # Import table and find velocities
    vel_table = [
        (float(lyr.split()[0]), float(lyr.split()[1]),
         float(lyr.split()[2]))
        for n, lyr in enumerate(open(vel_mod)) if n > 1]
    beta_src = [layer for n, layer in enumerate(vel_table) if layer[0] <= z_src
                and vel_table[n+1][0] > z_src and n < len(vel_table)-1][0][2]

    return beta_src * 1000.0


def buildG(a_event, α, POS, icase):
    """
    Build G matrix.
    """

    if POS == "P" and icase == 1:
        arrivals = a_event.p_arrivals_HQ
    elif POS == "P" and icase == 2:
        arrivals = a_event.p_arrivals_LQ
    elif POS == "P" and icase == 3:
        arrivals = a_event.p_arrivals_LQ_fitting
    elif POS == "S" and icase == 1:
        arrivals = a_event.s_arrivals_HQ
    elif POS == "S" and icase == 2:
        arrivals = a_event.s_arrivals_LQ
    elif POS == "S" and icase == 3:
        arrivals = a_event.s_arrivals_LQ_fitting

    for i_arr, arrival in enumerate(arrivals):
        freq_x = arrival.aspectrum.freq_good
        exponent = -1 * np.pi * freq_x * (freq_x ** (-1*α))
        Gblock = np.array([exponent]).transpose()
        if i_arr == 0:
            G = Gblock
        else:
            oldblock = np.hstack((G, np.zeros((G.shape[0], 1))))
            newblock = np.hstack((
                np.zeros((Gblock.shape[0], G.shape[1])),
                    Gblock))
            G = np.vstack((oldblock, newblock))
    G = np.hstack((np.ones((G.shape[0], 1)), G))
    return G


def buildd(a_event, fc, POS, icase, lnM=0):
    """
    Build data matrix
    """
    if POS == "P" and icase == 1:
        arrivals = a_event.p_arrivals_HQ
    elif POS == "P" and icase == 2:
        arrivals = a_event.p_arrivals_LQ
    elif POS == "P" and icase == 3:
        arrivals = a_event.p_arrivals_LQ_fitting
    elif POS == "S" and icase == 1:
        arrivals = a_event.s_arrivals_HQ
    elif POS == "S" and icase == 2:
        arrivals = a_event.s_arrivals_LQ
    elif POS == "S" and icase == 3:
        arrivals = a_event.s_arrivals_LQ_fitting

    for i_arr, arrival in enumerate(arrivals):
        correc = arrival.correction
        freq_x = arrival.aspectrum.freq_good
        spec_x = arrival.aspectrum.sig_good_dis

        # Displacement spectra (Brune, 1970). n=w for w^-2 models.
        stad = np.array([np.log(spec_x) - np.log(correc)
                         + np.log(1+(freq_x/fc)**2)-lnM]).transpose()
        if i_arr == 0:
            data = stad
        else:
            data = np.vstack((data, stad))

    return data


def invert_tstar(a_event, fc, phase, α, constr_MoS, icase):
    """
    Invert t* etc
    """
    data = buildd(a_event, fc, phase, icase)
    G = buildG(a_event, α, phase, icase)
    Ginv = np.linalg.inv(np.dot(G[:, :].transpose(), G[:, :]))
    try:
        model, residu = nnls(G[:, :], data[:, 0])
    except:
        print(data[:, 0])
    #if constr_MoS == 0:
    lnmomen = model[0]
    #else:
    #    lnmomen = a_event.lnMo_p
    tstar = model[1:]
    L2P = residu / np.sum(data[:, 0])
    vardat = residu / np.sqrt(data[:, 0].shape[0] - 2)
    lnmomenErr = np.sqrt(vardat*Ginv[0][0])
    estdataerr = np.dot(G[:, :], model)
    tstarerr = np.sqrt(vardat * Ginv.diagonal()[1:])
    return (data, model, residu, lnmomen, tstar, G, Ginv, vardat,
            lnmomenErr, estdataerr, tstarerr, L2P)


def invert_kappa(a_event):
    """
    Invert t* etc
    """
    data = buildd(a_event, fc, phase, icase)
    G = buildG(a_event, alpha, phase, icase)
    ialco = alpha.index(bestalpha)
    Ginv = np.linalg.inv(
        np.dot(G[:, :, ialco].transpose(), G[:, :, ialco]))
    model, residu = nnls(G[:, :, ialco], data[:, 0])
    if constrainMoS == 0:
        lnmomen = model[0]
    else:
        lnmomen = a_event.lnMo_p
    tstar = model[1:]
    L2P = residu / np.sum(data[:, 0])
    vardat = residu / np.sqrt(data[:, 0].shape[0] - 2)
    lnmomenErr = np.sqrt(vardat*Ginv[0][0])
    estdataerr = np.dot(G[:, :, ialco], model)
    tstarerr = np.sqrt(vardat * Ginv.diagonal()[1:])

    return (data, ialco, model, residu, lnmomen, tstar, G, Ginv, vardat,
            lnmomenErr, estdataerr, tstarerr, L2P)


def fitting(arrival, lnmomen, fc, alpha):
    """
    CALCULATE HOW WELL THE SYNTHETIC SPECTRUM FITS THE DATA
    IF THE FITTING CURVE IS BELOW THE NOISE, THEN resid = 999999.
    """

    corr = arrival.correction
    spec = arrival.aspectrum.sig_good_dis
    freq = arrival.aspectrum.freq_good
    frmin = arrival.aspectrum.fr_good[0]
    frmax = arrival.aspectrum.fr_good[1]
    invtstar = arrival.tstar
    synspec = (corr * np.exp(lnmomen) * np.exp(
        -np.pi*freq*(freq**(-alpha))*invtstar)
        / (1+(freq/fc)**2))
    indx = np.all([(freq >= frmin), (freq < frmax)], axis=0)
    specx = spec[indx]
    freqx = freq[indx]
    synx = synspec[indx]
    resid = (1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             / np.var(np.log(specx))))
    return resid


def smooth(x,window_len,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    if window_len % 2 ==0:
        window_len = window_len + 1
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2):-int(window_len/2)]
