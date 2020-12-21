#!/usr/bin/env python
"""Various routines for AttenPy."""

import os
import shutil
import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr
import obspy
from mtspec.multitaper import mtspec
import matplotlib.pyplot as plt
from obspy.signal.util import _npts2nfft
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import MultiLineString
from netCDF4 import Dataset as netcdf_dataset
import ScientificColourMaps6 as SCM6
import matplotlib
from obspy.imaging.spectrogram import _nearest_pow_2
import pickle
import pandas as pd

font = {'family': 'Latin Modern Sans',
        'weight': 'normal',
        'size':    7}
matplotlib.rc('font', **font)
matplotlib.rcParams['lines.linewidth'] = 0.7


def calc_spec(P_signal, P_noise, snrcrt, linresid, orig_id):
    """
    Compute amplitude spectrum using the multi-taper method.

    Parameters
    ----------
    P_signal : TYPE
        DESCRIPTION.
    P_noise : TYPE
        DESCRIPTION.
    snrcrt : TYPE
        DESCRIPTION.
    linresid : TYPE
        DESCRIPTION.
    orig_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #nft = 1024
    npi = 3.0
    smlen = 11
    # Low pass filter to ensure we don't get mtspec adaptspec converge errors
#    P_noise.filter("bandpass", freqmin=0.05,
#                   freqmax=0.94 * P_noise[0].stats.sampling_rate / 2)
#    P_signal.filter("bandpass", freqmin=0.05,
#                    freqmax=0.94 * P_noise[0].stats.sampling_rate / 2)
#    nft = _npts2nfft(len(P_signal[0].data))
    nft = 1024
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
    f_max = 0.95 * P_signal[0].stats.sampling_rate / 2
    if f_max > 20.0:
        f_max = 20

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
    ev_ids = []
    for evt in cat:
        orig = evt.preferred_origin()
        if (len(orig.arrivals) > 5
                and evt.event_descriptions[0].text not in cfg.evt_exclude
                and cfg.mindepth < orig.depth/1000 < cfg.maxdepth
                and evt.event_descriptions[0].text not in ev_ids):
            cat_final.append(evt)
            ev_ids.append(evt.event_descriptions[0].text)
    return cat_final


def get_fc_range(d_stress_minmax, Mo_Nm_m, Mo_Nm_p, src_beta):
    """
    Get range of corner frequencies from min and max stress drops.

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
    fc_m = (0.49 * ((d_stress_minmax[0] / Mo_Nm_p) ** (1.0 / 3.0)) * src_beta * 100)
    fc_p = (0.49 * ((d_stress_minmax[1] / Mo_Nm_m) ** (1.0 / 3.0)) * src_beta * 100)
    fc_minmax = [fc_m, fc_p]
    if fc_minmax[0] < 1 and fc_minmax[1] <= 1.1:
        fc_range = np.arange(fc_minmax[0], fc_minmax[1], 0.02)
    elif fc_minmax[0] < 1 and fc_minmax[1] > 1.1:
        fc_range = np.hstack((np.arange(fc_minmax[0], 1.09, 0.02),
                              np.arange(1.1, fc_minmax[1], 0.1)))
    else:
        fc_range = np.arange(fc_minmax[0], fc_minmax[1], 0.1)
    if np.max(fc_range) < fc_minmax[1]:
        fc_range = np.hstack((fc_range, fc_minmax[1]))

    return fc_range


def Mw_to_M0(Mw):
    """Compute moment from moment magnitude."""
    M0_dynecm = 10**(1.5 * (Mw + 10.7))
    M0_Nm = M0_dynecm * 1e-7
    return M0_Nm


def longest_segment(snr, _snrcrt, freq_sig, maxf, minf=0.5):
    """Find longest segment of spectra with SNR > SNRCRT."""
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
            w = 1
            m.append(w)
            bindex.append(kk-w)
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


def make_dirs():
    """Make directories."""
    if os.path.isdir("output"):
        shutil.rmtree("output")
    os.makedirs("output")
    os.makedirs("output/figures")
    os.makedirs("output/events")
    os.makedirs("output/events/P")
    os.makedirs("output/events/S")


def Mw_scaling(scale_type, M_in):
    """Types available: - "Mb_ISC_Das" -  Das et al. (2011), Nat. Hazards."""
    if scale_type == "Mb_ISC_Das":
        a = 1.54
        b = -2.54
    Mw = a * M_in + b

    return Mw


def comp_syn_spec(arrival, lnmo, spectrum, alpha_, fc):
    """Compute synthetic spectrum"""
    synspec = (arrival.correction * np.exp(lnmo)
               * np.exp(-np.pi * spectrum.freq_sig
               * (spectrum.freq_sig**(-alpha_))*arrival.tstar) /
                (1+(spectrum.freq_sig/fc)**2))
    return synspec


def plotsumm(event, arrival, snrcrt, icase, alpha_, show):
    """Make summary plots."""
    from matplotlib.gridspec import GridSpec
    Tpre = 4

    origin = event.origins[0]
    spectrum = arrival.aspectrum
    if arrival.phase == "P":
        fc = event.fc_p
        Mw = event.Mw_p
    elif arrival.phase == "S":
        fc = event.fc_s
        Mw = event.Mw_s
    lnmo = np.log(Mw_to_M0(Mw))

    fig = plt.figure(figsize=(8, 5.5))
    gs = GridSpec(2, 5)
    network = arrival.data[0].vel_corr[0].stats.network
    station = arrival.data[0].vel_corr[0].stats.station
    channel = arrival.data[0].vel_corr[0].stats.channel
    plt.suptitle(
        "Origin time: {:} Lat: {:5.2f}$^\circ$ Lon: {:5.2f}$^\circ$ Depth: {:2.0f} km"
        " $M_L$={:3.1f} $M_{{w}}$={:3.1f}\n"
        "NET.STA.CHA: {:}.{:}.{:}\n{:}-wave\n".format(
            event.origins[0].time.datetime.strftime('%Y-%m-%d %H:%M:%S'),
            event.origins[0].latitude,
            event.origins[0].longitude, event.origins[0].depth_km,
            event.mag, event.Mw_p, network, station, channel, arrival.phase))

    # Corrected, filtered displacement
    ax2 = fig.add_subplot(gs[0, 0:3])
    if len(arrival.data[0].dis_corr) != 0:
        trim = (arrival.data[0].dis_corr[0].filter(
            'bandpass', freqmin=spectrum.fr_good[0],
            freqmax=spectrum.fr_good[1]).detrend(type="demean")
            .slice(arrival.noise_win[0]-Tpre, arrival.sig_win[1]+Tpre))
        ax2.plot(trim.times(reftime=origin.time), trim.data/1e-9, "b-", lw=.8)
    i, j = ax2.get_ylim()
    ax2.axvline(arrival.time-origin.time, linestyle="--", lw=0.9, color="k",
               label="Picked onset")
    ax2.axvspan(arrival.sig_win[0]-origin.time,
                arrival.sig_win[1]-origin.time,
                label="Signal window", alpha=0.25, color="red")
    ax2.axvspan(arrival.noise_win[0]-origin.time,
                arrival.noise_win[1]-origin.time,
                label="Noise window", alpha=0.25, color="green")
    ax2.text(0.02, 0.92,
             "a) Instrument-corrected, filtered displacement waveform",
             fontsize=7, transform=ax2.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    ax2.set_xlabel("Time relative to origin (s)")
    ax2.set_ylabel("Displacement (nm)")
    ax2.legend(loc='lower left')

    # Spectra
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.grid(ls="--")
    stacorr = np.array(arrival.stacorr)
    stacorr[np.isnan(stacorr)] = 0
    if len(stacorr) == 0:
        stacorr = np.zeros(len(spectrum.freq_sig))
    spec = np.exp(np.log(spectrum.sig_full_dis*1e9) + stacorr)
    ax4.semilogy(spectrum.freq_sig, spec, label="Signal",
                 color="red", lw=0.8)
    ax4.semilogy(spectrum.freq_sig, spectrum.noise_dis/1e-9, label="Noise",
                 color="green", lw=0.8)
    if icase > 1:
        synspec = comp_syn_spec(arrival, lnmo, spectrum, alpha_, fc)
        ax4.semilogy(spectrum.freq_sig, synspec/1e-9,
                     label="Synthetic fit", linestyle="--", zorder=20, lw=0.85)

    ax4.set_ylabel('Displacement (nm)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_xlim(0, 20)
    ax4.axvline(fc, label="Best event $f_c$", lw=1.05, c="brown", ls="-.")
    if arrival.phase == "P":
        ax4.set_ylim(10**-2, 10**3.5)
    elif arrival.phase == "S":
        ax4.set_ylim(10**-2, 10**5)
    i, j = ax4.get_ylim()
    ax4.vlines(spectrum.fr_good, i, j, label="SNR ratio limits",
               linestyle="dotted", lw=1.3, color="k")
    ax4.legend(loc='upper right')
    ax4.text(0.4, 0.8, "$t^*_{:}$ = {:.3f}$\pm${:.3f}\n"
             "1000/$Q$ = {:2.1f}$\pm${:2.1f}\n% fit = {:2.0f}"
             .format(arrival.phase, arrival.tstar, arrival.err,
                     1000*arrival.tstar_pathave,
                     1000*arrival.tstar_pathave_err, arrival.fit*100),
             transform=ax4.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8, lw=0.7), va="top")
    ax4.text(0.02, 0.92, "b) Amplitude spectrum", transform=ax4.transAxes,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8), fontsize=7)

    # Map
    ax5 = fig.add_subplot(gs[:, 3:], projection=ccrs.PlateCarree())
    topo = netcdf_dataset("GEBCO_2014_2D-66_-57_9.2_18.5.grd")
    sst = topo.variables['z'][:]
    lats = topo.variables['lat'][:]
    lons = topo.variables['lon'][:]
    ax5.set_extent([-64, -59, 11, 18])
    ax5.contourf(lons, lats, sst, 60, transform=ccrs.PlateCarree(), cmap=SCM6.oleron, vmin=-7000, vmax=7000)

    shp = Reader("tectonicplates/PB2002_boundaries.shp")
    add_s = shp.records()
    for add in add_s:
        if( add.geometry == None):
            pass
        else:
            lines = add.geometry
            line_good=[]
            for l in lines:
                start_pt = list(l.coords)[0]
                for i in range(1,len(l.coords)):
                    end_pt = list(l.coords)[i]
                    simple_line = (start_pt, end_pt)
                    line_good.append(simple_line)
                    start_pt = end_pt
                #end for
            #end for
            lines = MultiLineString(line_good)

            ax5.add_geometries([lines], ccrs.PlateCarree(),\
                                 edgecolor='green', facecolor=None)
    ax5.coastlines(resolution="50m") 
    ax5.set_title("c) Event to station path", fontsize=7)
    ax5.scatter(arrival.station_lon, arrival.station_lat, s=50, marker="^", linewidth=0.5, zorder=10,
                edgecolor="k", c="white", label="Station", alpha=0.7)
    ax5.scatter(origin.longitude, origin.latitude, s=60, marker="*", linewidth=0.5, zorder=10,
                edgecolor="k", c="red", label="Epicentre", alpha=0.7)
    ax5.plot([arrival.station_lon, origin.longitude], [arrival.station_lat, origin.latitude])
    ax5.legend(loc="lower right")
    g1 = ax5.gridlines(draw_labels=True, xlocs=np.arange(-64, -58, 1),
                       ylocs=np.arange(11, 19, 1))
    g1.xlines = False
    g1.ylines = False
    g1.xlabels_top = False
    g1.ylabels_left = False
    g1.xformatter = LONGITUDE_FORMATTER
    g1.yformatter = LATITUDE_FORMATTER

    if show:
        plt.show()
    else:
        plt.savefig("output/figures/{:}/{:}.{:}.case{:}.pdf".format(
            str(event.origin_id).split('/')[-1],
            arrival.station, arrival.phase, icase), transparent=True, dpi=300)
        plt.gcf().subplots_adjust(bottom=0.25)
    plt.close()


def plot_corner_freq(result, L2all, bestresult, phase, event, show):
    font = {'family': 'Arial',
            'weight': 'normal',
            'size':    9}
    matplotlib.rc('font', **font)
    """Plot norm vs corner frequency and moment vs corner frequency."""
    fig = plt.figure(1, figsize=(2.15, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax1 = fig.add_subplot(1, 1, 1)
#    ax1.set_xlim([bestresult[0]*0.75, result[:, 0].max()])
#    ax1.set_ylim([min(L2all)*0.9998, min(L2all)*1.002])
    ax1.plot(result[:, 0], L2all, '-bo', markersize=2, lw=1)
    ax1.plot(bestresult[0], min(L2all), 'r*', ms=10)
    ax1.grid(ls="--")
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_xlabel('{:}-wave corner Frequency (Hz)'.format(phase))
    ax1.set_ylabel('L2 Norm')
#    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)
#    ax2.plot(result[:, 0], 2/3 * np.log10(np.exp(result[:, 1])*1e7)-10.73,
#             'b*-')
#    ax2.plot(bestresult[0], 2/3 * np.log10(np.exp(bestresult[1])*1e7)-10.73,
#             'r^', ms=10)
#    ax2.set_xlabel('Corner Frequency (Hz)')
#    ax2.set_ylabel('Mw')
    if show:
        plt.show()
    else:
        plt.savefig("output/figures/{:}/Fc_norm-{:}.pdf".format(
            str(event.origin_id).split('/')[-1], phase), dpi=300, transparent=True)


def plot_corner_freq_v_tstar(bestresult, phase, arrivals, tsfc, fcrange,
                             event, show):
    """Plot t* vs corner frequency for each station."""
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
    Get S-wave velocity at source from 1-D velocity model.

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
    """Build G matrix."""
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
        # RHS of Eq 4 in Wei & Wiens(2018, EPSL)
        exponent = -1 * np.pi * freq_x * freq_x ** -α 
        exponent = np.array([exponent]).transpose()
        Gblock = np.atleast_3d(exponent)
        if i_arr == 0:
            G = Gblock
        else:
            oldblock = np.hstack((G, np.zeros((G.shape[0], 1, 1))))
            newblock = np.hstack((
                np.zeros((Gblock.shape[0], G.shape[1], 1)), Gblock))
            G = np.vstack((oldblock, newblock))
    G = np.hstack((np.ones((G.shape[0], 1, 1)), G))
    return G


def buildd(a_event, fc, POS, icase, lnM=0):
    """Build data matrix."""
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
        stacorr = np.array(arrival.stacorr)
        stacorr[np.isnan(stacorr)] = 0
        stacorrf = np.array(arrival.stacorrf)
        freq_x = arrival.aspectrum.freq_good
        spec_x = arrival.aspectrum.sig_good_dis
        
        stacorr_good = stacorr[np.argwhere(np.in1d(stacorrf, freq_x))][:, 0]
        if len(stacorr_good) == 0:
            stacorr_good = np.zeros(len(spec_x))
        # Displacement spectra (Brune, 1970). n=w for w^-2 models.
        # LHS of Eq 4 in Wei & Wiens(2018, EPSL)
        stad = np.array([np.log(spec_x) + stacorr_good - np.log(correc)
                         + np.log(1+(freq_x/fc)**2) - lnM]).transpose()
        if i_arr == 0:
            data = stad
        else:
            data = np.vstack((data, stad))
    return data


def invert_tstar(a_event, fc, phase, α, constr_MoS, icase):
    """Invert t*."""
    data = buildd(a_event, fc, phase, icase)
    G = buildG(a_event, α, phase, icase)
    Ginv = np.linalg.inv(np.dot(G[:, :, 0].T, G[:, :, 0]))
    model, residu = nnls(G[:, :, 0], data[:, 0])
    if constr_MoS == 0:
        lnmomen = model[0]
    elif constr_MoS == 1:
        try:
            lnmomen = np.log(10**(1.5*(a_event.Mw_p+10.73))/1e7)
        except:
            pass
    tstar = model[1:]
    L2P = residu / np.sum(data[:, 0])
    vardat = L2P / np.sqrt(data[:, 0].shape[0] - 1)
    lnmomenErr = np.sqrt(vardat*Ginv[0][0])
    estdataerr = np.dot(G[:, :, 0], model)
    tstarerr = np.sqrt(vardat * Ginv.diagonal()[1:])
    return (data, model, residu, lnmomen, tstar, G, Ginv, vardat, lnmomenErr,
            estdataerr, tstarerr, L2P)


def invert_kappa(a_event):
    """Invert for site-term."""
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
    CALCULATE HOW WELL THE SYNTHETIC SPECTRUM FITS THE DATA.

    IF THE FITTING CURVE IS BELOW THE NOISE, THEN fit = 0.
    """
    corr = arrival.correction
    spec = arrival.aspectrum.sig_full_dis
    freq = arrival.aspectrum.freq_sig
    stacorr = np.array(arrival.stacorr)
    stacorr[np.isnan(stacorr)] = 0
    if len(stacorr) == 0:
        stacorr = np.zeros(len(spec))
    spec = np.exp(np.log(spec*1e9) + stacorr) / 1e9
    frmin, frmax = arrival.aspectrum.fr_good
    invtstar = arrival.tstar
    synspec = (corr * np.exp(lnmomen) * np.exp(
        -np.pi * freq * (freq**(-alpha)) * invtstar)
        / (1 + (freq / fc)**2))
    indx = np.all([(freq >= frmin), (freq < frmax)], axis=0)
    specx = spec[indx]
    freqx = freq[indx]
    synx = synspec[indx]
    resid = (1-((np.linalg.norm(np.log(synx)-np.log(specx)))**2/(len(freqx)-1)
             / np.var(np.log(specx))))
    residual_nm = np.log(synspec*1e9) - np.log(spec*1e9)
    residual_nm = np.where((freq >= frmin) & (freq <= frmax), residual_nm, np.nan)

    df=abs(freq[1]-freq[0])
    nlowf=0
    narea=0
    for ifreq in range(len(freq)):
        if (freq[ifreq]>frmax and freq[ifreq]<15):
            if (np.log(synspec[ifreq])<np.log(spec[ifreq]) or
                    np.log(synspec[ifreq])>np.log(spec[ifreq])+1):
                narea=narea+np.log(spec[ifreq])-np.log(synspec[ifreq])
            if np.log(synspec[ifreq])>np.log(spec[ifreq])+2:
                nlowf=nlowf+5
            elif np.log(synspec[ifreq])>np.log(spec[ifreq])+1:
                nlowf=nlowf+1
        if narea<-10 and nlowf*df>3:
            resid=0
    return resid, [freq, residual_nm]


def smooth(x, window_len, window='hanning'):
    """Smooth the amplitude spectrum."""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if window_len % 2 == 0:
        window_len = window_len + 1
    s = np.r_[x[window_len - 1: 0: -1], x, x[-1: -window_len: -1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2):-int(window_len/2)]


def compute_station_corrections(directory, phases):
    with open("{:}/residuals.pkl".format(directory), "rb") as f:
        p_resid_dic, s_resid_dic = pickle.load(f)

    MIN_OBS_PLOT_STA_P = 5
    MIN_OBS_PLOT_STA_S = 5
    GRIDX = 4

    plt.rcParams["font.size"] = "12"

    for phase in phases:
        dic_out = {}
        if phase == "P":
            dic = p_resid_dic
            min_obs_plot = MIN_OBS_PLOT_STA_P
        elif phase == "S":
            dic = s_resid_dic
            min_obs_plot = MIN_OBS_PLOT_STA_S
        number_stations = len([sta for sta in dic.keys() if len(dic[sta]) > min_obs_plot])
    
        gridy = int(np.ceil(number_stations/GRIDX))
        fig, axs = plt.subplots(13, GRIDX, figsize=(18, 26))
        axs = axs.flatten()
    
        # Remove excess subplots
        for n in range(GRIDX * gridy):
            if n > number_stations-1:
                fig.delaxes(axs[n])
    
        dic_filt = {k: v for k, v in dic.items() if len(v) > min_obs_plot}
    
        for n_sta, sta in enumerate(sorted(dic_filt.keys())):
            first_event = list(dic_filt[sta])[0]
            freq = dic_filt[sta][first_event][0]
            sta_all = np.empty((len(dic_filt[sta].keys()), len(freq)))
            sta_all[:] = np.nan
            for n_evt, evt in enumerate(dic_filt[sta].keys()):
                spec = dic_filt[sta][evt][1]
                axs[n_sta].plot(dic_filt[sta][evt][0], spec, c="gray", alpha=0.2)
                sta_all[n_evt, :] = spec
            median = np.nanmedian(sta_all, axis=0)
            stdev = np.nanstd(sta_all, axis=0)

            for i in range(len(median)):
                if sum(~np.isnan(sta_all[:, i])) <= min_obs_plot:
                    median[i] = np.nan

            median = smooth(median, 10)
            stdev = smooth(stdev, 10)
            dic_out[sta] = [freq, median]
            axs[n_sta].plot(freq, median, c="red", label="Median", alpha=0.8, lw=1.4)
            axs[n_sta].plot(freq, median+stdev, c="blue",  alpha=0.6)
            axs[n_sta].plot(freq, median-stdev, c="blue",  alpha=0.6)
            axs[n_sta].set_ylim([-3, 3])
            axs[n_sta].set_xlim([0, 20])
            axs[n_sta].set_title("{}: n = {}".format(sta, len(sta_all)), fontweight='bold', fontsize=13)
            axs[n_sta].grid(ls="--", alpha=0.4, c="gray")
            if n_sta == 0:
                axs[n_sta].legend(fontsize=14)
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Frequency (Hz)", fontsize=17, fontweight='bold')
        plt.ylabel("Residual spectrum, ln($A_{{syn}}$) - ln($A_{{obs}}$) (nm)", fontsize=17, fontweight='bold')
        plt.suptitle("{}-wave station residuals".format(phase), fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.0, 1, 0.98])
        plt.savefig("{}/figures/{}-wave_stationresiduals.png".format(directory, phase), dpi=250, bbox_inches="tight", transparent=True)
        if phase == "P":
            med_res_all_p = dic_out
        elif phase == "S":
            med_res_all_s = dic_out

    with open("output/residuals_med.pkl", "wb") as f:
        pickle.dump([med_res_all_p, med_res_all_s], f)
