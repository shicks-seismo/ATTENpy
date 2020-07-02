#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for passing objects between functions.

@author: Stephen Hicks, Imperial College London.
@date: July 2020.
"""


class Atten_DB:
    """Parent database containing events."""

    def __init__(self, Aevents=None):
        self.Aevents = Aevents or []

    def __getitem__(self, index):
        """Docstring."""
        return self.Aevents[index]


class AEvent:
    """Event structure."""

    def __init__(self, origin_id, origins=None, mag=None,
                 magnitude_type=None,
                 p_arrivals_HQ=[], p_arrivals_LQ=[],
                 s_arrivals_HQ=[], s_arrivals_LQ=[],
                 p_arrivals_LQ_fitting=[], s_arrivals_LQ_fitting=[],
                 Mw_p=None, lnMo_p=None, fc_p=None, fc_s=None, alpha_p=None,
                 Mw_s=None, lnMo_s=None, alpha_s=None):
        self.origin_id = origin_id or []
        self.mag = mag or []
        self.magnitude_type = magnitude_type or []
        self.origins = origins or []
        self.p_arrivals_HQ = p_arrivals_HQ or []
        self.p_arrivals_LQ = p_arrivals_LQ or []
        self.s_arrivals_HQ = s_arrivals_HQ or []
        self.s_arrivals_LQ = s_arrivals_LQ or []
        self.p_arrivals_LQ_fitting = p_arrivals_LQ_fitting or []
        self.s_arrivals_LQ_fitting = s_arrivals_LQ_fitting or []
        self.Mw_p = Mw_p or 99999.0
        self.fc_p = fc_p or 99999.0
        self.fc_s = fc_s or 99999.0
        self.alpha_p = alpha_p or 99999.0
        self.Mw_s = Mw_s or 99999.0
        self.alpha_s = alpha_s or 99999.0

    def to_dict(self):
        """Convert to dictionary - e.g. for making Pandas DataFrame."""
        return {
            'evt_id': self.origin_id, 'orig_time': self.origins[0].time,
            'lat': self.origins[0].latitude, 'lon': self.origins[0].longitude,
            'dep': self.origins[0].depth_km,
            'ML': self.mag, 'Mw_p': self.Mw_p, 'Mw_s': self.Mw_s,
            'fc_p': self.fc_p, 'fc_s': self.fc_s, 
            'no_p_hq': len(self.p_arrivals_HQ),
            'no_s_hq': len(self.s_arrivals_HQ),
            'no_p_lq': len(self.p_arrivals_LQ),
            'no_s_lq': len(self.s_arrivals_LQ),
            'no_p_lq_fit': len(self.p_arrivals_LQ_fitting), 
            'no_s_lq_fit': len(self.s_arrivals_LQ_fitting)
            }


class Aorigin:
    """Earthquake origin structure."""

    def __init__(self, time, latitude, longitude, depth_km, Mo=None,
                 alpha=None, beta_src=None):
        self.time = time or []
        self.latitude = latitude
        self.longitude = longitude
        self.depth_km = depth_km
        self.beta_src = beta_src or 0.0


class Aarrival:
    """Phase arrival structure."""

    def __init__(self, network, station, channel, station_lat, station_lon,
                 station_ele, back_azimuth, time, phase, correction, data=None,
                 aspectrum=None, sig_win=None, noise_win=None, tstar=None,
                 tstar_pathave=None, misfit=None, err=None, fitting=None,
                 fit=None):
        self.network = network or []
        self.station = station or []
        self.channel = channel or []
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.station_ele = station_ele or []
        self.back_azimuth = back_azimuth or []
        self.time = time or []
        self.phase = phase or []
        self.correction = correction or []
        self.data = data or []
        self.aspectrum = aspectrum or []
        self.sig_win = sig_win or []
        self.noise_win = noise_win or []
        self.tstar = tstar or 99999.0
        self.tstar_pathave = tstar_pathave or 99999.0
        self.misfit = misfit or 99999.0
        self.fit = fit or 99999.0
        self.err = err or 99999.0
        self.fitting = fitting or 99999.0


class Adata:
    """Waveform data structure."""

    def __init__(self, vel_corr, dis_corr, signal_vel_corr, noise_vel_corr):
        self.vel_corr = vel_corr
        self.dis_corr = dis_corr
        self.signal_vel_corr = signal_vel_corr
        self.noise_vel_corr = noise_vel_corr


class Aspectrum:
    """Waveform spectrum structure."""

    def __init__(self, freq_all=None, sig_full_dis=None, noise_dis=None,
                 SNR=None, freq_good=None,  sig_good_dis=None, fr_good=None):
        self.freq_all = freq_all or []
        self.sig_full_dis = sig_full_dis or []
        self.noise_dis = noise_dis or []
        self.SNR = SNR or []
        self.freq_good = freq_good or []
        self.sig_good_dis = sig_good_dis or []
        self.fr_good = fr_good or []
