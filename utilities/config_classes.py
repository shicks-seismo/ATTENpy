#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for passing configurations between functions.

@author: Stephen Hicks, Imperial College London.
@date: July 2020.
"""


class Earthmodel_cfg:
    """Earth model configuration."""

    def __init__(self, mod_name, beta_const, beta_src_const):
        self.mod_name = mod_name
        self.beta_const = beta_const
        self.beta_src_const = beta_src_const


class In_data_cfg:
    """Input data configuration."""

    def __init__(self, root_path, waveform_dir, metadata_dir,
                 min_lat, max_lat):
        self.root_path = root_path
        self.waveform_dir = waveform_dir
        self.metadata_dir = metadata_dir
        self.min_lat = min_lat
        self.max_lat = max_lat


class Plot_cfg:
    """Plotting options."""

    def __init__(self, plt_l2p_fck, plt_fc_tstar, plot_summ_case1,
                 plot_summ_case2, plot_summ_case3):
        self.plt_l2p_fck = plt_l2p_fck
        self.plt_fc_tstar = plt_fc_tstar
        self.plot_summ_case1 = plot_summ_case1
        self.plot_summ_case2 = plot_summ_case2
        self.plot_summ_case3 = plot_summ_case3


class Tstar_cfg:
    """Computed t^* parameters."""

    def __init__(self, fcps, prewin_p, prewin_s, wl_p, wl_s, snrcrtp1,
                 snrcrtp2, snrcrts1, snrcrts2, lincor, min_fit_p, min_fit_s,
                 d_stress, min_arr_fc, phases, constrainmos,
                 constrainfcs):
        self.fcps = fcps
        self.prewin_p = prewin_p
        self.prewin_s = prewin_s
        self.wl_p = wl_p
        self.wl_s = wl_s
        self.snrcrt_p1 = snrcrtp1
        self.snrcrt_p2 = snrcrtp2
        self.snrcrt_s1 = snrcrts1
        self.snrcrt_s2 = snrcrts2
        self.lincor = lincor
        self.minfit_p = min_fit_p
        self.minfit_s = min_fit_s
        self.d_stress = d_stress
        self.min_arr_fc = min_arr_fc
        self.phases = phases
        self.constrainmos = constrainmos
        self.constrainfcs = constrainfcs
