#!/usr/bin/env python

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys

def compute_station_corrections(directory, phases):
    with open("{:}/residuals.pkl".format(directory), "rb") as f:
        p_resid_dic, s_resid_dic = pickle.load(f)

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
                    stdev[i] = np.nan
            median = smooth(median, 10)
            stdev = smooth(stdev, 10)
            dic_out[sta] = median
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
        plt.savefig("{}/{}-wave_stationresiduals.png".format(directory, phase), dpi=250, bbox_inches=tight, transparent=True)
        if phase == "P":
            med_res_all_p = dic_out
        elif phase == "S":
            med_res_all_s = dic_out

    with open("{}/residuals_median.pkl".format(directory), "wb") as f:
        pickle.dump([ave_p_stacorr, ave_s_stacorr], f)
