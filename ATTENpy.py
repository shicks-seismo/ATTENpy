#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Script for running the t* inversion code."""


class obj(object):
    """Get simulation parameters from dictionary from yaml file."""

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(
                   self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def main_function():
    """t* function."""
    from main_event_waveforms import main_event_waveforms
    from obspy import read_events, Catalog
    import os
    import yaml
    from util import filter_cat, make_dirs
    import multiprocessing
    import pandas as pd
    from functools import partial
    from objects import Atten_DB
    import warnings
    import numpy as np
    from shapely.geometry.polygon import Polygon
    from shapely.geometry import Point

    warnings.filterwarnings("ignore", category=UserWarning)

    with open("config.yaml") as f:  # Read in config
        cfg = obj(yaml.safe_load(f))

    make_dirs()  # Clear previous combined directory and working directories

    # Read in and filter catalogue
    print("Reading in event catalogue")
    cat_start = read_events(cfg.dat.cat_file, format="QUAKEML")
    cat = filter_cat(cat_start, cfg.dat)

    # Get nuber of cores and split catalog
    if cfg.comp.nproc == -1:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = cfg.comp.nproci
    evts_all = [evt for evt in cat]
    if cfg.comp.debug:
        n_cores = 1
        evts_all = [evt for evt in cat if
                    evt.event_descriptions[0].text == cfg.comp.debug_evt_id]
    else:
        if len(evts_all) <= n_cores:
            n_cores = len(evts_all)
    evts_all = split(evts_all, n_cores)
    cat_segments = list(enumerate([Catalog(events=evts) for evts in evts_all]))
    if not cfg.comp.debug:
        os.system('cls' if os.name == 'nt' else 'clear')
        pool = multiprocessing.Pool(processes=n_cores)
        atten_db = pool.starmap(partial(main_event_waveforms, cfg),
                                cat_segments)
        atten_db_all = Atten_DB()  # Merge event databases
        for x in atten_db:
            for y in x:
                atten_db_all.Aevents.append(y)
        os.system('cls' if os.name == 'nt' else 'clear')

    else:
        atten_db_all = main_event_waveforms(cfg, 0, cat_segments[0][1])
    print('\n\nt* computation done\n')

    pd.DataFrame.from_records(
        [evt.to_dict() for evt in atten_db_all.Aevents]
        ).to_csv("output/event_atten.csv")

    for evt in atten_db_all.Aevents:
        pd.DataFrame.from_records(
            [arr.to_dict() for arr in evt.p_arrivals_LQ_fitting]
            ).to_csv("output/events/P/{:}.tstar.csv".format(evt.origin_id))
    for evt in atten_db_all.Aevents:
        pd.DataFrame.from_records(
            [arr.to_dict() for arr in evt.s_arrivals_LQ_fitting]
            ).to_csv("output/events/S/{:}.tstar.csv".format(evt.origin_id))
    n_P_fitting = len([arr for evt in atten_db_all.Aevents
                       for arr in evt.p_arrivals_LQ_fitting])
    n_S_fitting = len([arr for evt in atten_db_all.Aevents
                       for arr in evt.s_arrivals_LQ_fitting])
    n_evts_P_fitting = len([evt for evt in atten_db_all.Aevents
                            if len(evt.p_arrivals_LQ_fitting) > 3])
    n_evts_S_fitting = len([evt for evt in atten_db_all.Aevents
                            if len(evt.s_arrivals_LQ_fitting) > 4])
    tot_p_misfit = np.sum([evt.misfit_p for evt in atten_db_all.Aevents
                           if len(evt.p_arrivals_LQ_fitting) > 3]) / n_evts_P_fitting
    tot_s_misfit = np.sum([evt.misfit_s for evt in atten_db_all.Aevents
                           if len(evt.s_arrivals_LQ_fitting) > 4]) / n_evts_S_fitting
    tot_p_res = np.sum([evt.res_p for evt in atten_db_all.Aevents
                           if len(evt.p_arrivals_LQ_fitting) > 3]) / n_evts_P_fitting
    tot_s_res = np.sum([evt.res_s for evt in atten_db_all.Aevents
                           if len(evt.s_arrivals_LQ_fitting) > 4]) / n_evts_S_fitting
    print("Number of P obs = {}".format(n_P_fitting))
    print("Number of S obs = {}".format(n_S_fitting))
    print("Average t*_p = {:.3f}".format(np.mean(
        [arr.tstar for evt in atten_db_all.Aevents
         for arr in evt.p_arrivals_LQ_fitting])))
    print("Average t*_s = {:.3f}".format(np.mean(
        [arr.tstar for evt in atten_db_all.Aevents
         for arr in evt.s_arrivals_LQ_fitting])))
    print(tot_p_misfit, tot_s_misfit, tot_p_res, tot_s_res)


    forearc_poly = Polygon([
        (-63.01, 18.05), (-62.22, 17.18), (-61.55, 16.39), (-61.19, 15.77),
        (-60.92, 15.00), (-60.71, 14.53), (-60.73, 14.05), (-60.71, 13.49),
        (-60.90, 13.15), (-61.24, 12.45), (-61.51, 11.88), (-57.54, 11.72),
        (-57.72, 13.06), (-57.89, 13.84), (-57.83, 14.64), (-58.17, 14.98),
        (-58.69, 15.14), (-58.93, 16.37), (-59.42, 16.95), (-60.01, 17.52),
        (-60.30, 17.80), (-60.38, 17.92), (-60.39, 18.01), (-60.41, 18.07),
        (-63.01, 18.05)])
    backarc_poly = Polygon([
        (-63.48, 18.06), (-63.50, 11.89), (-62.07, 11.89), (-61.47, 13.05),
        (-61.27, 13.54), (-61.14, 14.18), (-61.30, 14.67), (-61.47, 15.16),
        (-61.88, 16.07), (-62.16, 16.50), (-63.03, 17.56), (-63.48, 18.06)
        ])
    arc_poly = Polygon([
        (-63.00, 18.06), (-63.46, 18.07), (-62.19, 16.51), (-61.91, 16.08),
        (-61.14, 14.34), (-61.39, 13.13), (-62.08, 11.90), (-61.51, 11.89),
        (-60.73, 13.51), (-60.71, 14.51), (-61.11, 15.59), (-61.52, 16.34),
        (-62.28, 17.27), (-63.00, 18.06)
        ])

    tstar_P_mean_backarc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.p_arrivals_LQ_fitting
        if backarc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    tstar_S_mean_backarc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.s_arrivals_LQ_fitting
        if backarc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    print("t* backarc mean P: {:.3f} S: {:.3f}".format(
        tstar_P_mean_backarc, tstar_S_mean_backarc))
    tstar_P_mean_forearc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.p_arrivals_LQ_fitting
        if forearc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    tstar_S_mean_forearc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.s_arrivals_LQ_fitting
        if forearc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    print("t* forearc mean P: {:.3f} S: {:.3f}".format(
        tstar_P_mean_forearc, tstar_S_mean_forearc))
    tstar_P_mean_arc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.p_arrivals_LQ_fitting
        if arc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    tstar_S_mean_arc = np.mean([
        arr.tstar for evt in atten_db_all.Aevents
        for arr in evt.s_arrivals_LQ_fitting
        if arc_poly.contains(Point(arr.station_lon, arr.station_lat))])
    print("t* arc mean P: {:.3f} S: {:.3f}".format(
        tstar_P_mean_arc, tstar_S_mean_arc))


#     p_res_all, p_misfitall, p_allt = list(*zip[
#         [float(l.split()[1]), float(l.split()[2]), float(l.split()[3])]
#         for n, l in enumerate(open("wd_{:}/{:}".format(n_proc, output_cfg.fits_p_out), "r"))
#         for n_proc in range(n_cores)])


# w = open("output_combined/{:}".format(FITS_P_OUT), "w")
# w.write("Alpha Normalised_residual Normalised_misfit N\n")
# w.write("{:} {:} {:} {:}\n".format(ALPHA, np.sum(p_resall)/np.sum(p_allt),
#                                     np.sum(p_misfitall)/np.sum(p_allt),
#                                     np.sum(p_allt)))
# w.close()

# if "S" in PHASES:
#     os.makedirs("output_combined/events/S")
#     os.system("mv wd_*/events/S/*.tstar output_combined/events/S")
#     os.system("cat wd_*/arrivals_s_atten.out > "
#               "output_combined/arrivals_s_atten.out")
#     w = open("output_combined/{:}".format(FITS_S_OUT), "w")
#     w.write("{:} {:} {:} {:}\n".format(ALPHA, np.sum(s_resall)/np.sum(s_allt),
#                                         np.sum(s_misfitall)/np.sum(s_allt),
#                                         np.sum(s_allt)))
#     w.close()


# # Compute path-averaged Q
# w = open("output_combined/average_Q.txt", "w")
# path_ave_QP = []
# tstar_files = glob.glob("output_combined/events/P/*.tstar")
# for file in tstar_files:
#     for n, l in enumerate(open(file)):
#         if n > 0 and float(l.split()[2]) > 0.001:
#             path_ave_QP.append(float(l.split()[1])/float(l.split()[2]))
# path_ave_QP_mean = np.mean(path_ave_QP)
# path_ave_QP_stdev = np.std(path_ave_QP)
# w.write("Phase MeanQ StDevQ\n")
# w.write("P {:} {:}\n".format(path_ave_QP_mean, path_ave_QP_stdev))



def split(a, n):
    """Split catalog into even chunks."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    main_function()
