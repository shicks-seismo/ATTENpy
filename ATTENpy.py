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
    import numpy as np
    import os
    import shutil
    import glob
    import yaml
    from util import filter_cat
    import multiprocessing
    import pandas as pd
    from functools import partial
    from objects import Atten_DB

    with open("config.yaml") as f:  # Read in config
        cfg = obj(yaml.safe_load(f))

    # Clear previous combined directory and working directories
    if os.path.isdir("output"):
        shutil.rmtree("output")
        os.system("rm -r wd_*")
    os.makedirs("output")
    os.makedirs("output/figures")
    os.makedirs("output/events")
    os.makedirs("output/events/P"); os.makedirs("output/events/S")

    # Read in and filter catalogue
    cat = read_events(cfg.dat.cat_file, format="QUAKEML")
    cat = filter_cat(cat, cfg.dat)

    # Get nuber of cores and split catalog
    if cfg.comp.nproc == -1:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = cfg.comp.nproc
    evts_all = [evt for evt in cat]
    if len(evts_all) <= n_cores:
        n_cores = len(evts_all)
    evts_all = split(evts_all, n_cores)
    cat_segments = list(enumerate([Catalog(events=evts) for evts in evts_all]))

    with multiprocessing.Pool(processes=n_cores) as pool:  # Run in parallel
        atten_db = pool.starmap(partial(main_event_waveforms, cfg),
                                cat_segments)
    atten_db_all = Atten_DB()  # Merge event databases
    for x in atten_db:
        for y in x:
            atten_db_all.Aevents.append(y)

    pd.DataFrame.from_records(
        [evt.to_dict() for evt in atten_db_all.Aevents]
        ).to_csv("output/event_atten.csv")


    # Calculate total misfit / residual values
    p_resall = []
    s_resall = []
    p_misfitall = []
    s_misfitall = []
    p_allt = []
    s_allt = []


#     # First get the P-waves
#     f = open("{:}/{:}".format(work_dir, output_cfg.fits_p_out), "r")

#     p_res_all, p_misfitall, p_allt = list(*zip[
#         [float(l.split()[1]), float(l.split()[2]), float(l.split()[3])]
#         for n, l in enumerate(open("wd_{:}/{:}".format(n_proc, output_cfg.fits_p_out), "r"))
#         for n_proc in range(n_cores)])
    
#     for l in f:
#         if int(l.split()[3]) > 0:
#             p_resall.append(float(l.split()[1]))
#             p_misfitall.append(float(l.split()[2]))
#             p_allt.append(int(l.split()[3]))
#     f.close()

#     # Now get the S-waves
#     if "S" in PHASES:
#         f = open("{:}/{:}".format(work_dir, output_cfg.fits_s_out), "r")
#         for l in f:
#             if (int(l.split()[3])) > 0:
#                 s_resall.append(float(l.split()[1]))
#                 s_misfitall.append(float(l.split()[2]))
#                 s_allt.append(int(l.split()[3]))
#         f.close()

# # Make directory for combined results
# os.makedirs("output_combined")
# os.makedirs("output_combined/figures")
# os.makedirs("output_combined/events")
# os.makedirs("output_combined/events/P")
# os.system("mv wd_*/figures/* output_combined/figures")
# os.system("mv wd_*/events/P/*.tstar output_combined/events/P")
# os.system("cat wd_*/arrivals_p_atten.out > "
#           "output_combined/arrivals_p_atten.out")
# os.system("cat wd_*/events_atten.out > "
#           "output_combined/events_atten.out")

# # Copy configuration to combined directory
# os.system("cp run_multi.py output_combined/")

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

# # Remove working directories
# for n_proc in range(0, NPROC):

#     if os.path.isdir(work_dir):
#         shutil.rmtree(work_dir)

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

# if "S" in PHASES:
#     path_ave_QS = []
#     tstar_files = glob.glob("output_combined/events/S/*.tstar")
#     for file in tstar_files:
#         for n, l in enumerate(open(file)):
#             if n > 0 and float(l.split()[2]) > 0.001:
#                 path_ave_QS.append(float(l.split()[1])/float(l.split()[2]))
#     path_ave_QS_mean = np.mean(path_ave_QS)
#     path_ave_QS_stdev = np.std(path_ave_QS)
#     w.write("S {:} {:}\n".format(path_ave_QS_mean, path_ave_QS_stdev))
# w.close()


def split(a, n):
    """Split catalog into even chunks."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    main_function()
