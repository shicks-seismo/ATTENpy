#!/usr/bin/env python
"""
This is the main script for running the tstar inversion code.

"""
from main_event_waveforms import main_event_waveforms
from config_classes import (Earthmodel_cfg, In_data_cfg, Output_cfg, Plot_cfg,
                            Tstar_cfg)
from obspy import read_events, Catalog
from multiprocessing import Process
import numpy as np
import os
import shutil
import glob
import sys

#################################################################
# Start of parameters to define
EVENT_OUT = "events_atten.out"
ARRIVALS_P_OUT = "arrivals_p_atten.out"
ARRIVALS_S_OUT = "arrivals_s_atten.out"
FITS_P_OUT = "fits_p.out"
FITS_S_OUT = "fits_s.out"
CAT_FILE = "VOILA_all_1D_cleaned_amplitudes_magnitudes.xml"
# CAT_FILE = "test_onevent.xml"
ROOT_PATH = "./"
WAVEFORM_DIR = "local_events/"
METADATA_DIR = "stations"
MAX_GAP = 250  # Maximum azimuthal gap of event
MINMAG = 2.2  # Minimum preferred magnitude of event to use
MINDEPTH = 0.0  # Minimum earthquake depth to use event/MIN
MAXDEPTH = 400.0  # Maximum earthquake depth to use
MIN_LAT = 13.8
MAX_LAT = 17.3
FCPS = 1.5  # P-S corner frequency ratio
PREWIN_P = 0.5  # Number of seconds to window before P-wave arrival
PREWIN_S = 1.0  # Number of seconds to window before P-wave arrival
WL_P = 5.0  # Window length for P-arrival in seconds # 3s - Stachnik
WL_S = 7.0  # Window length for S-arrival in seconds
SNRCRTP1 = [3.0, 4.]  # MIN. [SNR, BANDWIDTH] FOR FINDING BEST P-wave fc
SNRCRTP2 = [2.0, 3.]  # MIN. [SNR, BANDWIDTH] FOR P-wave t* INVERSION
SNRCRTS1 = [2.5, 2.]  # MIN. [SNR, BANDWIDTH] FOR BEST S-wave fc
SNRCRTS2 = [1.7, 1.]  # MIN. [SNR, BANDWIDTH] FOR S-wave t* INVERSION
LINCOR = [0.7, 0.7, 0.7]  # MINIMUM LINEAR CORRELATION COEFFICIENTS
MIN_FIT_P = 0.75
MIN_FIT_S = 0.70
BETA_SRC_CONST = 1  # 0 = Depth specific beta for stress drop; 1 = constant
BETA_CONST = 4000
D_STRESS = [0.1, 20.0]  # Stress drop in MPa
MIN_ARR_FC = 3  # Min no. HQ arrivals to determine fc
PHASES = ["P", "S"]
MOD_NAME = "lesserantilles_lb_noupperslow_190618"
NYQUIST_FRAC = 0.88
CONSTRAINMOS = 0  # Constrain MoS = to MoP (1=yes)
CONSTRAINFCS = 0  # 0: unconstrained; 1: fcs=fcp/1.5; 2: fcs<=fcp/1.5
NPROC = 48  # Number of processes to divie up catalogue into
BLACKLIST = ["e20160417.064427", "e20160831.223818", "e20161111.204639"]
# ALPHA = float(sys.argv[1])
ALPHA = 0.27
CONST_ALPHA = 1  # 1 = Use constant alpha; 0 = Depth-dependent alpha (below)
ALPHA_DEPTH = [[0, 30, 100, 150, 250], [0.8, 0.6, 0.5, 0.27, 0.27]]

# Plot options
PLT_L2P_FCK = 0
PLT_FC_TSTAR = 0
PLOT_SUMM_CASE1 = 0
PLOT_SUMM_CASE2 = 0
PLOT_SUMM_CASE3 = 0

# End of parameters to define
#########################################################################

# matplotlib.use('Agg')
# matplotlib.pyplot.ioff()

# Clear previous combined directory and working directories
if os.path.isdir("output_combined"):
    shutil.rmtree("output_combined")
os.system("rm -r wd_*")

# Read in catalogu
CAT_FULL = read_events(CAT_FILE, format="QUAKEML")

# Sort catalogue by origin date
k = [[event, event.origins[0].time] for event in CAT_FULL]
k.sort(key=lambda x: x[1])
events = [event[0] for event in k]
CATSORT = Catalog(events=events)

# Filter catalog
cat_filt = Catalog()
for event in CATSORT:
    origin = event.preferred_origin()
    if (origin.quality.azimuthal_gap < MAX_GAP and
            len(origin.arrivals) > 5 and len(event.magnitudes) != 0 and
            event.magnitudes[0].mag >= MINMAG
            and origin.depth / 1000 >= MINDEPTH
            and origin.depth / 1000 <= MAXDEPTH
			and origin.latitude >= MIN_LAT
			and origin.latitude <= MAX_LAT
            and event.event_descriptions[0].text not in BLACKLIST):
        cat_filt.append(event)
print(len(cat_filt))

# Split catalogue into chunks for different processes
ave_chunk = len(cat_filt) / float(NPROC)
#ave_chunk = len(cat_filt) / 200  # For testing
cat_segments = []
last = 0.0

print("Splitting catalogue for {:} processes".format(NPROC))
while last < len(cat_filt):
    cat_tmp = Catalog()
    for n, event in enumerate(cat_filt):
        if n >= last and n <= last + ave_chunk:
            cat_tmp.append(cat_filt[n])
    cat_segments.append(cat_tmp)
    last += ave_chunk

# Define configuration
earthmodel_cfg = Earthmodel_cfg(MOD_NAME, BETA_CONST, BETA_SRC_CONST)
in_data_cfg = In_data_cfg(ROOT_PATH, WAVEFORM_DIR, METADATA_DIR, MIN_LAT,
		                  MAX_LAT)
output_cfg = Output_cfg(
    ARRIVALS_P_OUT, ARRIVALS_S_OUT, FITS_P_OUT, FITS_S_OUT, EVENT_OUT)
plot_cfg = Plot_cfg(
    PLT_L2P_FCK, PLT_FC_TSTAR, PLOT_SUMM_CASE1, PLOT_SUMM_CASE2,
    PLOT_SUMM_CASE3)
tstar_cfg = Tstar_cfg(
    FCPS, PREWIN_P, PREWIN_S, WL_P, WL_S, SNRCRTP1, SNRCRTP2, SNRCRTS1,
    SNRCRTS2, LINCOR, MIN_FIT_P, MIN_FIT_S, D_STRESS, MIN_ARR_FC, PHASES,
    NYQUIST_FRAC, CONSTRAINMOS, CONSTRAINFCS)

processes = []
for n_proc in range(0, NPROC):
    work_dir = "wd_{:}".format(n_proc)
    print("Starting process {:}".format(n_proc))
    p = Process(target=main_event_waveforms, args=(
        earthmodel_cfg, in_data_cfg, output_cfg,
        plot_cfg, tstar_cfg, work_dir,
        cat_segments[n_proc], ALPHA, CONST_ALPHA, ALPHA_DEPTH,
        n_proc))
    p.start()
    processes.append(p)

# Ensure that all processes are complete before exiting
for p in processes:
    p.join()

# Calculate total misfit / residual values
p_resall = []
s_resall = []
p_misfitall = []
s_misfitall = []
p_allt = []
s_allt = []

for n_proc in range(0, NPROC):
    work_dir = "wd_{:}".format(n_proc)

    # First get the P-waves
    f = open("{:}/{:}".format(work_dir, output_cfg.fits_p_out), "r")
    for l in f:
        if int(l.split()[3]) > 0:
            p_resall.append(float(l.split()[1]))
            p_misfitall.append(float(l.split()[2]))
            p_allt.append(int(l.split()[3]))
    f.close()

    # Now get the S-waves
    if "S" in PHASES:
        f = open("{:}/{:}".format(work_dir, output_cfg.fits_s_out), "r")
        for l in f:
            if (int(l.split()[3])) > 0:
                s_resall.append(float(l.split()[1]))
                s_misfitall.append(float(l.split()[2]))
                s_allt.append(int(l.split()[3]))
        f.close()

# Make directory for combined results
os.makedirs("output_combined")
os.makedirs("output_combined/figures")
os.makedirs("output_combined/events")
os.makedirs("output_combined/events/P")
os.system("mv wd_*/figures/* output_combined/figures")
os.system("mv wd_*/events/P/*.tstar output_combined/events/P")
os.system("cat wd_*/arrivals_p_atten.out > "
          "output_combined/arrivals_p_atten.out")
os.system("cat wd_*/events_atten.out > "
          "output_combined/events_atten.out")

# Copy configuration to combined directory
os.system("cp run_multi.py output_combined/")

w = open("output_combined/{:}".format(FITS_P_OUT), "w")
w.write("Alpha Normalised_residual Normalised_misfit N\n")
w.write("{:} {:} {:} {:}\n".format(ALPHA, np.sum(p_resall)/np.sum(p_allt),
                                   np.sum(p_misfitall)/np.sum(p_allt),
                                   np.sum(p_allt)))
w.close()

if "S" in PHASES:
    os.makedirs("output_combined/events/S")
    os.system("mv wd_*/events/S/*.tstar output_combined/events/S")
    os.system("cat wd_*/arrivals_s_atten.out > "
              "output_combined/arrivals_s_atten.out")
    w = open("output_combined/{:}".format(FITS_S_OUT), "w")
    w.write("{:} {:} {:} {:}\n".format(ALPHA, np.sum(s_resall)/np.sum(s_allt),
                                       np.sum(s_misfitall)/np.sum(s_allt),
                                       np.sum(s_allt)))
    w.close()

# Remove working directories
for n_proc in range(0, NPROC):
    work_dir = "wd_{:}".format(n_proc)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

# Compute path-averaged Q
w = open("output_combined/average_Q.txt", "w")
path_ave_QP = []
tstar_files = glob.glob("output_combined/events/P/*.tstar")
for file in tstar_files:
    for n, l in enumerate(open(file)):
        if n > 0 and float(l.split()[2]) > 0.001:
            path_ave_QP.append(float(l.split()[1])/float(l.split()[2]))
path_ave_QP_mean = np.mean(path_ave_QP)
path_ave_QP_stdev = np.std(path_ave_QP)
w.write("Phase MeanQ StDevQ\n")
w.write("P {:} {:}\n".format(path_ave_QP_mean, path_ave_QP_stdev))

if "S" in PHASES:
    path_ave_QS = []
    tstar_files = glob.glob("output_combined/events/S/*.tstar")
    for file in tstar_files:
        for n, l in enumerate(open(file)):
            if n > 0 and float(l.split()[2]) > 0.001:
                path_ave_QS.append(float(l.split()[1])/float(l.split()[2]))
    path_ave_QS_mean = np.mean(path_ave_QS)
    path_ave_QS_stdev = np.std(path_ave_QS)
    w.write("S {:} {:}\n".format(path_ave_QS_mean, path_ave_QS_stdev))
w.close()
