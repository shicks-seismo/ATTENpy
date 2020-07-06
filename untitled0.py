#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:14:28 2020

@author: sph1r17
"""
evt_out.write("{:},{:},{:8.4f},{:8.4f},{:5.1f},{:9s},{:3.2f},{:8.2f},"
                      "{:8.2f},{:8.2f},{:8.2f},{:3.2f},{:3g},{:3g},{:3g},"
                      "{:3g},{:3g},{:3g}\n".format(
   a_event.origin_id, origin.time.datetime,
                          origin.latitude, origin.longitude, origin.depth_km,
                          a_event.mag,
                          a_event.Mw_p, a_event.Mw_s,
                          a_event.fc_p, a_event.fc_s, cfg.inv.α,
                          len(a_event.p_arrivals_HQ),
                          len(a_event.s_arrivals_HQ),
                          len(a_event.p_arrivals_LQ),
                          len(a_event.s_arrivals_LQ),
                          len(a_event.p_arrivals_LQ_fitting),
                          len(a_event.s_arrivals_LQ_fitting)))


    d_evt.append([a_event.origin_id, origin.time.datetime, origin.latitude,
                  origin.longitude, origin.depth_km, a_event.mag,
                  a_event.Mw_p, a_event.Mw_s, a_event.fc_p, a_event.fc_s,
                  cfg.inv.α,
                  len(a_event.p_arrivals_HQ),len(a_event.s_arrivals_HQ),
                  len(a_event.p_arrivals_LQ), len(a_event.s_arrivals_LQ),
                  len(a_event.p_arrivals_LQ_fitting),
                  len(a_event.s_arrivals_LQ_fitting)])
    
    
        # Write arrival details tosevent file
        w = open("{:}/events/P/{:}.tstar".format(wdir, a_event.origin_id),
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

        if "S" in cfg.inv.phases:
            w = open("{:}/events/S/{:}.tstar".format(wdir,
                                                     a_event.origin_id), "w")
            w.write("{:} {:} {:8.4f} {:8.4f} {:5.1f} {:3.2f} {:5.2f}\n".format(
                a_event.origin_id, origin.time.datetime, origin.latitude,
                origin.longitude, origin.depth_km, a_event.Mw_p, a_event.fc_s))
            for arrival in a_event.s_arrivals_LQ_fitting:
                if arrival.tstar > 0.0:
                    w.write("{:} {:6.2f} {:6.4f} {:4.3f}\n".format(
                        arrival.station, arrival.time-origin.time,
                        arrival.tstar, arrival.fit))
            w.close()

        # Write arrival details to file
        for arrival in a_event.p_arrivals_LQ_fitting:
            arr_p_out = open("{:}/arrivals_p_atten.out".format(wdir), "a")
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
            arr_s_out = open("{:}/arrivals_s_atten.out".format(wdir), "a")
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
        cfg.inv.α, p_resall, p_misfitall, p_allt))
    if "S" in cfg.inv.phases:
        fit_s_out.write("{:} {:} {:} {:}\n".format(
            cfg.inv.α, s_resall, s_misfitall, s_allt))
    fit_p_out.close()
    fit_s_out.close()
    
    
    