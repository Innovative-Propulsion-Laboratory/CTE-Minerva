# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:46:19 2020

@author: julien
"""
import numpy as np
import csv
from scipy.interpolate import PchipInterpolator


def canaux(profile_data, width_data, height_data, thickness_data, coefficients,
           tore_pos, nbc, plot_detail, write_in_csv, figure_dpi):
    """
    This function computes the caracteristics of channels on each point
    by interpolation between given values at injection plate (inj), end of cylindrical chamber (conv), 
    throat (col) and extremity of the nozzle (div).
    """

    x_value, y_value = profile_data
    lrg_inj, lrg_conv, lrg_col, lrg_tore = width_data
    ht_inj, ht_conv, ht_col, ht_tore = height_data
    e_conv, e_col, e_tore = thickness_data
    n1, n2, n3, n4, n5, n6 = coefficients

    xcanaux = []  # List of x where there are channels (before the manifold) (in m)
    y_coord_avec_canaux = []  # List of y where there are channels (before the manifold) (in m)
    i = 0
    while i < len(x_value) and x_value[i] <= tore_pos:
        xcanaux.append(x_value[i])
        y_coord_avec_canaux.append(y_value[i])
        i += 1

    pos_conv = 0  # Index of the end of cylindrical chamber
    while y_coord_avec_canaux[pos_conv] == y_coord_avec_canaux[pos_conv + 1]:
        pos_conv += 1
    pos_col = y_coord_avec_canaux.index(min(y_coord_avec_canaux))  # Index of the throat
    y_col = y_coord_avec_canaux[pos_col]  # y coordonate of the hot wall at the throat
    y_inj = y_coord_avec_canaux[0]  # y coordonate of the hot wall at the injection plate
    y_tore = y_coord_avec_canaux[-1]  # y coordonate of the hot wall at the manifold
    longc = len(xcanaux)  # Index of the end of the channels (at the manifold)

    wall_thickness = []  # Thickness of the chamber wall as a function of the engine axis (in m)
    acc = (e_conv - e_col) / (y_inj - y_col)
    for i in range(0, pos_col + 1):  # Chamber + convergent computation
        r = y_coord_avec_canaux[i]
        aug = (y_col - r) / (y_inj - y_col)
        epp_x = ((-aug) ** n5) * (r - y_col) * acc + e_col
        wall_thickness.append(epp_x)
    acc = (e_tore - e_col) / (y_tore - y_col)
    for i in range(pos_col + 1, longc):  # Divergent computation
        r = y_coord_avec_canaux[i]
        aug = (y_tore - r) / (y_tore - y_col)
        epp_x = ((1 - aug) ** n6) * (r - y_col) * acc + e_col
        wall_thickness.append(epp_x)

    angulaire = [0]
    ycanaux = [y_inj + wall_thickness[0]]  # Corrected thickness (to match with the geometry of the engine)
    for i in range(1, longc):
        vect = (xcanaux[i] - xcanaux[i - 1]) / ((((y_coord_avec_canaux[i] - y_coord_avec_canaux[i - 1]) ** 2) +
                                                 ((xcanaux[i] - xcanaux[i - 1]) ** 2)) ** 0.5)
        angulaire.append(np.rad2deg(np.arccos(vect)))
        """
        newep = y_coord_avec_canaux[i] + wall_thickness[i] / np.cos(np.deg2rad(angulaire[i]))
        ycanaux.append(newep)
        """
        ycanaux.append(y_coord_avec_canaux[i] + wall_thickness[i] / vect)

    veritas = []
    for i in range(0, longc):
        verifepe = (((ycanaux[i] - y_value[i]) ** 2) - (
                np.sin(np.deg2rad(angulaire[i])) * (ycanaux[i] - y_value[i])) ** 2) ** 0.5
        veritas.append(verifepe)

    y_col = ycanaux[pos_col]  # y coordonate of the cold wall at the throat
    y_inj = ycanaux[0]  # y coordonate of the cold wall at the injection plate
    y_tore = ycanaux[-1]  # y coordonate of the cold wall at the manifold
    larg_ailette = []  # Width of a rib as a function of the engine axis (in m)
    larg_canal = []  # Width of a channel as a function of the engine axis (in m)
    pente = (lrg_conv - lrg_inj) / (xcanaux[pos_conv] - xcanaux[0])
    for i in range(0, pos_conv + 1):  # Chamber computation
        r = ycanaux[i]
        lrg_x = pente * (xcanaux[i] - xcanaux[0]) + lrg_inj
        lrg_aill = (r * 2 * np.pi / nbc) - lrg_x
        larg_ailette.append(lrg_aill)
        larg_canal.append(lrg_x)
    acc = (lrg_conv - lrg_col) / (y_inj - y_col)
    for i in range(pos_conv + 1, pos_col + 1):  # Convergent computation
        r = ycanaux[i]
        aug = (y_col - r) / (y_inj - y_col)
        lrg_x = ((-aug) ** n1) * (r - y_col) * acc + lrg_col
        lrg_aill = (r * 2 * np.pi / nbc) - lrg_x
        larg_ailette.append(lrg_aill)
        larg_canal.append(lrg_x)
    acc = (lrg_tore - lrg_col) / (y_tore - y_col)
    for i in range(pos_col + 1, longc):  # Divergent computation
        r = ycanaux[i]
        aug = (y_tore - r) / (y_tore - y_col)
        lrg_x = ((1 - aug) ** n2) * (r - y_col) * acc + lrg_col
        lrg_aill = (r * 2 * np.pi / nbc) - lrg_x
        larg_ailette.append(lrg_aill)
        larg_canal.append(lrg_x)

    ht_canal = []  # Height of a channel as a function of the engine axis (in m)
    pente = (ht_conv - ht_inj) / (xcanaux[pos_conv] - xcanaux[0])
    for i in range(0, pos_conv + 1):  # Chamber computation
        htr_x = pente * (xcanaux[i] - xcanaux[0]) + ht_inj
        ht_canal.append(htr_x)
    acc = (ht_conv - ht_col) / (y_inj - y_col)
    for i in range(pos_conv + 1, pos_col):  # Convergent computation
        r = ycanaux[i]
        aug = (y_col - r) / (y_inj - y_col)
        htr_x = ((-aug) ** n3) * (r - y_col) * acc + ht_col
        ht_canal.append(htr_x)
    acc = (ht_tore - ht_col) / (y_tore - y_col)
    for i in range(pos_col, longc):  # Divergent computation
        r = ycanaux[i]
        aug = (y_tore - r) / (y_tore - y_col)
        htr_x = ((1 - aug) ** n4) * (r - y_col) * acc + ht_col
        ht_canal.append(htr_x)

    area_channel = [larg_canal[i] * ht_canal[i] for i in range(0, longc)]  # Area of a channel as a function of the engine axis (m²)

    if write_in_csv:
        "Writing the results of the study in a CSV file"
        file_name = "output/channel_macro_catia.csv"
        file = open(file_name, "w", newline="")
        writer = csv.writer(file)
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i]), 1000 * (larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i]), 1000 * (-larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i] + ht_canal[i]), 1000 * (larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i] + ht_canal[i]), 1000 * (- larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["End"])
        file.close()

    return xcanaux, ycanaux, larg_canal, larg_ailette, ht_canal, wall_thickness, area_channel, longc, \
           y_coord_avec_canaux


def canaux_library(profile_data, width_data, height_data, thickness_data, coefficients,
           tore_pos, nbc, plot_detail, write_in_csv, figure_dpi):
    """
    This function computes the caracteristics of channels on each point
    by interpolation between given values at injection plate (inj), end of cylindrical chamber (conv), 
    throat (col) and extremity of the nozzle (div) using library from python (scipy.interpolate)
    """

    x_value, y_value = profile_data
    lrg_inj, lrg_conv, lrg_col, lrg_tore = width_data
    ht_inj, ht_conv, ht_col, ht_tore = height_data
    e_conv, e_col, e_tore = thickness_data

    xcanaux = []  # List of x where there are channels (before the manifold) (in m)
    y_coord_avec_canaux = []  # List of y where there are channels (before the manifold) (in m)
    # WARNING ! ycanaux will later in this function become y on the coolant side (in m)
    i = 0
    while i < len(x_value) and x_value[i] <= tore_pos:
        xcanaux.append(x_value[i])
        y_coord_avec_canaux.append(y_value[i])
        i += 1

    pos_conv = 0  # Index of the end of cylindrical chamber
    while y_coord_avec_canaux[pos_conv] == y_coord_avec_canaux[pos_conv + 1]:
        pos_conv += 1
    pos_col = y_coord_avec_canaux.index(min(y_coord_avec_canaux))  # Index of the throat
    
    longc = len(xcanaux)  # Number of points for channels (end at the manifold)

    x_interpolate = [xcanaux[0], xcanaux[pos_conv], xcanaux[pos_col], xcanaux[longc-1]]

    y_e = [e_conv, e_conv, e_col, e_tore]
    wall_thickness = [x for x in PchipInterpolator(x_interpolate, y_e)(xcanaux)]  # Thickness of the chamber wall as a function of the engine axis (in m)

    angulaire = [0]
    ycanaux = [y_coord_avec_canaux[0] + wall_thickness[0]]  # y of wall on coolant side (matched with engine geometry)
    for i in range(1, longc):
        vect = (xcanaux[i] - xcanaux[i - 1]) / ((((y_coord_avec_canaux[i] - y_coord_avec_canaux[i - 1]) ** 2) +
                                                  ((xcanaux[i] - xcanaux[i - 1]) ** 2)) ** 0.5)
        angulaire.append(np.rad2deg(np.arccos(vect)))
        """
        newep = ycanaux[i] + epaiss_chemise[i] / np.cos(np.deg2rad(angulaire[i]))
        ycanaux.append(newep)
        """
        ycanaux.append(y_coord_avec_canaux[i] + wall_thickness[i] / vect)

    y_l = [lrg_inj, lrg_conv, lrg_col, lrg_tore]
    larg_canal = [x for x in PchipInterpolator(x_interpolate, y_l)(xcanaux)]  # Width of a channel as a function of the engine axis (in m)
    larg_ailette = [(ycanaux[i] * 2 * np.pi / nbc) - larg_canal[i] for i in range(0, longc)]  # Width of a rib as a function of the engine axis (in m)

    y_h = [ht_inj, ht_conv, ht_col, ht_tore]
    ht_canal = [x for x in PchipInterpolator(x_interpolate, y_h)(xcanaux)]  # Height of a channel as a function of the engine axis (in m)

    area_channel = [larg_canal[i] * ht_canal[i] for i in range(0, longc)]  # Area of a channel as a function of the engine axis (m²)

    if write_in_csv:
        "Writing the results of the study in a CSV file"
        file_name = "output/channel_macro_catia.csv"
        file = open(file_name, "w", newline="")
        writer = csv.writer(file)
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i]), 1000 * (larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i]), 1000 * (-larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i] + ht_canal[i]), 1000 * (larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["StartCurve"])
        for i in range(0, longc, 3):
            writer.writerow((1000 * xcanaux[i], 1000 * (ycanaux[i] + ht_canal[i]), 1000 * (- larg_canal[i] / 2)))
        writer.writerow(["EndCurve"])
        writer.writerow(["End"])
        file.close()

    return xcanaux, ycanaux, larg_canal, larg_ailette, ht_canal, wall_thickness, area_channel, longc, y_coord_avec_canaux

