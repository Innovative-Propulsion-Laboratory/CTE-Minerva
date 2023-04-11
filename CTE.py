"""
Created on Fri Nov 27 14:47:27 2020

Original author: Julien S

Refactored and improved by Mehdi D, Paul B, Paul M, Eve X and Antoine R
"""

import time
import csv

# Calculations
import numpy as np
import cte_tools as t
from main_solver import mainsolver

# Data
from Canaux import canaux, canaux_library
from CoolProp.CoolProp import PropsSI

# Graphics
from heatequationsolve import carto2D
from volume3d import carto3d
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
from plotter import plotter

start_time = time.perf_counter()  # Beginning of the timer

print("██████████████████████████ Cool The Engine V 2.0.0 █████████████████████████")
print("█                                                                          █")
print("█                  Innovative Propulsion Laboratory - IPL                  █")
print("█__________________________________________________________________________█")
print("█                                                                          █")
print("█ Initialisation                                                           █")
print("█                                                                          █")

# %% Initial definitions

mesh_size = 0.25  # Distance between two points of calculation
x_coords_filename = f"input/{mesh_size}/x.txt"  # X coordinates of the Minerva
y_coords_filename = f"input/{mesh_size}/y.txt"  # Y coordinates of the Minerva
input_CEA_data = "input/Minerva_project.txt"  # Minerva's parameters (found with CEA)

# Constant input_data_list
size2 = 16  # Used for the height of the display in 3D view
limitation = 0.05  # used to build the scales in 3D view
figure_dpi = 150  # Dots Per Inch (DPI) for all figures (lower=faster)
plot_detail = 3  # 0=No plots; 1=Important plots; 3=All plots
show_plots = True
save_plots = False
show_3d_plots = False
show_2D_temperature = False
do_final_3d_plot = False
write_in_csv = True

# %% Reading input data
input_data_reader = csv.reader(open(input_CEA_data, "r"))
input_data_list = [row[1] for row in input_data_reader]

# Store CEA output in lists
sound_speed_init = float(input_data_list[0])  # Sound velocity in the chamber
sound_speed_throat = float(input_data_list[1])  # Sound velocity in the throat
debit_LOX = float(input_data_list[2])  # LOX debit
debit_mass_coolant = float(input_data_list[3])  # Ethanol debit
rho_init = float(input_data_list[4])  # Initial density of the gases
Pc = float(input_data_list[5])  # Pressure in the chamber
Tc = float(input_data_list[6])  # Combustion temperature
gamma_c_input = float(input_data_list[7])  # Gamma in the chamber
gamma_t_input = float(input_data_list[8])  # Gamma in the throat
gamma_e_input = float(input_data_list[9])  # Gamma at the exit
molar_mass = float(input_data_list[10])  # Molar mass of the gases
c_star = float(input_data_list[11])  # Caracteristic velocity
combustion_efficiency = float(input_data_list[22])  # Caracteristic velocity efficiency
xH2O_c_input = float(input_data_list[16])  # Molar fraction of the H2O in the chamber
xH2O_t_input = float(input_data_list[17])  # Molar fraction of the H2O in the throat
xH2O_e_input = float(input_data_list[18])  # Molar fraction of the H2O at the exit
xCO2_c_input = float(input_data_list[19])  # Molar fraction of the CO2 in the chamber
xCO2_t_input = float(input_data_list[20])  # Molar fraction of the CO2 in the throat
xCO2_e_input = float(input_data_list[21])  # Molar fraction of the CO2 at the exit

# Store input dimensions in lists
curv_radius_pre_throat = float(input_data_list[12])  # Radius of curvature before the throat
curv_radius_after_throat = float(input_data_list[13])  # Radius of curvature after the throat
area_throat = float(input_data_list[14])  # Area at the throat
diam_throat = float(input_data_list[15])  # Throat diameter

# %% Import of the (X,Y) coordinates of the Minerva
x_coords_reader = csv.reader(open(x_coords_filename, "r"))
y_coords_reader = csv.reader(open(y_coords_filename, "r"))

# Storing the X,Y coordinates in lists
x_coord_list = [float(row[0]) / 1000 for row in x_coords_reader]
y_coord_list = [float(row[0]) / 1000 for row in y_coords_reader]
nb_points = len(x_coord_list)  # Number of points (or the index of the end of the divergent)

x_coord_list_mm = [x * 1000 for x in x_coord_list]
y_coord_list_mm = [y * 1000 for y in y_coord_list]
# %% Computation of the cross-sectional area along the engine
cross_section_area_list = [np.pi * r ** 2 for r in y_coord_list]
# %% Adiabatic constant (gamma) parametrization
print("█ Computing gamma                                                          █")

i_conv = 0  # Index of the beginning of the convergent
y1 = 1
y2 = 1
while y1 == y2:  # Read y values two per two in order to detect the beginning of the convergent
    y1 = y_coord_list[i_conv]
    i_conv += 1
    y2 = y_coord_list[i_conv]

i_throat = y_coord_list.index(min(y_coord_list))  # Throat index
"""
# Gamma in the cylindrical chamber
gamma_list = [gamma_c_input for i in range(0, i_conv)]  # Gamma is constant before the beginning of the convergent

# Gamma in the convergent
gamma_convergent = gamma_c_input
for m in range(-1, i_throat - i_conv - 1):
    # Linear interpolation between beginning and end of convergent:
    # (yi+1)=((y2-y1)/(x2-x1))*abs((xi+1)-(xi))
    gamma_convergent += ((gamma_t_input - gamma_c_input) / (x_coord_list[i_throat] - x_coord_list[i_conv])) * abs(
        x_coord_list[i_conv + 1 + m] - x_coord_list[i_conv + m])
    gamma_list.append(gamma_convergent)

# Gamma in the divergent nozzle
gamma_divergent = gamma_t_input
for q in range(-1, nb_points - i_throat - 1):  # Linear interpolation between beginning and end of divergent
    gamma_divergent += ((gamma_e_input - gamma_t_input) / (x_coord_list[-1] - x_coord_list[i_throat])) * abs(
        x_coord_list[i_throat + 1 + q] - x_coord_list[i_throat + q])
    gamma_list.append(gamma_divergent)
"""
x_given = [x_coord_list[0], x_coord_list[i_conv], x_coord_list[i_throat], x_coord_list[-1]]
gamma_given = [gamma_c_input, gamma_c_input, gamma_t_input, gamma_e_input]
gamma_list = [x for x in np.interp(x_coord_list, x_given, gamma_given)]

# %% Mach number computation
"Computation of gases mach number of the hot gases (and their initial velocity)"

v_init_gas = (debit_LOX + debit_mass_coolant) / (rho_init * cross_section_area_list[0])  # Initial velocity of the gases
mach_init_gas = v_init_gas / sound_speed_init  # Initial mach number
mach_gas = mach_init_gas
mach_list = [mach_init_gas]

with tqdm(total=nb_points - 1,
          desc="█ Computing mach number        ",
          unit="|   █", bar_format="{l_bar}{bar}{unit}",
          ncols=76) as progressbar:
    for i in range(0, nb_points - 1):
        mach_gas = t.mach_solv(cross_section_area_list[i], cross_section_area_list[i + 1],
                               mach_gas, gamma_list[i])
        mach_list.append(mach_gas)
        progressbar.update(1)

# %% Static pressure computation
pressure_list = [Pc]  # (in Pa)

with tqdm(total=nb_points - 1,
          desc="█ Computing static pressure    ",
          unit="|   █", bar_format="{l_bar}{bar}{unit}",
          ncols=76) as progressbar:
    for i in range(0, nb_points - 1):
        pressure = t.pressure_solv(mach_list[i], mach_list[i + 1], pressure_list[i], gamma_list[i])
        pressure_list.append(pressure)
        progressbar.update(1)
# %% Partial pressure computation and interpolation of the molar fraction
x_Molfrac = [x_coord_list[0], x_coord_list[i_throat], x_coord_list[-1]]  # Location associated to the molar mass

# Value of the molar fraction of the H20 after interpolation
Molfrac_H2O = np.interp(x_coord_list, x_Molfrac, [xH2O_c_input, xH2O_t_input, xH2O_e_input])

# Value of the molar fraction of the CO2 after interpolation
Molfrac_CO2 = np.interp(x_coord_list, x_Molfrac, [xCO2_c_input, xCO2_t_input, xCO2_e_input])

partial_p_H2O_list = [pressure_list[i] * Molfrac_H2O[i] for i in range(0, nb_points)]  # Partial pressure of the H2O
partial_p_CO2_list = [pressure_list[i] * Molfrac_CO2[i] for i in range(0, nb_points)]  # Partial pressure of the CO2

# %% Hot gas temperature computation
static_hotgas_temp_list = [Tc]
total_hotgas_temp = t.total_temp_calculation(Tc, gamma_c_input, mach_init_gas)
with tqdm(total=nb_points - 1,
          desc="█ Computing gas temperature    ",
          unit="|   █", bar_format="{l_bar}{bar}{unit}",
          ncols=76) as progressbar:
    for i in range(0, nb_points - 1):
        temperature = t.temperature_hotgas_solv(mach_list[i], mach_list[i + 1],
                                                static_hotgas_temp_list[i],
                                                gamma_list[i])
        static_hotgas_temp_list.append(temperature)
        progressbar.update(1)

static_hotgas_temp_list = [combustion_efficiency * T for T in static_hotgas_temp_list]
# List of corrected gas temperatures (max diff with original is about 75 K)
total_hotgas_temp_list = [combustion_efficiency * total_hotgas_temp for i in range(0, nb_points)]
recovery_hotgas_temp_list = [t.tempcorrige_pempie(total_hotgas_temp_list[i], gamma_list[i], mach_list[i]) for i in
                             range(0, nb_points)]

# %% Dimensions
print("█ Computing channel geometric                                              █")
print("█                                                                          █")

nbc = 42  # Number of channels
manifold_pos = 0.104  # Position of the manifold from the throat (in m)

# Widths
lrg_inj = 0.003  # Width of the channel in at the injection plate (in m)
lrg_conv = 0.002  # Width of the channel at the end of the cylindrical chamber (in m)
lrg_col = 0.002  # Width of the channel in the throat (in m)
lrg_tore = 0.002  # Width of the channel at the manifold (in m)

# Heights
ht_inj = 0.003  # Height of the channel at the injection plate (in m)
ht_conv = 0.002  # Height of the channel at the end of the cylindrical chamber (in m)
ht_col = 0.0015  # Height of the channel in the throat (in m)
ht_tore = 0.002  # Height of the channel at the manifold (in m)

# Thickness
e_conv = 0.001  # Thickness of the wall at the chamber (in m)
e_col = 0.001  # Thickness of the wall at the throat (in m)
e_tore = 0.001  # Thickness of the wall at the manifold (in m)

n1 = 1  # Width convergent
n2 = 1  # Width divergent
n3 = 1  # Height convergent
n4 = 1  # Height divergent
n5 = 1  # Thickness convergent
n6 = 1  # Thickness divergent

# %% Material selection
material = 2
if material == 0:
    material_name = "pure copper"
elif material == 1:
    material_name = "cucrzr"
elif material == 2:
    material_name = "inconel"

# %% Properties of the coolant
fluid = "Ethanol"
Temp_cool_init = 352  # Initial temperature of the coolant (in K)
Pressure_cool_init = 2700000  # Pressure of the coolant at inlet (in Pa)
density_cool_init = PropsSI("D", "T", Temp_cool_init, "P", Pressure_cool_init,
                            fluid)  # Density of the ethanol (in kg/m^3)
debit_volumique_total_cool = debit_mass_coolant / density_cool_init  # Total volumic flow rate of the coolant (in m^3/s)
roughness = 50e-6  # Roughness (m)

# %% Computation of channel geometry

profile = (x_coord_list, y_coord_list)
widths = (lrg_inj, lrg_conv, lrg_col, lrg_tore)
heights = (ht_inj, ht_conv, ht_col, ht_tore)
thicknesses = (e_conv, e_col, e_tore)
coeffs = (n1, n2, n3, n4, n5, n6)

# Compute dimensions
xcanaux, ycanaux, larg_canal, larg_ailette_list, ht_canal, wall_thickness, \
area_channel, nb_points_channel, y_coord_avec_canaux \
    = canaux(profile, widths, heights, thicknesses, coeffs, manifold_pos,
             debit_volumique_total_cool, nbc, plot_detail, write_in_csv, figure_dpi)

# Write the dimensions of the channels in a CSV file
file_name = "output/channelvalue.csv"
with open(file_name, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("Engine x", "Engine y", "y coolant wall", "Channel width", "Rib width",
                     "Channel height", "Chamber wall thickness", "Channel area"))
    for i in range(0, nb_points_channel):
        writer.writerow((xcanaux[i], y_coord_avec_canaux[i], ycanaux[i], larg_canal[i], larg_ailette_list[i],
                         ht_canal[i], wall_thickness[i], area_channel[i]))

end_init_time = time.perf_counter()  # End of the initialisation timer
time_elapsed = f"{round(end_init_time - start_time, 2)}"  # Initialisation elapsed time (in s)
if len(time_elapsed) <= 3:
    time_elapsed_i = f"   {time_elapsed} s"
elif len(time_elapsed) == 4:
    time_elapsed_i = f"  {time_elapsed} s"
elif len(time_elapsed) == 5:
    time_elapsed_i = f" {time_elapsed} s"
else:
    time_elapsed_i = f"{time_elapsed} s"
start_main_time = time.perf_counter()  # Start of the main solution timer

# %% Prepare the lists before main computation

wall_thickness.reverse()
xcanaux.reverse()
larg_canal.reverse()
area_channel.reverse()
ht_canal.reverse()
ycanaux.reverse()
y_coord_avec_canaux.reverse()
# We reverse the lists in order to calculate from the manifold to the injection

# Save the data for exporting, before altering the original lists
aire_saved = cross_section_area_list[:]
mach_list_saved = mach_list[:]
gamma_saved = gamma_list[:]
PH2O_list_saved = partial_p_H2O_list[:]
PCO2_list_saved = partial_p_CO2_list[:]

# Remove the data points before the manifold
recovery_hotgas_temp_list = recovery_hotgas_temp_list[:nb_points_channel]
cross_section_area_list = cross_section_area_list[:nb_points_channel]
mach_list = mach_list[:nb_points_channel]
gamma_list = gamma_list[:nb_points_channel]
partial_p_H2O_list = partial_p_H2O_list[:nb_points_channel]
partial_p_CO2_list = partial_p_CO2_list[:nb_points_channel]

gamma_list.reverse()
mach_list.reverse()
cross_section_area_list.reverse()
recovery_hotgas_temp_list.reverse()
partial_p_H2O_list.reverse()
partial_p_CO2_list.reverse()

# %% Main computation

data_hotgas = (recovery_hotgas_temp_list, molar_mass, gamma_list, Pc, c_star, partial_p_H2O_list, partial_p_CO2_list)
data_coolant = (Temp_cool_init, Pressure_cool_init, fluid, debit_mass_coolant)
data_channel = (xcanaux, ycanaux, larg_canal, larg_ailette_list, ht_canal,
                wall_thickness, area_channel, nb_points_channel)
data_chamber = (y_coord_avec_canaux, nbc, diam_throat, curv_radius_pre_throat, area_throat,
                roughness, cross_section_area_list, mach_list, material_name, combustion_efficiency)

# Call the main solving loop
hlcor_list, hlcor_list_2, hotgas_visc_list, hotgas_cp_list, hotgas_cond_list, \
hotgas_prandtl_list, hg_list, hotwall_temp_list, coldwall_temp_list, total_flux_list, \
sigma_list, coolant_reynolds_list, coolant_temp_list, coolant_viscosity_list, \
coolant_cond_list, coolant_cp_list, coolant_density_list, coolant_velocity_list, \
coolant_pressure_list, coolant_prandtl_list, wallcond_list, sound_speed_coolant_list, hlnormal_list, \
rad_flux_list, rad_CO2_list, rad_H2O_list, critical_heat_flux_list, Nu_list, Nu_corr_list, Dhy_list, \
vapor_quality_list \
    = mainsolver(data_hotgas, data_coolant, data_channel, data_chamber, chen=True)

end_m = time.perf_counter()  # End of the main solution timer
time_elapsed = f"{round(end_m - start_main_time, 2)}"  # Main computation elapsed time (in s)
if len(time_elapsed) <= 3:
    time_elapsed_m = f"   {time_elapsed} s"
elif len(time_elapsed) == 4:
    time_elapsed_m = f"  {time_elapsed} s"
elif len(time_elapsed) == 5:
    time_elapsed_m = f" {time_elapsed} s"
else:
    time_elapsed_m = f"{time_elapsed} s"

# %% Flux computation in 2D and 3D
"""2D flux computation"""
larg_ailette_list.reverse()

if show_2D_temperature:
    start_d2 = time.perf_counter()  # Start of the display of 2D timer
    # At the beginning of the chamber
    print("█ Results at the beginning of the chamber :                                █")
    dx = 0.00004  # *3.5
    location = " at the beginning of the chamber"
    carto2D(larg_ailette_list[-1] + larg_canal[-1], larg_canal[-1], e_conv, ht_canal[-1], dx, hg_list[-1],
            wallcond_list[-1], recovery_hotgas_temp_list[-1], hlcor_list[-1], coolant_temp_list[-1], 5, True, 1,
            location, False)

    # At the throat
    print("█ Results at the throat :                                                  █")
    pos_col = ycanaux.index(min(ycanaux))
    dx = 0.000025  # *3.5
    location = " at the throat"
    carto2D(larg_ailette_list[pos_col] + larg_canal[pos_col], larg_canal[pos_col], e_col, ht_canal[pos_col],
            dx, hg_list[pos_col], wallcond_list[pos_col], recovery_hotgas_temp_list[pos_col], hlcor_list[pos_col],
            coolant_temp_list[pos_col], 15, True, 2, location, False)
    # At the end of the divergent
    print("█ Results at the manifold :                                                █")
    dx = 0.00004
    location = " at the manifold"
    carto2D(larg_ailette_list[0] + larg_canal[0], larg_canal[0], e_tore, ht_canal[0], dx, hg_list[0],
            wallcond_list[0], recovery_hotgas_temp_list[0], hlcor_list[0], coolant_temp_list[0], 5, True, 1, location,
            False)

    end_d2 = time.perf_counter()  # End of the display of 2D timer
    time_elapsed = f"{round(end_d2 - start_d2, 2)}"  # 2D display elapsed time (in s)
    if len(time_elapsed) <= 3:
        time_elapsed_d2 = f"   {time_elapsed} s"
    elif len(time_elapsed) == 4:
        time_elapsed_d2 = f"  {time_elapsed} s"
    elif len(time_elapsed) == 5:
        time_elapsed_d2 = f" {time_elapsed} s"
    else:
        time_elapsed_d2 = f"{time_elapsed} s"

"Computation for 3D graph"
if do_final_3d_plot:
    start_d3 = time.perf_counter()
    temperature_slice_list = []
    lim1 = 0
    lim2 = 650
    dx = 0.0001

    # Compute a (low-resolution) 2D slice for each point in the engine
    with tqdm(total=nb_points_channel,
              desc="█ 3D graph computation         ",
              unit="|   █", bar_format="{l_bar}{bar}{unit}",
              ncols=76) as progressbar:
        for i in range(0, nb_points_channel):
            temperature_slice = carto2D(larg_ailette_list[i] + larg_canal[i], larg_canal[i], wall_thickness[i],
                                        ht_canal[i], dx, hg_list[i], wallcond_list[i], recovery_hotgas_temp_list[i],
                                        hlnormal_list[i], coolant_temp_list[i], 3, False, 1, "", True)
            temperature_slice_list.append(temperature_slice)
            progressbar.update(1)

    # Stack all these slices in a final 3D plot
    carto3d([0, 0, 0], xcanaux, ycanaux, temperature_slice_list, plt.cm.coolwarm,
            '3D view of wall temperatures (in K)', nbc, limitation)
    print("█                                                                          █")
    # End of the 3D display timer
    end_d3 = time.perf_counter()
    time_elapsed = f"{round(end_d3 - start_d3, 2)}"  # 3D display elapsed time (in s)
    if len(time_elapsed) <= 3:
        time_elapsed_d3 = f"   {time_elapsed} s"
    elif len(time_elapsed) == 4:
        time_elapsed_d3 = f"  {time_elapsed} s"
    elif len(time_elapsed) == 5:
        time_elapsed_d3 = f" {time_elapsed} s"
    else:
        time_elapsed_d3 = f"{time_elapsed} s"
start_e = time.perf_counter()  # Start of the end timer

# %% Reversion of the lists

cross_section_area_list.reverse()
gamma_list.reverse()
mach_list.reverse()
recovery_hotgas_temp_list.reverse()
xcanaux.reverse()
ycanaux.reverse()
larg_canal.reverse()
larg_ailette_list.reverse()
ht_canal.reverse()
area_channel.reverse()
hotgas_visc_list.reverse()
hotgas_cp_list.reverse()
hotgas_cond_list.reverse()
hg_list.reverse()
hotgas_prandtl_list.reverse()
sigma_list.reverse()
coldwall_temp_list.reverse()
hotwall_temp_list.reverse()
total_flux_list.reverse()
coolant_temp_list.reverse()
coolant_velocity_list.reverse()
coolant_reynolds_list.reverse()
hlnormal_list.reverse()
hlcor_list.reverse()
hlcor_list_2.reverse()
coolant_density_list.reverse()
coolant_viscosity_list.reverse()
coolant_cond_list.reverse()
coolant_cp_list.reverse()
coolant_prandtl_list.reverse()
coolant_pressure_list.reverse()
partial_p_H2O_list.reverse()
partial_p_CO2_list.reverse()
y_coord_avec_canaux.reverse()
Nu_list.reverse()
Nu_corr_list.reverse()
Dhy_list.reverse()
wallcond_list.reverse()
rad_CO2_list.reverse()
rad_H2O_list.reverse()
rad_flux_list.reverse()
critical_heat_flux_list.reverse()
# %% Preparation of the lists for CAD modelisation
"Changing the coordinates of the height of the channels (otherwise it is geometrically wrong)"

angles = [0]
newxhtre = [xcanaux[0]]
newyhtre = [ycanaux[0] + ht_canal[0]]
for i in range(1, nb_points_channel):
    if i == (nb_points_channel - 1):
        angle = angles[i - 1]
        angles.append(angle)
    else:
        vect1 = (xcanaux[i] - xcanaux[i - 1]) / (
                (((ycanaux[i] - ycanaux[i - 1]) ** 2) + ((xcanaux[i] - xcanaux[i - 1]) ** 2)) ** 0.5)
        vect2 = (xcanaux[i + 1] - xcanaux[i]) / (
                (((ycanaux[i + 1] - ycanaux[i]) ** 2) + ((xcanaux[i + 1] - xcanaux[i]) ** 2)) ** 0.5)
        angle1 = np.rad2deg(np.arccos(vect1))
        angle2 = np.rad2deg(np.arccos(vect2))
        angle = angle2
        angles.append(angle)
    newx = xcanaux[i] + ht_canal[i] * np.sin(np.deg2rad(angles[i]))
    newy = ycanaux[i] + ht_canal[i] * np.cos(np.deg2rad(angles[i]))
    newxhtre.append(newx)
    newyhtre.append(newy)

# Checking the height of channels
verification = []
print("█ Checking and computing channel height                                    █")
for i in range(0, nb_points_channel):
    verifhtre = (((newxhtre[i] - xcanaux[i]) ** 2) + ((newyhtre[i] - ycanaux[i]) ** 2)) ** 0.5
    verification.append(verifhtre)

# %% Display of the analysis results
print("█                                                                          █")

parameters_plotter = (plot_detail, show_3d_plots, show_2D_temperature,
                      do_final_3d_plot, figure_dpi, size2, limitation,
                      show_plots, save_plots)
data_plotter = (x_coord_list_mm, y_coord_list_mm, x_coord_list, y_coord_list, ycanaux, xcanaux,
                cross_section_area_list, gamma_list, mach_list, pressure_list, Molfrac_H2O, Molfrac_CO2,
                partial_p_H2O_list, partial_p_CO2_list, total_hotgas_temp_list, recovery_hotgas_temp_list,
                static_hotgas_temp_list, larg_ailette_list, larg_canal, ht_canal, wall_thickness, area_channel,
                hlnormal_list, hlcor_list, hlcor_list_2, hotwall_temp_list, coldwall_temp_list, total_flux_list,
                critical_heat_flux_list, coolant_temp_list, coolant_pressure_list, sound_speed_coolant_list,
                coolant_velocity_list, wallcond_list, material_name, hg_list, coolant_density_list, rad_CO2_list,
                rad_H2O_list, rad_flux_list, hotgas_visc_list, hotgas_cp_list, hotgas_cond_list,
                hotgas_prandtl_list, sigma_list, coolant_reynolds_list, coolant_cond_list, coolant_cp_list,
                coolant_viscosity_list, coolant_prandtl_list, newyhtre, verification, vapor_quality_list)

# Plot the results !
start_d1 = time.perf_counter()
plotter(parameters_plotter, data_plotter)
end_d1 = time.perf_counter()
time_elapsed = f"{round(end_d1 - start_d1, 2)}"  # 1D display elapsed time (in s)
if len(time_elapsed) <= 3:
    time_elapsed_d1 = f"   {time_elapsed} s"
elif len(time_elapsed) == 4:
    time_elapsed_d1 = f"  {time_elapsed} s"
elif len(time_elapsed) == 5:
    time_elapsed_d1 = f" {time_elapsed} s"
else:
    time_elapsed_d1 = f"{time_elapsed} s"

# %% Writing the results of the study in a CSV file

if write_in_csv:
    print("█ Writing results in .csv files                                            █")
    valuexport = open("output/valuexport.csv", "w", newline="")
    geometry1 = open("output/geometry1.csv", "w", newline="")
    geometry2 = open("output/geometry2.csv", "w", newline="")
    valuexport_writer = csv.writer(valuexport)
    geometry1_writer = csv.writer(geometry1)
    geometry2_writer = csv.writer(geometry2)
    valuexport_writer.writerow(
        ("x [mm]",
         "y [mm]",
         "Cross sectionnal area [m²]",
         "Gas gamma [-]",
         "Mach number [-]",
         "Gas pressure [bar]",
         "Channel x [mm]",
         "Channel y [mm]",
         "Channel width [mm]",
         "Channel height [mm]",
         "Channel area [m²]",
         "Hydraulic diameter [mm]",

         "Gas viscosity [µPa.s]",
         "Gas cp [J/kg.K]",
         "Gas conductivity [W/m.K]",
         "Gas Prandtl [-]",
         "Gas temperature [K]",
         "Adiabatic wall (recovery) gas temperature [K]",

         "hg [W/m².K]",
         "Sigma (Bartz) [-]",
         "hl [W/m².K]",
         "hl corrigé (Julien) [W/m².K]",
         "hl corrigé (L. Denies) [W/m².K]",
         "Cold side wall temperature [K]",
         "Hot side wall temperature [K]",
         "CO2 radiative heat flux [MW/m²]",
         "H2O radiative heat flux [MW/m²]",
         "Total radiative heat flux [MW/m²]",
         "Heat flux [MW/m²]",
         "Critical Heat Flux [MW/m²]",

         "Coolant temperature [K]",
         "Coolant Reynolds [-]",
         "Coolant density [kg/m^3]",
         "Coolant viscosity [µPa.s]",
         "Coolant conductivity [W/m.K]",
         "Coolant cp [J/kg.K]",
         "Coolant velocity [m/s]",
         "Coolant pressure [bar]",
         "Coolant Prandtl [-]",
         "Nusselt number [-]",
         "Corrected Nusselt (roughness) [-]",

         "Wall conductivity [W/m.K]",
         "x real height [mm]",
         "y real height [mm]"))

    geometry1_writer.writerow(("x real height", "y real height"))
    geometry2_writer.writerow(("Engine + chamber wall radius", "x real height"))

    for i in range(0, nb_points):
        if i < nb_points_channel:
            geometry1_writer.writerow((newxhtre[i] * (-1000), newyhtre[i] * 1000))
            geometry2_writer.writerow((ycanaux[i] * 1000, newxhtre[i] * (-1000)))
            valuexport_writer.writerow((x_coord_list[i] * 1000,
                                        y_coord_list[i] * 1000,
                                        aire_saved[i],
                                        gamma_saved[i],
                                        mach_list_saved[i],
                                        pressure_list[i] * 1e-5,
                                        xcanaux[i] * 1000,
                                        ycanaux[i] * 1000,
                                        larg_canal[i] * 1000,
                                        ht_canal[i] * 1000,
                                        area_channel[i],
                                        Dhy_list[i] * 1000,

                                        hotgas_visc_list[i] * 1e6,
                                        hotgas_cp_list[i],
                                        hotgas_cond_list[i],
                                        hotgas_prandtl_list[i],
                                        static_hotgas_temp_list[i],
                                        recovery_hotgas_temp_list[i],

                                        hg_list[i],
                                        sigma_list[i],
                                        hlnormal_list[i],
                                        hlcor_list[i],
                                        hlcor_list_2[i],
                                        coldwall_temp_list[i],
                                        hotwall_temp_list[i],
                                        rad_CO2_list[i] * 1e-6,
                                        rad_H2O_list[i] * 1e-6,
                                        rad_flux_list[i] * 1e-6,
                                        total_flux_list[i] * 1e-6,
                                        critical_heat_flux_list[i] * 1e-6,

                                        coolant_temp_list[i],
                                        coolant_reynolds_list[i],
                                        coolant_density_list[i],
                                        coolant_viscosity_list[i] * 1e6,
                                        coolant_cond_list[i],
                                        coolant_cp_list[i],
                                        coolant_velocity_list[i],
                                        coolant_pressure_list[i] * 1e-5,
                                        coolant_prandtl_list[i],
                                        Nu_list[i],
                                        Nu_corr_list[i],

                                        wallcond_list[i],
                                        newxhtre[i] * 1000,
                                        newyhtre[i] * 1000))
        else:
            valuexport_writer.writerow(
                (x_coord_list[i], y_coord_list[i], aire_saved[i], gamma_saved[i],
                 mach_list_saved[i], pressure_list[i],
                 ' ', ' ', ' ', ' ', ' ', ' ',
                 ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                 ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '))

    valuexport.close()
    geometry1.close()
    geometry2.close()

# %% Execution time display

end_t = time.perf_counter()  # End of the total timer
time_elapsed = f"{round(end_t - start_e, 2)}"  # End elapsed time (in s)
if len(time_elapsed) <= 3:
    time_elapsed_e = f"   {time_elapsed} s"
elif len(time_elapsed) == 4:
    time_elapsed_e = f"  {time_elapsed} s"
elif len(time_elapsed) == 5:
    time_elapsed_e = f" {time_elapsed} s"
else:
    time_elapsed_e = f"{time_elapsed} s"

time_elapsed = f"{round(end_t - start_time, 2)}"  # Total elapsed time
if len(time_elapsed) <= 3:
    time_elapsed_t = f"   {time_elapsed} s"
elif len(time_elapsed) == 4:
    time_elapsed_t = f"  {time_elapsed} s"
elif len(time_elapsed) == 5:
    time_elapsed_t = f" {time_elapsed} s"
else:
    time_elapsed_t = f"{time_elapsed} s"

print("█                                                                          █")
print("█__________________________________________________________________________█")
print("█                                                                          █")
print(f"█ Execution time for the initialisation       : {time_elapsed_i}                   █")
print("█                                                                          █")
print(f"█ Execution time for the main computation     : {time_elapsed_m}                   █")

if plot_detail >= 1:
    print("█                                                                          █")
    print(f"█ Execution time for the display of 1D        : {time_elapsed_d1}                   █")

if show_2D_temperature:
    print("█                                                                          █")
    print(f"█ Execution time for the display of 2D        : {time_elapsed_d2}                   █")

if do_final_3d_plot:
    print("█                                                                          █")
    print(f"█ Execution time for the display of 3D        : {time_elapsed_d3}                   █")

print("█                                                                          █")
print(f"█ Execution time for the end of the program   : {time_elapsed_e}                   █")
print("█                                                                          █")
print(f"█ Total execution time                        : {time_elapsed_t}                   █")
print("█                                                                          █")
print("███████████████████████████████████ END ████████████████████████████████████")
