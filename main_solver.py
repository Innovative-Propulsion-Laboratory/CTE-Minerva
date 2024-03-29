import numpy as np
import fluid_properties as flp
from tqdm import tqdm
import cte_tools as t
from CoolProp.CoolProp import PropsSI


def mainsolver(hotgas_data, coolant_data, channel_data, chamber_data, chen=False):
    """
    This is the main function used for solving the 1D case.
    The geometry is discretised into a 1 dimensionnal set of points.
    The function uses a marching algorithm, and computes all the relevant physical
    quantities at each point. The values obtained are then used on the next point.
    """

    hotgas_temp_list_with_film, hotgas_temp_list_no_film, molar_mass, gamma_list, Pc, c_star, PH2O, PCO2 = hotgas_data
    init_coolant_temp, init_coolant_pressure, fluid, \
    debit_mass_coolant = coolant_data
    xcanaux, ycanaux, larg_canal, larg_ailette_list, ht_canal, wall_thickness, \
    area_channel, nb_points_channel = channel_data
    y_coord_avec_canaux, nbc, diam_throat, curv_radius_pre_throat, area_throat, roughness, \
    cross_section_area_list, mach_list, material_name, combustion_efficiency = chamber_data

    # Lists containing the physical quantities at each point
    coolant_temp_list = [init_coolant_temp]
    coolant_pressure_list = [init_coolant_pressure]
    coolant_viscosity_list = [flp.viscosity(init_coolant_pressure, init_coolant_temp, fluid)]
    coolant_cond_list = [flp.conductivity(init_coolant_pressure, init_coolant_temp, fluid)]
    coolant_cp_list = [flp.cp(init_coolant_pressure, init_coolant_temp, fluid)]
    coolant_density_list = [flp.density(init_coolant_pressure, init_coolant_temp, fluid)]
    hotgas_viscosity_list = []
    hotgas_cp_list = []
    hotgas_cond_list = []
    hotgas_prandtl_list = []
    coolant_reynolds_list = []
    hg_list = []
    sigma_list = []
    wall_cond_list = []
    hotwall_temp_list = []
    coldwall_temp_list = []
    flux_list = []
    coolant_velocity_list = []
    sound_speed_list = []
    hl_normal_list = []
    hl_corrected_list = []
    hl_corrected_list_2 = []
    rad_H2O_list = []
    rad_CO2_list = []
    rad_flux_list = []
    critical_heat_flux_list = []
    Nu_list = []
    Nu_corr_list = []
    Dhy_list = []
    coolant_prandtl_list = []
    phase = 0
    vapor_quality = 0
    vapor_quality_list = []
    hotgas_temp_list_with_film.reverse()
    hotgas_temp_list_no_film.reverse()

    index_throat = y_coord_avec_canaux.index(min(y_coord_avec_canaux))

    # This is only relevant for the Taylor correlation for CH4 (leave it at 0)
    length_from_inlet = 0.00

    # Initial guess for the wall temperature
    coldwall_temp = 300
    hotwall_temp = 300

    debit_mass_coolant *= 0.93

    with tqdm(total=nb_points_channel,
              desc="█ Global resolution            ",
              unit="|   █", bar_format="{l_bar}{bar}{unit}",
              ncols=76) as pbar_main:

        # Main computation loop
        for i in range(0, nb_points_channel):
            # Hydraulic diameter (4 * Area/Perimeter )
            Dhy = (2 * ht_canal[i] * larg_canal[i]) / (ht_canal[i] + larg_canal[i])

            # Velocity of the coolant
            v_cool = debit_mass_coolant / (nbc * coolant_density_list[i] * area_channel[i])
            coolant_velocity_list.append(v_cool)

            # Reynolds number of the coolant
            Re_cool = (v_cool * Dhy * coolant_density_list[i]) / coolant_viscosity_list[i]
            coolant_reynolds_list.append(Re_cool)

            # Prandtl number of the coolant
            Pr_cool = (coolant_viscosity_list[i] * coolant_cp_list[i]) / coolant_cond_list[i]

            # Compute viscosity, Cp, conductivity and Prandtl number of the hot gases
            hotgas_visc, hotgas_cp, hotgas_cond, hotgas_prandtl = t.hotgas_properties(hotgas_temp_list_with_film[i],
                                                                                      molar_mass,
                                                                                      gamma_list[i])

            critical_heat_flux = t.compute_chf_1(P_SI=coolant_pressure_list[i],
                                                 T_SI=coolant_temp_list[i],
                                                 V_SI=coolant_velocity_list[i],
                                                 rho_SI=coolant_density_list[i],
                                                 Re=Re_cool)

            # If last point in the list
            if i == nb_points_channel - 1:
                # Distance between current point and the previous (Pythagoras)
                dl = ((xcanaux[i - 1] - xcanaux[i]) ** 2 + (ycanaux[i - 1] - ycanaux[i]) ** 2) ** 0.5
            else:
                # Distance between current point and the next (Pythagoras)
                dl = ((xcanaux[i + 1] - xcanaux[i]) ** 2 + (ycanaux[i + 1] - ycanaux[i]) ** 2) ** 0.5

            length_from_inlet += dl

            # Reuse the value at previous point for a more accurate first guess (and faster convergence)
            wall_cond = 15 if i == 0 else wall_cond_list[i - 1]
            sigma = 1 if i == 0 else sigma_list[i - 1]

            # Arbitrarely create a difference to enter the loop
            new_coldwall_temp = coldwall_temp + 10
            new_hotwall_temp = hotwall_temp + 10

            new_coolant_pressure = init_coolant_pressure
            new_coolant_temp = init_coolant_temp

            # if phase == 0 or phase == 2:
            #     vapor_quality = PropsSI("Q", "P", new_coolant_pressure, "T", new_coolant_temp, fluid)
            # elif phase == 1:
            #     H_liq = PropsSI("H", "Q", 0, "P", new_coolant_pressure, fluid)
            #     H_vap = PropsSI("H", "Q", 1, "P", new_coolant_pressure, fluid)
            #     vapor_quality = (H2 - H_liq) / (H_vap - H_liq)

            # This loop'slot_height goal is to find sigma and the wall conductivity
            # It iterates until the wall temperatures have converged
            while abs(new_coldwall_temp - coldwall_temp) > .2 and abs(new_hotwall_temp - hotwall_temp) > .2:
                coldwall_temp = new_coldwall_temp
                hotwall_temp = new_hotwall_temp

                c_star_corr = c_star * np.sqrt(combustion_efficiency)
                # Gas-side convective heat transfer coefficient (Bartz equation)
                hg = (0.026 / (diam_throat ** 0.2) * (((hotgas_visc ** 0.2) * hotgas_cp) / (hotgas_prandtl ** 0.6)) * (
                        (Pc / c_star_corr) ** 0.8) * ((diam_throat / curv_radius_pre_throat) ** 0.1) * (
                              (area_throat / cross_section_area_list[i]) ** 0.9)) * sigma

                if chen:
                    hl = t.chen_correlation(Re_l=Re_cool, Pr_l=Pr_cool,
                                            Dhy=Dhy,
                                            cp_l=coolant_cp_list[i],
                                            density_l=coolant_density_list[i],
                                            cond_l=coolant_cond_list[i],
                                            visc_l=coolant_viscosity_list[i],
                                            T_l=coolant_temp_list[i],
                                            P_l=coolant_pressure_list[i],
                                            T_wall=coldwall_temp)
                else:
                    # Coolant-side convective heat transfer coefficient (Modified Gnielinski by M.M. Sarafraz)
                    Nu = 0.02326 * (Re_cool ** 0.83 - 100) * Pr_cool ** 0.4201

                    # Nusselt number correction for the channel roughness
                    xi = t.darcy_weisbach(Dhy, Re_cool, roughness) / t.darcy_weisbach(Dhy, Re_cool, 0)
                    roughness_correction = xi * (
                            (1 + 1.5 * Pr_cool ** (-1 / 6) * Re_cool ** (-1 / 8) * (Pr_cool - 1)) / (
                            1 + 1.5 * Pr_cool ** (-1 / 6) * Re_cool ** (-1 / 8) * (Pr_cool * xi - 1)))

                    # Compute coolant-side convective heat-transfer coefficient
                    hl = Nu * roughness_correction * (coolant_cond_list[i] / Dhy)

                # Fin dimensions
                D = 2 * y_coord_avec_canaux[i]  # Diameter inside the engine
                fin_width = (np.pi * (D + ht_canal[i] + wall_thickness[i]) - nbc * larg_canal[
                    i]) / nbc  # Width of the fin

                # Correct for the fin effect (unknown source)
                m_ = ((2 * hl) / (fin_width * wall_cond)) ** 0.5
                hl_cor = hl * ((nbc * larg_canal[i]) / (np.pi * D)) + nbc * (
                        (2 * hl * wall_cond * (((np.pi * D) / nbc) - larg_canal[i])) ** 0.5) * (
                                 (np.tanh(m_ * ht_canal[i])) / (np.pi * D))

                # Correct for the fin effect (Luka Denies)
                intermediate_calc_1 = ((2 * hl * fin_width) / wall_cond) ** 0.5 * ht_canal[i] / fin_width
                nf = np.tanh(intermediate_calc_1) / intermediate_calc_1
                hl_cor2 = hl * (larg_canal[i] + 2 * nf * ht_canal[i]) / (larg_canal[i] + fin_width)

                # Compute radiative heat transfer of H2O (W) and CO2 (C) (Luka Denies)
                qW = 5.74 * ((PH2O[i] * y_coord_avec_canaux[i]) / 1e5) ** 0.3 * (
                        hotgas_temp_list_no_film[i] / 100) ** 3.5
                qC = 4 * ((PCO2[i] * y_coord_avec_canaux[i]) / 1e5) ** 0.3 * (hotgas_temp_list_no_film[i] / 100) ** 3.5
                qRad = qW + qC

                # Computing the heat flux and wall temperatures (Luka Denies)
                flux = (hotgas_temp_list_with_film[i] - coolant_temp_list[i] + qRad / hg) / (
                        1 / hg + 1 / hl_cor + wall_thickness[i] / wall_cond)
                new_hotwall_temp = hotgas_temp_list_with_film[i] + (qRad - flux) / hg
                new_coldwall_temp = coolant_temp_list[i] + flux / hl

                # Compute new value of sigma (used in the Bartz equation)
                T_hotgas_throat = hotgas_temp_list_with_film[index_throat]
                mach_hot_gases = mach_list[i]
                sigma = (((new_hotwall_temp / (2 * T_hotgas_throat)) * (
                        1 + (((gamma_list[i] - 1) / 2) * (mach_hot_gases ** 2))) + 0.5) ** -0.68) * (
                                (1 + (((gamma_list[i] - 1) / 2) * (mach_hot_gases ** 2))) ** -0.12)

                # Compute thermal conductivity of the solid at a given temperature
                wall_cond = t.conductivity(Twg=new_hotwall_temp, Twl=new_coldwall_temp, material_name=material_name)

            coldwall_temp = new_coldwall_temp
            hotwall_temp = new_hotwall_temp

            # Compute heat exchange area between two points
            # Cold-wall version (Julien)
            dA_1 = 2 * dl * (larg_canal[i] + ht_canal[i])
            # Hot-wall version (Luka Denies)
            dA_2 = (np.pi * D * dl) / nbc

            # New temperature at next point
            delta_T_coolant = ((flux * dA_2) / ((debit_mass_coolant / nbc) * coolant_cp_list[i]))
            new_coolant_temp = coolant_temp_list[i] + delta_T_coolant

            # Solving Colebrook'slot_height formula to obtain the Darcy-Weisbach friction factor
            frict_factor = t.darcy_weisbach(Dhy, Re_cool, roughness)

            # Computing pressure loss with the Darcy-Weisbach friction factor (no pressure loss taken into account)
            delta_p = 0.5 * frict_factor * (dl / Dhy) * coolant_density_list[i] * v_cool ** 2
            new_coolant_pressure = coolant_pressure_list[i] - delta_p

            # if phase == 0:
            #     H1 = PropsSI("H", "P", new_coolant_pressure, "T", new_coolant_temp, fluid)
            #     H2 = H1 + flux / debit_mass_coolant
            #     vapor_quality = PropsSI("Q", "H", H2, "P", new_coolant_pressure, fluid)
            #     phase = 1 if (0 < vapor_quality < 1) else 0
            # else:
            #     H1 = H2
            #     H2 = H1 + flux / debit_mass_coolant
            #     vapor_quality = PropsSI("Q", "H", H2, "P", new_coolant_pressure, fluid)
            #     phase = 1 if (0 < vapor_quality < 1) else 2

            # Computing the new properties of the ethanol (properties considered constant)
            if new_coolant_pressure < 0:
                print(f"Last coolant temperature :{new_coolant_temp:.2f} K")
                raise ValueError("Negative pressure ! Pressure drop is too high.")
            new_cool_visc = flp.viscosity(P=new_coolant_pressure, T=new_coolant_temp, fluid=fluid)
            new_cool_cond = flp.conductivity(P=new_coolant_pressure, T=new_coolant_temp, fluid=fluid)
            new_cool_cp = flp.cp(P=new_coolant_pressure, T=new_coolant_temp, fluid=fluid)
            new_cool_dens = flp.density(P=new_coolant_pressure, T=new_coolant_temp, fluid=fluid)
            new_cool_sound_spd = flp.sound_speed(P=new_coolant_pressure, T=new_coolant_temp, fluid=fluid)

            # Store the results
            vapor_quality_list.append(vapor_quality)
            hotgas_viscosity_list.append(hotgas_visc)
            hotgas_cp_list.append(hotgas_cp)
            hotgas_cond_list.append(hotgas_cond)
            hotgas_prandtl_list.append(hotgas_prandtl)

            hg_list.append(hg)
            sigma_list.append(sigma)
            hl_normal_list.append(hl)
            hl_corrected_list.append(hl_cor)
            hl_corrected_list_2.append(hl_cor2)

            hotwall_temp_list.append(hotwall_temp)
            coldwall_temp_list.append(coldwall_temp)
            flux_list.append(flux)
            critical_heat_flux_list.append(critical_heat_flux)
            rad_flux_list.append(qRad)
            rad_CO2_list.append(qC)
            rad_H2O_list.append(qW)
            wall_cond_list.append(wall_cond)

            coolant_pressure_list.append(new_coolant_pressure)
            coolant_temp_list.append(new_coolant_temp)
            coolant_viscosity_list.append(new_cool_visc)
            coolant_cond_list.append(new_cool_cond)
            coolant_cp_list.append(new_cool_cp)
            coolant_density_list.append(new_cool_dens)
            coolant_prandtl_list.append(Pr_cool)
            sound_speed_list.append(new_cool_sound_spd)
            if chen:
                Nu_list.append("Using chen, not available")
                Nu_corr_list.append("Using chen, not available")
            else:
                Nu_list.append(Nu)
                Nu_corr_list.append(Nu * roughness_correction)
            Dhy_list.append(Dhy)

            pbar_main.update(1)

        return hl_corrected_list, hl_corrected_list_2, hotgas_viscosity_list, \
               hotgas_cp_list, hotgas_cond_list, hotgas_prandtl_list, hg_list, \
               hotwall_temp_list, coldwall_temp_list, flux_list, sigma_list, \
               coolant_reynolds_list, coolant_temp_list, coolant_viscosity_list, \
               coolant_cond_list, coolant_cp_list, coolant_density_list, \
               coolant_velocity_list, coolant_pressure_list, coolant_prandtl_list, wall_cond_list, \
               sound_speed_list, hl_normal_list, rad_flux_list, rad_CO2_list, \
               rad_H2O_list, critical_heat_flux_list, Nu_list, Nu_corr_list, Dhy_list, vapor_quality_list
