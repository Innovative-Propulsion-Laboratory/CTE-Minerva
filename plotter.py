import matplotlib.pyplot as plt
import numpy as np
import cte_tools as t
from volume3d import view3d
import matplotlib.backends.backend_pdf


def plotter(parameters, data):
    # Disable warning about too many figure objects in memory.
    plt.rcParams.update({'figure.max_open_warning': 0})

    # List containing all figure objects
    figs = []

    plot_detail, show_3d_plots, show_2D_temperature, \
    do_final_3d_plot, figure_dpi, size2, limitation, show, save = parameters

    x_coord_list_mm, y_coord_list_mm, x_coord_list, y_coord_list, ycanaux, xcanaux, \
    cross_section_area_list, gamma_list, mach_list, pressure_list, Molfrac_H2O, Molfrac_CO2, \
    partial_p_H2O_list, partial_p_CO2_list, total_hotgas_temp_list, recovery_hotgas_temp_list, \
    static_hotgas_temp_list, larg_ailette_list, larg_canal, ht_canal, wall_thickness, area_channel, \
    vitesse_coolant, hlnormal_list, hlcor_list, hlcor_list_2, hotwall_temp_list, coldwall_temp_list, total_flux_list, \
    critical_heat_flux_list, coolant_temp_list, coolant_pressure_list, sound_speed_coolant_list, \
    coolant_velocity_list, wallcond_list, material_name, hg_list, coolant_density_list, rad_CO2_list, \
    rad_H2O_list, rad_flux_list, hotgas_visc_list, hotgas_cp_list, hotgas_cond_list, \
    hotgas_prandtl_list, sigma_list, coolant_reynolds_list, coolant_cond_list, coolant_cp_list, \
    coolant_viscosity_list, coolant_prandtl_list, newyhtre, verification, vapor_quality = data

    # Plot of the profile of the engine
    if plot_detail >= 3:
        figs.append(t.one_plot(x_coord_list_mm, y_coord_list_mm,
                               ylabel='Radius [mm]',
                               xlabel='x-coordinate [mm]',
                               title='Profile of the Minerva engine',
                               equal_axes=True, ymin=0, ymax=100, xmin=-200, dpi=figure_dpi, show=show))

    # Computation and plot of the mesh density of the engine
    if plot_detail >= 3 and show_3d_plots:
        dist_between_pts = [abs(x_coord_list[i] - x_coord_list[i + 1]) for i in range(0, len(x_coord_list) - 1)]
        dist_between_pts.append(dist_between_pts[-1])
        colormap = plt.cm.binary
        inv = 1, 1, 1  # 1 means should be reversed
        view3d(inv, x_coord_list, y_coord_list, dist_between_pts, colormap, 'Mesh density (in m)', size2,
               limitation, show=show)

    # Plots of the cross-sectionnal areas
    if plot_detail >= 3:
        figs.append(t.one_plot(x_coord_list_mm, cross_section_area_list,
                               title='Cross-sectional area inside the engine',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'Area [$m^2$]', ymin=0, ymax=0.018, xmin=-200, dpi=figure_dpi, show=show))

    # Plot of the gamma linearisation
    if plot_detail >= 3:
        figs.append(t.one_plot(x_coord_list_mm, gamma_list,
                               title=r'Adiabatic constant $\gamma$ of the combustion gases',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$\gamma$ [-]', xmin=-200, dpi=figure_dpi, show=show))

    # Plots of the Mach number in the engine (2D/3D)
    if plot_detail >= 1:
        figs.append(t.one_plot(x_coord_list_mm, mach_list,
                               title=r'Mach number',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$Ma$ [-]', ymin=0, xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 2 and show_3d_plots:
        colormap = plt.cm.Spectral
        inv = 1, 1, 1  # 1 means should be reversed
        view3d(inv, x_coord_list, y_coord_list, mach_list,
               colormap, 'Mach number of hot gases', size2, limitation, show=show)

    # Plot of the static pressure (2D/3D)
    if plot_detail >= 2:
        figs.append(t.one_plot(x_coord_list_mm, pressure_list,
                               title=r'Static pressure',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$P$ [Pa]', ymin=0, xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 2 and show_3d_plots:
        colormap = plt.cm.gist_rainbow_r
        inv = 1, 1, 1  # 1 means should be reversed
        view3d(inv, x_coord_list, y_coord_list, pressure_list,
               colormap, 'Static pressure (in Pa)', size2,
               limitation, show=show)

    # Plots of molar fraction and partial pressure
    if plot_detail >= 3:
        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[Molfrac_H2O, Molfrac_CO2],
                              y_label_list=[r'$H_2O$', r'$CO_2$'],
                              colors_list=['r', 'b'],
                              title=r'Molar fraction of combustion products',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'Molar fraction $x_i$ [-]',
                              ymin=0, xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[partial_p_H2O_list, partial_p_CO2_list],
                              y_label_list=[r'$H_2O$', r'$CO_2$'],
                              colors_list=['r', 'b'],
                              title=r'Partial pressure of combustion products',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'Partial pressure $p_i$ [Pa]',
                              ymin=0, xmin=-200, dpi=figure_dpi, show=show, sci_notation=True))

    # Plots of the temperature in the engine (2D/3D)
    if plot_detail >= 1:
        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[static_hotgas_temp_list, recovery_hotgas_temp_list, total_hotgas_temp_list],
                              y_label_list=[r'Static temperature $T_s$',
                                            r'Recovery temperature $T_{aw}$',
                                            r'Total temperature $T_{tot}$'],
                              colors_list=['r', 'b', 'k'],
                              title=r'Combustion gases temperature',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$T$ [K]', xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, recovery_hotgas_temp_list,
                               title=r'Recovery temperature $T_{aw}$',
                               xlabel=r'x-coordinate [$mm$]', fmt='b',
                               ylabel=r'$T$ [K]',
                               ymin=min(recovery_hotgas_temp_list) - 30,
                               ymax=max(recovery_hotgas_temp_list) + 30,
                               xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 2 and show_3d_plots:
        colormap = plt.cm.coolwarm
        inv = 1, 1, 1  # 1 means should be reversed
        view3d(inv, x_coord_list, y_coord_list, recovery_hotgas_temp_list,
               colormap, 'Temperature of the hot gas (in K)', size2, limitation, show=show)

    if plot_detail >= 3:
        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[np.array(larg_ailette_list) * 1000,
                                      np.array(larg_canal) * 1000,
                                      np.array(ht_canal) * 1000],
                              y_label_list=['Fin width', 'Channel width', 'Channel height'],
                              colors_list=['b', 'r', 'g'],
                              title='Cooling channels dimensions',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'Length [mm]', ymin=0, xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, np.array(wall_thickness) * 1000,
                               title=r'Chamber wall thickness',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'Wall thickness $t$ [$mm$]',
                               ymin=0, ymax=2, xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, np.array(vitesse_coolant),
                               title=r'Coolant velocity',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'Velocity [$m/s$]',
                               ymin=0, ymax=50, xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 1:
        figs.append(t.one_plot(x_coord_list_mm, area_channel,
                               title=r'Channel cross-sectional area',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'Area [$m^2$]', sci_notation=True,
                               ymin=0, xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 1:
        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[np.array(hlnormal_list) * 1000,
                                      np.array(hlcor_list) * 1000,
                                      np.array(hlcor_list_2) * 1000],
                              y_label_list=['No correction', 'Correction 1 (P. Pempie)',
                                            'Correction 2 (Popp & Schmidt)'],
                              colors_list=['b', 'r', 'g'],
                              title=r'Cold-side convective coefficient $h_l$',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$h_l$ [$\frac{W}{m^2 \cdot K}$]',
                              ymin=0, xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[hotwall_temp_list, coldwall_temp_list],
                              y_label_list=['Hot side', 'Cold side'],
                              colors_list=['r', 'b'],
                              title=r'Wall temperatures $T_{wg}$ and $T_{wl}$',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$T$ [K]', xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[total_flux_list, critical_heat_flux_list],
                              y_label_list=['Total heat flux', 'Critical Heat Flux'],
                              colors_list=['r', 'k'],
                              title=r'Total heat flux and CHF',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$\dot q$ [$\frac{W}{m^2}$]',
                              xmin=-200, dpi=figure_dpi, show=show))

        coolant_temp_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_temp_list,
                               title=r'Coolant temperature',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$T$ [K]', xmin=-200, dpi=figure_dpi, show=show))

        coolant_pressure_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_pressure_list,
                               title=r'Coolant pressure', fmt='-b',
                               xlabel=r'x-coordinate [$mm$]', sci_notation=True,
                               ylabel=r'$P$ [Pa]', xmin=-200, dpi=figure_dpi, show=show))

        mach_03 = [x * 0.3 for x in sound_speed_coolant_list]
        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[coolant_velocity_list, mach_03],
                              y_label_list=['Coolant', 'Mach 0.3 limit'],
                              colors_list=['b', 'k'],
                              title=r'Coolant velocity',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$V_l$ [$m/s$]', xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 2:
        figs.append(t.one_plot(x_coord_list_mm, wallcond_list,
                               title=f'Wall conductivity ({material_name})',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$\lambda_w$ [$\frac{W}{m \cdot K}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, hg_list,
                               title=r'Hot-side convection coefficient $h_g$',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$P$ [Pa]', xmin=-200, dpi=figure_dpi, show=show))

        coolant_density_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_density_list,
                               title=r'Coolant density $\rho$',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$\rho$ [$\frac{kg}{m^3}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.n_plots(x_coord_list_mm,
                              y_list=[rad_CO2_list, rad_H2O_list, rad_flux_list],
                              y_label_list=['CO2', 'H2O', 'Total'],
                              colors_list=['b', 'r', 'g'],
                              title=r'Radiative heat flux',
                              xlabel=r'x-coordinate [$mm$]',
                              ylabel=r'$\dot q_{rad}$ [$\frac{W}{m^2}$]',
                              ymin=0, xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, sigma_list,
                               title=r'Bartz equation coefficient $\sigma$',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$\sigma$ [-]',
                               xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 3:
        figs.append(t.one_plot(x_coord_list_mm, hotgas_visc_list,
                               title=r'Gas dynamic viscosity',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$\mu$ [$\mu Pa\cdot s$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, hotgas_cp_list,
                               title=r'Hot gas $c_p$',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$c_p$ [$\frac{J}{K\cdot kg}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, hotgas_cond_list,
                               title=r'Hot gas conductivity',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$\lambda_l$ [$\frac{W}{m \cdot K}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, hotgas_prandtl_list,
                               title=r'Hot gas Prandtl number',
                               xlabel=r'x-coordinate [$mm$]',
                               ylabel=r'$Pr_g$ [-]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, coolant_reynolds_list,
                               title=r'Coolant Reynolds number',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$Re_l$ [-]',
                               xmin=-200, dpi=figure_dpi, show=show))

        coolant_cond_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_cond_list,
                               title=r'Coolant conductivity',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$\lambda_l$ [$\frac{W}{m \cdot K}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        coolant_cp_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_cp_list,
                               title=r'Coolant $c_p$',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$c_p$ [$\frac{J}{K\cdot kg}$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        coolant_viscosity_list.pop()
        figs.append(t.one_plot(x_coord_list_mm, coolant_viscosity_list,
                               title=r'Coolant dynamic viscosity',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$\mu$ [$\mu Pa\cdot s$]',
                               xmin=-200, dpi=figure_dpi, show=show))

        figs.append(t.one_plot(x_coord_list_mm, coolant_prandtl_list,
                               title=r'Coolant Prandtl number',
                               xlabel=r'x-coordinate [$mm$]', fmt='-b',
                               ylabel=r'$Pr_l$ [-]',
                               xmin=-200, dpi=figure_dpi, show=show))

    if plot_detail >= 1 and show_3d_plots:
        colormap = plt.cm.plasma
        inv = 0, 0, 0
        view3d(inv, xcanaux, ycanaux, total_flux_list, colormap,
               "Heat flux (in MW/m²)", size2, limitation, show=show)

        colormap = plt.cm.coolwarm
        inv = 0, 0, 0
        view3d(inv, xcanaux, ycanaux, coolant_temp_list, colormap,
               "Temperature of the coolant (in K)", size2,
               limitation, show=show)

    if plot_detail >= 2 and show_3d_plots:
        colormap = plt.cm.magma
        inv = 0, 0, 0  # 1 means should be reversed
        view3d(inv, xcanaux, ycanaux, coldwall_temp_list, colormap,
               "Wall temperature on the gas side (in K)", size2,
               limitation, show=show)

    if plot_detail >= 3:
        t.n_plots(x_coord_list_mm,
                  y_list=[np.array(ycanaux) * 1000, np.array(newyhtre) * 1000],
                  y_label_list=['Former height', 'New height'],
                  colors_list=['r', 'b'], equal_axes=True,
                  title=r'Geometrical aspect of the channels',
                  xlabel=r'x-coordinate [$mm$]',
                  ylabel=r'$Height$ [mm]', xmin=-200, dpi=figure_dpi, show=show)

        t.one_plot(np.array(xcanaux) * 1000, verification,
                   title=r'Generated channel height check',
                   xlabel=r'x-coordinate [$mm$]',
                   ylabel=r'Length [$mm$]',
                   xmin=-200, dpi=figure_dpi, show=show)

    if plot_detail >= 1 and save:
        pdf = matplotlib.backends.backend_pdf.PdfPages("output/graphs.pdf")
        for fig in figs:
            fig.savefig(pdf, format='pdf')
        pdf.close()
