import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def mach_solv(area_1, area_2, mach_1, gamma):
    if area_1 == area_2:
        solution_mach = mach_1
    else:
        ome = (gamma + 1) / (2 * (gamma - 1))
        part_2 = (area_1 / area_2) * (mach_1 / ((1 + ((gamma - 1) / 2) * mach_1 * mach_1) ** ome))
        mach_2 = mach_1
        liste = []
        mach = []
        # Search of the mach_2 for which part_1 is minimum (600 iterations was ideal when tested)
        for i in range(0, 600):
            mach_2 += 0.00001
            part_1 = mach_2 * ((1 + (((gamma - 1) / 2) * mach_2 * mach_2)) ** (-ome))
            liste.append(abs(part_1 - part_2))
            mach.append(mach_2)
        solution_mach = mach[liste.index(min(liste))]

    return solution_mach


def pressure_solv(mach_1, mach_2, pressure_1, gamma):
    """
    Compute hot gas pressure at next point, given mach numbers and previous pressure
    """

    part1 = (gamma / (gamma - 1)) * np.log(mach_1 * mach_1 + (2 / (gamma - 1)))
    part2 = (gamma / (gamma - 1)) * np.log(mach_2 * mach_2 + (2 / (gamma - 1)))
    part3 = np.log(pressure_1)

    return np.exp(part1 - part2 + part3)


def temperature_hotgas_solv(mach_1, mach_2, temperature_1, gamma):
    """
    Compute hot gas temperature at next point, given mach numbers and previous temperature
    """

    part1 = np.log(abs(((gamma - 1) * mach_1 * mach_1) + 2))
    part2 = np.log(abs(((gamma - 1) * mach_2 * mach_2) + 2))
    part3 = np.log(temperature_1)

    return np.exp(part1 - part2 + part3)


def tempcorrige_pempie(temp_original, gamma, mach):
    """
    Compute the recovery temperature (adiabatic wall temperature) [P. Pempie]
    """

    Pr = 4 * gamma / (9 * gamma - 5)
    recovery_temp = temp_original * (
            (1 + (Pr ** 0.33) * ((gamma - 1) / 2) * mach ** 2) / (1 + ((gamma - 1) / 2) * mach ** 2))

    return recovery_temp


def tempcorrige_denies(temp_original, gamma, mach):
    """
    Compute the recovery temperature (adiabatic wall temperature) [L. Denies]
    """

    Pr = 4 * gamma / (9 * gamma - 5)
    recovery_temp = temp_original * (1 + (Pr ** 0.33) * (mach ** 2) * ((gamma - 1) / 2))

    return recovery_temp


def total_temp_calculation(temp_original, gamma, mach):
    """Compute the total temperature of the hot gases"""

    return temp_original * (1 + (mach ** 2) * ((gamma - 1) / 2))


def conductivity(Twg: float, Twl: float, material_name: str):
    """
    Compute the conductivity of the chamber wall, given temperature and material
    """

    T_avg = (Twg + Twl) / 2
    if material_name == "pure copper":
        return -0.065665 * T_avg + 421.82
    if material_name == "cucrzr":
        return -0.0269 * T_avg + 365.33
    if material_name == "inconel":
        return 0.0138 * T_avg + 5.577


def hotgas_properties(gas_temp, molar_mass, gamma):
    """
    Computes the properties of the hot gases according to [INSERT SOURCE].
    """

    dyn_viscosity = 17.858 * (46.61 * 10 ** (-10)) * (molar_mass ** 0.5) * ((9 / 5) * gas_temp) ** 0.6
    Cp = 8314 * gamma / ((gamma - 1) * molar_mass)
    Lamb = dyn_viscosity * (Cp + (8314.5 / (4 * molar_mass)))
    Pr = 4 * gamma / (9 * gamma - 5)

    return dyn_viscosity, Cp, Lamb, Pr


def flux_equations(guess, *data):
    """
    Used by scipy.optimize.fsolve() to compute hot and cold wall temperature.
    """

    t_hot, t_cold = guess  # Initial guess
    hg, hl, t_g, t_c, wall_conductivity, wall_thickness = data

    # System of equations to solve
    f1 = hg * (t_g - t_hot) - (wall_conductivity / wall_thickness) * (t_hot - t_cold)
    f2 = hl * (t_cold - t_c) - (wall_conductivity / wall_thickness) * (t_hot - t_cold)

    return [f1, f2]


def darcy_weisbach(Dhy, Re, roughness):
    friction_factor_1 = 1e-3
    friction_factor_2 = (1 / (-2 * np.log10(
        ((roughness / (Dhy * 3.7)) + 2.51 / (Re * (friction_factor_1 ** 0.5)))))) ** 2

    while abs((friction_factor_1 / friction_factor_2) - 1) > 0.0000001:
        friction_factor_1 = friction_factor_2
        friction_factor_2 = (1 / (-2 * np.log10(
            ((roughness / (Dhy * 3.7)) + 2.51 / (Re * (friction_factor_1 ** 0.5)))))) ** 2

    return friction_factor_2


def compute_chf_2(V_SI, T_bulk_SI, P_SI):
    T_bulk_IMP = (T_bulk_SI - 273.15) * (9. / 5.) + 32
    T_sat_SI = PropsSI('T', 'P', P_SI, 'Q', 1, 'Ethanol')
    T_sat_IMP = (T_sat_SI - 273.15) * (9. / 5.) + 32
    delta_T = T_sat_IMP - T_bulk_IMP if T_sat_IMP - T_bulk_IMP > 0 else 0
    V_IMP = V_SI * 3.28084
    CHF_IMP = 0.1003 + 0.05264 + np.sqrt(V_IMP * delta_T)
    CHF_SI = 1634246 * CHF_IMP
    return CHF_SI


def compute_chf_1(P_SI, T_SI, V_SI, rho_SI, Re):
    """
    Critical Heat Flux for a water/ethanol mixture using NASA/TMB1998-206612
    """

    # Mass flux
    G_SI = rho_SI * V_SI
    G_IMP = (1 / 703.07) * G_SI  # lbm/in².s

    # Latent heat of vaporisation
    # (http://www.coolprop.org/coolprop/HighLevelAPI.html#vapor-liquid-and-saturation-states)
    H_vap = PropsSI('H', 'P', P_SI, 'Q', 1, 'Ethanol')
    H_liq = PropsSI('H', 'P', P_SI, 'Q', 0, 'Ethanol')
    H_fg_SI = H_vap - H_liq  # J/kg
    H_fg_IMP = (1 / 2326) * H_fg_SI  # BTU/lb

    # Heat capacity at constant pressure
    cp_SI = PropsSI("CPMASS", "T", T_SI, "P", P_SI, 'Ethanol')  # J/(kg.K)
    cp_IMP = 4186 * cp_SI  # BTU/(lb.°F)

    # Saturation temperature
    T_sat_SI = PropsSI('T', 'P', P_SI, 'Q', 0, 'Ethanol')  # K
    T_sat_IMP = (T_sat_SI - 273.15) * (9. / 5.) + 32  # °F
    T_IMP = (T_SI - 273.15) * (9. / 5.) + 32  # °F

    X_ex = -cp_IMP * (T_sat_IMP - T_IMP) / H_fg_IMP
    if X_ex < -0.1:
        phi = 1.0
    elif 0 > X_ex >= -0.1:
        phi = 0.825 + 0.986 * X_ex
    else:  # X_ex >= 0
        phi = 1.0 / (2 + 30 * X_ex)

    P_MPa = P_SI / 10e5
    C = (0.216 + 0.0474 * P_MPa) * phi

    # Critical Heat Flux
    CHF_IMP = G_IMP * H_fg_IMP * C * Re ** (-0.5)  # BTU/(in².s)
    CHF_SI = 1634246 * CHF_IMP  # W/m²

    return CHF_SI


def chen_correlation(Re_l, Pr_l, Dhy, cp_l, density_l, cond_l,
                     visc_l, T_l, P_l, T_wall):
    F = 1
    surf_tension = PropsSI('SURFACE_TENSION', 'T', T_l, 'Q', 1, 'Ethanol')
    H_vap = PropsSI('H', 'P', P_l, 'Q', 1, 'Ethanol')
    H_liq = PropsSI('H', 'P', P_l, 'Q', 0, 'Ethanol')
    H_fg_SI = H_vap - H_liq
    density_g = PropsSI('D', 'P', P_l, 'Q', 1, 'Ethanol')
    T_sat = PropsSI('T', 'P', P_l, 'Q', 1, 'Ethanol')
    P_sat = PropsSI('P', 'T', T_l, 'Q', 1, 'Ethanol')

    delta_T = T_wall - T_sat
    delta_P = P_sat - P_l

    num = (cond_l ** 0.79) * (cp_l ** 0.45) * (density_l ** 0.49)
    denom = (surf_tension ** 0.5) * (visc_l ** 0.29) * (H_fg_SI ** 0.24) * (density_g ** 0.24)

    S = 1 / (1 + 2.53e-6 * (Re_l ** 1.17))

    h_macro = 0.023 * (Re_l ** 0.8) * (Pr_l ** 0.4) * (cond_l / Dhy) * F
    h_micro = 0.00122 * (num / denom) * (abs(delta_T) ** 0.24) * (abs(delta_P) ** 0.75) * S

    return h_macro + h_micro


def one_plot(x, y,
             xlabel=r'Default xlabel',
             ylabel=r'Default ylabel',
             xmin=None, xmax=None,
             ymin=None, ymax=None,
             title=r'Default title', equal_axes=False, show_grid=True,
             fmt='-k', lw=1.5, dpi=150, sci_notation=False, show=True):
    serif = {'fontname': 'DejaVu Serif'}

    margin = 0.05
    if xmin is None:
        xmin = min(x) - margin * min(x)
    if xmax is None:
        xmax = max(x) + margin * max(x)
    if ymin is None:
        ymin = min(y) - margin * min(y)
    if ymax is None:
        ymax = max(y) + margin * max(y)

    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.minorticks_on()
    ax.plot(x, y, fmt, linewidth=lw)
    ax.set_xlabel(xlabel, **serif)
    ax.set_ylabel(ylabel, **serif)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, **serif)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    if equal_axes:
        ax.set_aspect('equal', adjustable='box')
    if show_grid:
        ax.grid(ls=':')
    if sci_notation:
        ax.ticklabel_format(style='sci', axis='y')

    if show:
        plt.show()

    return fig


def n_plots(x, y_list,
            y_label_list, colors_list,
            xlabel=r'Default xlabel',
            ylabel=r'Default ylabel',
            xmin=None, xmax=None,
            ymin=None, ymax=None,
            title=r'Default title', equal_axes=False, show_grid=True,
            fmt='-', lw=1.5, dpi=150,
            label_loc='best', sci_notation=False, show=True):
    serif = {'fontname': 'DejaVu Serif'}

    margin = 0.05
    if xmin is None:
        xmin = min(x) - margin * min(x)
    if xmax is None:
        xmax = max(x) + margin * max(x)
    if ymin is None:
        ymin = np.min(y_list)
        ymin = ymin - margin * ymin
    if ymax is None:
        ymax = np.max(y_list)
        ymax = ymax + margin * ymax

    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.minorticks_on()
    for i, y in enumerate(y_list):
        ax.plot(x, y, fmt, linewidth=lw,
                label=y_label_list[i],
                color=colors_list[i])
    ax.set_xlabel(xlabel, **serif)
    ax.set_ylabel(ylabel, **serif)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, **serif)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend(loc=label_loc)
    if equal_axes:
        ax.set_aspect('equal', adjustable='box')
    if show_grid:
        ax.grid(ls=':')
    if sci_notation:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if show:
        plt.show()

    return fig

def film_cooling(s,slot_position=(-0.01),debit=1.955,pourcentage=(7/100),v_inf,,xcanaux,Nu_corr_list,coolant_prandtl_list,coolant_reynolds_list):
    nfilm_list = [] #ncool dans formule
    coolant_stanton_list = [] #St
    v_film = [] #v cool 
    blowing_ratio_list = [] #F 
    position_film = [xcanaux[i] - slot_position for i in range (len(xcanaux))] #x
    for i in range (len(position_film)) : 
            v_film.append((pourcentage*debit)/((density(coolant_pressure_list[i],new_coolant_temp_list[i], fluid="Ethanol")*area_channel[i]))) #mdot = ro v aire
            coolant_stanton_list.append(Nu_corr_list[i]/(coolant_reynolds_list[i]*coolant_prandtl_list[i]))
            blowing_ratio_list.append((density(coolant_pressure_list[i],new_coolant_temp_list[i], fluid="Ethanol")*v_film[i])/(ro_infini*v_infini))
            k1 = ((coolant_stanton_list[i]*position_film[i])/(blowing_ratio_list[i]*s))-0.04
            k2 = coolant_reynolds_list[i]*coolant_prandtl_list[i]*(v_inf[i]/v_film[i])
            nfilm_list.append(exp(-k1*k2**1/8))
    return nfilm_list

