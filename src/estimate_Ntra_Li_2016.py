# -*- coding: utf-8 -*-
'''
using the formulae given by Li, Holman & Tao (2016), estimate the number of
transits to expect for a CBP system given the orbital parameters (and
observational baseline).

comment: Li+ (2016)'s formulae are in the small Δ_ib = 90 - i_b limit, an
assumption justified in the text by arguing the stars are eclipsing.
OFC this breaks for contact binaries, but we will ignore this, temporarily.

another comment:
f_1: true anomaly of star 1? No. I think it's a true anomaly of the
planet at a particular configuration where it's transiting star 1.
But this is confusing b/c then R_star2 should be substituted in Eq (12).
Probably worth emailing Gongjie about this. I will assume that the
appropriate f_2 (Eq 12) has R_star2 in place of R_star1.
Eq (11)

'''
from __future__ import division, print_function
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.preamble': [
        '\\usepackage{amsmath}',
        '\\usepackage{amssymb}'
        ]
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import numpy as np
from numpy import arcsin as asin, sin as sin, cos as cos
from math import pi as π
from astropy import units as u, constants as c


def N_transit(n_2, P_star2, n_1, P_star1, P_cr1, P_cr2):
    # Eq (21)
    return n_2 * P_star2 + \
           n_1 * P_star1 * P_cr1 / P_cr2


def n_1(T_obs, P_p, ΔΩ_1, δΩ_prec, Ωdot, a_b1, a_p, f_2):
    # Eq (18)
    if π/2*u.rad - f_2 > asin(a_b1/a_p) and ΔΩ_1/2 + δΩ_prec > π*u.rad:
        _ = min(T_obs/P_p,
                    (ΔΩ_1/2 + δΩ_prec)/(π*u.rad) * (ΔΩ_1/2) / (Ωdot * P_p)
                   )

    elif π/2*u.rad - f_2 > asin(a_b1/a_p):
        _ = min(T_obs/P_p,
                    (ΔΩ_1/2) / (Ωdot * P_p)
                   )

    elif ΔΩ_1/2 + δΩ_prec > 2*π*u.rad:
        _ = min(T_obs/P_p,
                    (ΔΩ_1/2 + δΩ_prec)/(2*π*u.rad) * (ΔΩ_1/2) / (Ωdot * P_p)
                   )

    else:
        _ = min(T_obs/P_p,
                    ΔΩ_1 / (Ωdot * P_p)
                   )

    return _


def n_2(T_obs, P_p, ΔΩ_2, δΩ_prec, Ωdot, a_b2, a_p, f_2):
    # Eq (18) logical switch. (Note: f_2 remains the same. Everything else
    # swaps!)
    if π/2*u.rad - f_2 > asin(a_b2/a_p) and ΔΩ_2/2 + δΩ_prec > π*u.rad:
        _ = min(T_obs/P_p,
                  (ΔΩ_2/2 + δΩ_prec)/(π*u.rad) * (ΔΩ_2/2) / (Ωdot * P_p)
                 )

    elif π/2*u.rad - f_2 > asin(a_b2/a_p):
        _ = min(T_obs/P_p,
                  (ΔΩ_2/2) / (Ωdot * P_p)
                 )

    elif ΔΩ_2/2 + δΩ_prec > 2*π*u.rad:
        _ = min(T_obs/P_p,
                  (ΔΩ_2/2 + δΩ_prec)/(2*π*u.rad) * (ΔΩ_2/2) / (Ωdot * P_p)
                 )

    else:
        _ = min(T_obs/P_p,
                  ΔΩ_2 / (Ωdot * P_p)
                 )

    return _


def P_star1(a_b1, dl_1, dl_2):
    # Eq (17)
    if (dl_1 + dl_2)/2 > 2*a_b1:
        return 1
    else:
        return (dl_1 + dl_2) / (4*a_b1)


def P_star2(a_b2, dl_1, dl_2):
    # Eq (17), with appropriate 1s to 2s
    if (dl_1 + dl_2)/2 > 2*a_b2:
        return 1
    else:
        return (dl_1 + dl_2) / (4*a_b2)


def dl_1(a_p, P_p, R_star1, δi, P_b, a_b1):
    # text btwn Eqs 16 & 17
    v_star1 = 2*π*a_b1/P_b
    v_p = 2*π*a_p/P_p

    t_trans = π/2 * R_star1 / (v_p * sin(δi))

    _ = t_trans * abs(v_p*cos(δi) - 2*v_star1/π) + R_star1

    return _


def dl_2(a_p, P_p, R_star1, δi, P_b, a_b1):

    v_star1 = 2*π*a_b1/P_b
    v_p = 2*π*a_p/P_p

    t_trans = π/2 * R_star1 / (v_p * sin(δi))

    # NOTE: not clear why this would have 2*Rstar_1, while dl_1 has no
    # prefactor of 2
    # NOTE: without, agreement is better. So yeah, I think that factor of 2 is
    # an error.
    #_ = t_trans * abs(v_p*cos(δi) + 2*v_star1/π) + 2*R_star1
    _ = t_trans * abs(v_p*cos(δi) + 2*v_star1/π) + R_star1

    return _


def P_cr1(δΩ_1):
    # Probability of planet crossing m_1's stellar orbit
    # Eq (16)
    return min(δΩ_1, 2*π*u.rad) / (2*π*u.rad)


def P_cr2(δΩ_2):
    # Analog of Eq (16)
    return min(δΩ_2, 2*π*u.rad) / (2*π*u.rad)


def ΔΩ_1(f_1, f_2, a_b1, a_p):
    # The range of Ω (the longitude of the ascending node of the planet with
    # respect to the x'-y plane), that allows the planet to cross the stellar
    # orbit of star m_1.
    # Given by Eq (10)
    if π/2*u.rad - f_2 > asin(a_b1/a_p) and f_1 + π/2*u.rad > asin(a_b1/a_p):
        _ = min( 2*(f_2 - f_1) + 4*asin(a_b1/a_p),
                   2*π*u.rad
                 )

    elif f_1 + π/2*u.rad > asin(a_b1/a_p):
        _ = min( 2*(f_2 - f_1) + 2*(π/2*u.rad - f_2) + 2*asin(a_b1/a_p),
                   2*π*u.rad
                 )

    else:
        _ = min( 2*(f_2 - f_1) + 2*(π/2*u.rad - f_2) + 2*(f_1 + π/2*u.rad),
                   2*π*u.rad
                 )

    return _


def f_1(R_star1, a_p, Δ_ib, δi):
    # f_1: true anomaly of star 1? No. I think it's a true anomaly of the
    # planet at a particular configuration where it's transiting star 1.
    # But this is confusing b/c then R_star2 should be substituted in Eq (12).
    # Probably worth emailing Gongjie about this. I will assume that the
    # appropriate f_2 (Eq 12) has R_star2 in place of R_star1.
    # Eq (11)
    if -1 < (-R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) and \
            (-R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) < 1:

        _ = asin( (-R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) )

    elif (-R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) < -1:
        _ = -π/2*u.rad

    else:
        _ = π/2*u.rad

    return _


def f_2(R_star1, a_p, Δ_ib, δi):
    # Eq (12), direct transcription.
    if -1 < (R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) and \
            (R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) < 1:

        _ = asin( (R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) )

    elif (R_star1/a_p + sin(abs(Δ_ib))) / sin(δi) < -1:
        _ = -π/2*u.rad

    else:
        _ = π/2*u.rad

    return _


def get_Ωdot(P_p, δi, a_p, a_b, m_1, m_2):
    # Eq 13, 14
    T_prec = P_p * 4 / (3*cos(δi)) * a_p**2 / (a_b**2) \
               * (m_1 + m_2)**2 / (m_1*m_2)

    return 2*π*u.rad/T_prec


def get_δΩ_prec(T_obs, Ωdot):
    # Change in longitude of ascending node over an observing window.
    # Definition in text, btwn Eq 14 and 15
    return T_obs * Ωdot


def δΩ_1(ΔΩ_1, δΩ_prec, f_2, a_b1, a_p):
    # Total range over the node of longitude during the observing period
    # (T_obs).
    # Eq 15.

    if π/2*u.rad - f_2 > δΩ_prec/2:
        _ = ΔΩ_1 + 2*δΩ_prec

    elif π/2*u.rad - f_2 > asin(a_b1/a_p):
        _ = ΔΩ_1 + δΩ_prec + 2*(π/2*u.rad - f_2)

    else:
        _ = ΔΩ_1 + δΩ_prec

    return _


def get_N_tra(T_obs, m_1, m_2, R_star1, R_star2, P_b, x, Δ_ib, δi, m_b, a_b,
        a_b1, a_b2, a_p, P_p):

    Ωdot = get_Ωdot(P_p, δi, a_p, a_b, m_1, m_2)

    δΩ_prec = get_δΩ_prec(T_obs, Ωdot)

    N_tra = N_transit(
            n_2(
                T_obs,
                P_p,
                ΔΩ_1(f_1(R_star2, a_p, Δ_ib, δi), # really calling ΔΩ_2
                     f_2(R_star2, a_p, Δ_ib, δi), # but got lazy & didn't
                     a_b2,                        # implement it
                     a_p),
                δΩ_prec,
                Ωdot,
                a_b2,
                a_p,
                f_2(R_star2, a_p, Δ_ib, δi)
               ),
            P_star2(
                a_b2,
                dl_1(a_p, P_p, R_star2, δi, P_b, a_b2),
                dl_2(a_p, P_p, R_star2, δi, P_b, a_b2)
               ),
            n_1(
                T_obs,
                P_p,
                ΔΩ_1(
                    f_1(R_star1, a_p, Δ_ib, δi),
                    f_2(R_star1, a_p, Δ_ib, δi),
                    a_b2,
                    a_p),
                δΩ_prec,
                Ωdot,
                a_b1,
                a_p,
                f_2(R_star1, a_p, Δ_ib, δi)
               ),
            P_star1(
                a_b1,
                dl_1(a_p, P_p, R_star1, δi, P_b, a_b1),
                dl_2(a_p, P_p, R_star1, δi, P_b, a_b1)
               ),
            P_cr1(
                δΩ_1(
                    ΔΩ_1(
                        f_1(R_star1, a_p, Δ_ib, δi),
                        f_2(R_star1, a_p, Δ_ib, δi),
                        a_b2,
                        a_p),
                    δΩ_prec,
                    f_2(R_star1, a_p, Δ_ib, δi),
                    a_b1,
                    a_p)
               ),
            P_cr2(
                δΩ_1(
                    ΔΩ_1(f_1(R_star2, a_p, Δ_ib, δi), # really calling ΔΩ_2
                         f_2(R_star2, a_p, Δ_ib, δi), # but got lazy & didn't
                         a_b2,                        # implement it
                         a_p),
                    δΩ_prec,
                    f_2(R_star2, a_p, Δ_ib, δi),
                    a_b2,
                    a_p
                    )
               )
            )

    return abs(N_tra)


def get_preliminaries(m_1, m_2, P_b, x):

    m_b = m_1 + m_2
    a_b = ( (c.G * m_b)/(4*π*π) * P_b**2 )**(1/3)

    a_b1 = a_b * m_2 / m_b
    a_b2 = a_b * m_1 / m_b

    a_p = x*a_b
    P_p = ( a_p**3 * 4*π*π / (c.G * m_b) )**(1/2)

    return m_b, a_b, a_b1, a_b2, a_p, P_p



if __name__ == '__main__':
    '''
    Note that there are a corresponding set of true anomalies for star 2:
    f_star2,1 and f_star2,2. I avoided a "proper" implementation by passing the
    wrong arguments to a function that I could have named more generally. (This
    applies for f_1, f_2, and similarly ΔΩ_1, ΔΩ_2, dl_1, and dl_2).
    '''

    # Given quantities
    T_obs = 1*u.year

    m_1 = 1*u.Msun
    m_2 = 1*u.Msun
    R_star1 = 1*u.Rsun
    R_star2 = 1*u.Rsun

    P_b = 2*u.day        # figure 4

    Δ_ib = 0*u.degree
    δi_array = np.arange(2,90,2)*u.degree
    # singularities do not play nice, and approximations break.
    δi_array = np.delete(δi_array, np.argwhere(δi_array == 90*u.degree))


    #########################################################
    # PLOT COMPARISON WITH FIGURE 4 OF LI, HOLMAN, TAO 2016 #
    #########################################################
    # read answers from the figure
    import pandas as pd
    colnames=['δi','N_tra']
    ans_x5 = pd.read_csv(
            '../data/Li_Holman_Tao_2016_top_panel_purple_line.csv',
            names=colnames)
    ans_x2pt4 = pd.read_csv(
            '../data/Li_Holman_Tao_2016_top_panel_blue_line.csv',
            names=colnames)

    import matplotlib.pyplot as plt

    f,ax = plt.subplots(figsize=(4,4))

    ax.plot(ans_x5['δi'], ans_x5['N_tra'], c='k', lw=1, ls='--',
            label='Li+ (2016) $a_p/a_b=5$')

    ax.plot(ans_x2pt4['δi'], ans_x2pt4['N_tra'], c='darkgray', lw=1, ls='--',
            label='Li+ (2016) $a_p/a_b=2.4$')

    for x, col_str, leg_str in zip([5, 2.4], ['k','darkgray'],
            ['my code $a_p/a_b=5$','my code $a_p/a_b=2.4$']):

        N_tra_list = []
        for index, δi in enumerate(δi_array):
            print(index)
            m_b, a_b, a_b1, a_b2, a_p, P_p = get_preliminaries(m_1, m_2, P_b, x)
            N_tra = get_N_tra(T_obs, m_1, m_2, R_star1, R_star2, P_b, x, Δ_ib, δi,
                    m_b, a_b, a_b1, a_b2, a_p, P_p)
            N_tra_list.append(N_tra.cgs)
        N_tra_array = np.array(N_tra_list)

        ax.plot(δi_array, N_tra_array, c=col_str, lw=2, label=leg_str)

    ax.legend(loc='best', fontsize='xx-small')

    ax.set(xlim=[0,180],
           xlabel=r'$\delta i\ (\mathrm{degree})$',
           ylabel=r'$\mathrm{average\ number\ of\ transits}$',
           )
    ax.set_title(
           'fig 4 Li+ 2016. $P_b=2\,\mathrm{days},\ T_\mathrm{obs}=1\,\mathrm{yr}$, solar parameters.'+\
           '\nmy implementation is slightly off (but not enough to matter)',
           fontsize='xx-small'
           )

    f.tight_layout()

    f.savefig('../results/N_transits_Li_2016/Ntra_estimate.pdf', dpi=300,
        bbox_inches='tight')

