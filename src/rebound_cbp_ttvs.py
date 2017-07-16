# -*- coding: utf-8 -*-
'''
numerics to work out the order of magnitude of geometric effect TTVs
'''
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import rebound
from math import pi as π
from astropy import units as u, constants as c


def _get_preliminaries(m_1, m_2, P_b, x):

    m_b = m_1 + m_2
    # semimajor axis of the relative orbit.
    # approximate that m_p << m_b (i.e. it's a test particle, and has no
    # influence on their orbits.
    a_b = ( (c.G * m_b)/(4*π*π) * P_b**2 )**(1/3)

    # semimajor axes of the barycentric orbits for mass 1 and mass 2,
    # respectively.
    a_b1 = a_b * m_2 / m_b
    a_b2 = a_b * m_1 / m_b

    # definition of x
    a_p = x*a_b
    # NOTE: this equation assumes a_p >> a_b. In this limit, the monopole term
    # matters much more than the quadrupole, and we are back to the Keplerian
    # two-body problem. This is wrong for detailed cases of a close-in
    # circumbinary planet.
    P_p_if_ap_much_bigger_than_ab = ( a_p**3 * 4*π*π / (c.G * m_b) )**(1/2)

    return m_b, a_b, a_b1, a_b2, a_p, P_p_if_ap_much_bigger_than_ab


def _get_transit_times_of_star(star_index, N_transits, Δt, m_0_in_Msun,
        m_1_in_Msun, m_p_in_Msun, a_b_in_AU, a_p_in_AU):
    '''
    Set up a REBOUND simulation that will give the transit times of a CBP in
    front of one star in a CBP system. The bisection scheme for measuring
    transit times means this routine must be called twice to get the transit
    times of each star, separately (but the integration is deterministic, so
    this is OK).

    args:
        star_index must be 0 or 1.
        N_transits (int) to run the integration
        Δt: timestep between which to check for transits
    '''
    sim = rebound.Simulation()
    sim.units = ('day', 'AU', 'Msun')

    # Set up a binary.
    sim.add(m=m_0_in_Msun)
    # Set up the secondary by calling `a` for the relative orbit, $a = a_0 +
    # a_1$, not the center of mass orbit of m_1 (a_1).
    sim.add(m=m_1_in_Msun, a=a_b_in_AU)

    # Move to the center of mass frame.
    sim.move_to_com()

    # Add a CBP on a circular orbit.  This sets up the planet relative to the
    # binary center of mass, which is what we want for the circumbinary case.
    # For the circumprimary case, we might want to explicitly pass the primary.
    # That's in the docs.
    sim.add(m=m_p_in_Msun, a=a_p_in_AU)

    # Keep track of the individual times of transit for the given star in
    # star_index.
    transittimes = np.zeros(N_transits)
    p = sim.particles
    ind = 0
    while ind < N_transits:
        # y pos of planet relative to each star
        y0_old = p[2].y - p[star_index].y
        y1_old = p[2].y - p[star_index].y
        t_old = sim.t

        # check for transits every Δt days. this is shorter than the
        # binary period, and much shorter than the planet period
        sim.integrate(sim.t+Δt)
        t_new = sim.t

        # check for transits of star 0
        # sign changed (y_old*y<0), planet in front of star (x>0)
        if y0_old*(p[2].y-p[star_index].y)<0 and p[2].x-p[star_index].x>0:

            # bisect until prec of 1e-6 days ~= 0.1 seconds reached
            while t_new-t_old>1e-6:
                if y0_old*(p[2].y-p[star_index].y)<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)

            transittimes[ind] = sim.t
            ind += 1
            sim.integrate(sim.t+Δt) # integrate to be past the transit

    return transittimes


def _get_conjunction_counter(N_transits, transittimes):
    '''
    one conjunction, multiple transits. count correct.
    '''
    conjunction_counter = []
    conj_ind, tra_ind = 0, 0
    while tra_ind < N_transits:

        if tra_ind == 0:
            conjunction_counter.append(conj_ind)
            conj_ind += 1
            tra_ind += 1
            continue

        if np.diff(transittimes)[tra_ind-1] < 0.5*np.percentile(np.diff(transittimes), 90):
            conjunction_counter.append(conj_ind-1)
            tra_ind += 1
            continue

        conjunction_counter.append(conj_ind)
        conj_ind += 1
        tra_ind += 1

    return conjunction_counter


def main(m_1):

    N_transits = 50
    m_0 = 1.0*u.Msun
    #m_1 = 1.0*u.Msun
    m_p = 1e-6*u.Msun
    P_b = 0.5*u.day
    x = 5
    Δt = 0.04 # days

    # Compute *initial* binary semimajor axis and planet semimajor axis. These
    # change during the numerical integration.
    m_b, a_b, a_b1, a_b2, a_p, P_p_Keplerian = \
            _get_preliminaries(m_0, m_1, P_b, x)
    print(P_b)
    print(P_p_Keplerian)

    # Get transit times, and compute linear least squares model as the linear
    # ephemeris. (Keeping in mind that you need to be careful with the
    # conjunction count).
    transittimes0 = _get_transit_times_of_star( 0, N_transits, Δt, m_0.value,
            m_1.value, m_p.value, a_b.to(u.AU).value, a_p.to(u.AU).value)

    conjcount0 = np.array(
            _get_conjunction_counter(N_transits, transittimes0))
    A = np.vstack([np.ones(N_transits), conjcount0]).T
    c0, m0 = np.linalg.lstsq(A, transittimes0)[0]

    tt0_reprod = _get_transit_times_of_star( 0, N_transits, Δt, m_0.value,
            m_1.value, m_p.value, a_b.to(u.AU).value, a_p.to(u.AU).value)
    np.testing.assert_array_equal(transittimes0, tt0_reprod)

    transittimes1 = _get_transit_times_of_star(1, N_transits, Δt, m_0.value,
            m_1.value, m_p.value, a_b.to(u.AU).value, a_p.to(u.AU).value)
    conjcount1 = np.array(
            _get_conjunction_counter(N_transits, transittimes1))
    A = np.vstack([np.ones(N_transits), conjcount1]).T
    c1, m1 = np.linalg.lstsq(A, transittimes1)[0]

    # Make plots
    f, axs = plt.subplots(nrows=2, ncols=1)

    axs[0].scatter(conjcount0,
                   (transittimes0-m0*conjcount0-c0)*24,
                   c='k', alpha=0.5)
    axs[1].scatter(conjcount1,
                   (transittimes1-m1*conjcount1-c1)*24,
                   c='k', alpha=0.5)

    axs[0].set_title('amplitude of TTV 0: {:.2f} hr'.format(
        max((transittimes0-m0*conjcount0-c0)*24) \
       -min((transittimes0-m0*conjcount0-c0)*24)) + \
       ', amplitude of TTV 1: {:.2f} hr'.format(
        max((transittimes1-m1*conjcount1-c1)*24) \
       -min((transittimes1-m1*conjcount1-c1)*24)  ),
       fontsize='small')

    txt = 'm_0: {:.1f}\nm_1: {:.1f}\nP_b: {:.1f}\nP_p_kep: {:.1f}\nx: {:.1f}'.format(
            m_0, m_1, P_b, P_p_Keplerian, x)

    t = axs[0].text(0.03, 0.97, txt, horizontalalignment='left',
                verticalalignment='top', fontsize='xx-small',
                transform=axs[0].transAxes)
    t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='black'))


    for ix, [ax, tt] in enumerate(zip(axs, [transittimes0, transittimes1])):
        ax.set_ylabel('TTV star {:d} [hrs]'.format(ix))
        ax.set_xlim([0, max(max(conjcount0), max(conjcount1))])

    axs[1].set_xlabel('conjunction count')

    #f.tight_layout()

    txt = 'm0_{:.1f}_m1_{:.1f}_Pb_{:.1f}_x_{:.1f}'.format(
            m_0, m_1, P_b, x)

    savedir = '../results/rebound_cbp_ttvs/'
    f.savefig(savedir+'ttv_'+txt+'.pdf', dpi=250, bbox_inches='tight')
    print('saved {:s}'.format(savedir+'ttv_'+txt+'.pdf'))


if __name__ == '__main__':

    m1s = np.arange(0.2, 1, 0.1)*u.Msun

    for m1 in m1s:
        main(m1)

