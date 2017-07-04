# -*- coding: utf-8 -*-
'''
Assuming a certain fraction of the close binaries in our sample have the median
Kepler circumbinary planet (e.g., say all of them do), how many detections
would we expect?

This calculation assumes:
* morph>0.7 stars (N_star ~= 1000) -- from earlier May run.
* The noise in these stars' lightcurves is just the dumb RMS post-whitening.
* Noise scales following Poisson statistics
* Planet R_p, a_p/a_b, and δi mutual inclination are median values from Kepler
    CBPs.
* The stars have masses that are the median of Kepler CBP star masses. The
    stars all have radii 1 Rsun and 0.5 Rsun.
* The planets are observed for their true Kepler baselines.
* Dilution factor of 50% (1.5x smaller signal b/c of third light)
* The Li, Holman, and Tao formulae hold to predict the number of CBPs that
    transit at least once. This means:
* The "independent transits" assumption for Li+ 2016's N_tra equation 21 is
    "good enough" for an estimate to a factor of ~2.
* "Transits" are singular events (each transit happens across both the
    primary and second), so that the number of transits predicted by Eq 21
    should be divided by 2.

* 100% of the stars in the sample have a planet with the above properties.

Then:

N_det = N_stars
        * fraction of stars that have a planet with above properties
        * prob(planet transits at least once)
        * prob(detection | SNR_pf, P_CBP, at least transits once)

where fraction of stars that have a planet with above properties = 1, for now.
'''
from __future__ import division, print_function

import numpy as np, pandas as pd
from astropy import units as u, constants as c
import pickle, os

import estimate_Ntra_Li_2016 as estimate_Ntra

def calc_SNR_pf(N_tra, R_p, R_star_eff, D, noise_postwhitening):
    '''
    Phase folded signal to noise ratio assuming the noise scales following
    Poisson statistics.
    '''
    return (N_tra**(1/2) * (R_p/R_star_eff)**2 / D / noise_postwhitening).cgs

# FIXME NOTE
# These are the results from the "realsearch" (with included whitening
# assumptions, number of stars, etc) done ~Apr/June 2017. We need to do an
# updated version of this with the paper's assumptions.
df = pd.read_csv('../results/real_search/irresult_sap_top1.csv')

noise_pws = np.array(list(map(float,(df['rms_biased']))))
P_bs = np.array(list(map(float,df['kebc_period'])))*u.day
T_obss = np.array(list(map(float,df['baseline'])))*u.day

# Assume each close binary gets a median Kepler circumbinary planet.
cbps = pd.read_csv('../data/all_kepler_cbps.csv', delimiter=';')
cbps = cbps.drop([0]) # row 0 is units, which I don't need.
R_p = np.nanmedian(list(map(float,np.array(cbps['R_p']))))*u.R_earth
δi =  np.nanmedian(list(map(float,np.array(cbps['\Delta I_p,in']))))*u.degree
x = np.nanmedian(list(map(float,np.array(cbps['a_p'])))) / \
        np.nanmedian(list(map(float,np.array(cbps['a_b']))))
# Assume median Kepler circumbinary planet host star masses and binary period.
m_1 = np.nanmedian(list(map(float,np.array(cbps['M_1']))))*u.Msun
m_2 = np.nanmedian(list(map(float,np.array(cbps['M_2']))))*u.Msun
# Assume unjustified effective host star radius.
#R_star_eff = 1.5*u.Rsun
R_star_eff = 1.5*u.Rsun
R_star1 = 1*u.Rsun
R_star2 = 0.5*u.Rsun
# Assume D = (total flux in aperture)/(flux in aperture from target star)
#          = 1.5
D = 1.5

##############################
##############################

np.random.seed(42)
N_montecarlo_trials = 20

workdir = '../results/N_transits_Li_2016/'

if os.path.exists(workdir+'Ndet_estimate_MonteCarlo.csv'):

    mc_df = pd.read_csv(workdir+'Ndet_estimate_MonteCarlo.csv')

else:

    for ix in range(N_montecarlo_trials):
        print('\n')
        print('\n')
        print(ix)
        print('\n')
        print('\n')

        # Draw the host binary inclinations from uniform in cos(i_b), because
        # we don't know them very well (although they should have more
        # preference towards being aligned).
        cos_ibs = np.random.uniform(low=0, high=1,size=len(df))
        Δ_ibs = (90*u.degree - np.rad2deg(np.arccos(cos_ibs))*u.degree)

        N_tra_list, P_tra_list, P_p_list = [], [], []

        for index, (noise_pw, P_b, T_obs, Δ_ib) in enumerate(
                zip(noise_pws,P_bs, T_obss, Δ_ibs)):

            print('{:d}/{:d}'.format(index, len(P_bs)))

            m_b, a_b, a_b1, a_b2, a_p, P_p = \
                    estimate_Ntra.get_preliminaries(m_1, m_2, P_b, x)

            P_tra = estimate_Ntra.get_P_tra(T_obs, m_1, m_2, R_star1, R_star2, P_b,
                    x, Δ_ib, δi, m_b, a_b, a_b1, a_b2, a_p, P_p)

            m_b, a_b, a_b1, a_b2, a_p, P_p = \
                    estimate_Ntra.get_preliminaries(m_1, m_2, P_b, x)

            N_tra = estimate_Ntra.get_N_tra(T_obs, m_1, m_2, R_star1, R_star2, P_b, x,
                    Δ_ib, δi, m_b, a_b, a_b1, a_b2, a_p, P_p)

            P_tra_list.append(P_tra.value)
            N_tra_list.append(N_tra.value)
            P_p_list.append(P_p.value)

        P_tra_array = np.array(P_tra_list)
        N_tra_array = np.array(N_tra_list)
        P_p_array = np.array(P_p_list)*u.day

        # The number of transits we've computed is over either star. We're treating
        # the host binary as a single object (kind of). So divide it by two (we're not
        # being any more precise than that anyway -- and Li 2016 mentions that the
        # independent transits approximation is crappy in this regime anyway).
        # FIXME: do this better.
        N_tra_array /= 2

        SNR_pfs = calc_SNR_pf(N_tra_array, R_p, R_star_eff, D, noise_pws)

        outdf = pd.DataFrame({'SNR_pf':SNR_pfs, 'P_p':P_p_array,
            'N_tra':N_tra_array, 'T_obs':T_obss, 'P_tra':P_tra_array,
            'trial':np.ones_like(N_tra_array)*ix})

        # B/c of these numerical issues, some of these have BS results.
        clearly_wonky = outdf[ outdf['N_tra'] > 2*outdf['T_obs']/outdf['P_p'] ]
        print('dropping {:d} clearly wonky cases:'.format(len(clearly_wonky)))
        print(clearly_wonky)
        outdf = outdf.drop(clearly_wonky.index)

        if ix == 0:
            mc_df = outdf
        else:
            mc_df = pd.concat([mc_df, outdf])


    mc_df.to_csv(workdir+'Ndet_estimate_MonteCarlo.csv', index=False)

##########################################
# Now estimate the number of detections. #
##########################################
compd = pickle.load(open(workdir+'completeness_SNR_vs_P_CBP_grid.pkl','rb'))

Pgrid =   compd['Pgrid']
SNRgrid = compd['SNRgrid']
results = compd['results']

from scipy.interpolate import interp2d

np.all(np.testing.assert_array_equal(Pgrid, np.logspace(-1,3,13)))
np.all(np.testing.assert_array_equal(SNRgrid, np.logspace(0,4,13)))

Pgrid_mids = np.logspace(-1+1/(3*2), 3+1/(3*2), 13)[:-1]
#SNRgrid_mids = np.logspace(0+1/(6*2), 3+1/(6*2), 19)[:-1]
SNRgrid_mids = np.logspace(0+1/(3*2), 4+1/(3*2), 13)[:-1]

f = interp2d(np.log10(Pgrid_mids), np.log10(SNRgrid_mids), results.T, fill_value=0)

prob_det = []

for P_p, SNR_pf, prob_tra in zip(mc_df['P_p'],mc_df['SNR_pf'],mc_df['P_tra']):

    prob_given_SNR_and_period_and_tra = f(np.log10(P_p), np.log10(SNR_pf))

    prob_det.append(prob_given_SNR_and_period_and_tra * prob_tra)

prob_det = np.ravel(prob_det)
mc_df['P_det'] = prob_det

N_montecarlo_trials = int(len(np.unique(mc_df['trial'])))

N_det_list = []

for ix in range(N_montecarlo_trials):

    N_det_list.append(np.sum(mc_df[mc_df['trial'] == ix]['P_det']))

N_det = np.average(N_det_list)
N_det_err = np.std(N_det_list)

mc_df.to_csv(workdir+'Ndet_estimate_df.csv', index=False)

print("you'd have detected {:.1f} +/- {:.1f} of these CBPs (Nstars: {:d})".
    format(N_det, N_det_err, int(len(mc_df)/N_montecarlo_trials) ) )
