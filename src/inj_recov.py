'''
Short period search:
    Injection/recovery of planet transits on the P<1 day Kepler Eclipsing
    Binary Catalog entries.
'''

import pickle, os, logging, pdb
from astropy.io import ascii
import astropy.units as u, astropy.constants as c
from datetime import datetime
from astrobase import astrokep, periodbase, lcmath
from astrobase.varbase import lcfit as lcf
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr
from numpy.polynomial.legendre import Legendre, legval
import batman
import pandas as pd

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.kepler' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )

#############
# INJECTION #
#############

def inject_transit_known_depth(lcd, δ):
    '''
    Same as inject_random_drawn_transit, but made easier by gridding over
    transit depth from δ = (2,1,1/2,1/4,1/8,1/16,1/32)% in the calling
    routine.

    Args:
        lcd (dict): the dictionary with everything, before any processing has
        been done. (Keys are quarter numbers).

        δ (float): transit depth in relative flux units

    Returns:
        lcd (dict): lcd, with injected fluxes keyed as 'sap_inj_flux' and
        'pdcsap_inj_flux' in each quarter. Separately, return (over the
        baseline of the entire set of observations) 'perfect_inj_fluxes' and
        'perfect_times'. These are in a separate dictionary.
    '''

    # Find max & min times (& grid appropriately). If there are many nans at 
    # the beginning/end of the data, this underestimates the total timespan.
    # This should be negligible.
    qnums = np.sort(list(lcd.keys()))
    maxtime = np.nanmax(lcd[max(qnums)]['time'])
    mintime = np.nanmin(lcd[min(qnums)]['time'])

    for ix, qnum in enumerate(qnums):
        if ix == 0:
            times = lcd[qnum]['time']
        else:
            times = np.append(times, lcd[qnum]['time'])

    # Injected model: select a random phase, so that the time of inferior
    # conjunction is not initially right over the main EB transit.
    # X q1 and q2 from q~U(0,1)
    # X Argument of periapsis ω (which, per Winn Ch of Exoplanets, is the same
    #   as curly-pi up to 180deg) ω~U(-180deg,180deg).
    # X P_CBP [days] from exp(lnP_CBP) ~ exp(U(ln P_EB*4, ln P_EB*40)), i.e. a
    #   log-uniform distribution, where P_EB is the value reported by the
    #   KEBWG.
    # X Reference transit time t_0 [days] from t_0 ~ U(0,P_CBP) (plus the
    #   minimum time of the timeseries).
    # X Radius ratio Rp/Rstar from lnRp/Rstar ~ U(ln0.04,ln0.2)
    # X Eccentricity e~Beta(0.867,3.03)
    # X Assume M1+M2 = 2Msun. R_star=1.5Rsun. Draw cosi~U(0,1) (so that 
    #   pdf(i)*di=sini*di). Compute the impact parameter, b_tra. If b_tra is 
    #   within [-1,1] continue, else redraw cosi, repeat (we only want 
    #   transiting planets). Then set the inclination as arccos of the
    #   first inclination drawn that transits.
    # Note that argument of periapsis = longitude of periastron, w=ω. The
    # parameters are inspired by Foreman-Mackey et al (2015), Table 1 (K2).
    # Calculate the models for 10 samples over each full ~29.4 minute exposure
    # time, and then average to get the data point over the full exposure time.

    params = batman.TransitParams()

    q1, q2 = np.random.uniform(low=0.0,high=1.0), \
             np.random.uniform(low=0.0,high=1.0)
    params.u = [q1,q2]
    params.limb_dark = "quadratic"
    w = np.random.uniform(
            low=np.rad2deg(-np.pi),
            high=np.rad2deg(np.pi))
    params.w = w

    period_eb = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    lowpref, highpref = 4, 40
    ln_period_cbp = np.random.uniform(
            low=np.log(lowpref*period_eb),
            high=np.log(highpref*period_eb))
    period_cbp = np.e**ln_period_cbp
    params.per = period_cbp

    t0 = np.random.uniform(
            low=0.,
            high=period_cbp)
    params.t0 = mintime + t0

    params.rp = δ**(1/2.)

    ecc = np.random.beta(0.867,3.03)
    params.ecc = ecc

    # Prefactor of 2^(1/3) comes from assuming M1+M2=2Msun. (Kepler's 3rd).
    a_in_AU = 2**(1/3.)*(period_cbp*u.day/u.yr)**(2/3.)
    Rstar_in_AU = (1.5*u.Rsun/u.au)
    a_by_Rstar = a_in_AU.cgs.value / Rstar_in_AU.cgs.value
    b_tra = 42. # initialize to whatever.
    while b_tra > 1:
        cosi = np.random.uniform(low=0.0,high=1.0)
        # Equation (7) of Winn's Ch in Exoplanets textbook.
        b_tra = a_by_Rstar*cosi * (1-ecc*ecc)/(1 + ecc*np.sin(w))

    params.a = a_by_Rstar
    inc = np.arccos(cosi)
    params.inc = np.rad2deg(inc)

    exp_time_minutes = 29.423259
    exp_time_days = exp_time_minutes / (24.*60)

    ss_factor = 10
    # Initialize model
    m_toinj = batman.TransitModel(params,
                            times,
                            supersample_factor = ss_factor,
                            exp_time = exp_time_days)

    # We also want a "perfect times" model, to assess whether the reasonable 
    # fraction of the above have non-zero quality flags, nans, etc. are
    # important. The independent time grid starts and ends at the same times 
    # (& with 30minute cadence). Nb. that batman deals with nans in times by 
    # returning zeros (which, in injection below, have no effect -- as you'd 
    # hope!)
    perftimes = np.arange(mintime, maxtime, exp_time_days)
    m_perftimes = batman.TransitModel(params,
                            perftimes,
                            supersample_factor = ss_factor,
                            exp_time = exp_time_days)

    # Calculate light curve
    fluxtoinj = m_toinj.light_curve(params)
    perfflux = m_perftimes.light_curve(params)

    # Append perfect times and injected fluxes.
    allq = {}
    allq['perfect_times'] = perftimes
    allq['perfect_inj_fluxes'] = perfflux
    allq['inj_model'] = {'params':params} # has things like the period.

    # Inject, by quarter
    kbegin, kend = 0, 0
    for ix, qnum in enumerate(qnums):

        qsapflux = lcd[qnum]['sap']['sap_flux']
        qpdcflux = lcd[qnum]['pdc']['pdcsap_flux']
        qtimes = lcd[qnum]['time']

        kend += len(qtimes)

        qfluxtoinj = fluxtoinj[kbegin:kend]

        qinjsapflux = qsapflux + (qfluxtoinj-1.)*np.nanmedian(qsapflux)
        qinjpdcflux = qpdcflux + (qfluxtoinj-1.)*np.nanmedian(qsapflux)

        lcd[qnum]['sap_inj_flux'] = qinjsapflux
        lcd[qnum]['pdcsap_inj_flux'] = qinjpdcflux

        kbegin += len(qtimes)

    kicid = str(lcd[qnum]['objectinfo']['keplerid'])
    LOGINFO('KICID {:s}: injected {:f} transit to both SAP and PDC fluxes.'.\
            format(kicid, δ))

    return lcd, allq


def inject_random_drawn_transit(lcd):
    '''
    Inject a transit with the following parameters into both the SAP and PDC
    fluxes of the passed light curve dictionary.

    # Injected model: select a random phase, so that the time of inferior
    # conjunction is not initially right over the main EB transit.
    # X q1 and q2 from q~U(0,1)
    # X Argument of periapsis ω (which, per Winn Ch of Exoplanets, is the same
    #   as curly-pi up to 180deg) ω~U(-180deg,180deg).
    # X P_CBP [days] from exp(lnP_CBP) ~ exp(U(ln P_EB*2.5, ln365)), i.e. a
    #   log-uniform distribution, where P_EB is the value reported by the
    #   KEBWG.
    # X Reference transit time t_0 [days] from t_0 ~ U(0,P_CBP) (plus the
    #   minimum time of the timeseries).
    # X Radius ratio Rp/Rstar from lnRp/Rstar ~ U(ln0.04,ln0.2)
    # X Eccentricity e~Beta(0.867,3.03)
    # X Assume M1+M2 = 2Msun. R_star=1.5Rsun. Draw cosi~U(0,1) (so that
    #   pdf(i)*di=sini*di). Compute the impact parameter, b_tra. If b_tra is
    #   within [0,1] continue, else redraw cosi, repeat (we only want
    #   transiting planets). Then set the inclination as arccos of the
    #   first inclination drawn that transits.

    Note that argument of periapsis = longitude of periastron, w=ω. The
    parameters are inspired by Foreman-Mackey et al (2015), Table 1 (K2).
    Calculate the models for 10 samples over each full ~29.4 minute exposure
    time, and then average to get the data point over the full exposure time.
    Note also that this means the EB signal subtraction has to be pretty good-
    the 50th percentile of transit depth is 0.1%.

    Note this duplicates most of the code from inject_fixed_transit

    Args:
        lcd (dict): the dictionary with everything, before any processing has
        been done. (Keys are quarter numbers).

    Returns:
        lcd (dict): lcd, with injected fluxes keyed as 'sap_inj_flux' and
        'pdcsap_inj_flux' in each quarter. Separately, return (over the
        baseline of the entire set of observations) 'perfect_inj_fluxes' and
        'perfect_times'. These are in a separate dictionary.
    '''

    # Find max & min times (& grid appropriately). If there are many nans at 
    # the beginning/end of the data, this underestimates the total timespan.
    # This should be negligible.
    qnums = np.sort(list(lcd.keys()))
    maxtime = np.nanmax(lcd[max(qnums)]['time'])
    mintime = np.nanmin(lcd[min(qnums)]['time'])

    for ix, qnum in enumerate(qnums):
        if ix == 0:
            times = lcd[qnum]['time']
        else:
            times = np.append(times, lcd[qnum]['time'])

    # Injected model: select a random phase, so that the time of inferior
    # conjunction is not initially right over the main EB transit.
    # X q1 and q2 from q~U(0,1)
    # X Argument of periapsis ω (which, per Winn Ch of Exoplanets, is the same
    #   as curly-pi up to 180deg) ω~U(-180deg,180deg).
    # X P_CBP [days] from exp(lnP_CBP) ~ exp(U(ln P_EB*4, ln P_EB*40)), i.e. a
    #   log-uniform distribution, where P_EB is the value reported by the
    #   KEBWG.
    # X Reference transit time t_0 [days] from t_0 ~ U(0,P_CBP) (plus the
    #   minimum time of the timeseries).
    # X Radius ratio Rp/Rstar from lnRp/Rstar ~ U(ln0.04,ln0.2)
    # X Eccentricity e~Beta(0.867,3.03)
    # X Assume M1+M2 = 2Msun. R_star=1.5Rsun. Draw cosi~U(0,1) (so that 
    #   pdf(i)*di=sini*di). Compute the impact parameter, b_tra. If b_tra is 
    #   within [-1,1] continue, else redraw cosi, repeat (we only want 
    #   transiting planets). Then set the inclination as arccos of the
    #   first inclination drawn that transits.
    # Note that argument of periapsis = longitude of periastron, w=ω. The
    # parameters are inspired by Foreman-Mackey et al (2015), Table 1 (K2).
    # Calculate the models for 10 samples over each full ~29.4 minute exposure
    # time, and then average to get the data point over the full exposure time.

    params = batman.TransitParams()

    q1, q2 = np.random.uniform(low=0.0,high=1.0), \
             np.random.uniform(low=0.0,high=1.0)
    params.u = [q1,q2]
    params.limb_dark = "quadratic"
    w = np.random.uniform(
            low=np.rad2deg(-np.pi),
            high=np.rad2deg(np.pi))
    params.w = w

    period_eb = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    lowpref, highpref = 4, 40
    ln_period_cbp = np.random.uniform(
            low=np.log(lowpref*period_eb),
            high=np.log(highpref*period_eb))
    period_cbp = np.e**ln_period_cbp
    params.per = period_cbp

    t0 = np.random.uniform(
            low=0.,
            high=period_cbp)
    params.t0 = mintime + t0

    ln_Rp = np.random.uniform(
            low=np.log(0.04),
            high=np.log(0.2))
    params.rp = np.e**ln_Rp

    ecc = np.random.beta(0.867,3.03)
    params.ecc = ecc

    # Prefactor of 2^(1/3) comes from assuming M1+M2=2Msun. (Kepler's 3rd).
    a_in_AU = 2**(1/3.)*(period_cbp*u.day/u.yr)**(2/3.)
    Rstar_in_AU = (1.5*u.Rsun/u.au)
    a_by_Rstar = a_in_AU.cgs.value / Rstar_in_AU.cgs.value
    b_tra = 42. # initialize to whatever.
    while b_tra > 1:
        cosi = np.random.uniform(low=0.0,high=1.0)
        # Equation (7) of Winn's Ch in Exoplanets textbook.
        b_tra = a_by_Rstar*cosi * (1-ecc*ecc)/(1 + ecc*np.sin(w))

    params.a = a_by_Rstar
    inc = np.arccos(cosi)
    params.inc = inc

    exp_time_minutes = 29.423259
    exp_time_days = exp_time_minutes / (24.*60)

    ss_factor = 10
    # Initialize model
    m_toinj = batman.TransitModel(params,
                            times,
                            supersample_factor = ss_factor,
                            exp_time = exp_time_days)

    # We also want a "perfect times" model, to assess whether the reasonable 
    # fraction of the above have non-zero quality flags, nans, etc. are
    # important. The independent time grid starts and ends at the same times 
    # (& with 30minute cadence). Nb. that batman deals with nans in times by 
    # returning zeros (which, in injection below, have no effect -- as you'd 
    # hope!)
    perftimes = np.arange(mintime, maxtime, exp_time_days)
    m_perftimes = batman.TransitModel(params,
                            perftimes,
                            supersample_factor = ss_factor,
                            exp_time = exp_time_days)

    # Calculate light curve
    fluxtoinj = m_toinj.light_curve(params)
    perfflux = m_perftimes.light_curve(params)

    # Append perfect times and injected fluxes.
    allq = {}
    allq['perfect_times'] = perftimes
    allq['perfect_inj_fluxes'] = perfflux
    allq['inj_model'] = {'params':params} # has things like the period.

    # Inject, by quarter
    kbegin, kend = 0, 0
    for ix, qnum in enumerate(qnums):

        qsapflux = lcd[qnum]['sap']['sap_flux']
        qpdcflux = lcd[qnum]['pdc']['pdcsap_flux']
        qtimes = lcd[qnum]['time']

        kend += len(qtimes)

        qfluxtoinj = fluxtoinj[kbegin:kend]

        qinjsapflux = qsapflux + (qfluxtoinj-1.)*np.nanmedian(qsapflux)
        qinjpdcflux = qpdcflux + (qfluxtoinj-1.)*np.nanmedian(qsapflux)

        lcd[qnum]['sap_inj_flux'] = qinjsapflux
        lcd[qnum]['pdcsap_inj_flux'] = qinjpdcflux

        kbegin += len(qtimes)

    kicid = str(lcd[qnum]['objectinfo']['keplerid'])
    LOGINFO('KICID {:s}: injected transit to both SAP and PDC fluxes.'.\
            format(kicid))

    return lcd, allq


####################################
# KEPLER LCS PARSING AND SELECTION #
####################################

def retrieve_random_lc():
    '''
    Retrieve the light curves for all quarters of a randomly selected entry
    of the KEBC.
    Returns a dictionary with keys of quarter number.
    '''

    kebc = get_kepler_ebs_info()
    kebc = kebc[kebc['morph']>0.6]
    kebc_kic_ids = np.array(kebc['KIC'])
    ind = int(np.random.randint(0, len(kebc['KIC']), size=1))

    rd = get_all_quarters_lc_data(kebc_kic_ids[ind])
    if len(rd) < 1:
        lcflag = True
        LOGERROR('Error getting LC data. KICID-{:s}'.format(
            str(kebc_kic_ids[ind])))
    else:
        lcflag = False

    return rd, lcflag


def get_kepler_ebs_info():
    '''
    Read in the nicely formatted astropy table of the Kepler Eclipsing
    Binary Catalog information. (Includes morphology parameter, kepler
    magnitudes, whether short cadence data exists, and periods).
    See Prsa et al. (2011) and the subsequent papers in the KEBC series.
    '''

    #get Kepler EB data (e.g., the period)
    keb_path = '../data/kepler_eb_catalog_v3.csv'
    cols = 'KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC'
    cols = tuple(cols.split(','))
    tab = ascii.read(keb_path,comment='#')
    currentcols = tab.colnames
    for ix, col in enumerate(cols):
        tab.rename_column(currentcols[ix], col)

    tab.remove_column('col12') # now table has correct features

    return tab


def get_all_quarters_lc_data(kicid):
    '''
    Given a KIC ID, get a dictionary with all the FITS data (i.e. the
    lightcurves, for each quarter) downloaded from MAST for that KIC ID.

    Args:
        kicid (np.int64): KIC ID of system to try and find LC data for

    Returns:
        lcdat (dict): dictionary with keys of quarter number. The value of the
        keys are the lightcurve dictionaries returned by
        `astrokep.read_kepler_fitslc`.

    '''
    assert type(kicid) == np.int64, 'kicid was {:s}'.format(str(type(kicid)))

    lcdir = '../data/morph_gt0.6_kepler_MAST/'

    fs = [lcdir+f for f in os.listdir(lcdir) if f.endswith('.fits') and
            str(kicid) in f]

    rd = {}
    for fits_path in fs:
        lcd = astrokep.read_kepler_fitslc(fits_path)
        quarter_number = np.unique(lcd['quarter'])
        assert len(quarter_number)==1, 'Expect each fits file to correspond '+\
            ' to a given quarter'
        lcd['kebwg_info'] = get_kebwg_info(lcd['objectinfo']['keplerid'])
        rd[int(quarter_number)] = lcd

    return rd


def get_kebwg_info(kicid):
    '''
    Given a KIC ID, get the EB period reported by the Kepler Eclipsing Binary
    Working Group in v3 of their catalog.
    '''
    keb_path = '../data/kepler_eb_catalog_v3.csv'
    #fast read
    f = open('../data/kepler_eb_catalog_v3.csv', 'r')
    ls = f.readlines()
    thisentry = [l for l in ls if l.startswith(str(kicid))]
    assert len(thisentry) == 1

    cols = 'KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC'
    cols = cols.split(',')
    thesevals = thisentry.pop().split(',')[:-1]

    kebwg_info = dict(zip(cols, thesevals))

    return kebwg_info


#####################
# UTILITY FUNCTIONS #
#####################

def _legendre_dtr(times, fluxs, errs, legendredeg=10):

    p = Legendre.fit(times, fluxs, legendredeg)
    fitfluxs = p(times)

    fitchisq = npsum(
        ((fitfluxs - fluxs)*(fitfluxs - fluxs)) / (errs*errs)
    )

    nparams = legendredeg + 1
    fitredchisq = fitchisq/(len(fluxs) - nparams - 1)

    LOGINFO(
        'legendre detrend applied. chisq = %.5f, reduced chisq = %.5f' %
        (fitchisq, fitredchisq)
    )

    return fitfluxs, fitchisq, fitredchisq


def _polynomial_dtr(times, fluxs, errs, polydeg=2):
    '''
    Following F. Dai's implementation, Least-squares fit a 2nd-order
    polynomial of the form

        p(x) = a_0 + a_1 * x + a_2 * x^2 + .. + a_n * x^n.

    polydeg in the args is n.
    '''

    n = polydeg
    a = np.polyfit(times, fluxs, n)

    fitfluxs = np.zeros_like(times)
    for k in range(n+1):
        fitflux += times**(n-k)*a[k]

    fitchisq = npsum(
        ((fitfluxs - fluxs)*(fitfluxs - fluxs)) / (errs*errs)
    )

    nparams = n + 1
    fitredchisq = fitchisq/(len(fluxs) - nparams - 1)

    LOGINFO(
        'polynomial detrend applied. chisq = %.5f, reduced chisq = %.5f' %
        (fitchisq, fitredchisq)
    )

    return fitfluxs, fitchisq, fitredchisq


def detrend_allquarters(lcd, σ_clip=None, legendredeg=10, inj=False):
    '''
    Wrapper for detrend_lightcurve that detrends all the quarters of Kepler
    data passed in `lcd`, a dictionary of dictionaries, keyed by quarter
    numbers.
    "Detrend" means: select finite, sigma-clipped, not-near-gaps-in-timeseries
    points, and fit each resulting time group with a variable-order finite
    Legendre series.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = detrend_lightcurve(lcd[k], k, σ_clip=σ_clip,
                legendredeg=legendredeg, inj=inj)
        LOGINFO('KIC ID %s, detrended quarter %s.'
            % (str(lcd[k]['objectinfo']['keplerid']), str(k)))

    return rd


def detrend_lightcurve(lcd, qnum, detrend='legendre', legendredeg=10,
        polydeg=2, σ_clip=None, inj=False):
    '''
    You are given a dictionary, for a *single quarter* of kepler data, returned
    by `astrokep.read_kepler_fitslc`. It has keys like

    ['pdcsap_flux', 'sap_bkg_err', 'channel', 'psf_centr2_err', 'sap_flux_err',
    'psf_centr2', 'pdcsap_flux_err', 'mom_centr2', 'mom_centr2_err', 'season',
    'sap_flux', 'module', 'columns', 'time', 'mom_centr1', 'lcinfo', 'varinfo',
    'obsmode', 'cadenceno', 'output', 'mom_centr1_err', 'objectinfo',
    'psf_centr1_err', 'skygroup', 'psf_centr1', 'datarelease', 'quarter',
    'objectid', 'sap_bkg', 'sap_quality', 'timecorr'].

    This module returns this same dictionary, appending detrended magnitude
    values, for both the SAP and PDC fluxes.

    Args:
        lcd (dict): the dictionary returned by astrokep.read_kepler_fitslc (as
        described above).

        detrend (str): method by which to detrend the LC. 'legendre' and
        'polynomial' are accepted.

        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries

        inj (bool): whether or not the passed LC has injected transits. This
        changes which fluxes are detrended.

    Returns:
        lcd (dict): lcd, with the detrended times, magnitudes, and fluxes in a
        sub-dictionary, accessible as lcd['dtr'], which gives the
        dictionary:

            dtr = {
            'sap':{'times':,
                    'mags':,
                    'fitfluxs_legendre':,
                    'errs':
                   },
            'pdc':{'times':,
                    'mags':,
                    'fitfluxs_poly':,
                    'errs':
                   }
            },

        where the particular subclass of fitfluxs is specified by the detrend
        kwarg.
    '''

    assert detrend == 'legendre' or detrend == 'polynomial'

    # Get finite, good-quality times, mags, and errs for both SAP and PDC.
    # Take data with non-zero saq_quality flags. Fraquelli & Thompson (2012),
    # or perhaps newer papers, give the list of exclusions (following Armstrong
    # et al. 2014).

    nbefore = lcd['time'].size
    times = lcd['time'][lcd['sap_quality'] == 0]

    if inj:
        sapfluxs = lcd['sap_inj_flux'][lcd['sap_quality'] == 0]
        pdcfluxs = lcd['pdcsap_inj_flux'][lcd['sap_quality'] == 0]
    else:
        sapfluxs = lcd['sap']['sap_flux'][lcd['sap_quality'] == 0]
        pdcfluxs = lcd['pdc']['pdcsap_flux'][lcd['sap_quality'] == 0]

    saperrs = lcd['sap']['sap_flux_err'][lcd['sap_quality'] == 0]
    find = npisfinite(times) & npisfinite(sapfluxs) & npisfinite(saperrs)
    fsaptimes, fsapfluxs, fsaperrs = times[find], sapfluxs[find], saperrs[find]
    ssaptimes, ssapfluxs, ssaperrs = lcmath.sigclip_magseries(
            fsaptimes, fsapfluxs, fsaperrs,
            magsarefluxes=True, sigclip=σ_clip)
    nqflag = ssaptimes.size

    # Drop intra-quarter and interquarter gaps in the SAP lightcurves. These
    # are the same limits set by Armstrong et al (2014): split each quarter's
    # timegroups by whether points are within 0.5 day limits. Then drop points
    # within 0.5 days of any boundary.
    # Finally, since the interquarter burn-in time is more like 1 day, drop a
    # further 0.5 days from the edges of each quarter.
    # A nicer way to implement this would be with numpy masks, but this
    # approach just constructs the full arrays for any given quarter.
    mingap = 0.5 # days
    ngroups, groups = lcmath.find_lc_timegroups(ssaptimes, mingap=mingap)
    tmptimes, tmpfluxs, tmperrs = [], [], []
    for group in groups:
        tgtimes = ssaptimes[group]
        tgfluxs = ssapfluxs[group]
        tgerrs  = ssaperrs[group]
        sel = (tgtimes > npmin(tgtimes)+mingap) & \
                 (tgtimes < npmax(tgtimes)-mingap)
        tmptimes.append(tgtimes[sel])
        tmpfluxs.append(tgfluxs[sel])
        tmperrs.append(tgerrs[sel])
    ssaptimes,ssapfluxs,ssaperrs = nparr([]),nparr([]),nparr([])
    # N.b.: works fine with empty arrays.
    for ix, _ in enumerate(tmptimes):
        ssaptimes = np.append(ssaptimes, tmptimes[ix])
        ssapfluxs = np.append(ssapfluxs, tmpfluxs[ix])
        ssaperrs =  np.append(ssaperrs, tmperrs[ix])
        # Extra inter-quarter burn-in of 0.5 days.
        ssapfluxs = ssapfluxs[(ssaptimes>(npmin(ssaptimes)+mingap)) & \
                              (ssaptimes<(npmax(ssaptimes)-mingap))]
        ssaperrs  = ssaperrs[(ssaptimes>(npmin(ssaptimes)+mingap)) & \
                              (ssaptimes<(npmax(ssaptimes)-mingap))]
        ssaptimes = ssaptimes[(ssaptimes>(npmin(ssaptimes)+mingap)) & \
                              (ssaptimes<(npmax(ssaptimes)-mingap))]

    nafter = ssaptimes.size

    LOGINFO('CLIPPING (SAP), qnum: {:d}'.format(qnum)+\
            '\nndet before qflag & sigclip: {:d} ({:.3g}),'.format(
                nbefore, 1.)+\
            '\nndet after qflag & finite & sigclip: {:d} ({:.3g})'.format(
                nqflag, nqflag/float(nbefore))+\
            '\nndet after dropping pts near gaps: {:d} ({:.3g})'.format(
                nafter, nafter/float(nbefore)))

    # Ensure PDC data is finite and sigclipped.
    pdcerrs = lcd['pdc']['pdcsap_flux_err'][lcd['sap_quality'] == 0]
    find = npisfinite(times) & npisfinite(pdcfluxs) & npisfinite(pdcerrs)
    fpdctimes, fpdcfluxs, fpdcerrs = times[find], pdcfluxs[find], pdcerrs[find]
    spdctimes, spdcfluxs, spdcerrs = lcmath.sigclip_magseries(
            fpdctimes, fpdcfluxs, fpdcerrs,
            magsarefluxes=True, sigclip=σ_clip)

    # Drop intra-quarter and interquarter gaps in the PDC lightcurves.
    ngroups, groups = lcmath.find_lc_timegroups(spdctimes, mingap=mingap)
    tmptimes, tmpfluxs, tmperrs = [], [], []
    for group in groups:
        tgtimes = spdctimes[group]
        tgfluxs = spdcfluxs[group]
        tgerrs  = spdcerrs[group]
        sel = (tgtimes > npmin(tgtimes)+mingap) & \
                 (tgtimes < npmax(tgtimes)-mingap)
        tmptimes.append(tgtimes[sel])
        tmpfluxs.append(tgfluxs[sel])
        tmperrs.append(tgerrs[sel])
    spdctimes,spdcfluxs,spdcerrs = nparr([]),nparr([]),nparr([])
    for ix, _ in enumerate(tmptimes):
        spdctimes = np.append(spdctimes, tmptimes[ix])
        spdcfluxs = np.append(spdcfluxs, tmpfluxs[ix])
        spdcerrs =  np.append(spdcerrs, tmperrs[ix])
        # Extra inter-quarter burn-in of 0.5 days.
        spdcfluxs = spdcfluxs[(spdctimes>(npmin(spdctimes)+mingap)) & \
                              (spdctimes<(npmax(spdctimes)-mingap))]
        spdcerrs  = spdcerrs[(spdctimes>(npmin(spdctimes)+mingap)) & \
                              (spdctimes<(npmax(spdctimes)-mingap))]
        spdctimes = spdctimes[(spdctimes>(npmin(spdctimes)+mingap)) & \
                              (spdctimes<(npmax(spdctimes)-mingap))]

    nafter = spdctimes.size

    LOGINFO('CLIPPING (PDC), qnum: {:d}'.format(qnum)+\
            '\nndet before qflag & sigclip: {:d} ({:.3g}),'.format(
                nbefore, 1.)+\
            '\nndet after qflag & finite & sigclip: {:d} ({:.3g})'.format(
                nqflag, nqflag/float(nbefore))+\
            '\nndet after dropping pts near gaps: {:d} ({:.3g})'.format(
                nafter, nafter/float(nbefore)))

    # DETREND: fit a legendre series or polynomial, save it to the output
    # dictionary.
    tfe = {'sap':(ssaptimes, ssapfluxs, ssaperrs),
           'pdc':(spdctimes, spdcfluxs, spdcerrs)}
    dtr = {}

    for k in tfe.keys():

        times,fluxs,errs = tfe[k]

        if detrend == 'legendre':
            fitfluxs, fitchisq, fitredchisq = _legendre_dtr(times,fluxs,errs,
                    legendredeg=legendredeg)

        elif detrend == 'polynomial':
            fitfluxs, fitchisq, fitredchisq = _polynomial_dtr(times,fluxs,errs,
                    polydeg=polydeg)

        dtr[k] = {'times':times,
                  'fluxs':fluxs,
                  'fitfluxs_'+detrend:fitfluxs,
                  'errs':errs
                 }

    lcd['dtr'] = dtr

    return lcd



def redetrend_allquarters(lcd, σ_clip=None, legendredeg=10):
    '''
    Wrapper for redetrend_lightcurve, as with detrend_allquarters.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = redetrend_lightcurve(lcd[k], σ_clip=σ_clip,
                    legendredeg=legendredeg)
        LOGINFO('KIC ID %s, detrended quarter %s.'
            % (str(lcd[k]['objectinfo']['keplerid']), str(k)))

    return rd



def redetrend_lightcurve(lcd,
        detrend='legendre', legendredeg=10, σ_clip=None):

    '''
    Once you have whitened fluxes, re-detrend (and re sigma-clip). (Note that
    this deals implicitly with injected magnitudes if they're what you started
    with).

    Args:
        lcd (dict): the dictionary with everything, after whitening out the
        main eclipsing binary signal.
        detrend (str): method by which to detrend the LC. 'legendre' is the
        only thing currently implemented.
        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries

    Returns:
        lcd (dict): lcd, with the redetrended times, magnitudes, and fluxes in a
        sub-dictionary, accessible as lcd['redtr'], which gives the
        dictionary:

            redtr = {
            'sap':{'times':,
                    'fluxs':,
                    'fitfluxs_legendre':,
                    'errs':
                   },
            'pdc':{'times':,
                    'fluxs':,
                    'fitfluxs_poly':,
                    'errs':
                   }
            },

        where the particular subclass of fitfluxs is specified by the detrend
        kwarg.
    '''

    assert detrend == 'legendre'

    lcd['redtr'] = {}
    for ap in ['sap', 'pdc']:

        times = lcd['white'][ap]['whiteseries']['times']
        nbefore = times.size
        # times, mags, and errs are already finite.
        fluxes = lcd['white'][ap]['whiteseries']['fluxes']
        errs = lcd['white'][ap]['whiteseries']['errs']

        #FIXME: could choose to cut out data within 0.5 days of time gaps here, per
        #Armstrong+ (2014)'s approach
        #FIXME: OR you could choose to fit the legendre series/polynomials TO EACH
        #TIME GROUP. Then(automatically) "fine-tune" the order (degree) of the fit
        #based on the duration of the  timegroup. 
        stimes, sfluxes, serrs = lcmath.sigclip_magseries(
                times, fluxes, errs,
                magsarefluxes=True, sigclip=σ_clip)

        nafter = stimes.size
        LOGINFO('for refilter & sigclip ({:s}), '.format(ap)+\
                'ndet before: {:d}, ndet after: {:d}'.format(nbefore, nafter))

        #DETREND: fit a legendre series or polynomial, save it to the output
        #dictionary.
        if detrend == 'legendre':
            fitfluxs, fitchisq, fitredchisq = _legendre_dtr(
                    stimes,sfluxes,serrs,
                    legendredeg=legendredeg)

        redtr = {'times':stimes,
                  'fluxs_pre_redtr':sfluxes,
                  'fitfluxs_'+detrend:fitfluxs,
                  'fluxs_redtr':sfluxes-fitfluxs,
                  'errs':serrs
                 }

        lcd['redtr'][ap] = redtr

    return lcd



def normalize_lightcurve(lcd, qnum, dt='dtr'):
    '''
    Once detrended fits are computed, this function computes the residuals, and
    also expresses them in normalized flux units, saving the keys to
    `lcd['dtr']['sap']['*_rsdl']`.

    Args:
        dt = 'dtr' or 'redtr', if you've just detrended, or you're
        redtrending. (`dt` for "detrending type").
    '''

    for ap in ['sap','pdc']:

        dtrtype = [k for k in list(lcd[dt][ap].keys()) if
            k.startswith('fitfluxs_')]
        assert len(dtrtype) == 1, 'Single type of fit assumed.'
        dtrtype = dtrtype.pop()

        flux = lcd[dt][ap]['fluxs']
        flux_norm = flux / np.median(flux)
        fitflux = lcd[dt][ap][dtrtype]
        fitflux_norm = fitflux / np.median(flux)

        # NOTE: there is some abiguity about whether to divide by the fit
        # (preserving relative variability) or to subtract.
        flux_dtr_norm = flux_norm / fitflux_norm

        errs = lcd[dt][ap]['errs']
        errs_dtr_norm =  errs / np.median(flux)

        lcd[dt][ap]['fluxs_dtr_norm'] = flux_dtr_norm
        lcd[dt][ap]['errs_dtr_norm'] = errs_dtr_norm

        LOGINFO('KIC ID %s, normalized quarter %s. (%s)'
            % (str(lcd['objectinfo']['keplerid']), str(qnum), ap))


    return lcd



def normalize_allquarters(lcd, dt='dtr'):
    '''
    Wrapper to normalize_lightcurve, to run for all quarters.
    Saves to keys `lcd[qnum]['dtr']['sap']['*_rsdl']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = normalize_lightcurve(lcd[k], k, dt=dt)

    return rd





def run_fineperiodogram_allquarters(lcd, iter_n=0):
    '''
    Wrapper to run_periodogram, to run for all quarters after selected best
    period has been identified.
    Saves periodogram results to keys `lcd[qnum]['fineper']['sap']['*']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = run_fineperiodogram(lcd[k], k, 'pdm', iter_n=iter_n)

    return rd


def run_periodograms_allquarters(lcd, iter_n=0):
    '''
    Wrapper to run_periodogram, to run for all quarters.
    Saves periodogram results to keys `lcd[qnum]['per']['sap']['*']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = run_periodogram(lcd[k], k, 'pdm', iter_n=iter_n)

    return rd


def run_fineperiodogram(dat, qnum, pertype='pdm', iter_n=0):
    '''
    See run_periodogram docstring. It's that, but for narrowing down the period
    of an EB once select_eb_period has been run.
    '''
    assert pertype=='pdm'

    dat['fineper'] = {}

    keyd = {
            0: {'dtr': ['times', 'fluxs_dtr_norm', 'errs_dtr_norm']},
            1: {'redtr': ['times', 'fluxs_redtr', 'errs']}
           }
    mk = list(keyd[iter_n].keys()).pop()

    for ap in ['sap', 'pdc']:

        times = dat[mk][ap][keyd[iter_n][mk][0]]
        fluxs = dat[mk][ap][keyd[iter_n][mk][1]]
        errs =  dat[mk][ap][keyd[iter_n][mk][2]]

        if len(times) < 50 or len(fluxs) < 50:
            LOGERROR('Got quarter with too few points. Continuing.')
            continue

        # Range of interesting periods (morph>0.7), now that the selected
        # period has been chosen: +/- 1% above/below it. If difference between
        # KEBC period and selperiod is greater than 1%, use whatever the
        # relative difference is as the bound.
        kebc_period = float(dat['kebwg_info']['period'])
        selperiod = dat['per'][ap]['selperiod']

        rdiff = max(0.01, abs(kebc_period - selperiod)/abs(kebc_period))

        smallest_p = selperiod - rdiff*selperiod
        biggest_p = selperiod + rdiff*selperiod

        if pertype == 'pdm':
            # periodepsilon: required distance [days] to ID different peaks
            pgd = periodbase.stellingwerf_pdm(times,fluxs,errs,
                autofreq=False,
                startp=smallest_p,
                endp=biggest_p,
                normalize=False,
                stepsize=2.0e-6,
                phasebinsize=0.05,
                mindetperbin=9,
                nbestpeaks=5,
                periodepsilon=rdiff*selperiod*0.1,
                sigclip=None, # no sigma clipping
                nworkers=None)

        if isinstance(pgd, dict):
            LOGINFO('KIC ID %s computed fine periodogram (%s) quarter %s. (%s)'
                % (str(dat['objectinfo']['keplerid']), pertype, str(qnum), ap))
        else:
            LOGERROR('Error in KICID %s fine periodogram %s, quarter %s (%s)'
                % (str(dat['objectinfo']['keplerid']), pertype, str(qnum), ap))

        dat['fineper'][ap] = pgd

    return dat



def run_periodogram(dat, qnum, pertype='pdm', iter_n=0):
    '''
    Given normalized, detrended fluxes, this function computes periodograms of
    choice. Most importantly in the context of being able to subsequently
    whiten, it also finds the maximum of the periodogram, and calls that the
    "period", presumably of the EB (which has already been identified by the
    KEBWG).

    This primarily wraps around astrobase.periodbase's routines.

    Args:
        dat (dict): the dictionary returned by astrokep.read_kepler_fitslc (as
        described above), post-normalization & detrending.
        pertype (str): periodogram type. 'pdm', 'bls', and 'lsp' are accepted
        for Stellingwerf phase dispersion minimization, box least squares, and
        a generalized Lomb-Scargle periodogram respectively. (See astrobase
        for details).
        iter_n (int): same as for whiten_allquarters.

    Returns:
        dat (dict): dat, with the periodogram information in a sub-dictionary,
        accessible as dat['per'], which gives the
        dictionary:

            per =
                {'nbestlspvals':,
                 'lspvals':,
                 'method':,
                 'nbestpeaks':,
                 'nbestperiods':,
                 'bestperiod':,
                 'periods':,
                 'bestlspval':}

        which is all the important things computed in the periodogram.
    '''

    assert pertype=='pdm' or pertype=='bls' or pertype=='lsp'

    dat['per'] = {}

    keyd = {
            0: {'dtr': ['times', 'fluxs_dtr_norm', 'errs_dtr_norm']},
            1: {'redtr': ['times', 'fluxs_redtr', 'errs']}
           }
    mk = list(keyd[iter_n].keys()).pop()

    for ap in ['sap', 'pdc']:

        times = dat[mk][ap][keyd[iter_n][mk][0]]
        fluxs = dat[mk][ap][keyd[iter_n][mk][1]]
        #fluxs = dat['redtr'][ap]['fluxs'] - dat['redtr'][ap]['fitfluxs_legendre']
        errs =  dat[mk][ap][keyd[iter_n][mk][2]]

        # Range of interesting periods (morph>0.7): 0.05days (1.2hr)-20days.
        # BLS can only search for periods < half the light curve observing 
        # baseline. (Nb longer signals are almost always stellar rotation)

        if len(times) < 50 or len(fluxs) < 50:
            LOGERROR('Got quarter with too few points. Continuing.')
            continue

        smallest_p = 0.05
        biggest_p = min((times[-1] - times[0])/2.01, 20.)

        if pertype == 'pdm':
            pgd = periodbase.stellingwerf_pdm(times,fluxs,errs,
                autofreq=True,
                startp=smallest_p,
                endp=biggest_p,
                normalize=False,
                stepsize=5.0e-5,
                phasebinsize=0.05,
                mindetperbin=9,
                nbestpeaks=5,
                periodepsilon=0.05, # 0.05 days
                sigclip=None, # no sigma clipping
                nworkers=None)
        elif pertype == 'bls':
            pgd = periodbase.bls_parallel_pfind(times,fluxs,errs,
                startp=smallest_p,
                endp=biggest_p, # don't search full timebase
                stepsize=5.0e-5,
                mintransitduration=0.01, # minimum transit length in phase
                maxtransitduration=0.2,  # maximum transit length in phase
                nphasebins=100,
                autofreq=False, # figure out f0, nf, and df automatically
                nbestpeaks=5,
                periodepsilon=0.1, # 0.1 days
                nworkers=None,
                sigclip=None)
        elif pertype == 'lsp':
            pgd = periodbase.pgen_lsp(times,fluxs,errs,
                startp=smallest_p,
                endp=biggest_p,
                autofreq=True,
                nbestpeaks=5,
                periodepsilon=0.1,
                stepsize=1.0e-4,
                nworkers=None,
                sigclip=None)

        if isinstance(pgd, dict):
            LOGINFO('KIC ID %s, computed periodogram (%s) quarter %s. (%s)'
                % (str(dat['objectinfo']['keplerid']), pertype, str(qnum), ap))
        else:
            LOGERROR('Error in KICID %s, periodogram %s, quarter %s (%s)'
                % (str(dat['objectinfo']['keplerid']), pertype, str(qnum), ap))

        dat['per'][ap] = pgd


    return dat


def _select_whiten_period(dat, rtol=5e-2, fine=False, inum=0, ap='sap',
        want_eb_period=False):
    '''
    Related to select_eb_period, but different control flow.

    Select the period at which to whiten for a given quarter, given the
    periodogram information and the KEBC information.

    Logic:
    If want EB period:
        If within rtol% of KEBC period, take the periodogram period.
        Else, look for periods in the best 5 from the periodogram that are
        within rtol% of the KEBC period. If there is only one, take that one
        period.
        Otherwise, use the KEBC period.
    Else:
        Take best period. Run fine periodogram over +/- rtol (e.g., if rtol is
        5e-2, +5% and -5% of the best period).

    Args:
        dat: dictionary with specified qnum.

        rtol (float): relative tolerance for accepting close periods from
        periodogram

        fine (bool): False if you have not run a "fine search" for the best
        period. True if the fine periodogram search has been run, and thus the
        results of that should be used in selecting the EB period.

        want_eb_period (bool): whether you want periods in the data close to
        the KEBC EB periods.

    Returns:
        dat['per'] (or dat['fineper']) with 'selperiod' and
        'selforcedkebc' keys. These give the selected period (float) and a
        string that takes values of either ('forcedkebc','switch','correct')
        for cases when we were forced to take the period from the KEBC, where
        given the KEBC value we switched to a different peak in the
        periodogram, or when the periodogram got it right on the first try.
    '''

    pgkey = 'per' if not fine else 'fineper'
    pgd = dat['white'][inum][ap][pgkey]

    my_period = nparr(pgd['bestperiod'])
    my_periods = nparr(pgd['nbestperiods'])

    if fine:
        my_period = nparr(pgd['bestperiod'])
        my_periods = nparr(pgd['nbestperiods'])

        pgd['selperiod'] = my_period
        pgd['selforcedkebc'] = 'finebest'

    elif not fine and want_eb_period:

        kebc_period = nparr(float(dat['kebwg_info']['period']))
        rightperiod = npabs(my_period - kebc_period)/npabs(kebc_period) <= rtol

        if rightperiod:
            pgd['selperiod'] = my_period
            pgd['selforcedkebc'] = 'correct'

        else:
            sel = npabs(my_periods - kebc_period)/npabs(kebc_period) <= rtol

            if not np.any(sel) or len(sel[sel==True]) > 1:
                pgd['selperiod'] = kebc_period
                pgd['selforcedkebc'] = 'forcedkebc'

            else:
                pgd['selperiod'] = float(my_periods[sel])
                pgd['selforcedkebc'] = 'switch'

    elif not fine and not want_eb_period:

        pgd['selperiod'] = my_period
        pgd['selforcedkebc'] = 'notwantebperiod'

    dat['white'][inum][ap][pgkey] = pgd

    return dat



def select_eb_period(lcd, rtol=1e-1, fine=False):
    '''
    Select the "correct" EB period for each quarter given the periodogram
    information and the KEBC information.
    Logic:
    For a given quarter:
        If within 10% of KEBC period, take the periodogram period.
        Else, look for periods in the best 5 from the periodogram that are
        within 10% of the KEBC period. If there is only one, take that one
        period.
        Otherwise, use the KEBC period.

    Args:
        rtol (float): relative tolerance for accepting close periods from
        periodogram

        fine (bool): False if you have not run a "fine search" for the best
        period. True if the fine periodogram search has been run, and thus the
        results of that should be used in selecting the EB period.

    Returns:
        lcd[qnum]['per'] (or lcd[qnum]['fineper']) with 'selperiod' and
        'selforcedkebc' keys. These give the selected period (float) and a
        string that takes values of either ('forcedkebc','switch','correct')
        for cases when we were forced to take the period from the KEBC, where
        given the KEBC value we switched to a different peak in the
        periodogram, or when the periodogram got it right on the first try.
    '''

    kebc_period = nparr(float(lcd[list(lcd.keys())[0]]['kebwg_info']['period']))

    for k in lcd.keys():
        for ap in ['sap','pdc']:

            my_period = nparr(lcd[k]['per'][ap]['bestperiod'])
            my_periods = nparr(lcd[k]['per'][ap]['nbestperiods'])

            rightperiod = npabs(my_period - kebc_period)/npabs(kebc_period) <= rtol

            if fine:
                my_period = nparr(lcd[k]['fineper'][ap]['bestperiod'])
                my_periods = nparr(lcd[k]['fineper'][ap]['nbestperiods'])

                lcd[k]['fineper'][ap]['selperiod'] = my_period
                lcd[k]['fineper'][ap]['selforcedkebc'] = 'finebest'

            elif not fine:
                if rightperiod:
                    lcd[k]['per'][ap]['selperiod'] = my_period
                    lcd[k]['per'][ap]['selforcedkebc'] = 'correct'

                else:
                    sel = npabs(my_periods - kebc_period)/npabs(kebc_period) <= rtol

                    if not np.any(sel) or len(sel[sel==True]) > 1:
                        lcd[k]['per'][ap]['selperiod'] = kebc_period
                        lcd[k]['per'][ap]['selforcedkebc'] = 'forcedkebc'

                    else:
                        lcd[k]['per'][ap]['selperiod'] = float(my_periods[sel])
                        lcd[k]['per'][ap]['selforcedkebc'] = 'switch'

    return lcd


def _get_legendre_deg(npts):
    from scipy.interpolate import interp1d

    degs = np.array([2,5,10,20,30])
    pts = np.array([1e2,5e2,1e3,2e3,3e3])
    fn = interp1d(pts, degs, kind='linear',
                 bounds_error=False,
                 fill_value=(2, 30))
    legendredeg = int(np.floor(fn(npts)))

    return legendredeg


def _iter_run_periodogram(dat, qnum, inum=0, ap='sap', fine=False,
        dynamical_prefactor=5):

    # Initialize periodogram or fineperiodogram dictionaries.
    kebc_period = nparr(float(dat['kebwg_info']['period']))
    pgkey = 'per' if not fine else 'fineper'
    if not fine:
        dat['white'][inum][ap] = {}

    # Get times and fluxes.
    if inum == 0:
        times = dat['dtr'][ap]['times']
        fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
        errs =  dat['dtr'][ap]['errs_dtr_norm']
    else:
        times = dat['white'][inum-1][ap]['legdict']['whiteseries']['times']
        fluxs = dat['white'][inum-1][ap]['legdict']['whiteseries']['wfluxs']
        errs =  dat['white'][inum-1][ap]['legdict']['whiteseries']['errs']

    if len(times) < 50 or len(fluxs) < 50:
        LOGERROR('Got quarter with too few points. Continuing.')
        dat['white'][inum][ap][pgkey] = np.nan
    else:
        if not fine:
            # Range of interesting periods (morph>0.7): 0.05days(1.2hr)-20days.
            # BLS can only search for periods < half the light curve observing 
            # baseline. (Nb longer signals are almost always stellar rotation).
            smallest_p = 0.05
            biggest_p = min((times[-1] - times[0])/2.01,
                    kebc_period*dynamical_prefactor)
            periodepsilon = 0.01 # days
            stepsize=42 # because it doesn't matter
            autofreq=True
        elif fine:
            # Range of interesting periods, now that the selected period has 
            # been chosen: +/- 1% above/below it.
            selperiod = dat['white'][inum][ap]['per']['selperiod']
            rdiff = 0.01
            smallest_p = selperiod - rdiff*selperiod
            biggest_p = selperiod + rdiff*selperiod
            periodepsilon = rdiff*selperiod*0.1
            stepsize=2e-6
            autofreq=False

        pgd = periodbase.stellingwerf_pdm(times,fluxs,errs,
            autofreq=autofreq,
            startp=smallest_p,
            endp=biggest_p,
            normalize=False,
            stepsize=stepsize,
            phasebinsize=0.05,
            mindetperbin=9,
            nbestpeaks=5,
            periodepsilon=periodepsilon,
            sigclip=None, # no sigma clipping
            nworkers=None)

        if isinstance(pgd, dict):
            LOGINFO('KIC ID %s, computed periodogram (inum %s) quarter %s. (%s)'
                % (str(dat['objectinfo']['keplerid']), str(inum), str(qnum), ap))
        else:
            LOGERROR('Error in KICID %s, periodogram %s, quarter %s (%s)'
                % (str(dat['objectinfo']['keplerid']), str(inum), str(qnum), ap))

        dat['white'][inum][ap][pgkey] = pgd

    return dat


def iterative_whiten_allquarters(lcd, σ_clip=[30.,5.], nwhiten_max=10,
        nwhiten_min=2, rms_floor=0.1):
    '''
    Wrapper to iterative_whiten_lightcurve, to run for all quarters. NOTE that
    this doesn't make whiten_allquarters redundant because the data structures
    are all different (because I unsurprisingly didn't make the dictionaries
    flexible enough to begin with).

    Saves iteratively whitened results to keys
    `lcd[qnum]['white']['legdict'][inum][ap]['w*']`, for * in (fluxs,errs,times,phases)
    '''
    rd = {}
    for k in lcd.keys():
        rd[k] = iterative_whiten_lightcurve(lcd[k],
                k,
                σ_clip=σ_clip,
                nwhiten_max=nwhiten_max,
                nwhiten_min=nwhiten_min,
                rms_floor=rms_floor)

    return rd


def iterative_whiten_lightcurve(dat, qnum, method='legendre',
        legendredeg='best', rescaletomedian=True, σ_clip=None, nwhiten_max=10,
        nwhiten_min=2, rms_floor=0.001):
    '''
    Given the normalized, detrended fluxes, and the known period computed from
    the periodogram routines, iteratively fit for the eclipsing binary signal
    in phase and subtract it out. In pseudocode:

    while (rms>rms_floor or nwhiten < nwhiten_min) and nwhiten < nwhiten_max:
        * Stellingwerf PDM (coarse freq bins), get coarseperiod
        * Repeat over narrow bins centered on peak signal, get selperiod
        * Whiten at selperiod by phase-folding, and fitting a finite-order
          Legendre series (order chosen to avoid overfitting & underfitting. It
          should be low enough to gloss over transits. Methods for finding good
          orders include cross validation and AIC/BIC, but we do something
          entirely empirical, i.e. w/out rigorous statistical justification,
          but it works).
        * Subtract fit, compute new RMS.

    Args:
        dat (dict): the single-quarter dictionary returned by
        astrokep.read_kepler_fitslc, post-normalization, detrending, and
        selecting the appropriate EB period.

        detrend (str): method by which to whiten the LC. 'legendre' is
        currently the only one accepted (although astrobase.varbase has other
        options, e.g., Fourier Series, Savitzky-Golay filter, etc, which could
        be implemented).

        rescaletomedian (bool): rescales the whitened fluxes to their median
        value.

        legendredeg: integer to be fixed, else (str) "best".

        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries.

        rms_floor (float): unbiased RMS [%] at which iterative whitening can
        stop. E.g., set to 0.1 to stop at 0.1%.

        nwhiten_max (int): maximum number of times to iterate the whitening
        process.

        nwhiten_min (int): converse of nwhiten_max. Must be >=1.

    Returns:
        dat (dict): dat, with the phased times, fluxes and errors in a
        sub-dictionary, accessible as dat['white'], which gives the
        dictionary:

        dat['white'] is keyed by number of whitenings that have been applied
        (1,2,3,...,nwhiten_max). Each one of these keys, `inum`, is keyed by
        sap/pdc, and contains:

        dat['white'][inum]['sap'].keys():
            ['magseries', 'whiteseries', 'fitinfo', 'data_rms', 'resid_rms',
            'per', # the coarse periodogram for this iteation
            'fineper', # the fine periodogram for this iteation
            'fittype', 'fitplotfile', 'fitchisq', 'fitredchisq'].

        data_rms is the RMS of the data used in this iterationnumber (inum).
        resid_rms is the RMS of the residual once the fit is subtracted.

        The subkeys that lead to other dictionaries:
        ...['whiteseries'].keys() # whitened, time-sorted, sigma clipped.
            ['wfluxs', 'errs', 'times']
            (so wfluxs is the *residual* of the inum^{th} iteration)

        ...['magseries'].keys() # NOT whitened, but phase-sorted & sigclipped
            ['mags', 'errs', 'times', 'phase']

        ...['fitinfo'].keys()
            ['fitepoch', 'legendredeg', 'fitmags']
    '''

    assert nwhiten_min >= 1
    assert isinstance(legendredeg, int) or \
            (isinstance(legendredeg, str) and legendredeg=='best')
    assert method == 'legendre'

    dat['white'] = {}

    sap_rms = 42 # placeholder RMS
    nwhiten = 0 # number of whitenings that have been done

    while (sap_rms>rms_floor or nwhiten<=nwhiten_min) and nwhiten<=nwhiten_max:

        dat['white'][nwhiten] = {}

        for ap in ['sap', 'pdc']:
            # Be sure to whiten at the EB period first.
            whiten_at_eb_period = True if nwhiten == 0 else False

            # Run coarse periodogram and get initial period guess.
            dat = _iter_run_periodogram(dat, qnum, inum=nwhiten, ap=ap, fine=False)
            dat = _select_whiten_period(dat, fine=False, inum=nwhiten, ap=ap,
                    want_eb_period=whiten_at_eb_period)
            # Run fine periodogram and get precise period at which to whiten.
            dat = _iter_run_periodogram(dat, qnum, inum=nwhiten, ap=ap, fine=True)
            dat = _select_whiten_period(dat, fine=True, inum=nwhiten, ap=ap,
                    want_eb_period=whiten_at_eb_period)

            # `period` is the period at which we will whiten.
            # `times`, `fluxs`, `errs` are the things to be whitened.
            period = dat['white'][nwhiten][ap]['fineper']['selperiod']
            if nwhiten == 0:
                times = dat['dtr'][ap]['times']
                fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
                errs =  dat['dtr'][ap]['errs_dtr_norm']
            else:
                times = dat['white'][nwhiten-1][ap]['legdict']['whiteseries']['times']
                fluxs = dat['white'][nwhiten-1][ap]['legdict']['whiteseries']['wfluxs']
                errs =  dat['white'][nwhiten-1][ap]['legdict']['whiteseries']['errs']

            if legendredeg=='best':
                npts = len(fluxs)
                legdeg = _get_legendre_deg(npts)
            try:
                legdict = lcf.legendre_fit_magseries(
                    times,fluxs,errs,period,
                    legendredeg=legdeg,
                    sigclip=σ_clip,
                    plotfit=False,
                    magsarefluxes=True)
                LOGINFO('Whitened KICID %s, quarter %s, (%s, inum %s).'
                    % (str(dat['objectinfo']['keplerid']), str(qnum), ap,
                       str(nwhiten)))
            except:
                LOGERROR('Whitening error. KIC %s, qnum %s, (%s, inum %s).'
                    % (str(dat['objectinfo']['keplerid']), str(qnum), ap,
                       str(nwhiten)))

            # Now compute the residual.
            tms, tfi = legdict['magseries'], legdict['fitinfo']
            # This magseries (`tms`) is in phase-sorted order.
            phase = tms['phase']
            ptimes = tms['times']
            pfluxs = tms['mags']
            perrs = tms['errs']
            presiduals = tms['mags'] - tfi['fitmags']
            # Get times, residuals (whitened fluxs), errs in time-sorted order.
            wtimeorder = np.argsort(ptimes)
            wtimes = ptimes[wtimeorder]
            wphase = phase[wtimeorder]
            wfluxs = presiduals[wtimeorder]
            werrs = perrs[wtimeorder]

            if rescaletomedian:
                wfluxs += np.median(wfluxs)

            # Compute inital and post-whitened RMS.
            meanflux = np.mean(fluxs)
            init_rms = np.sqrt(np.sum((fluxs-meanflux)**2)/\
                        (float(len(fluxs))-1))
            legdict['data_rms'] = init_rms
            meanwflux = np.mean(wfluxs)
            w_rms = np.sqrt(np.sum((wfluxs-meanwflux)**2)/\
                        (float(len(wfluxs))-1))
            legdict['resid_rms'] = w_rms
            if ap == 'sap':
                sap_rms = w_rms

            # Save whitened times, fluxes, and errs to 'whiteseries' subdict.
            whitedict = {'times':wtimes,
                    'phase': wphase,
                    'wfluxs':wfluxs,
                    'errs':werrs}

            legdict['whiteseries'] = whitedict

            dat['white'][nwhiten][ap]['legdict'] = legdict

        nwhiten += 1
        LOGINFO('nwhiten (inum) {:d}, sap_rms {:.3g}, rms_floor {:.3g}'.format(
            nwhiten, sap_rms, rms_floor))

    return dat


def whiten_allquarters(lcd, σ_clip=None, iter_n=0):
    '''
    Wrapper to whiten_lightcurve, to run for all quarters.
    Saves whitened results to keys `lcd[qnum]['white']['sap']['w_*']`, for * in
    (fluxs,errs,times,phases)
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = whiten_lightcurve(lcd[k], k, σ_clip=σ_clip, iter_n=iter_n)

    return rd


def whiten_lightcurve(dat, qnum, method='legendre', legendredeg=80,
        rescaletomedian=True, σ_clip=None, iter_n=0):
    '''
    Given the normalized, detrended fluxes, and the known period computed from
    the periodogram routines, fit for the eclipsing binary signal in phase and
    subtract it out.

    Args:
        dat (dict): the dictionary returned by astrokep.read_kepler_fitslc,
        after normalization, detrending, and selecting the appropriate EB
        period.

        detrend (str): method by which to whiten the LC. 'legendre' is
        currently the only one accepted (although astrobase.varbase has other
        options, e.g., Fourier Series, Savitzky-Golay filter, etc, which could
        be implemented).

        rescaletomedian (bool): rescales the whitened fluxes to their median
        value.

        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries.

        iter_n (int): 0: 'dtr', no whitening has yet been done
                     1: 'redtr', the first round of whitening has been done,
                     2: etc.

    Returns:
        dat (dict): dat, with the phased times, fluxes and errors in a
        sub-dictionary, accessible as dat['white'], which gives the
        dictionary:

        dat['white'] = {'sap':ldict, 'pdc':ldict}, where ldict contains:

        ldict.keys():
            ['fitplotfile', 'fitchisq', 'fitredchisq', 'magseries',
            'fitinfo', 'fittype', 'whiteseries']

        ldict['whiteseries'].keys() # time-sorted, and sigma clipped.
            ['mags', 'errs', 'times']

        legdict['magseries'].keys() # phase-sorted
            ['mags', 'errs', 'times', 'phase']

        legdict['fitinfo'].keys()
            ['fitepoch', 'legendredeg', 'fitmags']
    '''

    dat['white'] = {}

    keyd = {
            0: {'dtr': ['times', 'fluxs_dtr_norm', 'errs_dtr_norm']},
            1: {'redtr': ['times', 'fluxs_redtr', 'errs']}
           }

    mk = list(keyd[iter_n].keys()).pop()

    for ap in ['sap', 'pdc']:

        period = dat['fineper'][ap]['selperiod']
        times = dat[mk][ap][keyd[iter_n][mk][0]]
        fluxs = dat[mk][ap][keyd[iter_n][mk][1]]
        errs =  dat[mk][ap][keyd[iter_n][mk][2]]

        try:
            legdict = lcf.legendre_fit_magseries(
                times,fluxs,errs,period,
                legendredeg=legendredeg,
                sigclip=σ_clip,
                plotfit=False,
                magsarefluxes=True)
            LOGINFO('Whitened KICID %s, quarter %s, (%s) (Legendre).'
                % (str(dat['objectinfo']['keplerid']), str(qnum), ap))

        except:
            LOGERROR('Legendre whitening error in KICID %s, quarter %s, (%s).'
                % (str(dat['objectinfo']['keplerid']), str(qnum), ap))

        tms = legdict['magseries']
        tfi = legdict['fitinfo']

        #everything in phase-sorted order:
        phase = tms['phase']
        ptimes = tms['times']
        pfluxs = tms['mags']
        perrs = tms['errs']
        presiduals = tms['mags'] - tfi['fitmags']

        #get it all in time-sorted order:
        wtimeorder = np.argsort(ptimes)
        wtimes = ptimes[wtimeorder]
        wphase = phase[wtimeorder]
        wfluxes = presiduals[wtimeorder]
        werrs = perrs[wtimeorder]

        if rescaletomedian:
            median_mag = np.median(wfluxes)
            wfluxes = wfluxes + median_mag

        whitedict = {'times':wtimes,
                'phase': wphase,
                'fluxes':wfluxes,
                'errs':werrs}

        legdict['whiteseries'] = whitedict

        dat['white'][ap] = legdict

    return dat


def find_dips(lcd, allq, method='bls'):
    '''
    Find dips (e.g., what planets would do) over the entire magnitude time
    series.

    Args:
        method (str): currently only "bls": just do BLS.

    Returns:
        `allq`, with "dipfind" key that holds the results of the dip
        search (e.g., periodogram information) for each technique. They are
        sub-directories:

        ["dipfind"]["tfe"][ap] =
                  {'times':times,
                   'fluxs':fluxs,
                   'errs':errs,
                   'qnums':quarter}
        ["dipfind"][method][ap] = periodogram dictionary.
    '''

    #Concatenate all the quarters when running dip-finder.
    qnums = np.sort(list(lcd.keys()))
    tfe = {}
    for ap in ['sap','pdc']:
        for ix, qnum in enumerate(qnums):
            lc = lcd[qnum]['redtr'][ap]
            if ix == 0:
                times = lc['times']
                fluxs = lc['fluxs'] - lc['fitfluxs_legendre']
                errs = lc['errs']
                quarter = np.ones_like(times)*qnum
            else:
                times = np.append(times, lc['times'])
                fluxs = np.append(fluxs, lc['fluxs'] - lc['fitfluxs_legendre'])
                errs = np.append(errs, lc['errs'])
                quarter = np.append(quarter, np.ones_like(lc['times'])*qnum)
        tfe[ap] = {'times':times,
                   'fluxs':fluxs,
                   'errs':errs,
                   'qnums':quarter}

    # Range of interesting periods determined by Holman & Weigert (1999)
    # stability criterion on the inner edge. On the outer, note BLS can search
    # for periods < half the observed baseline.
    keplerid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])
    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])

    smallfactor, bigfactor = 4., 40.
    smallest_p = smallfactor * kebc_period
    biggest_p = bigfactor * kebc_period
    minTdur_φ = 0.005 # minimum transit length in phase
    maxTdur_φ = 0.2 # maximum transit length in phase

    df_dict = {}
    for ap in ['sap','pdc']:

        df_dict[ap] = {}
        df_dict[ap]['finebls'] = {}

        times, fluxs, errs = \
                tfe[ap]['times'], tfe[ap]['fluxs'], tfe[ap]['errs']
        finiteind = np.isfinite(fluxs) & np.isfinite(times) & np.isfinite(errs)
        times = times[finiteind]
        fluxs = fluxs[finiteind]
        errs = errs[finiteind]

        if method != 'bls':
            raise Exception

        pgd = periodbase.bls_parallel_pfind(times,fluxs,errs,
            magsarefluxes=True,
            startp=smallest_p,
            endp=biggest_p, # don't search full timebase
            stepsize=5.0e-5,
            mintransitduration=minTdur_φ,
            maxtransitduration=maxTdur_φ,
            autofreq=True, # auto-find frequencies and nphasebins
            nbestpeaks=5,
            periodepsilon=0.1, # 0.1 days btwn period peaks to be distinct.
            nworkers=None,
            sigclip=None)
        pgd['nphasebins'] = int(np.ceil(2.0/minTdur_φ))

        nbestperiods = pgd['nbestperiods']

        LOGINFO('KICID: {:s}. Finished coarsebls ({:s}) ap:{:s}'.format(
                keplerid, method.upper(), ap.upper()))

        df_dict[ap]['coarsebls'] = pgd

        # coarsebls has blsresults from each worker's frequency chunk. To 
        # get epochs, you need to go from the BIN INDEX at ingress of the 
        # best transit to that of egress. As eebls.f is written, it only 
        # returns the "best transit" from whatever frequency chunk was 
        # given to it. So we need to run SERIAL BLS around the nbestperiods
        # that we want epochs for, and wrangle the results of that to get 
        # good epochs.
        for nbestperiod in nbestperiods:
            # do ~10000 frequencies within +/- 0.5% of nbestperiod
            rdiff = 0.005
            startp = nbestperiod - rdiff*nbestperiod
            endp = nbestperiod + rdiff*nbestperiod
            maxfreq = 1/startp
            minfreq = 1/endp
            nfreqdesired = 1e4
            stepsize = (maxfreq-minfreq)/nfreqdesired
            nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))
            nphasebins = int(np.ceil(2.0/minTdur_φ))

            LOGINFO('Narrow serial BLS: {:d} freqs, '.format(nfreq)+\
                    'start {:.6g} d, end {:.6g} d. {:d} phasebins.'.format(
                    startp, endp, nphasebins))

            sd = periodbase.bls_serial_pfind(times, fluxs, errs,
                 magsarefluxes=True,
                 startp=startp,
                 endp=endp,
                 stepsize=stepsize,
                 mintransitduration=minTdur_φ,
                 maxtransitduration=maxTdur_φ,
                 nphasebins=nphasebins,
                 autofreq=False,
                 periodepsilon=rdiff*nbestperiod/3.,
                 nbestpeaks=1,
                 sigclip=None)

            df_dict[ap]['finebls'][nbestperiod] = {}
            df_dict[ap]['finebls'][nbestperiod]['serialdict'] = sd
            #serialdict (sd) keys:['nfreq', 'maxtransitduration', 'nphasebins',
            #'nbestpeaks', 'nbestperiods', 'lspvals', 'blsresult', 'bestperiod',
            #'method', 'mintransitduration', 'stepsize', 'bestlspval',
            #'nbestlspvals', 'periods', 'frequencies'].
            #blsresult['transingressbin'] and blsresult['transegressbin'] are
            #what are relevant for finding the epoch.

            # -1 because fortran is 1-based
            ingbinind = int(sd['blsresult']['transingressbin']-1)
            egrbinind = int(sd['blsresult']['transegressbin']-1)

            # Kovacs' eebls.f uses the first time as the epoch when computing
            # the phase. (Note he keeps the phases in time-sorted order).
            tempepoch = np.min(times)
            phasedlc = lcmath.phase_magseries(times,
                                              fluxs,
                                              nbestperiod,
                                              tempepoch,
                                              wrap=False,
                                              sort=False)

            # N.b. these phases and fluxes (at the corresponding phase-values) 
            # are in time-sorted order.
            φ = phasedlc['phase']
            flux_φ = phasedlc['mags']

            phasebin = 1./sd['nphasebins']
            binphasedlc = lcmath.phase_bin_magseries(φ,flux_φ,binsize=phasebin)

            φ_bin = binphasedlc['binnedphases']
            flux_φ_bin = binphasedlc['binnedmags']
            if ingbinind < len(φ_bin):
                φ_ing = φ_bin[ingbinind]
                φ_egr = φ_bin[egrbinind]
            else:
                LOGERROR('ingbinind from eebls.f shorter than it should be.')
                LOGERROR('Hard-setting ingress and egress phases.')
                φ_ing, φ_egr = 0.0, 0.01

            # Get φ_0, the phase of central transit.
            if φ_ing < φ_egr:
                halfwidth = (φ_egr-φ_ing)/2.
                φ_0 = φ_ing + halfwidth
            elif φ_ing > φ_egr:
                halfwidth = (1+φ_egr-φ_ing)/2.
                if 1-φ_ing > φ_egr:
                    φ_0 = (φ_egr - halfwidth)%1.
                elif 1-φ_ing < φ_egr:
                    φ_0 = (φ_ing + halfwidth)%1.
                elif 1-φ_ing == φ_egr:
                    φ_0 = 0.

            df_dict[ap]['finebls'][nbestperiod]['φ'] = phasedlc['phase']
            df_dict[ap]['finebls'][nbestperiod]['flux_φ'] = phasedlc['mags']
            df_dict[ap]['finebls'][nbestperiod]['binned_φ'] = φ_bin
            df_dict[ap]['finebls'][nbestperiod]['binned_flux_φ'] = flux_φ_bin
            df_dict[ap]['finebls'][nbestperiod]['φ_ing'] = φ_ing
            df_dict[ap]['finebls'][nbestperiod]['φ_egr'] = φ_egr
            df_dict[ap]['finebls'][nbestperiod]['φ_0'] = φ_0

        LOGINFO('KICID: {:s}. Finished finebls ({:s}) ap:{:s}'.format(
                keplerid, method.upper(), ap.upper()))

    allq['dipfind'] = {}
    allq['dipfind'][method] = df_dict
    allq['dipfind']['tfe'] = tfe

    return allq


def save_lightcurve_data(lcd, allq=None, stage=False):
    '''
    Args:
        lcd (anytype): the thing you want to write to a pickle file. Always
        "lcd" object, keyed by quarters.

    Kwargs:
        allq (anytype): if you also have "all of the quarters" concatenated
        data that you'd like to save in a separate pickle.

        stage (str): a short string that will be appended to the pickle file
        name to simplify subsequent reading. E.g., "pw" for "post-whitening",
        or "redtr_inj" if it's post-redetrending, and you've injected fake
        transits.
    '''

    kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

    if stage:
        pklname = kicid+'_'+stage+'.p'
        pklallqname = kicid+'_allq_'+stage+'.p'
    else:
        pklname = kicid+'.p'

    predir = ''
    if 'no_inj' not in stage:
        predir += 'inj/'
    elif 'no_inj' in stage:
        predir += 'no_inj/'
    spath = '../data/injrecov_pkl/'+predir+pklname
    sallqpath = '../data/injrecov_pkl/'+predir+pklallqname

    if 'eb_sbtr' in stage:
        #override; save pickles somewhere else for EB subtraction tests.
        spath = '../data/eb_subtr_pkl/'+pklname
        sallqpath = '../data/eb_subtr_pkl/'+pklallqname

    pickle.dump(lcd, open(spath, 'wb'))
    LOGINFO('Saved (pickled) lcd data to %s' % spath)
    if isinstance(allq, dict):
        pickle.dump(allq, open(sallqpath, 'wb'))
        LOGINFO('Saved (pickled) allquarter data to %s' % sallqpath)

    return kicid


def load_lightcurve_data(kicid, stage=None, δ=None):

    # `stage` includes the transit depth string
    pklname = str(kicid)+'_'+stage+'.p'
    if δ == 'whatever':
        pklname = str(kicid)+'_'+stage+'_0.00125.p'

    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    elif 'inj' not in stage:
        predir += 'no_inj/'

    lpath = '../data/injrecov_pkl/'+predir+pklname

    if 'eb_sbtr' in stage:
        #saved pickles somewhere else for EB subtraction tests.
        lpath = '../data/eb_subtr_pkl/'+pklname

    dat = pickle.load(open(lpath, 'rb'))
    LOGINFO('Loaded pickled data from %s' % lpath)

    return dat


def load_allq_data(kicid, stage=None, δ=None):

    pklname = str(kicid)+'_allq_'+stage+'.p'
    if δ == 'whatever':
        pklname = str(kicid)+'_'+stage+'_0.00125.p'

    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    elif 'inj' not in stage:
        predir += 'no_inj/'

    lpath = '../data/injrecov_pkl/'+predir+pklname

    if 'eb_sbtr' in stage:
        #saved pickles somewhere else for EB subtraction tests.
        lpath = '../data/eb_subtr_pkl/'+pklname

    allq = pickle.load(open(lpath, 'rb'))
    LOGINFO('Loaded allquarter pickled data from %s' % lpath)

    return allq


def drop_gaps_in_lightcurves(times, mags, errs):
    '''
    For any gap in the time-series of more than 0.5 days, drop the neighboring
    0.5 day windows (i.e. 0.5 days before the gap, and 0.5 days after). This is
    to avoid the systematic noise that tends to be near them.
    (N.b. this follows exactly Armstrong et al. (2014)'s description)

    Args:
    '''
    pass
