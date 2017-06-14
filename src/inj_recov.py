'''
See docstrings in run_the_machine.py
'''

import pickle, os, logging, pdb, socket
from astropy.io import ascii
import astropy.units as u, astropy.constants as c
from datetime import datetime
from astrobase import astrokep, periodbase, lcmath
from astrobase.varbase import lcfit as lcf
import matplotlib
matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr
from numpy.polynomial.legendre import Legendre, legval
import batman
import pandas as pd

#############
## GLOBALS ##
#############
global HOSTNAME, DATADIR
HOSTNAME = socket.gethostname()
DATADIR = '../data/' if 'della' not in HOSTNAME else '/tigress/lbouma/data/'

#############
## LOGGING ##
#############
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
    Same as inject_random_drawn_transit, but made easier by getting the transit
    depth from the calling routine.

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
    # X P_CBP [days] from exp(lnP_CBP) ~ exp(U(ln P_EB*2, ln 150 d)), i.e. a
    #   log-uniform distribution, where P_EB is the value reported by the
    #   KEBWG.
    # X Reference transit time t_0 [days] from t_0 ~ U(0,P_CBP) (plus the
    #   minimum time of the timeseries).
    # X Radius ratio Rp/Rstar from lnRp/Rstar ~ U(..., ...)
    # X Eccentricity e~Beta(0.867,3.03)
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
    lowpref = 2
    max_period = 150 # days
    ln_period_cbp = np.random.uniform(
            low=np.log(lowpref*period_eb),
            high=np.log(max_period))
    period_cbp = np.e**ln_period_cbp
    params.per = period_cbp

    t0 = np.random.uniform(
            low=0.,
            high=period_cbp)
    params.t0 = mintime + t0

    params.rp = δ**(1/2.)

    ecc = np.random.beta(0.867,3.03) # Kipping 2013
    params.ecc = ecc

    ln_ρstarbyρsun = np.random.uniform(
            low=np.log(0.1),
            high=np.log(100))
    ρsun = 3*u.Msun / (4*np.pi*u.Rsun**3)

    # Eq (30) Winn 2010, once we've sampled the stellar density
    a_by_Rstar = ( ( c.G * (period_cbp*u.day)**2 / \
            (3*np.pi) * (e**ln_ρstarbyρsun)*ρsun )**(1/3) ).cgs.value

    import IPython; IPython.embed() # FIXME: verify gives good a_by_Rstar

    b_tra = 42. # initialize to whatever.
    while b_tra > 1:
        cosi = np.random.uniform(low=0.0,high=1.0)
        # Eq (7) of Winn 2010
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
    Returns a dictionary with keys of quarter number and a flag if failed.
    '''

    kebc = get_kepler_ebs_info()
    kebc = kebc[kebc['morph']>0.6]
    kebc_kic_ids = np.array(kebc['KIC'])
    ind = int(np.random.randint(0, len(kebc['KIC']), size=1))

    rd, errflag = get_all_quarters_lc_data(kebc_kic_ids[ind])
    if len(rd) < 1:
        lcflag = True
        LOGERROR('Error getting LC data. KICID-{:s}'.format(
            str(kebc_kic_ids[ind])))
    else:
        lcflag = False

    return rd, lcflag


def retrieve_next_lc(stage=None, blacklist=None, kicid=None):
    '''
    Get the next LC from the KEBWG list of morph>0.6 or period<3 day EBs.
    Returns a dictionary with keys of quarter number and a flag if failed.

    If no KIC ID is passed in kicid, finds the next one to search based on what
    dipsearchplots, pkls, and whitened diagnostics exist locally.
    '''

    if not kicid:
        kebc_kic_ids = np.genfromtxt(
                DATADIR+'morph_gt0.6_OR_period_lt3_kepler_MAST/'+\
                        'morph_gt_0.6_OR_per_lt_3_ids.txt',
                dtype=int)
        ind = 0
        # Find the next KIC ID to search (requires dipsearchplot, pkl, and whitened
        # diagnostic to proceed).
        while True:
            if ind == len(kebc_kic_ids):
                return np.nan, 'finished', np.nan
            this_id = kebc_kic_ids[ind]
            if this_id in blacklist:
                ind += 1
                continue

            pklmatch = [f for f in os.listdir(DATADIR+'injrecov_pkl/real/') if
                    f.endswith('.p') and f.startswith(str(this_id)) and stage in f]
            dspmatch = [f for f in os.listdir('../results/dipsearchplot/real/') if
                    f.endswith('.png') and f.startswith(str(this_id)) and stage in f]
            wdmatch = [f for f in os.listdir('../results/whitened_diagnostic/real/') if
                    f.endswith('.png') and f.startswith(str(this_id)) and stage in f]

            if len(pklmatch)>0 and len(dspmatch)>0 and len(wdmatch)>0:
                LOGINFO('Found {:s} pkl, dipsearchplt, whitened_diagnostic'.format(
                    str(this_id)))
                blacklist.append(this_id)
                ind += 1
                continue
            else:
                break
    elif kicid:
        this_id = np.int64(kicid)

    # Retrieve the LC data.
    rd, errflag = get_all_quarters_lc_data(this_id)
    if len(rd) < 1 or errflag:
        lcflag = True
        blacklist.append(this_id)
        LOGERROR('Error getting LC data. KICID-{:s}'.format(
            str(this_id)))
    else:
        lcflag = False

    return rd, lcflag, blacklist


def retrieve_injrecov_lc(kicid=None):
    '''
    Given the KIC ID parsed from /data/N_to_KICID.txt, get lightcurve data.
    '''
    assert isinstance(kicid, int)
    kicid = np.int64(kicid)

    rd, errflag = get_all_quarters_lc_data(kicid)
    if len(rd) < 1 or errflag:
        lcflag = True
        LOGERROR('Error getting LC data. KICID-{:s}'.format(str(kicid)))
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
    keb_path = DATADIR+'kebc_v3_170611.csv'
    cols = 'KIC,period,period_err,bjd0,bjd0_err,pdepth,sdepth,pwidth,swidth,'+\
           'sep,morph,GLon,GLat,kmag,Teff,SC,'
    cols = tuple(cols.split(','))
    tab = ascii.read(keb_path,comment='#')
    currentcols = tab.colnames
    for ix, col in enumerate(cols):
        tab.rename_column(currentcols[ix], col)

    tab.remove_column('') # now table has correct features

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

    lcdir = DATADIR+'morph_gt0.6_OR_period_lt3_kepler_MAST/'

    fs = [lcdir+f for f in os.listdir(lcdir) if f.endswith('.fits') and
            str(kicid) in f]

    rd = {}
    for fits_path in fs:
        try:
            lcd = astrokep.read_kepler_fitslc(fits_path)
        except:
            LOGERROR('{:s} failed in read_kepler_fitslc, {:s}. Escaping.'\
                .format(str(lcd['objectinfo']['keplerid']), str(fits_path)))
            break
        quarter_number = np.unique(lcd['quarter'])
        assert len(quarter_number)==1, 'Expect each fits file to correspond '+\
            ' to a given quarter'
        lcd['kebwg_info'], errflag = \
            get_kebwg_info(lcd['objectinfo']['keplerid'])
        if errflag:
            LOGERROR('{:s} gave errflag for {:s}'.format(
                str(lcd['objectinfo']['keplerid']), str(fits_path)))
            break
        rd[int(quarter_number)] = lcd

    return rd, errflag


def get_kebwg_info(kicid):
    '''
    Given a KIC ID, get the EB period reported by the Kepler Eclipsing Binary
    Working Group in v3 of their catalog.
    '''
    keb_path = DATADIR+'kebc_v3_170611.csv'
    #fast read
    f = open(DATADIR+'kebc_v3_170611.csv', 'r')
    ls = f.readlines()
    thisentry = [l for l in ls if l.startswith(str(kicid))]
    if len(thisentry) == 1:
        errflag = False
    else:
        LOGERROR('{:s} gave too many (or not enough) entries'.format(
            str(kicid)))
        errflag = True

    cols = 'KIC,period,period_err,bjd0,bjd0_err,pdepth,sdepth,pwidth,swidth,'+\
           'sep,morph,GLon,GLat,kmag,Teff,SC,'
    cols = cols.split(',')
    thesevals = thisentry.pop().split(',')[:-1]

    kebwg_info = dict(zip(cols, thesevals))

    return kebwg_info, errflag


#####################
# UTILITY FUNCTIONS #
#####################

def _legendre_dtr(times, fluxs, errs, legendredeg=10):

    try:
        p = Legendre.fit(times, fluxs, legendredeg)
        fitfluxs = p(times)
    except:
        fitfluxs = np.zeros_like(fluxs)

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


def detrend_allquarters(lcd, σ_clip=None, inj=False):
    '''
    Wrapper for detrend_lightcurve that detrends all the quarters of Kepler
    data passed in `lcd`, a dictionary of dictionaries, keyed by quarter
    numbers.
    "Detrend" means: select finite, sigma-clipped, not-near-gaps-in-timeseries
    points, and fit each resulting time group with a variable-order finite
    Legendre series.
    If `errflag` is raised in `detrend_lightcurve`, this quarter gets skipped
    (this was required because of an exception raised in a quarter with a
    single good time).
    '''

    rd = {}
    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    LOGINFO('Beginning detrend. KIC {:s}'.format(str(keplerid)))

    for k in lcd.keys():
        tempdict, errflag = detrend_lightcurve(lcd[k], k, σ_clip=σ_clip,
                inj=inj)
        if not errflag:
            rd[k] = tempdict
        LOGINFO('KIC ID %s, detrended quarter %s.'
            % (str(lcd[k]['objectinfo']['keplerid']), str(k)))

    return rd


def detrend_lightcurve(lcd, qnum, detrend='legendre', σ_clip=None, inj=False):
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

        detrend (str): method by which to detrend the LC. 'legendre' is the
        only thing implemented.

        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries

        inj (bool): whether or not the passed LC has injected transits. This
        changes which fluxes are detrended.

    Returns:
        lcd (dict) and errflag (bool).

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

        errflag (bool): boolean error flag.
    '''

    assert detrend == 'legendre'

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
        try:
            sel = (tgtimes > npmin(tgtimes)+mingap) & \
                     (tgtimes < npmax(tgtimes)-mingap)
        except ValueError:
            # If tgtimes is empty, continue to next timegroup.
            continue
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
    try:
        ssapfluxs = ssapfluxs[(ssaptimes>(npmin(ssaptimes)+mingap)) & \
                              (ssaptimes<(npmax(ssaptimes)-mingap))]
    except:
        # Case: ssaptimes is WONKY, all across this quarter.
        LOGERROR('DETREND FAILED, qnum {:d}'.format(qnum))
        return np.nan, True
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
            mingap = 0.5 # days
            ngroups, groups = lcmath.find_lc_timegroups(times, mingap=mingap)
            tmpfluxslegfit, legdegs = [], []
            for group in groups:
                tgtimes = times[group]
                tgfluxs = fluxs[group]
                tgerrs  = errs[group]

                legdeg = _get_legendre_deg_time(len(tgtimes))
                tgfluxslegfit, _, _ = _legendre_dtr(tgtimes,tgfluxs,tgerrs,
                        legendredeg=legdeg)

                tmpfluxslegfit.append(tgfluxslegfit)
                legdegs.append(legdeg)

            fitfluxs = nparr([])
            for ix, _ in enumerate(tmpfluxslegfit):
                fitfluxs = np.append(fitfluxs, tmpfluxslegfit[ix])

        dtr[k] = {'times':times,
                  'fluxs':fluxs,
                  'fitfluxs_'+detrend:fitfluxs,
                  'errs':errs
                 }

    lcd['dtr'] = dtr

    return lcd, False


def normalize_lightcurve(lcd, qnum, dt='dtr'):
    '''
    Once detrended fits are computed, this function computes the residuals, and
    also expresses them in normalized flux units, saving the keys to
    `lcd['dtr']['sap']['*_rsdl']`.

    Args:
        dt = 'dtr'
    '''

    for ap in ['sap']:

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


def _get_legendre_deg_time(npts):
    from scipy.interpolate import interp1d

    degs = np.array([2,4,5,6,10,20])
    pts = np.array([5e1,1e2,3e2,5e2,1e3,2e3])
    fn = interp1d(pts, degs, kind='linear',
                 bounds_error=False,
                 fill_value=(min(degs), max(degs)))

    if len(npts)==1:
        legendredeg = int(np.floor(fn(npts)))
    else:
        legendredeg = list(map(int,np.floor(fn(npts))))

    return legendredeg


def _get_legendre_deg_phase(npts, norbs):
    from scipy.interpolate import interp1d

    if norbs > 10:
        # better phase-coverage means you can raise the fit order
        degs = np.array([5,15,40,50])
        pts = np.array([1e2,5e2,2e3,3e3])
    else:
        degs = np.array([4,7,15,22,30])
        pts = np.array([1e2,5e2,1e3,2e3,3e3])
    fn = interp1d(pts, degs, kind='linear',
                 bounds_error=False,
                 fill_value=(min(degs), max(degs)))

    if len(npts)==1:
        legendredeg = int(np.floor(fn(npts)))
    else:
        legendredeg = list(map(int,np.floor(fn(npts))))

    return legendredeg


def _iter_run_periodogram(dat, qnum, inum=0, ap='sap', fine=False,
        dynamical_prefactor=2.01, nworkers=None):

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
        fluxs = dat['white'][inum-1][ap]['legdict']['whiteseries']['wfluxsresid']
        errs =  dat['white'][inum-1][ap]['legdict']['whiteseries']['errs']

    if len(times) < 50 or len(fluxs) < 50:
        LOGERROR('Got quarter with too few points. Continuing.')
        dat['white'][inum][ap][pgkey] = np.nan
    else:
        if not fine:
            # Range of interesting periods (morph>0.6): 0.05days(1.2hr)-20days.
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
            stepsize=1e-5
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
            nworkers=nworkers)

        if isinstance(pgd, dict):
            LOGINFO('KIC ID %s, computed periodogram (inum %s) quarter %s. (%s)'
                % (str(dat['objectinfo']['keplerid']), str(inum), str(qnum), ap))
        else:
            LOGERROR('Error in KICID %s, periodogram %s, quarter %s (%s)'
                % (str(dat['objectinfo']['keplerid']), str(inum), str(qnum), ap))

        dat['white'][inum][ap][pgkey] = pgd

    return dat


def iterative_whiten_allquarters(lcd, σ_clip=[30.,5.], nwhiten_max=10,
        nwhiten_min=2, rms_floor=0.1, nworkers=None):
    '''
    Wrapper to iterative_whiten_lightcurve, to run for all quarters. NOTE that
    this doesn't make whiten_allquarters redundant because the data structures
    are all different (because I unsurprisingly didn't make the dictionaries
    flexible enough to begin with). However, it practice whiten_allquarters
    will just never be used (since the iterative data structure is better, and
    all plot changes were made to not be backwards compatible).

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
                rms_floor=rms_floor,
                nworkers=nworkers)

    return rd


def iterative_whiten_lightcurve(dat, qnum, method='legendre',
        legendredeg='best', rescaletomedian=True, σ_clip=None, nwhiten_max=10,
        nwhiten_min=1, rms_floor=0.001, mingap=0.5, nworkers=None):
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

        mingap = 0.5 (float): the minimum number of days between adjacent
        points to qualify them as separate timegroups (nb. separate timegroups
        get culled b/c of variability near edges).

        rescaletomedian (bool): rescales the whitened fluxes to their median
        value.

        legendredeg: integer to be fixed, else (str) "best".

        σ_clip (float or list): to pass to astrobase.lcmath.sigclip_magseries.

        rms_floor (float): unbiased RMS [%] at which iterative whitening can
        stop. E.g., set to 0.1 to stop at 0.1%.

        nwhiten_max (int): maximum number of times to iterate the whitening
        process.

        nwhiten_min (int): converse of nwhiten_max. Must be >=1.

        nworkers (int): number of workers.

    Returns:
        dat (dict): dat, with the phased times, fluxes and errors in a
        sub-dictionary, accessible as dat['white'], which gives the
        dictionary:

        dat['white'] is keyed by number of whitenings that have been applied
        (1,2,3,...,nwhiten_max). Each one of these keys, `inum`, is keyed by
        sap/pdc, and contains:

        dat['white'][inum]['sap'].keys():
            ['magseries', 'whiteseries', 'fitinfo', 'data_rms', 'resid_rms',
            'white_rms', 'per', # the coarse periodogram for this iteation
            'fineper', # the fine periodogram for this iteation
            'fittype', 'fitplotfile', 'fitchisq', 'fitredchisq'].

        data_rms is the RMS of the data used in this iterationnumber (inum).
        white_rms is the RMS of the residual once the phase-fit is subtracted.
        resid_rms is the RMS of the residual once the phase & time-fits are
            subtracted (in time, it's a "high pass filter").

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

    fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
    sap_rms = np.sqrt(np.sum((fluxs-np.mean(fluxs))**2)/\
                        (float(len(fluxs))-1))

    nwhiten = 0 # number of whitenings that have been done

    while (sap_rms>rms_floor or nwhiten<nwhiten_min) and nwhiten<=nwhiten_max:

        dat['white'][nwhiten] = {}

        for ap in ['sap']:
            # Be sure to whiten at the EB period first.
            whiten_at_eb_period = True if nwhiten == 0 else False

            # Run coarse periodogram and get initial period guess.
            dat = _iter_run_periodogram(dat, qnum, inum=nwhiten, ap=ap,
                    fine=False, nworkers=nworkers)
            dat = _select_whiten_period(dat, fine=False, inum=nwhiten, ap=ap,
                    want_eb_period=whiten_at_eb_period)
            # Run fine periodogram and get precise period at which to whiten.
            dat = _iter_run_periodogram(dat, qnum, inum=nwhiten, ap=ap,
                    fine=True, nworkers=nworkers)
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
                fluxs = dat['white'][nwhiten-1][ap]['legdict']['whiteseries']['wfluxsresid']
                errs =  dat['white'][nwhiten-1][ap]['legdict']['whiteseries']['errs']

            if legendredeg=='best':
                npts = len(fluxs)
                duty_cycle = 0.8 # as an estimate, 80% of the data are out
                norbs = int(duty_cycle*(max(times)-min(times))/period)
                legdeg = _get_legendre_deg_phase(npts, norbs)
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
                # Errors to be parsed from LOG files.
                LOGERROR('Whitening error. KIC %s, qnum %s, (%s, inum %s).'
                    % (str(dat['objectinfo']['keplerid']), str(qnum), ap,
                       str(nwhiten)))
                continue

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
            legdict['white_rms'] = w_rms

            # Fit a low-order polynomial (as always, in the form of a finite
            # Legendre series) to each timegroup.
            ngroups, groups = lcmath.find_lc_timegroups(wtimes, mingap=mingap)
            tmpwfluxslegfit, legdegs = [], []
            for group in groups:
                tgtimes = wtimes[group]
                tgfluxs = wfluxs[group]
                tgerrs  = werrs[group]

                legdeg = _get_legendre_deg_time(len(tgtimes))
                tgwfluxslegfit, _, _ = _legendre_dtr(tgtimes,tgfluxs,tgerrs,
                        legendredeg=legdeg)

                tmpwfluxslegfit.append(tgwfluxslegfit)
                legdegs.append(legdeg)

            wfluxslegfit = nparr([])
            for ix, _ in enumerate(tmpwfluxslegfit):
                wfluxslegfit = np.append(wfluxslegfit, tmpwfluxslegfit[ix])

            LOGINFO('Got n={:s} legendre fit, ({:s}, {:s}, inum {:s})'.format(
                   ','.join(list(map(str,legdegs))),
                   str(dat['objectinfo']['keplerid']),
                   str(qnum),
                   ap,
                   str(nwhiten)))

            wfluxshighpass = wfluxs - wfluxslegfit
            meanwfluxhp = np.mean(wfluxshighpass)
            whp_rms = np.sqrt(np.sum((wfluxshighpass-meanwfluxhp)**2)/\
                        (float(len(wfluxshighpass))-1))
            if ap == 'sap':
                sap_rms = whp_rms
            legdict['resid_rms'] = w_rms

            # Save whitened times, fluxes, and errs to 'whiteseries' subdict.
            whitedict = {'times':wtimes,
                    'phase': wphase,
                    'wfluxs':wfluxs, # the whitened fluxes
                    'wfluxslegfit':wfluxslegfit, # a legendre fit to wfluxs
                    'wfluxsresid':wfluxshighpass, # wfluxs-wfluxslegfit
                    'legdegs':legdegs, # degrees for timegroups in legendre fit
                    'errs':werrs}

            legdict['whiteseries'] = whitedict

            dat['white'][nwhiten][ap]['legdict'] = legdict

        nwhiten += 1
        LOGINFO('nwhiten (inum) {:d}, sap_rms {:.3g}, rms_floor {:.3g}'.format(
            nwhiten, sap_rms, rms_floor))

    return dat


def find_dips(lcd, allq, method='bls', nworkers=None):
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
        ["dipfind"][method] = periodogram dictionary.
    '''

    #Concatenate all the quarters when running dip-finder.
    qnums = np.sort(list(lcd.keys()))
    tfe = {}
    for ap in ['sap']:
        for ix, qnum in enumerate(qnums):
            max_inum = np.max(list(lcd[qnum]['white'].keys()))
            lc = lcd[qnum]['white'][max_inum][ap]['legdict']['whiteseries']
            if ix == 0:
                times = lc['times']
                fluxs = lc['wfluxsresid']
                errs = lc['errs']
                quarter = np.ones_like(times)*qnum
            else:
                times = np.append(times, lc['times'])
                fluxs = np.append(fluxs, lc['wfluxsresid'])
                errs = np.append(errs, lc['errs'])
                quarter = np.append(quarter, np.ones_like(lc['times'])*qnum)

        if quarter[quarter==0].size > 0:
            # If quarter 0 has much worse RMS than other quarters, drop it.

            m = (quarter == 0)
            rms_q0 = np.sqrt(np.sum((fluxs[m]-np.mean(fluxs[m]))**2)/\
                                (float(len(fluxs[m]))-1))
            rms_otherquarters = \
                     np.sqrt(np.sum((fluxs[~m]-np.mean(fluxs[~m]))**2)/\
                                (float(len(fluxs[~m]))-1))

            import IPython; IPython.embed()
            if rms_q0 > 2*rms_otherquarters:
                times = times[~m]
                fluxs = fluxs[~m]
                errs = errs[~m]
                quarter = quarter[~m]

        tfe[ap] = {'times':times,
                   'fluxs':fluxs,
                   'errs':errs,
                   'qnums':quarter}

    # Range of interesting periods determined by Holman & Weigert (1999)
    # stability criterion on the inner edge. On the outer, note BLS can search
    # for periods < half the observed baseline.
    keplerid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])
    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])

    smallfactor, bigfactor = 2.02, None # slightly above 2*P_EB to avoid harmonic
    smallest_p = smallfactor * kebc_period
    biggest_p = 150. # days
    minTdur_φ = 0.0025 # minimum transit length in phase
    maxTdur_φ = 0.25 # maximum transit length in phase

    df_dict = {}
    for ap in ['sap']:

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
            nbestpeaks=30, # large number now, filter for positive dips after
            periodepsilon=0.1, # 0.1 days btwn period peaks to be distinct.
            nworkers=nworkers,
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
        # good epochs. The same applies when we want to impose only positive
        # dips (for dimmings of the star) or want to get depths of nbestperiods
        ntokeep, ix = 5, 0
        while len(df_dict[ap]['finebls']) < ntokeep:
            if ix == len(nbestperiods):
                LOGERROR('Could not find enough positive dips. Breaking out.')
                break
            nbestperiod = nbestperiods[ix]
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
                    '\nstart {:.6g} d, end {:.6g} d. {:d} phasebins.'.format(
                    startp, endp, nphasebins)+\
                    '\ncurrent len df_dict: {:d}, ix: {:d}'.format(
                        len(df_dict[ap]['finebls']), ix))

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

            # ensure that at the end of the day, we keep only positive dips.
            if sd['blsresult']['transdepth'] < 0:
                LOGINFO('Skip P={:.4g} (transdepth<0). ix={:d}'.format(
                            nbestperiod, ix))
                ix += 1
                continue

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
            if ingbinind < len(φ_bin) and egrbinind < len(φ_bin):
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

            df_dict[ap]['finebls'][nbestperiod]['δ'] = sd['blsresult']['transdepth']
            df_dict[ap]['finebls'][nbestperiod]['φ'] = phasedlc['phase']
            df_dict[ap]['finebls'][nbestperiod]['flux_φ'] = phasedlc['mags']
            df_dict[ap]['finebls'][nbestperiod]['binned_φ'] = φ_bin
            df_dict[ap]['finebls'][nbestperiod]['binned_flux_φ'] = flux_φ_bin
            df_dict[ap]['finebls'][nbestperiod]['φ_ing'] = φ_ing
            df_dict[ap]['finebls'][nbestperiod]['φ_egr'] = φ_egr
            df_dict[ap]['finebls'][nbestperiod]['φ_0'] = φ_0

            ix += 1

        LOGINFO('KICID: {:s}. Finished finebls ({:s}) ap:{:s}'.format(
                keplerid, method.upper(), ap.upper()))

    allq['dipfind'] = {}
    allq['dipfind'][method] = df_dict
    allq['dipfind']['tfe'] = tfe

    return allq


def save_lightcurve_data(lcd, allq=None, stage=False, tossiterintermed=True):
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

        tossiterintermed (bool): whether to look for, and throw out, all
        intermediate data from the whitening (periodograms and timeseries
        copies). This saves a factor of ~2-4x in hard disk memory
        (100Mb->~25Mb). However, significant intermediate steps should as a
        matter of principle be kept unless storage space prohibits it.
    '''

    kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

    if stage:
        pklname = kicid+'_'+stage+'.p'
        pklallqname = kicid+'_allq_'+stage+'.p'
    else:
        pklname = kicid+'.p'

    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    else:
        predir += 'real/'
    spath = DATADIR+'injrecov_pkl/'+predir+pklname
    sallqpath = DATADIR+'injrecov_pkl/'+predir+pklallqname

    if 'eb_sbtr' in stage:
        #override; save pickles somewhere else for EB subtraction tests.
        spath = DATADIR+'eb_subtr_pkl/'+pklname
        sallqpath = DATADIR+'eb_subtr_pkl/'+pklallqname

    if tossiterintermed:
        for qnum in list(lcd.keys()):
            iterkeys = list(lcd[qnum]['white'].keys())
            for inum in list(range(min(iterkeys)+1,max(iterkeys))):
                # This leaves the first and last iterations of whitening. The
                # last one is what the plots and dipsearch actually need.
                del lcd[qnum]['white'][inum]


    pickle.dump(lcd, open(spath, 'wb'))
    LOGINFO('Saved (pickled) lcd data to %s' % spath)
    if isinstance(allq, dict):
        pickle.dump(allq, open(sallqpath, 'wb'))
        LOGINFO('Saved (pickled) allquarter data to %s' % sallqpath)

    return kicid


def load_lightcurve_data(kicid, stage=None, δ=None, datapath=None):
    '''
    Args:
        kicid (int): Kepler Input Catalog ID number
        stage (str): e.g., "dipsearch_0.01". Read `src/run_inj_recov.py` to get
        examples.

    Returns:
        the LC data, and a boolean flag for whether the load failed/succeeded.
    '''

    # `stage` includes the transit depth string
    poststr = '_real' if datapath else ''
    pklname = str(kicid)+'_'+stage+poststr+'.p'
    if δ == 'whatever':
        pklname = str(kicid)+'_'+stage+'_0.00125.p'

    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    else:
        predir += 'real/'

    if not datapath:
        prepath = DATADIR+'injrecov_pkl/'+predir
    else:
        prepath = datapath

    lpath = prepath+pklname

    if 'eb_sbtr' in stage:
        #saved pickles somewhere else for EB subtraction tests.
        lpath = DATADIR+'eb_subtr_pkl/'+pklname

    try:
        dat = pickle.load(open(lpath, 'rb'))
        LOGINFO('Loaded pickled data from %s' % lpath)
        return dat, False
    except:
        LOGERROR('Trying to load from %s failed. Continue.' % lpath)
        return np.nan, True


def load_allq_data(kicid, stage=None, δ=None, datapath=None):

    poststr = '_real' if datapath else ''
    pklname = str(kicid)+'_allq_'+stage+poststr+'.p'
    if δ == 'whatever':
        pklname = str(kicid)+'_'+stage+'_0.00125.p'

    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    else:
        predir += 'real/'

    if not datapath:
        prepath = DATADIR+'injrecov_pkl/'+predir
    else:
        prepath = datapath
    lpath = prepath+pklname

    if 'eb_sbtr' in stage:
        #saved pickles somewhere else for EB subtraction tests.
        lpath = DATADIR+'eb_subtr_pkl/'+pklname

    try:
        allq = pickle.load(open(lpath, 'rb'))
        LOGINFO('Loaded allquarter pickled data from %s' % lpath)
        return allq, False
    except:
        LOGERROR('Trying to load from %s failed. Continue.' % lpath)
        return np.nan, True
