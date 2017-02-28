'''
Short period search:
    Injection/recovery of planet transits on the P<1 day Kepler Eclipsing
    Binary Catalog entries.
'''

import pickle, os, logging, pdb
from astropy.io import ascii
from datetime import datetime
from astrobase import astrokep, periodbase, lcmath
from astrobase.varbase import lcfit as lcf
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr
from numpy.polynomial.legendre import Legendre, legval

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

##############
# UNIT TESTS #
##############


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
    assert len(rd) > 1, 'failed for KIC ID {:s}'.format(str(kebc_kic_ids[ind]))

    return rd


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



def detrend_allquarters(lcd, σ_clip=None, legendredeg=10):
    '''
    Wrapper for detrend_lightcurve that detrends all the quarters of Kepler
    data passed in `lcd`, a dictionary of dictionaries, keyed by quarter
    numbers.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = detrend_lightcurve(lcd[k], σ_clip=σ_clip,
                legendredeg=legendredeg)
        LOGINFO('KIC ID %s, detrended quarter %s.'
            % (str(lcd[k]['objectinfo']['keplerid']), str(k)))

    return rd



def detrend_lightcurve(lcd, detrend='legendre', legendredeg=10, polydeg=2,
        σ_clip=None):
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
        σ_clip (float or list): to pass to astrobase.lcmath.sigmaclip_lc

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

    # GET finite, good-quality times, mags, and errs for both SAP and PDC.
    # Take data with non-zero saq_quality flags. Fraquelli & Thompson (2012),
    # or perhaps newer papers, give the list of exclusions (following Armstrong
    # et al. 2014). Nb. astrokep.filter_kepler_lcdict should do this, but is 
    # currently trying to do too many other things.

    nbefore = lcd['time'].size
    times = lcd['time'][lcd['sap_quality'] == 0]

    sapfluxs = lcd['sap_flux'][lcd['sap_quality'] == 0]
    saperrs = lcd['sap_flux_err'][lcd['sap_quality'] == 0]
    find = npisfinite(times) & npisfinite(sapfluxs) & npisfinite(saperrs)
    fsaptimes, fsapfluxs, fsaperrs = times[find], sapfluxs[find], saperrs[find]
    ssaptimes, ssapfluxs, ssaperrs = lcmath.sigmaclip_lc(
            fsaptimes, fsapfluxs, fsaperrs,
            isflux=True, sigclip=σ_clip)

    nafter = ssaptimes.size
    LOGINFO('for quality flag filter & sigclip (SAP), '+\
            'ndet before = %s, ndet after = %s'
            % (nbefore, nafter))

    pdcfluxs = lcd['pdcsap_flux'][lcd['sap_quality'] == 0]
    pdcerrs = lcd['pdcsap_flux_err'][lcd['sap_quality'] == 0]
    find = npisfinite(times) & npisfinite(pdcfluxs) & npisfinite(pdcerrs)
    fpdctimes, fpdcfluxs, fpdcerrs = times[find], pdcfluxs[find], pdcerrs[find]
    spdctimes, spdcfluxs, spdcerrs = lcmath.sigmaclip_lc(
            fpdctimes, fpdcfluxs, fpdcerrs,
            isflux=True, sigclip=σ_clip)


    nafter = fpdctimes.size
    LOGINFO('for quality flag filter & sigclip (PDC), '+\
            'ndet before = %s, ndet after = %s'
            % (nbefore, nafter))


    #DETREND: fit a legendre series or polynomial, save it to the output
    #dictionary.

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
    Once you have whitened fluxes, re-detrend (and re sigma-clip).

    Args:
        lcd (dict): the dictionary with everything, after whitening out the
        main eclipsing binary signal.
        detrend (str): method by which to detrend the LC. 'legendre' is the
        only thing currently implemented.
        σ_clip (float or list): to pass to astrobase.lcmath.sigmaclip_lc

    Returns:
        lcd (dict): lcd, with the redetrended times, magnitudes, and fluxes in a
        sub-dictionary, accessible as lcd['redtr'], which gives the
        dictionary:

            redtr = {
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

    assert detrend == 'legendre'

    lcd['redtr'] = {}
    for ap in ['sap', 'pdc']:

        times = lcd['white'][ap]['whiteseries']['times']
        nbefore = times.size
        # times, mags, and errs are already finite.
        fluxes = lcd['white'][ap]['whiteseries']['fluxes']
        errs = lcd['white'][ap]['whiteseries']['errs']

        stimes, sfluxes, serrs = lcmath.sigmaclip_lc(
                times, fluxes, errs,
                isflux=True, sigclip=σ_clip)


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
                  'fluxs':sfluxes,
                  'fitfluxs_'+detrend:fitfluxs,
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

        flux_dtr_norm = flux_norm - fitflux_norm + 1

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





def run_fineperiodogram_allquarters(lcd):
    '''
    Wrapper to run_periodogram, to run for all quarters after selected best
    period has been identified.
    Saves periodogram results to keys `lcd[qnum]['fineper']['sap']['*']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = run_fineperiodogram(lcd[k], k, 'pdm')

    return rd


def run_periodograms_allquarters(lcd):
    '''
    Wrapper to run_periodogram, to run for all quarters.
    Saves periodogram results to keys `lcd[qnum]['per']['sap']['*']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = run_periodogram(lcd[k], k, 'pdm')

    return rd


def run_fineperiodogram(dat, qnum, pertype='pdm'):
    '''
    See run_periodogram docstring. It's that, but for narrowing down the period
    of an EB once select_eb_period has been run.
    '''
    assert pertype=='pdm'

    dat['fineper'] = {}
    for ap in ['sap','pdc']:

        times = dat['dtr'][ap]['times']
        fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
        errs = dat['dtr'][ap]['errs_dtr_norm']

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



def run_periodogram(dat, qnum, pertype='pdm'):
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
    for ap in ['sap','pdc']:

        times = dat['dtr'][ap]['times']
        fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
        errs = dat['dtr'][ap]['errs_dtr_norm']

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



def whiten_allquarters(lcd, σ_clip=None):
    '''
    Wrapper to whiten_lightcurve, to run for all quarters.
    Saves whitened results to keys `lcd[qnum]['dtr']['sap']['w_*']`, for * in
    (fluxs,errs,times,phases)
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = whiten_lightcurve(lcd[k], k, σ_clip=σ_clip)

    return rd


def whiten_lightcurve(dat, qnum, method='legendre', legendredeg=80,
        rescaletomedian=True, σ_clip=None):
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

        σ_clip (float or list): to pass to astrobase.lcmath.sigmaclip_lc.

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

    for ap in ['sap', 'pdc']:

        period = dat['fineper'][ap]['selperiod']
        times = dat['dtr'][ap]['times']
        fluxs = dat['dtr'][ap]['fluxs_dtr_norm']
        errs = dat['dtr'][ap]['errs_dtr_norm']

        try:
            legdict = lcf.legendre_fit_magseries(
                times,fluxs,errs,period,
                legendredeg=legendredeg,
                sigclip=σ_clip,#doesn't support asymmetric but ok for now
                plotfit=False,
                isnormalizedflux=True)
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


def trim_near_gaps(lcd):
    '''
    '''

    pass



def save_lightcurve_data(dat, stage=False):
    '''
    Args:
        dat (anytype): the thing you want to write to a pickle file.

        stage (str): a short string that will be appended to the pickle file
        name to simplify subsequent reading. E.g., "pw" for "post-whitening",
        or anything along these terms.
    '''

    kicid = str(dat[list(dat.keys())[0]]['objectinfo']['keplerid'])
    pklname = kicid

    if stage:
        pklname = pklname+'_'+stage+'.p'
    else:
        pklname = pklname+'.p'

    spath = '../data/injrecov_pkl/'+pklname

    pickle.dump(dat, open(spath, 'wb'))
    LOGINFO('Saved data to %s' % spath)

    return kicid


def load_lightcurve_data(kicid, stage='pw'):

    pklname = str(kicid)+'_'+stage+'.p'
    lpath = '../data/injrecov_pkl/'+pklname

    dat = pickle.load(open(lpath, 'rb'))
    LOGINFO('Loaded data from %s' % lpath)

    return dat
