'''
Short period search:
    Injection/recovery of planet transits on the P<1 day Kepler Eclipsing
    Binary Catalog entries.
'''

import os
import pdb
from astrobase import astrokep, periodbase
import numpy as np, matplotlib.pyplot as plt
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum
from numpy.polynomial.legendre import Legendre, legval

from astropy.io import ascii
import logging
from datetime import datetime

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



###########
# SANDBOX #
###########
def single_lc_check():

    datdir = '../data/morph_gt0.6_kepler_MAST/trial/'
    fs = [datdir+f for f in os.listdir(datdir) if f.endswith('.fits')]
    lcd = astrokep.read_kepler_fitslc(fs[0])

    time = lcd['time']
    sap_flux = lcd['sap_flux']
    sap_flux_err = lcd['sap_flux_err']

    time = lcd['time']
    pdc_flux = lcd['pdcsap_flux']
    pdc_flux_err = lcd['pdcsap_flux_err']




##############
# UNIT TESTS #
##############
def test_retrieve_lcs():
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



######################
# PLOTTING UTILITIES #
######################
def orosz_style_flux_vs_time(lcdat, flux_to_use='sap'):
    '''
    Make a plot in the style of Fig S4 of Orosz et al. (2012) showing Kepler
    data, colored by quarter.

    Args:
    lcdat (dict): dictionary with keys of quarter number for a single object.
        The value of the keys are the lightcurve dictionaries returned by
        `astrokep.read_kepler_fitslc`. The lightcurves have been detrended,
        using `detrend_lightcurve`, and normalized.

    flux_to_use (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    Returns: nothing, but saves the plot with a smart name to
        ../results/colored_flux_vs_time/
    '''

    assert flux_to_use == 'sap' or flux_to_use == 'pdc'

    keplerid = lcdat[list(lcdat.keys())[0]]['objectinfo']['keplerid']

    colors = ['r', 'g', 'b', 'k']
    f, axs = plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True)

    for ix, quarter_number in enumerate(lcdat.keys()):

        for axix, ax in enumerate(axs):

            lc = lcdat[quarter_number]['dtr'][flux_to_use]

            if axix == 0:
                times = lc['times']
                fluxs = lc['fluxs']
                errs = lc['errs']

            elif axix == 1:
                times = lc['times']
                fluxs = lc['fluxs_dtr_norm']
                errs = lc['errs_dtr_norm']

            thiscolor = colors[ix%len(colors)]

            ax.plot(times, fluxs, c=thiscolor, linestyle='-',
                    marker='o', markerfacecolor=thiscolor,
                    markeredgecolor=thiscolor, ms=0.1, lw=0.1)
            if axix == 0:
                fitfluxs = lc['fitfluxs_legendre']
                ax.plot(times, fitfluxs, c='k', linestyle='-',
                    lw=0.5)

        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = npmin(times)
            max_time = npmax(times)
        elif ix > 0:
            if npmin(times) < min_time:
                min_time = npmin(times)
            if npmax(times) > max_time:
                max_time = npmax(times)


    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time

    axs[0].set(xlabel='', ylabel=flux_to_use+' Flux [counts/s]',
            xlim=[min_time-timelen*0.03,max_time+timelen*0.03],
            title='KIC ID {:s}, {:s}, quality flag > 0.'.\
            format(str(keplerid), flux_to_use))
    axs[1].set(xlabel='Time (BJD something)',
            ylabel='Normalized flux',
            title='Detrended (n=10 Legendre series), normalized by median.')

    f.tight_layout()

    savedir = '../results/colored_flux_vs_time/'
    plotname = str(keplerid)+'_'+flux_to_use+'_os.png'
    f.savefig(savedir+plotname, dpi=300)


#####################
# UTILITY FUNCTIONS #
#####################

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



def detrend_allquarters(lcd):
    '''
    Wrapper for detrend_lightcurve that detrends all the quarters of Kepler
    data passed in `lcd`, a dictionary of dictionaries, keyed by quarter
    numbers.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = detrend_lightcurve(lcd[k])
        LOGINFO('KIC ID %s, detrended quarter %s.'
            % (str(lcd[k]['objectinfo']['keplerid']), str(k)))

    return rd



def detrend_lightcurve(lcd, detrend='legendre', legendredeg=10, polydeg=2):
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

    nafter = fsaptimes.size
    LOGINFO('for quality flag filter (SAP), ndet before = %s, ndet after = %s'
            % (nbefore, nafter))

    pdcfluxs = lcd['pdcsap_flux'][lcd['sap_quality'] == 0]
    pdcerrs = lcd['pdcsap_flux_err'][lcd['sap_quality'] == 0]
    find = npisfinite(times) & npisfinite(pdcfluxs) & npisfinite(pdcerrs)
    fpdctimes, fpdcfluxs, fpdcerrs = times[find], pdcfluxs[find], pdcerrs[find]

    nafter = fpdctimes.size
    LOGINFO('for quality flag filter (PDC), ndet before = %s, ndet after = %s'
            % (nbefore, nafter))


    #DETREND: fit a legendre series or polynomial, save it to the output
    #dictionary.

    tfe = {'sap':(fsaptimes, fsapfluxs, fsaperrs),
           'pdc':(fpdctimes, fpdcfluxs, fpdcerrs)}
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





    '''
    Quoting Armstrong et al. (2014):
    ```We elected to detrend the light curves from instrumental and sys-
    tematic effects using covariance basis vectors. These were used over
    the Presearch Data Conditioning (PDC) detrended data available as
    the PDC data is not robust against long-duration events, as warned
    in Fanelli et al. (2011), and the transits of CB planets may in theory
    last for half the orbital period of the binary. While it would be ideal
    to individually tune the detrending of all light curves, the sample
    size made this impractical. Detrending was enacted using the PYKE
    code (Still & Barclay 2012). At this stage, data with a non-zero
    SAP_QUALITY flag was cut (see Fraquelli & Thompson 2012 for
    full list of exclusions). Once detrended, quarter data were stitched
    together through dividing by the median flux value of each quarter,
    forming single light curves for each binary.```
    '''

    '''
    Q1-Q12, took SAP from MAST. `orosz2012_kep47_dtr.png` is the thing to
    emulate. Doing detrending separately by quarter, tune the "aggressiveness" for the
    task. To model eclipses & transits, remove both instrumental trends and spot
    modulations -- to do this, mask out eclipses & transits, a fit a high order
    cubic spline to short segments whose end points are defined by gaps in the
    data collection (due to monthly data downlink, rolls btwn Quarters, or spacecrft
    safe modes). Normalize the segments to the spline fits, and reassemble the
    segments.
    '''



def normalize_lightcurve(lcd, qnum):
    '''
    Once detrended fits are computed, this function computes the residuals, and
    also expresses them in normalized flux units, saving the keys to
    `lcd['dtr']['sap']['*_rsdl']`.
    '''

    for ap in ['sap','pdc']:

        dtrtype = [k for k in list(lcd['dtr'][ap].keys()) if
            k.startswith('fitfluxs_')]
        assert len(dtrtype) == 1, 'Single type of fit assumed.'
        dtrtype = dtrtype.pop()

        flux = lcd['dtr'][ap]['fluxs']
        flux_norm = flux / np.median(flux)
        fitflux = lcd['dtr'][ap][dtrtype]
        fitflux_norm = fitflux / np.median(flux)

        flux_dtr_norm = flux_norm - fitflux_norm + 1

        errs = lcd['dtr'][ap]['errs']
        errs_dtr_norm =  errs / np.median(flux)

        lcd['dtr'][ap]['fluxs_dtr_norm'] = flux_dtr_norm
        lcd['dtr'][ap]['errs_dtr_norm'] = errs_dtr_norm

        LOGINFO('KIC ID %s, normalized quarter %s. (%s)'
            % (str(lcd['objectinfo']['keplerid']), str(qnum), ap))


    return lcd



def normalize_allquarters(lcd):
    '''
    Wrapper to normalize_lightcurve, to run for all quarters.
    Saves to keys `lcd[qnum]['dtr']['sap']['*_rsdl']`.
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = normalize_lightcurve(lcd[k], k)

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

        dat['per'] = pgd


    return dat




def whiten_allquarters(lcd):
    '''
    Wrapper to whiten_lightcurve, to run for all quarters.
    Saves whitened results to keys `lcd[qnum]['dtr']['sap']['w_*']`, for * in
    (fluxs,errs,times,phases)
    '''

    rd = {}
    for k in lcd.keys():
        rd[k] = whiten_lightcurve(lcd[k], k)

    return rd


def whiten_lightcurve(lcd):
    pass

def save_lightcurve_data(lcd):
    pass
