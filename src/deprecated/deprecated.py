'''
where old functions go to die.
'''

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

    for ap in ['sap']:

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

    for ap in ['sap']:

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
        for ap in ['sap']:

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

    for ap in ['sap']:

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
