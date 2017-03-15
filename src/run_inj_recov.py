import numpy as np, os
import inj_recov as ir
import inj_recov_plots as irp
import injrecovresult_analysis as irra

def N_lc_fixedinjrecov(N,
        ors=False, whitened=True, ds=True,
        stage=None,
        inj=None):
    '''
    Run injection-recovery on N KEBC objects. There are two important objects:
    `lcd` organizes everything by quarter. `allq` stitches over all quarters.

    The current processing is as follows:

        inject a transit δ = 1% at (20-25)x the binary
            period->
        detrend (n=20 legendre and [30,30]σ sigclip)->
        normalize (median by quarter)->
        run periodograms (stellingwerf_pdm)->
        get nominal EB period->
        run finer periodogram (stellingwerf_pdm narrow window)->
        select a real EB period->
        "whiten" (phase-fold [30,5]σ sigclip, fit & subtract out fit)->
        redetrend (n=20 legendre, and [30,5]σ  sigclip)->
        normalize (median by quarter)->
        find dips (BLS, over all the quarters)

    stages (in order that they could be saved in):

        'pw' if post-whitening.
        'redtr' if post-redetrending.
        'ds' if up to recovering the dips.

    inj (bool): True if you're injecting (fixes names of things).

    ors (bool), whitened (bool), ds (bool): whether to create the Orosz,
    whitened 6row, and dipsearch plots respectively.
    '''

    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    if inj:
        stage += '_inj'
    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    elif 'inj' not in stage:
        predir += 'no_inj/'


    for s in seeds:
        np.random.seed(s)

        lcd, lcflag = ir.retrieve_random_lc()
        if lcflag:
            continue
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pklmatch = [f for f in os.listdir('../data/injrecov_pkl/'+predir) if
                f.endswith('.p') and f.startswith(kicid) and stage in f]

        if len(pklmatch) < 1:

            #INJECT TRANSITS AND PROCESS LIGHTCURVES
            lcd, allq = ir.inject_fixed_transit(lcd)
            lcd = ir.detrend_allquarters(lcd,
                    σ_clip=30., legendredeg=20, inj=inj)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=[30.,5.])
            if 'pw' in stage:
                kicid = ir.save_lightcurve_data(lcd, stage=stage)
            lcd = ir.redetrend_allquarters(lcd, σ_clip=[30.,5.], legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='redtr')
            if 'redtr' in stage:
                kicid = ir.save_lightcurve_data(lcd, stage=stage)
            allq = ir.find_dips(lcd, allq, method='bls')
            if 'dipsearch' in stage:
                kicid = ir.save_lightcurve_data(lcd, allq=allq, stage=stage)

        #LOAD STUFF
        lcd = ir.load_lightcurve_data(kicid, stage=stage)
        if 'dipsearch' in stage:
            allq = ir.load_allq_data(kicid, stage=stage)

        #WRITE RESULTS TABLES
        if 'dipsearch' in stage:
            if inj:
                ir.write_injrecov_result(lcd, allq, stage=stage)

        #MAKE PLOTS
        if ors:
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap', stage=stage)
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc', stage=stage)

        if ds:
            doneplots = os.listdir('../results/dipsearchplot/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound dipsearchplot, continuing.\n')
                continue

            if 'dipsearch' in stage:
                irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)
                irp.dipsearchplot(lcd, allq, ap='pdc', stage=stage, inj=inj)

        if whitened:
            doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound whitened_diagnostic, continuing.\n')
                continue

            if 'pw' in stage:
                irp.whitenedplot_5row(lcd, ap='sap', stage=stage)
                irp.whitenedplot_5row(lcd, ap='pdc', stage=stage)
            elif 'redtr' in stage:
                irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=inj)
            elif 'dipsearch' in stage:
                irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=inj)

    if inj:
        irra.summarize_injrecov_result()


def N_lc_injrecov(N,
        ors=False, whitened=True, ds=True,
        stage=None,
        inj=None):
    '''
    Run injection-recovery on N KEBC objects. There are two important objects:
    `lcd` organizes everything by quarter. `allq` stitches over all quarters.

    The current processing is as follows:

        inject a realistic transit signal at δ=(2,1,1/2,1/4,1/8,1/16,1/32)%
            depth, anywhere from P_CBP=(4-40)x P_EB.
        detrend (n=20 legendre and [30,30]σ sigclip)->
        normalize (median by quarter)->
        run periodograms (stellingwerf_pdm)->
        get nominal EB period->
        run finer periodogram (stellingwerf_pdm narrow window)->
        select a real EB period->
        "whiten" (phase-fold [30,5]σ sigclip, fit & subtract out fit)->
        redetrend (n=20 legendre, and [30,5]σ  sigclip)->
        normalize (median by quarter)->
        find dips (BLS, over all the quarters)

    stages (in order that they could be saved in):

        'pw' if post-whitening.
        'redtr' if post-redetrending.
        'ds' if up to recovering the dips.

    inj (bool): True if you're injecting (fixes names of things).

    ors (bool), whitened (bool), ds (bool): whether to create the Orosz,
    whitened 6row, and dipsearch plots respectively.
    '''

    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    if inj:
        stage += '_inj'
    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    elif 'inj' not in stage:
        predir += 'no_inj/'

    origstage = stage

    for s in seeds:
        np.random.seed(s)

        lcd, lcflag = ir.retrieve_random_lc()
        if lcflag:
            continue
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        #INJECT TRANSITS AND PROCESS LIGHTCURVES
        δarr = np.array([2.,1.,1/2.,1/4.,1/8.,1/16.])/100.

        for δ in δarr:
            stage = origstage + '_' + str(δ)
            pklmatch = [f for f in os.listdir('../data/injrecov_pkl/'+predir) if
                    f.endswith('.p') and f.startswith(kicid) and stage in f]
            if len(pklmatch) > 0:
                print('Found {:s}, {:f}, continue'.format(kicid, δ))
                continue
            lcd, allq = ir.inject_transit_known_depth(lcd, δ)
            lcd = ir.detrend_allquarters(lcd,
                    σ_clip=30., legendredeg=20, inj=inj)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=[30.,5.])
            if 'pw' in stage:
                kicid = ir.save_lightcurve_data(lcd, stage=stage)
            lcd = ir.redetrend_allquarters(lcd, σ_clip=[30.,5.], legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='redtr')
            if 'redtr' in stage:
                kicid = ir.save_lightcurve_data(lcd, stage=stage)
            allq = ir.find_dips(lcd, allq, method='bls')
            if 'dipsearch' in stage:
                kicid = ir.save_lightcurve_data(lcd, allq=allq, stage=stage)


        for δ in δarr:
            stage = origstage + '_' + str(δ)
            #LOAD STUFF
            lcd = ir.load_lightcurve_data(kicid, stage=stage)
            if 'dipsearch' in stage:
                allq = ir.load_allq_data(kicid, stage=stage)

            #WRITE RESULTS TABLES
            if 'dipsearch' in stage:
                if inj:
                    irra.write_injrecov_result(lcd, allq, stage=stage)

            #MAKE PLOTS
            if ors:
                irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap', stage=stage)
                irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc', stage=stage)

            if ds:
                doneplots = os.listdir('../results/dipsearchplot/'+predir)
                plotmatches = [f for f in doneplots if f.startswith(kicid) and
                        stage in f]
                if len(plotmatches)>0:
                    print('\nFound dipsearchplot, continuing.\n')
                    continue

                if 'dipsearch' in stage:
                    irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)
                    irp.dipsearchplot(lcd, allq, ap='pdc', stage=stage, inj=inj)

            if whitened:
                doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
                plotmatches = [f for f in doneplots if f.startswith(kicid) and
                        stage in f]
                if len(plotmatches)>0:
                    print('\nFound whitened_diagnostic, continuing.\n')
                    continue

                if 'pw' in stage:
                    irp.whitenedplot_5row(lcd, ap='sap', stage=stage)
                    irp.whitenedplot_5row(lcd, ap='pdc', stage=stage)
                elif 'redtr' in stage:
                    irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                    irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=inj)
                elif 'dipsearch' in stage:
                    irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                    irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=inj)

        if inj:
            irra.summarize_injrecov_result()

def test_EB_subraction(N,
        ors=False, whitened=True, ds=True,
        stage=None,
        inj=None):
    '''
    Calls routines for testing and improving subtraction of the eclipsing
    binary signal.

    Call this with stage (str) set to 'eb_sbtr'.
    '''

    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    if inj:
        stage += '_inj'
    predir = ''
    if 'inj' in stage:
        predir += 'inj/'
    elif 'inj' not in stage:
        predir += 'no_inj/'

    origstage = stage

    for s in seeds:
        np.random.seed(s)

        lcd, lcflag = ir.retrieve_random_lc()
        if lcflag:
            continue
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        #INJECT TRANSITS AND PROCESS LIGHTCURVES
        δarr = np.array([1/2.,1/4.])/100.
        δ = δarr[0] # nominal 0.5% transit injection. #TODO could generalize.

        stage = origstage + '_' + str(δ)
        pklmatch = [f for f in os.listdir('../data/eb_subtr_pkl') if
                f.endswith('.p') and f.startswith(kicid) and stage in f]
        if len(pklmatch) > 0:
            print('Found {:s}, {:f}, continue'.format(kicid, δ))
            continue
        lcd, allq = ir.inject_transit_known_depth(lcd, δ)
        lcd = ir.detrend_allquarters(lcd,
                σ_clip=30., legendredeg=20, inj=inj)
        lcd = ir.normalize_allquarters(lcd, dt='dtr')
        lcd = ir.run_periodograms_allquarters(lcd)
        lcd = ir.select_eb_period(lcd, fine=False)
        lcd = ir.run_fineperiodogram_allquarters(lcd)
        lcd = ir.select_eb_period(lcd, fine=True)
        #lcd = ir.whiten_allquarters(lcd, σ_clip=[30.,5.])
        #FIXME: implement
        lcd = ir.iterative_whiten_allquarters(lcd, σ_clip=[30.,5.], n_iter=5)
        if 'pw' in stage:
            kicid = ir.save_lightcurve_data(lcd, stage=stage)
        lcd = ir.redetrend_allquarters(lcd, σ_clip=[30.,5.], legendredeg=20)
        lcd = ir.normalize_allquarters(lcd, dt='redtr')
        if 'redtr' in stage or 'eb_sbtr' in stage:
            kicid = ir.save_lightcurve_data(lcd, stage=stage)

        # LOADING AND PLOTTING
        stage = origstage + '_' + str(δ)
        lcd = ir.load_lightcurve_data(kicid, stage=stage)
        if 'dipsearch' in stage or 'eb_sbtr' in stage:
            allq = ir.load_allq_data(kicid, stage=stage)

        # Write results tables for injection&recovery.
        if 'dipsearch' in stage:
            if inj:
                irra.write_injrecov_result(lcd, allq, stage=stage)

        # Make plots.
        if ors:
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap', stage=stage)
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc', stage=stage)

        if ds:
            doneplots = os.listdir('../results/dipsearchplot/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound dipsearchplot, continuing.\n')
                continue
            if 'dipsearch' in stage:
                irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)
                irp.dipsearchplot(lcd, allq, ap='pdc', stage=stage, inj=inj)

        if whitened:
            doneplots = os.listdir(
                    '../results/whitened_diagnostic/eb_subtraction')
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound whitened_diagnostic, continuing.\n')
                continue
            if 'pw' in stage:
                irp.whitenedplot_5row(lcd, ap='sap', stage=stage)
                irp.whitenedplot_5row(lcd, ap='pdc', stage=stage)
            elif 'redtr' in stage or 'dipsearch' in stage \
                    or 'eb_sbtr' in stage:
                irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=inj)

    if inj and 'eb_sbtr' not in stage:
        irra.summarize_injrecov_result()





def get_lcd(stage='redtr', inj=None, allq=None):
    datapath = '../data/injrecov_pkl/'
    if inj:
        datapath += 'inj/'
        stage += '_inj'
    else:
        datapath += 'no_inj/'
    pklnames = [f for f in os.listdir(datapath) if stage in f]

    kicids = np.unique([pn.split('_')[0] for pn in pklnames])
    sind = np.random.randint(0, len(kicids))
    kicid = kicids[sind]

    lcd = ir.load_lightcurve_data(kicid, stage=stage)
    if 'dipsearch' in stage:
        allq = ir.load_allq_data(kicid, stage=stage)

    if isinstance(allq, dict):
        return lcd, allq
    else:
        return lcd


if __name__ == '__main__':

    ## If you just need a quick `lcd` to play with:
    #lcd = get_lcd(stage='dtr', inj=False)
    #lcd, allq = get_lcd(stage='dipsearch', inj=True)

    ## Testing out injection:
    #N_lc_fixedinjrecov(100, stage='redtr', inj=True)

    ## Testing out recovery:
    #N_lc_fixedinjrecov(100, stage='dipsearch', inj=True)

    ## Run a "bonafide test" of the injection/recovery routines sufficient to
    ## understand completeness vs various parameters (e.g., δ, P_CBP, RMS).
    #N_lc_injrecov(1000, stage='dipsearch', inj=True)

    ## Just test EB subtraction (transits are injected, but no recovery)
    test_EB_subraction(100, stage='eb_sbtr', inj=True, ds=False, whitened=True)
