import numpy as np, os
import inj_recov as ir
import inj_recov_plots as irp

######
#TODO#
######
'''
TODO
* implement Armstrong+ 2014, Sec 3.1. Box with local polynomial detrending
  (add it in astrobase, then call in find_dips)

* In "redtrending": cut out data within "0.5d regions" of gaps (defined by
    >0.5day space btwn points)
  -> nb. lcmath.find_lc_timegroups(fsaptimes, mingap=mingap) is most relevant.
  -> implement as trim_near_gaps. (nb. requires "stitching" to get full LC)

    ```
    # Drop intra-quarter and interquarter gaps in the lightcurves
    mingap = 0.5 # days
    ngroups, groups = lcmath.find_lc_timegroups(fsaptimes, mingap=mingap)

    for group in groups:
    ```

* Similarly, fine-tune the sigma clipping based on the actual RMS across a
    quarter. **If (once detrended+whitened) it's very small, we must allow
    bigger dips (w/out clipping them).**




----------

Consider:
* Iteratively whiten out more frequencies. (Say ~few more) (Nb. this depends on
  whether there are strong periodicities in current redtrended residuals, or
  whether the spot-movement is "pseudo-periodic")
* Possible to high/loss pass filter at ~<2x EB period? (Read Feigelson text & see)
* How to fit out spots: e.g., with `george` (GP regression)?
* Email Johan for his code -- more general fourier approach (more expensive
  too).

Longer-term ideas:
*Detrend+normalize:
    Match the KEBC detrending? As-is, I think I'm leaving in trends that are
    too big.
*Period finder:
    Assess for what fraction we need to revert to KEBC period.

astrokep.find_lightcurve_gaps

 ALSO:
lcmath.find_lc_timegroups -- basically an implementation of astrokep's
 find_lightcurve_gaps, already done.

astrokep.stitch_lightcurve_gaps

astrokep.keplerflux_to_keplermag

ALSO:
 PyKE is also worth assessing.
'''


def N_lc_injrecov(N,
        ors=False, whitened=True, ds=True,
        stage=None,
        inj=None):
    '''
    Run injection-recovery on N KEBC objects. There are two important objects:
    `lcd` organizes everything by quarter. `allq` stitches over all quarters.

    The current processing is as follows:

        inject a 1% transit at (20-25)x the binary period->
        detrend (n=20 legendre and [20,20]σ sigclip)->
        normalize (median by quarter)->
        run periodograms (stellingwerf_pdm)->
        get nominal EB period->
        run finer periodogram (stellingwerf_pdm narrow window)->
        select a real EB period->
        "whiten" (phase-fold [20,5]σ sigclip, fit & subtract out fit)->
        redetrend (n=20 legendre, and [20,5]σ  sigclip)->
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

        lcd = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pklmatch = [f for f in os.listdir('../data/injrecov_pkl/'+predir) if
                f.endswith('.p') and f.startswith(kicid) and stage in f]

        if len(pklmatch) < 1:

            #INJECT TRANSITS AND PROCESS LIGHTCURVES
            lcd, allq = ir.inject_transits(lcd)
            lcd = ir.detrend_allquarters(lcd,
                    σ_clip=20., legendredeg=20, inj=inj)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=[20.,5.])
            if 'pw' in stage:
                kicid = ir.save_lightcurve_data(lcd, stage=stage)
            lcd = ir.redetrend_allquarters(lcd, σ_clip=[20.,5.], legendredeg=20)
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

        #MAKE PLOTS
        if ors:
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap', stage=stage)
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc', stage=stage)

        if ds:
            doneplots = os.listdir('../results/dipsearchplot/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('Found dipsearchplot, continuing.')
                continue

            if 'dipsearch' in stage:
                irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)
                irp.dipsearchplot(lcd, allq, ap='pdc', stage=stage, inj=inj)

        if whitened:
            doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('Found whitened_diagnostic, continuing.')
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



def get_lcd(stage='redtr'):
    N = 100
    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)
    np.random.seed(seeds[0])

    lcd = ir.retrieve_random_lc()
    kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

    lcd = ir.load_lightcurve_data(kicid, stage=stage)

    return lcd


if __name__ == '__main__':

    # If you just need a quick `lcd` to play with:
    #lcd = get_lcd()

    # Testing out injection:
    #N_lc_injrecov(100, stage='redtr', inj=True)

    # Testing out recovery:
    N_lc_injrecov(100, stage='dipsearch', inj=True)
