import numpy as np, os
import inj_recov as ir
import inj_recov_plots as irp

######
#TODO#
######
'''
TODO
* Cut out data within "0.5d regions" of gaps (defined by >0.5day space btwn
  points)
  -> add "gapfind" to astrokep (e.g., from lcmath) & then drop points near gaps
  -> implement as trim_near_gaps. (nb. requires "stitching" to get full LC)
* implement boxfilter search
  (e.g., Armstrong+ 2014, Sec 3.1. Box with local polynomial detrending)
* PLOT: P<5 day cut (the actual armstrong et al statistical claim)
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

def N_lc_injrecov(N, ors=False, whitened=True, stage='redtr', inj=None):
    '''
    Run injection-recovery on N KEBC objects. This currently means:
        inject a 1% transit at 10x the binary period->
        detrend (n=20 legendre)->
        normalize (median by quarter)->
        run periodograms (stellingwerf_pdm)->
        get nominal EB period->
        run finer periodogram (stellingwerf_pdm narrow window)->
        select a real EB period->
        "whiten" (phase-fold to "real" EB period, fit, subtract out fit)->
        redetrend (n=20 legendre)->
        normalize (median by quarter)->

    stages (in order that they could be saved in):

        'pw' if post-whitening.
        'redtr' if post-redetrending.

    inj (bool): True if you're injecting (fixes names of things).
    '''
    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    if inj:
        stage += '_inj'

    for s in seeds:
        np.random.seed(s)

        lcd = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pklmatch = [f for f in os.listdir('../data/injrecov_pkl/') if
                f.endswith('.p') and f.startswith(kicid) and stage in f]

        if len(pklmatch) < 1:

            #`lcd` organizes everything by quarter. `allq` stitches.
            lcd, allq = ir.inject_transits(lcd)
            lcd = ir.detrend_allquarters(lcd,
                    σ_clip=8., legendredeg=20, inj=True)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=6.)
            #kicid = ir.save_lightcurve_data(lcd, stage='pw')
            lcd = ir.redetrend_allquarters(lcd, σ_clip=6., legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='redtr')
            kicid = ir.save_lightcurve_data(lcd, stage=stage)

        lcd = ir.load_lightcurve_data(kicid, stage=stage)

        dones = os.listdir('../results/whitened_diagnostic/')
        plotmatches = [f for f in dones if f.startswith(kicid) and
                stage in f]
        if len(plotmatches)>0:
            print('Found whitened_diagnostic, continuing.')
            continue

        if ors:
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap', stage=stage)
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc', stage=stage)

        if whitened:
            if 'pw' in stage:
                irp.whitenedplot_5row(lcd, ap='sap', stage=stage)
                irp.whitenedplot_5row(lcd, ap='pdc', stage=stage)
            elif 'redtr' in stage:
                irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=True)
                irp.whitenedplot_6row(lcd, ap='pdc', stage=stage, inj=True)


def N_lc_process(N, ors=False, whitened=True, stage='redtr'):
    '''
    Run LC processing on N KEBC objects. This currently means:
        detrend (n=20 legendre)->
        normalize (median by quarter)->
        run periodograms (stellingwerf_pdm)->
        get nominal EB period->
        run finer periodogram (stellingwerf_pdm narrow window)->
        select a real EB period->
        "whiten" (phase-fold to "real" EB period, fit, subtract out fit)->
        redetrend (n=20 legendre)->
        normalize (median by quarter)->

    stages (in order that they could be saved in):


        'pw' if post-whitening.
        'redtr' if post-redetrending.
    '''
    #np.random.seed(42)
    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)
    for s in seeds:
        np.random.seed(s)

        lcd = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pwdumpmatch = [f for f in os.listdir('../data/injrecov_pkl/') if
                f.endswith('.p') and f.startswith(kicid)]

        if len(pwdumpmatch) < 1:

            lcd = ir.detrend_allquarters(lcd, σ_clip=8., legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=6.)
            #kicid = ir.save_lightcurve_data(lcd, stage='pw')
            lcd = ir.redetrend_allquarters(lcd, σ_clip=6., legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='redtr')
            kicid = ir.save_lightcurve_data(lcd, stage=stage)

        #lcd = ir.load_lightcurve_data(kicid, stage='pw')
        lcd = ir.load_lightcurve_data(kicid, stage=stage)

        dones = os.listdir('../results/whitened_diagnostic/')
        matches = [f for f in dones if f.startswith(kicid)]
        if len(matches)>0:
            print('Found whitened_diagnostic, continuing.')
            continue

        if ors:
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
            irp.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

        if whitened:
            if stage == 'pw':
                irp.whitenedplot_5row(lcd, ap='sap')
                irp.whitenedplot_5row(lcd, ap='pdc')
            elif stage == 'redtr':
                irp.whitenedplot_6row(lcd, ap='sap')
                irp.whitenedplot_6row(lcd, ap='pdc')


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

    #N_lc_check(100, stage='redtr')

    # If you just need a quick `lcd` to play with:
    #lcd = get_lcd()

    # Testing out injection:
    N_lc_injrecov(100, stage='redtr', inj=True)
